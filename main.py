import argparse
import copy
import json
from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from util.commons import resume_from_checkpoint, adapter_state_dict
import datasets.samplers as samplers
from datasets import build_dataset
from models.sansa.sansa import build_sansa
from inference_fss import eval_fss
from engine import train_one_epoch
import opts
from util.commons import setup_logging, make_deterministic


def main(args):
    utils.init_distributed_mode(args)
    rank = utils.get_rank()
    setup_logging(save_dir=args.output_dir, console="info", rank=rank)
    make_deterministic(args.seed+rank) # fix the seed for reproducibility

    print(args)

    device = torch.device(args.device)
    model = build_sansa(args.sam2_version, args.adaptformer_stages, args.channel_factor, args.device)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    n_parameters_tot = sum(p.numel() for p in model.parameters())
    print(f'number of params: {n_parameters_tot}')

    head, fix = [], []
    for k, v in model_without_ddp.named_parameters():
        (head if v.requires_grad else fix).append(v)

    print(f'Trainable parameters: {sum(p.numel() for p in head)}')
    print(f'Parameters fixed: {sum(p.numel() for p in fix)}')

    param_list = [{'params': head, 'initial_lr': args.lr}]
    optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), fused=True)

    cfg = copy.deepcopy(args)
    cfg.shots = args.J
    dataset_train = build_dataset(args.dataset_file, image_set='train', args=cfg)

    args.batch_size = int(args.batch_size / args.ngpu)
    if args.distributed:
        sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(data_loader_train)
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        model_without_ddp, optimizer, lr_scheduler = resume_from_checkpoint(args.resume, model_without_ddp, optimizer, lr_scheduler, args)
        
    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
                model, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, lr_scheduler=lr_scheduler, args=args)

        if args.output_dir:
            print("Save Model")
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': adapter_state_dict(model_without_ddp),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

            print(f"Start validation")
            eval_fss(model, args)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters_tot}

        print(json.dumps(log_stats))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SANSA training', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.name_exp)

    main(args)

