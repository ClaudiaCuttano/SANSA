r""" LVIS-92i few-shot semantic segmentation dataset """
import os
import pickle

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from datasets.transform_utils import polygons_to_bitmask
import pycocotools.mask as mask_util


class DatasetLVIS(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 10
        self.benchmark = 'lvis'
        self.shot = shot
        self.anno_path = os.path.join(datapath, "LVIS")
        self.base_path = os.path.join(datapath, "LVIS", 'coco')
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize

        self.nclass, self.class_ids_ori, self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.class_ids_c = {cid: i for i, cid in enumerate(self.class_ids_ori)}
        self.class_ids = sorted(list(self.class_ids_c.values()))

        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 2300

    def __getitem__(self, idx):
        idx %= len(self.class_ids)

        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame(idx)

        query_img = self.transform(query_img)
        query_mask = query_mask.float()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
        support_masks = torch.stack(support_masks)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
        }
        if not self.split == 'trn':
            batch['class_id'] = torch.tensor(self.class_ids_c[class_sample])
            batch['class_sample'] = torch.tensor(class_sample)
            
        return batch

    def build_img_metadata_classwise(self):

        with open(os.path.join(self.anno_path, 'lvis_train.pkl'), 'rb') as f:
            train_anno = pickle.load(f)
        with open(os.path.join(self.anno_path, 'lvis_val.pkl'), 'rb') as f:
            val_anno = pickle.load(f)

        # below: keep only classes with > 1 sample
        train_cat_ids = [i for i in list(train_anno.keys()) if len(train_anno[i]) > self.shot]
        val_cat_ids = [i for i in list(val_anno.keys()) if len(val_anno[i]) > self.shot]

        trn_nclass = len(train_cat_ids)
        val_nclass = len(val_cat_ids)

        nclass_val_spilt = val_nclass // self.nfolds

        if self.fold != -1:
            class_ids_val = [val_cat_ids[self.fold + self.nfolds * v] for v in range(nclass_val_spilt)]
            class_ids_trn = [x for x in train_cat_ids if x not in class_ids_val]
        else:
            class_ids_val = [val_cat_ids[0 + self.nfolds * v] for v in range(nclass_val_spilt)]
            class_ids_trn = train_cat_ids
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val
        nclass = trn_nclass if self.split == 'trn' else val_nclass
        img_metadata_classwise = train_anno if self.split == 'trn' else val_anno

        return nclass, class_ids, img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata.extend(list(self.img_metadata_classwise[k].keys()))
        return sorted(list(set(img_metadata)))

    def get_mask(self, segm, image_size):

        if isinstance(segm, list):
            # polygon
            # polygons = [np.asarray(p).reshape(-1, 2)[:,::-1] for p in segm]
            # polygons = [p.reshape(-1) for p in polygons]
            polygons = [np.asarray(p) for p in segm]
            mask = polygons_to_bitmask(polygons, *image_size[::-1])
        elif isinstance(segm, dict):
            # COCO RLE
            mask = mask_util.decode(segm)
        elif isinstance(segm, np.ndarray):
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            mask = segm
        else:
            raise NotImplementedError

        return torch.tensor(mask)

    def load_frame(self, idx):

        class_sample = self.class_ids_ori[idx]

        # class_sample = np.random.choice(self.class_ids_ori, 1, replace=False)[0]
        query_name = np.random.choice(list(self.img_metadata_classwise[class_sample].keys()), 1, replace=False)[0]
        query_info = self.img_metadata_classwise[class_sample][query_name]
        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        org_qry_imsize = query_img.size
        query_annos = query_info['annotations']
        segms = []

        for anno in query_annos:
            segms.append(self.get_mask(anno['segmentation'], org_qry_imsize)[None, ...].float())
        query_mask = torch.cat(segms, dim=0)
        query_mask = query_mask.sum(0) > 0

        support_names = []
        support_pre_masks = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(list(self.img_metadata_classwise[class_sample].keys()), 1, replace=False)[0]
            if query_name != support_name:
                support_names.append(support_name)
                support_info = self.img_metadata_classwise[class_sample][support_name]
                support_annos = support_info['annotations']

                support_segms = []
                for anno in support_annos:
                    support_segms.append(anno['segmentation'])
                support_pre_masks.append(support_segms)

            if len(support_names) == self.shot:
                break


        support_imgs = []
        support_masks = []
        for support_name, support_pre_mask in zip(support_names, support_pre_masks):
            support_img = Image.open(os.path.join(self.base_path, support_name)).convert('RGB')
            support_imgs.append(support_img)
            org_sup_imsize = support_img.size
            sup_masks = []
            for pre_mask in support_pre_mask:
                sup_masks.append(self.get_mask(pre_mask, org_sup_imsize)[None, ...].float())
            support_mask = torch.cat(sup_masks, dim=0)
            support_mask = support_mask.sum(0) > 0

            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize


def build(image_set, args):
    from torchvision import transforms
    img_size = 640
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = DatasetLVIS(datapath=args.data_root, fold=args.fold, transform=transform,
                 shot=args.shots, use_original_imgsize=False, split=image_set)

    return dataset
