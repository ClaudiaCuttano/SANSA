import argparse

def get_args_parser() -> argparse.ArgumentParser:
    """Argument parser for SANSA training and inference (parent parser)."""
    parser = argparse.ArgumentParser("SANSA training and inference", add_help=False)

    # General
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device.")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from.")

    # Experiment I/O
    parser.add_argument("--output_dir", type=str, default="output", help="Root directory for outputs.")
    parser.add_argument("--name_exp", type=str, default="prova", help="Experiment name (subfolder in output_dir).")

    # Data
    parser.add_argument("--data_root", type=str, default="data", help="Root directory for datasets.")
    parser.add_argument("--dataset_file", type=str, default="coco", choices=["coco", "lvis", "fss", "pascal_voc", "pascal_voc_cd", "pascal_part", "paco_part", "deepglobe", "isic",
                 "lung", "ade20k", "multi"], help="Dataset name. Use 'multi' for training the generalist model.")
    parser.add_argument("--multi_train", nargs="+", type=str, default=["lvis", "coco", "ade20k", "paco_part"], help="Datasets to mix when dataset_file='multi'.")
    parser.add_argument("--ds_weight", nargs="+", type=float, default=[0.4, 0.45, 0.1, 0.05], help="Sampling weights for datasets in --multi_train.")
    parser.add_argument("--num_workers", type=int, default=6, help="DataLoader workers.")

    # Prompting / Shots / Folds
    parser.add_argument("--prompt", type=str, default="mask", choices=["mask", "scribble", "box", "point", "multi"], help="Prompt type for support frames; 'multi' samples a type at random.")
    parser.add_argument("--shots", type=int, default=1, help="Number of support frames per episode.")
    parser.add_argument("--J", type=int, default=3, help="Number of unlabeled target images per training episode.")
    parser.add_argument("--fold", type=int, default=0, help="Training fold (if applicable).")

    # SAM2 / Backbone
    parser.add_argument("--sam2_version", type=str, default="large", choices=["tiny", "base", "large"], help="Version of SAM2 image encoder.")
    parser.add_argument("--adaptformer_stages", nargs="+", type=int, default=[-1], help="Adapter/AdaptFormer stages to enable (model-specific).")
    parser.add_argument("--channel_factor", type=float, default=0.3, help="Adapter channel scaling factor (model-specific).")

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--start_epoch', default=0, type=int, help="Epoch to start training from (for resuming)")
    parser.add_argument("--clip_max_norm", type=float, default=0.1, help="Gradient clipping max norm (0 disables clipping).")
    parser.add_argument("--batch_size", type=int, default=2, help="Global batch size (may be split per GPU).")

    # Logging / Runtime
    parser.add_argument("--no_distributed", action="store_true", default=False, help="Force single-process training.")

    # Inference
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold to binarize masks at eval.")
    parser.add_argument("--visualize", action="store_true", default=False, help="Save qualitative results.")

    return parser
