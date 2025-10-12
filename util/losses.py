from typing import Dict

import torch
import torch.nn.functional as F
import einops


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_boxes: int) -> torch.Tensor:
    """
    Compute the Dice loss for binary masks.

    Args:
        inputs (Tensor): raw logits, shape [N, ...].
        targets (Tensor): binary masks with same shape as inputs.
        num_boxes (int): normalization factor (usually batch size).

    Returns:
        Tensor: scalar Dice loss.
    """
    inputs = inputs.sigmoid().flatten(1).float()
    targets = targets.flatten(1).float()

    numerator   = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Compute the sigmoid focal loss (used in RetinaNet).

    Args:
        inputs (Tensor): raw logits, shape [N, ...].
        targets (Tensor): binary masks with same shape as inputs.
        num_boxes (int): normalization factor (usually batch size).
        alpha (float): class balancing factor (0–1). Default: 0.25.
        gamma (float): focusing parameter. Default: 2.0.

    Returns:
        Tensor: scalar focal loss.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def loss_masks(
    outputs: torch.Tensor,
    masks: torch.Tensor,
    num_frames: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute focal and dice loss on predicted masks.

    Args:
        outputs (Tensor): B*T, h, W.
        masks (Tensor): B, T, H, W (binary 0/1).
        num_frames (int): number of last frames to include (T → all, T-1 → skip first).

    Returns:
        dict: {"loss_mask": Tensor, "loss_dice": Tensor}
    """
    bs, T = masks.shape[:2]
    start = max(0, T - num_frames)

    src = einops.rearrange(outputs, '(b t) h w -> b t h w', b=bs)[:, start:T]
    tgt = masks[:, start:]

    # flatten to [B, F*H*W]
    tgt = tgt.to(src)
    src = src.flatten(1).to(torch.float32)
    tgt = tgt.flatten(1).to(src.dtype)

    # drop NaN rows (if any)
    keep = ~torch.isnan(src).any(dim=1)
    if not keep.all():
        src, tgt = src[keep], tgt[keep]
        bs = int(keep.sum())

    return {
        "loss_mask": sigmoid_focal_loss(src, tgt, bs),
        "loss_dice":  dice_loss(src, tgt, bs),
    }

