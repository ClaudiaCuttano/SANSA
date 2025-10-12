from typing import List, Union, Optional, Dict
import os
from os.path import join
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets.transform_utils import denormalize

def visualize_episode(
    support_imgs: List[torch.Tensor],                  # K * [3,H,W] tensors (display size, e.g., 640×640)
    query_img: torch.Tensor,                           # [3,H,W]
    query_gt: Union[np.ndarray, torch.Tensor],         # [H,W]
    query_pred: Union[np.ndarray, torch.Tensor],       # [H,W]
    prompt_dict: Dict,                                 # prompts for supports (in src_size coords)
    out_dir: str,
    idx: int,
    src_size: Union[int, tuple],                       # e.g., model.sam.image_size (int or (H,W))
    iou: Optional[float] = None,
) -> str:
    """
    Save: [Support 1 (+prompt)] ... [Support K (+prompt)] [GT] [Prediction]
    Prompts are assumed in src_size space (e.g., 1024×1024). They are rescaled to each
    support image display size before rendering.
    """
    os.makedirs(join(out_dir, "vis"), exist_ok=True)

    # ------------ helpers ------------
    # overlay color and alpha
    OVERLAY_RGBA = (30/255.0, 144/255.0, 255/255.0, 0.6)

    SRC_H, SRC_W = src_size, src_size

    def _to_img_np(x):
        x = denormalize(x)
        x = x.detach().clamp(0.0, 1.0).cpu()  # clamp after de-norm
        x = x.permute(1, 2, 0)  # [H,W,C]
        return (x * 255.0 + 0.5).to(torch.uint8).numpy()

    def _to_bool_np(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim == 3 and x.shape[0] == 1:  # [1,H,W]
            x = x[0]
        if x.ndim == 3 and x.shape[-1] == 1: # [H,W,1]
            x = x[..., 0]
        return (x > 0).astype(np.uint8)

    def _scale_coords(coords, dst_h, dst_w):
        c = np.asarray(coords, dtype=np.float32)
        sx, sy = float(dst_w) / float(SRC_W), float(dst_h) / float(SRC_H)
        c[..., 0] *= sx
        c[..., 1] *= sy
        return c

    def _resize_mask(mask, dst_h, dst_w):
        m = mask.detach().float()
        out = F.interpolate(m, size=(dst_h, dst_w), mode="nearest")
        return (out[0, 0] > 0).cpu().numpy().astype(np.float32)


    def draw_mask(ax, img, mask, title):
        ax.imshow(img)
        h, w = mask.shape[:2]
        overlay = np.ones((h, w, 4), dtype=np.float32)
        overlay[..., 0:4] = OVERLAY_RGBA
        ax.imshow(overlay, alpha=mask.astype(float))
        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.axis("off")

    def draw_points(ax, img, coords, title):
        ax.imshow(img); ax.axis("off"); ax.set_title(title, fontsize=18, fontweight="bold")
        c = np.asarray(coords)[0]
        ax.scatter(c[:,0], c[:,1], c='lime', marker='*', s=250, edgecolor='white', linewidth=1.2)

    def draw_boxes(ax, img, boxes_xyxy, title):
        ax.imshow(img); ax.axis("off"); ax.set_title(title, fontsize=18, fontweight="bold")
        for b in boxes_xyxy:
            x0, y0, x1, y1 = map(float, b)
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='lime', facecolor=(0,0,0,0), lw=2))

    # pick batch key (assume 0; fallback to first int key)
    batch_keys = [k for k in prompt_dict.keys() if isinstance(k, int)]
    batch_idx = 0 if 0 in batch_keys else (batch_keys[0] if batch_keys else None)

    # convert images
    K = len(support_imgs)
    s_imgs = [_to_img_np(im) for im in support_imgs]
    q_img  = _to_img_np(query_img)
    q_gt   = _to_bool_np(query_gt)
    q_pr   = _to_bool_np(query_pred)

    # canvas
    cols = K + 2
    fig_w = max(3.0 * cols, 9.0)
    fig, axes = plt.subplots(1, cols, figsize=(fig_w, 3.6), dpi=150, constrained_layout=True)
    axes = np.asarray([axes] if cols == 1 else axes).reshape(cols)

    # draw supports + prompts
    for i in range(K):
        ax = axes[i]
        img = s_imgs[i]
        H_disp, W_disp = img.shape[:2]
        spec = prompt_dict.get(batch_idx, {}).get(i, None)

        ptype = spec['prompt_type']
        pd = spec.get('prompt', {})

        if ptype == 'mask':
            m = pd if not isinstance(pd, dict) else pd.get('mask', pd.get('m', pd))
            m = _resize_mask(m, H_disp, W_disp)
            draw_mask(ax, img, m, f"Support {i+1} (mask)")

        elif ptype == 'box':
            coords = pd.get('point_coords', None)
            c = coords.detach().cpu().numpy() if isinstance(coords, torch.Tensor) else np.asarray(coords, dtype=np.float32)
            c = _scale_coords(c.copy(), H_disp, W_disp)
            boxes = np.stack([c[:,0,0], c[:,0,1], c[:,1,0], c[:,1,1]], axis=1)
            draw_boxes(ax, img, boxes, f"Support {i+1} (box)")

        elif ptype in ('point', 'scribble'):
            coords = pd.get('point_coords', None)
            c = coords.detach().cpu().numpy() if isinstance(coords, torch.Tensor) else np.asarray(coords, dtype=np.float32)
            c = _scale_coords(c.copy(), H_disp, W_disp)
            draw_points(ax, img, c, f"Support {i+1} ({ptype})")

    # GT & Prediction
    draw_mask(axes[K],     q_img, q_gt, "GT")
    draw_mask(axes[K + 1], q_img, q_pr, "Prediction")

    if iou is not None:
        fig.suptitle(f"IoU: {iou:.2f}", fontsize=16, y=1.1)

    out_path = join(out_dir, "vis", f"{idx}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
