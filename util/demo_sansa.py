# util/demo_sansa.py
# Helpers for SANSA demos (points/box/scribble/mask)
# For Jupyter/Colab: works with %matplotlib inline (no ipympl requirement)

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import base64
import io
import os
from string import Template

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import HTML, JSON as IPyJSON, display
from PIL import Image
from torchvision import transforms

# Colab callback (if available)
try:
    from google.colab import output  # for kernel.invokeFunction
except Exception:  # not in Colab
    output = None  # type: ignore

# -------------------- Layout (consistent margins) --------------------
_SUBPLOT_KW = dict(left=0.06, right=0.98, bottom=0.08, top=0.90, wspace=0.10)

# Keep at most one live figure per demo type to avoid piling up figures
_LIVE_FIGS = {}

def _two_panel_fig(fig_size: float, key: str):
    """Create a 1x2 figure; close the previous one with the same key if it exists."""
    old = _LIVE_FIGS.get(key)
    try:
        # Close only our previous figure (if still open)
        if old is not None and plt.fignum_exists(old.number):
            plt.close(old)
    except Exception:
        pass
    fig, axes = plt.subplots(1, 2, figsize=(fig_size * 2, fig_size))
    _LIVE_FIGS[key] = fig
    return fig, axes


# -------------------- Types --------------------
Points = List[Tuple[int, int]]

@dataclass
class DemoCtx:
    model: Any
    reference_img: Image.Image
    target_img: Image.Image
    device: torch.device
    prompt_type: str      # 'point'|'box'|'scribble'|'mask'
    img_size: int = 640

# -------------------- Viz helpers --------------------
def show_mask(mask, ax, random_color=False, borders=True, reference=False):
    # Predicted masks: RED; Prompts/reference masks: GREEN
    if reference:
        color = np.array([0/255, 170/255, 0/255, 0.6])  # green for prompts/reference
    elif random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 0/255, 0/255, 0.6])  # red for predictions
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, ax, marker_size=110):
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    if coords.size == 0: return
    x, y = coords[:, 0], coords[:, 1]
    # White halo + GREEN fill for prompts
    ax.scatter(x, y, s=marker_size * 1.6, facecolors='none', edgecolors='white', linewidths=2.0, zorder=3)
    ax.scatter(x, y, s=marker_size, c='#00aa00', marker='o', edgecolors='white', linewidths=0.8, zorder=4)

def show_box(box, ax):
    b = np.asarray(box, dtype=np.float32).reshape(-1, 4)
    for bb in b:
        x0, y0, x1, y1 = bb.tolist()
        w, h = x1 - x0, y1 - y0
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='#00aa00',
                                   facecolor=(0, 0, 0, 0), lw=2))  # GREEN

def show_scribble_points(coords, ax, line=True, marker_size=60, linewidth=2.0):
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    if coords.size == 0: return
    x, y = coords[:, 0], coords[:, 1]
    if line and coords.shape[0] >= 2:
        ax.plot(x, y, color='#00aa00', linewidth=linewidth, alpha=0.95)  # GREEN
    ax.scatter(x, y, s=marker_size, c='#00aa00', marker='o',
               edgecolor='white', linewidth=0.8, alpha=0.95)  # GREEN

def generate_scribble(start, end, n_points=10, k=120.0, s_curve=True):
    start = np.asarray(start, dtype=np.float32); end = np.asarray(end, dtype=np.float32)
    v = end - start; norm = np.linalg.norm(v) + 1e-6
    perp = np.array([-v[1], v[0]], dtype=np.float32) / norm
    p0, p3 = start, end; sign2 = -1.0 if s_curve else 1.0
    p1 = start + 0.25 * v + k * perp
    p2 = start + 0.75 * v + sign2 * k * perp
    t = np.linspace(0.0, 1.0, int(n_points), dtype=np.float32)[:, None]
    pts = ((1 - t)**3)*p0 + 3*((1 - t)**2)*t*p1 + 3*(1 - t)*(t**2)*p2 + (t**3)*p3
    return pts.astype(np.float32)

# -------------------- SANSA helpers --------------------
img_size = 640
_transform = transforms.Compose([
    transforms.Resize(size=(img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def pseudo_video_from_pil(sup_img: Image.Image, q_img: Image.Image, device: torch.device = torch.device('cuda')):
    sup_t, q_t = _transform(sup_img), _transform(q_img)
    video = torch.stack([sup_t, q_t], dim=0)      # [2, 3, H, W]
    return video[None].to(device)                 # [1, 2, 3, H, W]

def format_prompt(n_shots: int, prompt_input, prompt_type: str, device: torch.device = torch.device('cuda')):
    assert prompt_type in ['mask', 'point', 'box', 'scribble']
    prompt_dict = {0: {}, 'shots': n_shots}
    if prompt_type in ['point', 'scribble']:
        pts = torch.as_tensor(prompt_input, dtype=torch.float32, device=device).view(-1, 2)
        prompt_d = {'point_coords': pts.view(1, -1, 2),
                    'point_labels': torch.ones(1, pts.shape[0], dtype=torch.int32, device=device)}
    elif prompt_type == 'box':
        b = torch.as_tensor(prompt_input, dtype=torch.float32, device=device).view(-1, 4)
        x0y0 = torch.minimum(b[:, :2], b[:, 2:])
        x1y1 = torch.maximum(b[:, :2], b[:, 2:])
        point_coords = torch.stack([x0y0, x1y1], dim=1).view(1, -1, 2)
        n = point_coords.shape[1] // 2
        point_labels = torch.tensor([2, 3], dtype=torch.int32, device=device).repeat(1, n)
        prompt_d = {'point_coords': point_coords, 'point_labels': point_labels}
    else:  # 'mask'
        prompt_d = prompt_input
    prompt_dict[0][0] = {'prompt_type': prompt_type, 'prompt': prompt_d}
    return prompt_dict

def predict(model, reference_img, target_img, prompt_input, device, prompt_type: str):
    """Unified forward -> returns np.uint8 mask [H,W] in {0,1}."""
    if prompt_type == 'point' and (prompt_input is None or len(prompt_input) == 0):
        return None
    if prompt_type == 'box' and (prompt_input is None or np.asarray(prompt_input).size != 4):
        return None
    if prompt_type == 'mask' and isinstance(prompt_input, np.ndarray):
        m = (prompt_input > 0.5).astype(np.float32)
        prompt_input = torch.from_numpy(m[None, None, ...]).to(device)

    clip = pseudo_video_from_pil(reference_img, target_img, device=device)
    prompt = format_prompt(n_shots=1, prompt_input=prompt_input, prompt_type=prompt_type, device=device)
    with torch.inference_mode():
        out = model(clip, prompt)
        target_prob = out['pred_masks'][1].sigmoid()
    return (target_prob.detach().cpu().numpy() > 0.5).astype(np.uint8)

def load_ref_mask(reference_path: str, class_name: str, size: int = img_size):
    """Load a binary mask from '<ref folder>/mask_<class_name>.png' at the demo resolution."""
    mask_path = os.path.join(os.path.dirname(reference_path), f"mask_{class_name}.png")
    m = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
    return (np.array(m) > 0).astype(np.uint8)

# ---- tiny helper to call predict from a single context ----
def predict_with_ctx(ctx: DemoCtx, prompt_input):
    return predict(ctx.model, ctx.reference_img, ctx.target_img, prompt_input, ctx.device, ctx.prompt_type)

# -------------------- Renderers --------------------
def _as_numpy_rgb(img, size_hw: Tuple[int, int] = None):
    """PIL or np -> uint8 HxWx3; optional resize to (H, W)."""
    if isinstance(img, Image.Image):
        if size_hw is not None:
            img = img.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
        arr = np.array(img)
        if arr.ndim == 2: arr = np.repeat(arr[..., None], 3, axis=2)
        return arr
    arr = np.asarray(img)
    if arr.ndim == 2: arr = np.repeat(arr[..., None], 3, axis=2)
    if size_hw is not None and (arr.shape[0] != size_hw[0] or arr.shape[1] != size_hw[1]):
        arr = cv2.resize(arr, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_LINEAR)
    return arr

def _binarize_np_mask(mask_like, size_hw: Tuple[int, int] = None):
    if mask_like is None: return None
    m = np.asarray(mask_like)
    if m.ndim > 2: m = np.squeeze(m)
    if size_hw is not None and (m.shape[0] != size_hw[0] or m.shape[1] != size_hw[1]):
        m = cv2.resize(m, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_NEAREST)
    if m.dtype != np.uint8: m = (m > 0.5).astype(np.uint8)
    else: m = (m > 0).astype(np.uint8)
    return m

def show_ref_tgt(
    reference_img,
    target_img,
    prompt_input=None,
    mask=None,
    prompt_type: str = None,
    img_size_for_mask: int = img_size,
    preview_grid_step: int = 80,
    is_manual: bool = False
):
    # Reuse/replace a live figure per category ("preview", "point", "box", "scribble", "mask")
    key = f"show::{(prompt_type or 'preview').lower()}"
    # Our helper expects a base size; previous figsize=(11,5) => per-panel base ≈ 5.5
    fig, axes = _two_panel_fig(fig_size=5.5, key=key)
    fig.subplots_adjust(**_SUBPLOT_KW)
    ax_ref, ax_tgt = axes[0], axes[1]

    # Draw base images
    ax_ref.clear(); ax_tgt.clear()
    ax_ref.imshow(reference_img); ax_ref.axis("off")
    ax_tgt.imshow(target_img);    ax_tgt.axis("off")

    is_preview_only = (prompt_type is None and prompt_input is None and mask is None)
    if is_preview_only:
        ax_ref.set_title("Reference Image (Preview)")
        # Light preview grid helps users estimate coordinates
        ax_ref.set_xticks(np.arange(0, img_size_for_mask + 1, preview_grid_step))
        ax_ref.set_yticks(np.arange(0, img_size_for_mask + 1, preview_grid_step))
        ax_ref.grid(True, linewidth=0.4, alpha=0.4)
        ax_tgt.set_title("Target Image (Preview)")
        plt.show()
        return

    # Overlay prompt on reference (GREEN)
    pt = (prompt_type or "").lower()
    if pt == "point":
        show_points(prompt_input, ax_ref, marker_size=110)
        ax_ref.set_title("Reference (+point)")
    elif pt == "box":
        show_box(prompt_input, ax_ref)
        ax_ref.set_title("Reference (+box)")
    elif pt == "scribble":
        show_scribble_points(prompt_input, ax_ref, line=True, marker_size=50, linewidth=2.0)
        ax_ref.set_title("Reference (+scribble)")
    elif pt == "mask":
        m_ref = _binarize_np_mask(prompt_input, (img_size_for_mask, img_size_for_mask))
        if m_ref is not None:
            show_mask(m_ref, ax_ref, borders=True, reference=True)  # GREEN
        ax_ref.set_title("Reference (+mask)")
    else:
        ax_ref.set_title("Reference")
    if is_manual:
        ax_ref.axis("on")

    # Overlay predicted mask on target (RED)
    m = _binarize_np_mask(mask, (img_size_for_mask, img_size_for_mask))
    if m is not None:
        show_mask(m, ax_tgt, borders=True, reference=False)  # RED
    ax_tgt.set_title("Target (prediction)")

    plt.show()
    return


def render_ref_tgt_with_prompt(
    ax_ref, ax_tgt,
    reference_img, target_img,
    prompt_input=None, pred_mask=None,
    prompt_type: str = None,                # None => pure preview
    img_size_for_mask: int = img_size,
    show_grid: bool = False,
    ref_title: str = None,
    tgt_title: str = None,
):
    """Draw on existing axes (resizes for display) with consistent margins upstream."""
    H = W = img_size_for_mask
    ref_vis = _as_numpy_rgb(reference_img, (H, W))
    tgt_vis = _as_numpy_rgb(target_img,    (H, W))

    ax_ref.clear(); ax_tgt.clear()
    ax_ref.imshow(ref_vis); ax_ref.axis("off")
    ax_tgt.imshow(tgt_vis); ax_tgt.axis("off")

    if prompt_type and (prompt_input is not None):
        pt = prompt_type.lower()
        if pt == "point":
            show_points(prompt_input, ax_ref, marker_size=110)  # GREEN
        elif pt == "scribble":
            show_scribble_points(prompt_input, ax_ref, line=True, marker_size=50, linewidth=2.0)  # GREEN
        elif pt == "box":
            show_box(prompt_input, ax_ref)  # GREEN
        elif pt == "mask":
            m = _binarize_np_mask(prompt_input, (H, W))
            if m is not None: show_mask(m, ax_ref, borders=True, reference=True)  # GREEN

    if pred_mask is not None:
        m = _binarize_np_mask(pred_mask, (H, W))
        if m is not None: show_mask(m, ax_tgt, borders=True, reference=False)  # RED

    if ref_title is None or tgt_title is None:
        if prompt_type and (prompt_input is not None or pred_mask is not None):
            tag = {"point": "point", "box": "box", "scribble": "scribble", "mask": "mask"}.get((prompt_type or "").lower(), "prompt")
            ref_title = ref_title or f"Reference (+{tag})"
            tgt_title = tgt_title or "Target (prediction)"
        else:
            ref_title = ref_title or "Reference Image (Preview)"
            tgt_title = tgt_title or "Target Image (Preview)"

    ax_ref.set_title(ref_title)
    ax_tgt.set_title(tgt_title)

# -------------------- Interactive utils --------------------
def _clamp_xy(x: float, y: float, size: int = 640) -> Tuple[int, int]:
    x = int(np.clip(round(x), 0, size - 1))
    y = int(np.clip(round(y), 0, size - 1))
    return x, y

# -------------------- Interactive: Points --------------------
def start_point_picker_live_predict(
    ctx: DemoCtx,
    store: Optional[Points],
    fig_size: float = 5.5,
    echo_clicks: bool = True,
    show_count_in_title: bool = True,
):
    """
    Two-pane figure. Left = clickable reference; Right = live prediction.
    Controls: left-click=add, U=undo, C=clear
    """
    if store is None:
        store = []

    fig, axes = _two_panel_fig(fig_size, key="points")
    fig.subplots_adjust(**_SUBPLOT_KW)
    ax_ref, ax_tgt = axes[0], axes[1]

    def _echo(msg: str):
        if echo_clicks: print(msg, flush=True)

    def _titles():
        t_ref = "Reference — click to add points (U undo, C clear)"
        if show_count_in_title: t_ref += f" — {len(store)}"
        return t_ref, "Target (prediction)"

    def _refresh():
        mask = predict_with_ctx(ctx, np.array(store, dtype=np.float32)) if len(store) > 0 else None
        tr, tt = _titles()
        render_ref_tgt_with_prompt(
            ax_ref, ax_tgt, ctx.reference_img, ctx.target_img,
            prompt_input=(np.array(store, dtype=np.float32) if len(store) > 0 else None),
            pred_mask=mask, prompt_type="point", img_size_for_mask=ctx.img_size,
            show_grid=False, ref_title=tr, tgt_title=tt,
        )
        fig.canvas.draw_idle()

    def _onclick(event):
        if event.inaxes is not ax_ref or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        x, y = _clamp_xy(event.xdata, event.ydata, ctx.img_size)
        store.append((x, y)); _echo(f"+ ({x}, {y})"); _refresh()

    def _onkey(event):
        k = (event.key or "").lower()
        if k == "u" and store:
            px, py = store.pop(); _echo(f"↩ undo ({px}, {py})"); _refresh()
        elif k == "c" and store:
            store.clear(); _echo("✂ clear"); _refresh()

    fig.canvas.mpl_connect("button_press_event", _onclick)
    fig.canvas.mpl_connect("key_press_event",   _onkey)
    _refresh()
    plt.show()
    return

# --------------------- BOX ----------------------------
def start_box_picker_live_predict(
    ctx: DemoCtx,
    store,                      # mutable list: [x0, y0, x1, y1]
    fig_size: float = 5.5,
    echo_clicks: bool = True,
    show_count_in_title: bool = True,
    min_span: int = 2,          # ignore tiny drags
):
    """
    Jupyter:
      - drag to draw rectangle (green rubber-band while dragging)
      - predict ON RELEASE
      - U or C = clear
    """
    import matplotlib.patches as patches

    if store is None: store = []

    fig, axes = _two_panel_fig(fig_size, key="box")

    fig.subplots_adjust(**_SUBPLOT_KW)
    ax_ref, ax_tgt = axes[0], axes[1]

    dragging = {"active": False, "p0": None, "p1": None}
    box_patch = None

    def _echo(m):
        if echo_clicks: print(m, flush=True)

    def _normalize_xyxy(p0, p1):
        x0, y0 = p0; x1, y1 = p1
        x0, x1 = sorted([x0, x1]); y0, y1 = sorted([y0, y1])
        x0 = float(np.clip(x0, 0, ctx.img_size - 1))
        x1 = float(np.clip(x1, 0, ctx.img_size - 1))
        y0 = float(np.clip(y0, 0, ctx.img_size - 1))
        y1 = float(np.clip(y1, 0, ctx.img_size - 1))
        return np.array([x0, y0, x1, y1], dtype=np.float32)

    def _titles(n=0, recording=False):
        left = "Reference — drag a box (release to predict)"
        if recording: left = "Reference (recording…) release to predict"
        if show_count_in_title: left += f" — {n}"
        return left, "Target (prediction)"

    def _clear_preview():
        nonlocal box_patch
        if box_patch is not None:
            try: box_patch.remove()
            except Exception: pass
            box_patch = None

    def _draw_preview(p0, p1):
        nonlocal box_patch
        _clear_preview()
        x0, y0 = p0; x1, y1 = p1
        x = min(x0, x1); y = min(y0, y1)
        w = abs(x1 - x0); h = abs(y1 - y0)
        box_patch = patches.Rectangle((x, y), w, h, fill=False, ec="#00aa00", lw=2)  # GREEN
        ax_ref.add_patch(box_patch)
        ax_ref.set_title(_titles(recording=True)[0]); ax_tgt.set_title(_titles()[1])
        fig.canvas.draw_idle()

    def _render_final(box_xyxy):
        store.clear(); store.extend(box_xyxy.tolist())
        mask = predict_with_ctx(ctx, np.asarray(store, dtype=np.float32))
        left_title, right_title = _titles(n=1 if len(store)==4 else 0, recording=False)
        render_ref_tgt_with_prompt(
            ax_ref, ax_tgt, ctx.reference_img, ctx.target_img,
            prompt_input=np.asarray(store, dtype=np.float32),
            pred_mask=mask, prompt_type="box", img_size_for_mask=ctx.img_size,
            show_grid=False, ref_title=left_title, tgt_title=right_title,
        )
        fig.canvas.draw_idle()

    def _on_press(event):
        if event.inaxes is not ax_ref or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        dragging["active"] = True
        dragging["p0"] = _clamp_xy(event.xdata, event.ydata, ctx.img_size)
        dragging["p1"] = dragging["p0"]
        _draw_preview(dragging["p0"], dragging["p1"])

    def _on_move(event):
        if not dragging["active"] or event.inaxes is not ax_ref or event.xdata is None or event.ydata is None:
            return
        dragging["p1"] = _clamp_xy(event.xdata, event.ydata, ctx.img_size)
        _draw_preview(dragging["p0"], dragging["p1"])

    def _on_release(event):
        if not dragging["active"]: return
        dragging["active"] = False
        if event.inaxes is ax_ref and event.xdata is not None and event.ydata is not None:
            p1 = _clamp_xy(event.xdata, event.ydata, ctx.img_size)
        else:
            p1 = dragging["p1"]
        p0 = dragging["p0"]
        _clear_preview()

        if abs(p1[0] - p0[0]) < min_span or abs(p1[1] - p0[1]) < min_span:
            render_ref_tgt_with_prompt(
                ax_ref, ax_tgt, ctx.reference_img, ctx.target_img,
                prompt_input=None, pred_mask=None, prompt_type="box",
                img_size_for_mask=ctx.img_size, show_grid=False,
                ref_title=_titles(n=0)[0], tgt_title=_titles()[1],
            )
            fig.canvas.draw_idle()
            return

        box = _normalize_xyxy(p0, p1)
        _echo(f"□ [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
        _render_final(box)

    def _on_key(event):
        k = (event.key or "").lower()
        if k in ("u", "c"):
            store.clear()
            _clear_preview()
            render_ref_tgt_with_prompt(
                ax_ref, ax_tgt, ctx.reference_img, ctx.target_img,
                prompt_input=None, pred_mask=None, prompt_type="box",
                img_size_for_mask=ctx.img_size, show_grid=False,
                ref_title=_titles(n=0)[0], tgt_title=_titles()[1],
            )
            fig.canvas.draw_idle()
            _echo("✂ clear")

    render_ref_tgt_with_prompt(
        ax_ref, ax_tgt, ctx.reference_img, ctx.target_img,
        prompt_input=None, pred_mask=None, prompt_type="box",
        img_size_for_mask=ctx.img_size, show_grid=False,
        ref_title=_titles(n=0)[0], tgt_title=_titles()[1],
    )
    fig.canvas.mpl_connect("button_press_event",   _on_press)
    fig.canvas.mpl_connect("motion_notify_event",  _on_move)
    fig.canvas.mpl_connect("button_release_event", _on_release)
    fig.canvas.mpl_connect("key_press_event",      _on_key)
    plt.show()
    return

# --------------------- SCRIBBLE (drag-to-draw) ----------------------------
def start_scribble_drag_picker_live_predict(
    ctx: DemoCtx,
    store,                      # mutable list of (x,y)
    fig_size: float = 5.5,
    min_dist: float = 3.0,      # sample spacing (px) while dragging
    echo_clicks: bool = True,
    show_count_in_title: bool = True,
):
    """Jupyter scribble picker; release to predict; U/C to clear."""
    if store is None: store = []

    fig, axes = _two_panel_fig(fig_size, key="scribble")

    fig.subplots_adjust(**_SUBPLOT_KW)
    ax_ref, ax_tgt = axes[0], axes[1]

    drawing = {"active": False}
    xs, ys = [], []
    # GREEN live stroke
    line_ref = ax_ref.plot([], [], '-', color='#00aa00', linewidth=2.0, alpha=0.95)[0]

    def _echo(m):
        if echo_clicks: print(m, flush=True)

    def _titles(recording: bool = False):
        left = "Reference — draw a stroke (release to predict)"
        if recording: left = "Reference (recording…) release to predict"
        if show_count_in_title: left += f" — {len(store)}"
        return left, "Target (prediction)"

    def _clear_live():
        xs.clear(); ys.clear()
        line_ref.set_data([], [])

    def _refresh_preview(recording=False):
        render_ref_tgt_with_prompt(
            ax_ref, ax_tgt, ctx.reference_img, ctx.target_img,
            prompt_input=None, pred_mask=None, prompt_type="scribble",
            img_size_for_mask=ctx.img_size, show_grid=False,
            ref_title=_titles(recording)[0], tgt_title=_titles()[1],
        )
        if len(xs) > 0:
            line_ref.set_data(xs, ys)
        fig.canvas.draw_idle()

    def _finalize_and_predict():
        if len(xs) < 2:
            _refresh_preview(False)
            return
        pts = np.array(list(zip(xs, ys)), dtype=np.float32)
        store[:] = [(int(x), int(y)) for x, y in pts]
        mask = predict_with_ctx(ctx, pts)
        render_ref_tgt_with_prompt(
            ax_ref, ax_tgt, ctx.reference_img, ctx.target_img,
            prompt_input=pts, pred_mask=mask, prompt_type="scribble",
            img_size_for_mask=ctx.img_size, show_grid=False,
            ref_title=_titles(False)[0], tgt_title=_titles()[1],
        )
        fig.canvas.draw_idle()

    def _on_press(event):
        if event.inaxes is not ax_ref or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        drawing["active"] = True
        _clear_live()
        x, y = _clamp_xy(event.xdata, event.ydata, ctx.img_size)
        xs.append(x); ys.append(y)
        _refresh_preview(True)

    def _on_move(event):
        if not drawing["active"] or event.inaxes is not ax_ref or event.xdata is None or event.ydata is None:
            return
        x, y = _clamp_xy(event.xdata, event.ydata, ctx.img_size)
        if len(xs) == 0:
            xs.append(x); ys.append(y)
        else:
            dx = x - xs[-1]; dy = y - ys[-1]
            if (dx*dx + dy*dy) >= (min_dist*min_dist):
                xs.append(x); ys.append(y)
        line_ref.set_data(xs, ys)
        ax_ref.set_title(_titles(True)[0]); ax_tgt.set_title(_titles()[1])
        fig.canvas.draw_idle()

    def _on_release(event):
        if not drawing["active"]:
            return
        drawing["active"] = False
        if event.inaxes is ax_ref and event.xdata is not None and event.ydata is not None:
            x, y = _clamp_xy(event.xdata, event.ydata, ctx.img_size)
            if len(xs) == 0 or xs[-1] != x or ys[-1] != y:
                xs.append(x); ys.append(y)
        _echo(f"stroke points: {len(xs)}")
        _finalize_and_predict()

    def _on_key(event):
        k = (event.key or "").lower()
        if k in ("u", "c"):
            store.clear()
            _clear_live()
            _refresh_preview(False)
            _echo("✂ clear")

    _refresh_preview(False)
    fig.canvas.mpl_connect("button_press_event",   _on_press)
    fig.canvas.mpl_connect("motion_notify_event",  _on_move)
    fig.canvas.mpl_connect("button_release_event", _on_release)
    fig.canvas.mpl_connect("key_press_event",      _on_key)
    plt.show()
    return

# --------------------- MASK (freehand lasso) ----------------------------
def start_freehand_mask_picker_live_predict(
    ctx: DemoCtx,
    store,                      # mutable list that will hold one np.ndarray[H,W] mask (0/1)
    fig_size: float = 5.5,
    min_dist: float = 2.0,      # sampling step while dragging (px)
    echo_clicks: bool = True,
    show_count_in_title: bool = True,
):
    """Jupyter freehand mask picker; release to predict; U/C to clear."""
    if store is None: store = []

    fig, axes = _two_panel_fig(fig_size, key="mask")
    fig.subplots_adjust(**_SUBPLOT_KW)
    ax_ref, ax_tgt = axes[0], axes[1]

    drawing = {"active": False}
    xs, ys = [], []
    # GREEN live lasso
    line_ref = ax_ref.plot([], [], '-', color='#00aa00', linewidth=2.0, alpha=0.95)[0]

    def _echo(msg):
        if echo_clicks: print(msg, flush=True)

    def _title_ref(recording: bool = False):
        base = "Reference — draw an outline around the object (release to predict)"
        if recording: base = "Reference (recording…) release to predict"
        if show_count_in_title:
            base += f" — {1 if (store and len(store) > 0) else 0}"
        return base

    def _clear_live():
        xs.clear(); ys.clear()
        line_ref.set_data([], [])

    def _refresh(ref_mask=None, pred_mask=None, recording=False):
        render_ref_tgt_with_prompt(
            ax_ref, ax_tgt, ctx.reference_img, ctx.target_img,
            prompt_input=ref_mask, pred_mask=pred_mask,
            prompt_type='mask', img_size_for_mask=ctx.img_size,
            show_grid=False, ref_title=_title_ref(recording), tgt_title="Target (prediction)",
        )
        if recording and len(xs) > 0:
            line_ref.set_data(xs, ys)
        fig.canvas.draw_idle()

    def _rasterize_polygon(int_points):
        if len(int_points) < 3: return None
        pts = np.array(int_points, dtype=np.int32)
        mask = np.zeros((ctx.img_size, ctx.img_size), dtype=np.uint8)
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 1)
        return mask

    def _finalize_and_predict():
        if len(xs) < 3:
            _refresh(None, None, recording=False)
            return
        if xs[0] != xs[-1] or ys[0] != ys[-1]:
            xs.append(xs[0]); ys.append(ys[0])

        pts_int = [(int(np.clip(x, 0, ctx.img_size-1)),
                    int(np.clip(y, 0, ctx.img_size-1))) for x, y in zip(xs, ys)]
        m = _rasterize_polygon(pts_int)

        store.clear()
        if m is not None:
            store.append(m)

        pred = predict_with_ctx(ctx, m) if m is not None else None
        _refresh(m, pred, recording=False)
        _echo(f"mask area(px)={int(m.sum()) if m is not None else 0}")

    def _on_press(event):
        if event.inaxes is not ax_ref or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        drawing["active"] = True
        _clear_live()
        x, y = _clamp_xy(event.xdata, event.ydata, ctx.img_size)
        xs.append(x); ys.append(y)
        _refresh(None, None, recording=True)

    def _on_move(event):
        if not drawing["active"] or event.inaxes is not ax_ref or event.xdata is None or event.ydata is None:
            return
        x, y = _clamp_xy(event.xdata, event.ydata, ctx.img_size)
        if len(xs) == 0:
            xs.append(x); ys.append(y)
        else:
            dx = x - xs[-1]; dy = y - ys[-1]
            if (dx*dx + dy*dy) >= (min_dist*min_dist):
                xs.append(x); ys.append(y)
        line_ref.set_data(xs, ys)
        ax_ref.set_title(_title_ref(True)); ax_tgt.set_title("Target (prediction)")
        fig.canvas.draw_idle()

    def _on_release(event):
        if not drawing["active"]:
            return
        drawing["active"] = False
        if event.inaxes is ax_ref and event.xdata is not None and event.ydata is not None:
            x, y = _clamp_xy(event.xdata, event.ydata, ctx.img_size)
            if len(xs) == 0 or xs[-1] != x or ys[-1] != y:
                xs.append(x); ys.append(y)
        _finalize_and_predict()

    def _on_key(event):
        k = (event.key or "").lower()
        if k in ("u", "c"):
            store.clear()
            _clear_live()
            _refresh(None, None, recording=False)

    _refresh(None, None, recording=False)
    fig.canvas.mpl_connect("button_press_event",   _on_press)
    fig.canvas.mpl_connect("motion_notify_event",  _on_move)
    fig.canvas.mpl_connect("button_release_event", _on_release)
    fig.canvas.mpl_connect("key_press_event",      _on_key)
    plt.show()
    return


# ===== Colab canvas add-ons (box / scribble / point / mask) =====
# Internal; required by demo_* when in_jupyter=False in Colab.

def _resize_to_ctx(img, size: int):
    if isinstance(img, Image.Image):
        arr = np.array(img.resize((size, size), Image.BILINEAR))
    else:
        arr = np.asarray(img)
        if arr.shape[:2] != (size, size):
            arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr

def _np_to_data_url(arr: np.ndarray) -> str:
    a = arr
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format='PNG')
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def _overlay_mask_np(im, mask, color=(255, 0, 0), alpha=0.55):
    # Default overlay color set to RED for predictions
    im = im.copy()
    m = (mask.astype(np.uint8) > 0).astype(np.uint8) if mask is not None else None
    if m is not None:
        if m.ndim > 2: m = m.squeeze()
        if m.shape[:2] != im.shape[:2]:
            m = cv2.resize(m, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = np.zeros_like(im, dtype=np.uint8); overlay[:] = color
        im = np.where(m[...,None]==1, (alpha*overlay + (1-alpha)*im).astype(np.uint8), im)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, contours, -1, (255,255,255), 3, lineType=cv2.LINE_AA)
    return im

def start_box_picker_live_predict_colab_canvas_drag(ctx, store, panel_px: int = 520, echo_clicks: bool = True):
    if output is None:
        raise RuntimeError("google.colab.output not available; run this in Colab.")
    ref_np = _resize_to_ctx(ctx.reference_img, ctx.img_size)
    tgt_np = _resize_to_ctx(ctx.target_img,    ctx.img_size)
    ref_url = _np_to_data_url(ref_np); tgt_url = _np_to_data_url(tgt_np)

    def _normalize_xyxy(p0, p1):
        x0, y0 = map(int, p0); x1, y1 = map(int, p1)
        x0, x1 = sorted([x0, x1]); y0, y1 = sorted([y0, y1])
        x0 = int(np.clip(x0, 0, ctx.img_size-1)); x1 = int(np.clip(x1, 0, ctx.img_size-1))
        y0 = int(np.clip(y0, 0, ctx.img_size-1)); y1 = int(np.clip(y1, 0, ctx.img_size-1))
        return [x0, y0, x1, y1]

    def _py_on_box_drag(p0, p1):
        try:
            box = _normalize_xyxy(p0, p1)
            store.clear(); store.extend(box)
            pred = predict_with_ctx(ctx, np.asarray(box, dtype=np.float32))
            mask_sum = int(pred.sum()) if pred is not None else 0
            over = _overlay_mask_np(tgt_np, pred) if pred is not None else tgt_np  # RED overlay
            return IPyJSON({"ok": True, "box": box, "mask_sum": mask_sum,
                            "tgt_url": _np_to_data_url(over)})
        except Exception as e:
            if echo_clicks: print(f"[box_cb][ERROR] {type(e).__name__}: {e}", flush=True)
            return IPyJSON({"ok": False, "error": str(e),
                            "tgt_url": _np_to_data_url(tgt_np)})

    cb_name = 'notebook.sansa_box_drag'
    output.register_callback(cb_name, _py_on_box_drag)

    html_tpl = Template(r"""
<div style="font-family:system-ui,-apple-system,Segoe UI,Roboto;max-width:980px">
  <div style="margin:0 0 8px 0;font-weight:600">Drag a box (release to predict). U=Undo, C=Clear.</div>
  <div style="display:flex;gap:16px;align-items:flex-start">
    <div style="display:flex;flex-direction:column;align-items:center">
      <canvas id="ref" width="${IMG_SIZE}" height="${IMG_SIZE}"
              style="width:${PANEL}px;height:${PANEL}px;border:1px solid #ddd;border-radius:8px"></canvas>
      <div style="margin-top:8px;display:flex;gap:8px">
        <button id="undoBtn">U: Undo</button>
        <button id="clearBtn">C: Clear</button>
      </div>
    </div>
    <canvas id="tgt" width="${IMG_SIZE}" height="${IMG_SIZE}"
            style="width:${PANEL}px;height:${PANEL}px;border:1px solid #ddd;border-radius:8px"></canvas>
  </div>
  <script>
  (function(){
    const ref = document.getElementById('ref');
    const tgt = document.getElementById('tgt');
    const cR = ref.getContext('2d'), cT = tgt.getContext('2d');
    const ri = new Image(); ri.src = '${REF_URL}';
    const ti = new Image(); ti.src = '${TGT_URL}';

    function drawRef(){ cR.clearRect(0,0,ref.width,ref.height); cR.drawImage(ri,0,0); }
    function drawTgt(img){ cT.clearRect(0,0,tgt.width,tgt.height); cT.drawImage(img ?? ti, 0, 0); }
    ri.onload = ()=>drawRef(); ti.onload = ()=>drawTgt();

    function toCanvasXY(ev, el){
      const r = el.getBoundingClientRect();
      return [Math.round((ev.clientX-r.left)*el.width/r.width),
              Math.round((ev.clientY-r.top) *el.height/r.height)];
    }

    let dragging = false, p0 = null, p1 = null;

    ref.addEventListener('mousedown', (ev)=>{
      if (ev.button!==0) return;
      dragging = true; p0 = toCanvasXY(ev, ref); p1 = p0;
    });

    ref.addEventListener('mousemove', (ev)=>{
      if (!dragging) return;
      p1 = toCanvasXY(ev, ref);
      drawRef();
      const [x0,y0] = p0, [x1,y1] = p1;
      cR.save(); cR.strokeStyle='#00aa00'; cR.lineWidth=6;  // GREEN rubber-band
      cR.strokeRect(Math.min(x0,x1), Math.min(y0,y1), Math.abs(x1-x0), Math.abs(y1-y0));
      cR.restore();
    });

    ref.addEventListener('mouseup', async (ev)=>{
      if (!dragging) return;
      dragging = false;
      p1 = toCanvasXY(ev, ref);
      try {
        const r = await google.colab.kernel.invokeFunction('${CB_NAME}', [p0, p1], {});
        const payload = r && r.data ? r.data['application/json'] : null;
        const [x0,y0] = p0, [x1,y1] = p1;
        drawRef();
        cR.save(); cR.strokeStyle='#00aa00'; cR.lineWidth=6;  // GREEN final box
        cR.strokeRect(Math.min(x0,x1), Math.min(y0,y1), Math.abs(x1-x0), Math.abs(y1-y0));
        cR.restore();
        if (!payload || !payload.ok || !payload.tgt_url) { drawTgt(); return; }
        const nt = new Image(); nt.onload = ()=>drawTgt(nt); nt.src = payload.tgt_url;  // RED overlay already baked
      } catch (e) { console.error('Callback failed', e); drawTgt(); }
    });

    // Buttons
    document.getElementById('clearBtn').addEventListener('click', ()=>{ drawRef(); drawTgt(); });
    document.getElementById('undoBtn').addEventListener('click',  ()=>{ drawRef(); drawTgt(); });

    // Keyboard shortcuts: U / C
    window.addEventListener('keydown', (ev)=>{
      const k = (ev.key || '').toLowerCase();
      if (k === 'u' || k === 'c'){ drawRef(); drawTgt(); }
    });
  })();
  </script>
</div>
""")
    html = html_tpl.substitute(IMG_SIZE=ctx.img_size, PANEL=panel_px, REF_URL=ref_url, TGT_URL=tgt_url, CB_NAME=cb_name)
    display(HTML(html))

def start_scribble_drag_picker_live_predict_colab_canvas_drag(ctx, store, panel_px: int = 520, min_dist: float = 3.0, echo_clicks: bool = True):
    if output is None:
        raise RuntimeError("google.colab.output not available; run this in Colab.")
    ref_np = _resize_to_ctx(ctx.reference_img, ctx.img_size)
    tgt_np = _resize_to_ctx(ctx.target_img,    ctx.img_size)
    ref_url = _np_to_data_url(ref_np); tgt_url = _np_to_data_url(tgt_np)

    def _downsample_pts(pts, min_d: float):
        if not pts: return []
        keep = [pts[0]]
        md2 = float(min_d) * float(min_d)
        for (x, y) in pts[1:]:
            dx = x - keep[-1][0]; dy = y - keep[-1][1]
            if (dx*dx + dy*dy) >= md2: keep.append([x, y])
        return keep

    def _py_on_scribble(points_js):
        try:
            if not points_js or len(points_js) < 2:
                return IPyJSON({"ok": False, "error": "empty stroke",
                                "tgt_url": _np_to_data_url(tgt_url)})
            pts = []
            for x, y in points_js:
                xi = int(np.clip(int(x), 0, ctx.img_size-1))
                yi = int(np.clip(int(y), 0, ctx.img_size-1))
                pts.append([xi, yi])
            pts = _downsample_pts(pts, min_dist)
            store[:] = [(int(x), int(y)) for x, y in pts]

            pred = predict_with_ctx(ctx, np.asarray(pts, dtype=np.float32))
            mask_sum = int(pred.sum()) if pred is not None else 0
            over = _overlay_mask_np(tgt_np, pred) if pred is not None else tgt_np  # RED overlay
            return IPyJSON({"ok": True, "mask_sum": mask_sum,
                            "tgt_url": _np_to_data_url(over)})
        except Exception as e:
            if echo_clicks: print(f"[scrib_cb][ERROR] {type(e).__name__}: {e}", flush=True)
            return IPyJSON({"ok": False, "error": str(e),
                            "tgt_url": _np_to_data_url(tgt_np)})

    cb_name = 'notebook.sansa_scrib_drag'
    output.register_callback(cb_name, _py_on_scribble)

    html_tpl = Template(r"""
<div style="font-family:system-ui,-apple-system,Segoe UI,Roboto;max-width:980px">
  <div style="margin:0 0 8px 0;font-weight:600">Draw a stroke (release to predict). U=Undo, C=Clear.</div>
  <div style="display:flex;gap:16px;align-items:flex-start">
    <div style="display:flex;flex-direction:column;align-items:center">
      <canvas id="ref" width="${IMG_SIZE}" height="${IMG_SIZE}"
              style="width:${PANEL}px;height:${PANEL}px;border:1px solid #ddd;border-radius:8px"></canvas>
      <div style="margin-top:8px;display:flex;gap:8px">
        <button id="undoBtn">U: Undo</button>
        <button id="clearBtn">C: Clear</button>
      </div>
    </div>
    <canvas id="tgt" width="${IMG_SIZE}" height="${IMG_SIZE}"
            style="width:${PANEL}px;height:${PANEL}px;border:1px solid #ddd;border-radius:8px"></canvas>
  </div>
  <script>
  (function(){
    const ref = document.getElementById('ref');
    const tgt = document.getElementById('tgt');
    const cR = ref.getContext('2d'), cT = tgt.getContext('2d');
    const ri = new Image(); ri.src = '${REF_URL}';
    const ti = new Image(); ti.src = '${TGT_URL}';

    function drawRef(){ cR.clearRect(0,0,ref.width,ref.height); cR.drawImage(ri,0,0); }
    function drawTgt(img){ cT.clearRect(0,0,tgt.width,tgt.height); cT.drawImage(img ?? ti, 0, 0); }
    ri.onload = ()=>drawRef(); ti.onload = ()=>drawTgt();

    function toCanvasXY(ev, el){
      const r = el.getBoundingClientRect();
      return [Math.round((ev.clientX-r.left)*el.width/r.width),
              Math.round((ev.clientY-r.top) *el.height/r.height)];
    }

    let drawing = false;
    let pts = [];

    function drawStroke(points){
      if (!points || points.length < 1) return;

      // white halo for visibility
      cR.save();
      cR.lineJoin = 'round';
      cR.lineCap  = 'round';
      cR.strokeStyle = 'white';
      cR.lineWidth = 8;
      cR.beginPath();
      cR.moveTo(points[0][0], points[0][1]);
      for (let i = 1; i < points.length; i++) cR.lineTo(points[i][0], points[i][1]);
      cR.stroke();
      cR.restore();

      // GREEN main stroke
      cR.save();
      cR.lineJoin = 'round';
      cR.lineCap  = 'round';
      cR.strokeStyle = '#00aa00';
      cR.lineWidth = 5;
      cR.beginPath();
      cR.moveTo(points[0][0], points[0][1]);
      for (let i = 1; i < points.length; i++) cR.lineTo(points[i][0], points[i][1]);
      cR.stroke();
      cR.restore();
    }

    ref.addEventListener('mousedown', (ev)=>{
      if (ev.button!==0) return;
      drawing = true; pts = [toCanvasXY(ev, ref)];
      drawRef(); drawStroke(pts);
    });

    ref.addEventListener('mousemove', (ev)=>{
      if (!drawing) return;
      pts.push(toCanvasXY(ev, ref));
      drawRef(); drawStroke(pts);
    });

    async function finishStroke(){
      drawing = false;
      if (!pts || pts.length < 2) { drawRef(); return; }
      try {
        const r = await google.colab.kernel.invokeFunction('${CB_NAME}', [pts], {});
        let payload = r && r.data ? (r.data['application/json'] ?? null) : null;
        if (!payload) {
          const txt = r && r.data ? (r.data['text/plain'] ?? null) : null;
          if (typeof txt === 'string') { try { payload = JSON.parse(txt); } catch(e) { payload = null; } }
        }
        drawRef(); drawStroke(pts);
        if (!payload || !payload.ok || !payload.tgt_url) { drawTgt(); return; }
        const nt = new Image(); nt.onload = ()=>drawTgt(nt); nt.src = payload.tgt_url; // RED overlay baked
      } catch(e){ console.error('Callback failed', e); drawTgt(); }
    }

    ref.addEventListener('mouseup',   finishStroke);
    ref.addEventListener('mouseleave', ()=>{ if (drawing) finishStroke(); });

    // Buttons
    document.getElementById('clearBtn').addEventListener('click', ()=>{ pts = []; drawRef(); drawTgt(); });
    document.getElementById('undoBtn').addEventListener('click',  ()=>{ pts = []; drawRef(); drawTgt(); });

    // Keyboard shortcuts: U / C
    window.addEventListener('keydown', (ev)=>{
      const k = (ev.key || '').toLowerCase();
      if (k === 'u'){ pts = []; drawRef(); drawTgt(); }
      else if (k === 'c'){ pts = []; drawRef(); drawTgt(); }
    });
  })();
  </script>
</div>
""")
    html = html_tpl.substitute(IMG_SIZE=ctx.img_size, PANEL=panel_px, REF_URL=ref_url, TGT_URL=tgt_url, CB_NAME=cb_name)
    display(HTML(html))

def start_point_picker_live_predict_colab_canvas_click(ctx, store, panel_px: int = 520, echo_clicks: bool = True):
    if output is None:
        raise RuntimeError("google.colab.output not available; run this in Colab.")
    ref_np = _resize_to_ctx(ctx.reference_img, ctx.img_size)
    tgt_np = _resize_to_ctx(ctx.target_img,    ctx.img_size)
    ref_url = _np_to_data_url(ref_np); tgt_url = _np_to_data_url(tgt_np)

    def _py_on_points_update(points_js):
        try:
            pts = []
            for x, y in (points_js or []):
                xi = int(np.clip(int(x), 0, ctx.img_size-1))
                yi = int(np.clip(int(y), 0, ctx.img_size-1))
                pts.append([xi, yi])
            store[:] = [(int(x), int(y)) for x, y in pts]

            if len(pts) == 0:
                return IPyJSON({"ok": True, "n": 0, "mask_sum": 0, "tgt_url": tgt_url})

            pred = predict_with_ctx(ctx, np.asarray(pts, dtype=np.float32))
            mask_sum = int(pred.sum()) if pred is not None else 0
            over = _overlay_mask_np(tgt_np, pred) if pred is not None else tgt_np
            return IPyJSON({"ok": True, "n": len(pts), "mask_sum": mask_sum,
                            "tgt_url": _np_to_data_url(over)})
        except Exception as e:
            return IPyJSON({"ok": False, "error": str(e), "tgt_url": tgt_url})

    cb_name = 'notebook.sansa_point_update'
    output.register_callback(cb_name, _py_on_points_update)

    from string import Template
    html_tpl = Template(r"""
<div style="font-family:system-ui,-apple-system,Segoe UI,Roboto;max-width:980px">
  <div style="margin:0 0 8px 0;font-weight:600">Click to add points. U=Undo, C=Clear.</div>
  <div style="display:flex;gap:16px;align-items:flex-start">
    <div style="display:flex;flex-direction:column;align-items:center">
      <canvas id="ref" width="${IMG_SIZE}" height="${IMG_SIZE}"
              style="width:${PANEL}px;height:${PANEL}px;border:1px solid #ddd;border-radius:8px"></canvas>
      <div style="margin-top:8px;display:flex;gap:8px">
        <button id="undoBtn">U: Undo</button>
        <button id="clearBtn">C: Clear</button>
      </div>
    </div>
    <canvas id="tgt" width="${IMG_SIZE}" height="${IMG_SIZE}"
            style="width:${PANEL}px;height:${PANEL}px;border:1px solid #ddd;border-radius:8px"></canvas>
  </div>
  <script>
  (function(){
    const ref = document.getElementById('ref');
    const tgt = document.getElementById('tgt');
    const cR = ref.getContext('2d'), cT = tgt.getContext('2d');
    const ri = new Image(); ri.src = '${REF_URL}';
    const ti = new Image(); ti.src = '${TGT_URL}';

    function drawRef(){ cR.clearRect(0,0,ref.width,ref.height); cR.drawImage(ri,0,0); }
    function drawTgt(img){ cT.clearRect(0,0,tgt.width,tgt.height); cT.drawImage(img ?? ti, 0, 0); }
    ri.onload = ()=>drawRef(); ti.onload = ()=>drawTgt();

    function toCanvasXY(ev, el){
      const r = el.getBoundingClientRect();
      return [Math.round((ev.clientX-r.left)*el.width/r.width),
              Math.round((ev.clientY-r.top) *el.height/r.height)];
    }

    function drawPoints(pts){
      for (const [x,y] of pts){
        // white ring
        cR.save();
        cR.beginPath(); cR.arc(x, y, 12, 0, Math.PI*2);
        cR.strokeStyle = 'white'; cR.lineWidth = 3; cR.stroke(); cR.restore();
        // GREEN fill + thin white outline
        cR.save();
        cR.beginPath(); cR.arc(x, y, 8, 0, Math.PI*2);
        cR.fillStyle = '#00aa00'; cR.fill();
        cR.strokeStyle = 'white'; cR.lineWidth = 1; cR.stroke();
        cR.restore();
      }
    }

    let pts = [];

    async function updateMask(){
      try {
        const r = await google.colab.kernel.invokeFunction('${CB_NAME}', [pts], {});
        let payload = r && r.data ? (r.data['application/json'] ?? null) : null;
        if (!payload) {
          const txt = r && r.data ? (r.data['text/plain'] ?? null) : null;
          if (typeof txt === 'string') { try { payload = JSON.parse(txt); } catch(e) { payload = null; } }
        }
        if (!payload || !payload.ok || !payload.tgt_url) { drawTgt(); return; }
        const nt = new Image(); nt.onload = ()=>drawTgt(nt); nt.src = payload.tgt_url;
      } catch(e){ drawTgt(); }
    }

    ref.addEventListener('click', async (ev)=>{
      const p = toCanvasXY(ev, ref);
      pts = pts.concat([[p[0], p[1]]]);
      drawRef(); drawPoints(pts);
      await updateMask();
    });

    // Buttons
    document.getElementById('undoBtn').addEventListener('click', async ()=>{
      if (pts.length > 0) pts.pop();
      drawRef(); drawPoints(pts);
      await updateMask();
    });
    document.getElementById('clearBtn').addEventListener('click', async ()=>{
      pts = [];
      drawRef(); drawTgt();
      await updateMask();
    });

    // Keyboard shortcuts: U / C
    window.addEventListener('keydown', async (ev)=>{
      const k = (ev.key || '').toLowerCase();
      if (k === 'u'){
        if (pts.length > 0) pts.pop();
        drawRef(); drawPoints(pts);
        await updateMask();
      } else if (k === 'c'){
        pts = [];
        drawRef(); drawTgt();
        await updateMask();
      }
    });
  })();
  </script>
</div>
""")
    html = html_tpl.substitute(IMG_SIZE=ctx.img_size, PANEL=panel_px, REF_URL=ref_url, TGT_URL=tgt_url, CB_NAME=cb_name)
    display(HTML(html))


def start_freehand_mask_picker_live_predict_colab_canvas_lasso(ctx, store, panel_px: int = 520, echo_clicks: bool = True):
    if output is None:
        raise RuntimeError("google.colab.output not available; run this in Colab.")
    ref_np = _resize_to_ctx(ctx.reference_img, ctx.img_size)
    tgt_np = _resize_to_ctx(ctx.target_img,    ctx.img_size)
    ref_url = _np_to_data_url(ref_np); tgt_url = _np_to_data_url(tgt_np)

    def _poly_to_mask(verts, size: int):
        if verts is None or len(verts) < 3: return None
        pts = np.array(verts, dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, size-1)
        pts[:, 1] = np.clip(pts[:, 1], 0, size-1)
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 1)
        return mask

    def _py_on_mask_lasso(verts_js):
        try:
            m = _poly_to_mask(verts_js, ctx.img_size)
            if m is None or m.sum() == 0:
                return IPyJSON({"ok": True, "area": 0, "ref_url": ref_url, "tgt_url": tgt_url})
            store.clear(); store.append(m)
            pred = predict_with_ctx(ctx, m)
            # Reference prompt overlay: GREEN; Target prediction overlay: RED
            ref_over = _overlay_mask_np(ref_np, m, color=(0, 170, 0), alpha=0.35)  # GREEN
            tgt_over = _overlay_mask_np(tgt_np, pred) if pred is not None else tgt_np  # RED default
            return IPyJSON({"ok": True, "area": int(m.sum()),
                            "ref_url": _np_to_data_url(ref_over),
                            "tgt_url": _np_to_data_url(tgt_over)})
        except Exception as e:
            if echo_clicks: print(f"[mask_lasso_cb][ERROR] {type(e).__name__}: {e}", flush=True)
            return IPyJSON({"ok": False, "error": str(e), "ref_url": ref_url, "tgt_url": tgt_url})

    cb_name = 'notebook.sansa_mask_lasso'
    output.register_callback(cb_name, _py_on_mask_lasso)

    html_tpl = Template(r"""
<div style="font-family:system-ui,-apple-system,Segoe UI,Roboto;max-width:980px">
  <div style="margin:0 0 8px 0;font-weight:600">Mask (lasso) — press-drag-release on the left. U=Undo, C=Clear.</div>
  <div style="display:flex;gap:16px;align-items:flex-start">
    <div style="display:flex;flex-direction:column;align-items:center">
      <canvas id="ref" width="${IMG_SIZE}" height="${IMG_SIZE}"
              style="width:${PANEL}px;height:${PANEL}px;border:1px solid #ddd;border-radius:8px;touch-action:none"></canvas>
      <div style="margin-top:8px;display:flex;gap:8px">
        <button id="undoBtn">U: Undo</button>
        <button id="clearBtn">C: Clear</button>
      </div>
    </div>
    <canvas id="tgt" width="${IMG_SIZE}" height="${IMG_SIZE}"
            style="width:${PANEL}px;height:${PANEL}px;border:1px solid #ddd;border-radius:8px;touch-action:none"></canvas>
  </div>
  <script>
  (function(){
    const ref = document.getElementById('ref');
    const tgt = document.getElementById('tgt');
    const cR = ref.getContext('2d'), cT = tgt.getContext('2d');
    const ri = new Image(); ri.src = '${REF_URL}';
    const ti = new Image(); ti.src = '${TGT_URL}';

    function drawRef(img){ cR.clearRect(0,0,ref.width,ref.height); cR.drawImage(img ?? ri, 0, 0); }
    function drawTgt(img){ cT.clearRect(0,0,tgt.width,tgt.height); cT.drawImage(img ?? ti, 0, 0); }
    ri.onload = ()=>drawRef(); ti.onload = ()=>drawTgt();

    function toCanvasXY(ev, el){
      const r = el.getBoundingClientRect();
      return [Math.round((ev.clientX-r.left)*el.width/r.width),
              Math.round((ev.clientY-r.top) *el.height/r.height)];
    }

    let drawing = false;
    let verts = [];

    function strokeRefPath(pts){
      if (pts.length < 2) return;
      cR.save();
      cR.lineWidth = 6;
      cR.strokeStyle = '#00aa00';  // GREEN lasso stroke
      cR.beginPath();
      cR.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; ++i){ cR.lineTo(pts[i][0], pts[i][1]); }
      cR.stroke();
      cR.restore();
    }

    function redraw(){
      if (verts.length === 0){ drawRef(); return; }
      drawRef(); strokeRefPath(verts);
    }

    async function finalizeAndPredict(){
      if (verts.length < 3){ drawRef(); drawTgt(); return; }
      try {
        const r = await google.colab.kernel.invokeFunction('${CB_NAME}', [verts], {});
        let payload = r && r.data ? (r.data['application/json'] ?? null) : null;
        if (!payload){
          const txt = r && r.data ? (r.data['text/plain'] ?? null) : null;
          if (typeof txt === 'string'){ try { payload = JSON.parse(txt); } catch(e) { payload = null; } }
        }
        if (!payload || !payload.ok){ drawRef(); drawTgt(); return; }
        const ri2 = new Image(); const ti2 = new Image();
        ri2.onload = ()=>drawRef(ri2); ti2.onload = ()=>drawTgt(ti2);
        ri2.src = payload.ref_url; ti2.src = payload.tgt_url; // GREEN prompt overlay / RED prediction
      } catch(e){ drawRef(); drawTgt(); }
    }

    // Mouse handlers
    ref.addEventListener('mousedown', (ev)=>{
      if (ev.button !== 0) return;
      drawing = true; verts = [toCanvasXY(ev, ref)];
      redraw();
    });
    ref.addEventListener('mousemove', (ev)=>{
      if (!drawing) return;
      const p = toCanvasXY(ev, ref);
      const last = verts[verts.length-1];
      const dx = p[0]-last[0], dy = p[1]-last[1];
      if ((dx*dx + dy*dy) >= 4){
        verts.push(p);
        redraw();
      }
    });
    ref.addEventListener('mouseup', async (ev)=>{
      if (!drawing) return;
      drawing = false;
      await finalizeAndPredict();
    });

    // Touch handlers
    ref.addEventListener('touchstart', (ev)=>{
      const t = ev.touches[0]; if (!t) return;
      drawing = true;
      const fake = {clientX: t.clientX, clientY: t.clientY};
      verts = [toCanvasXY(fake, ref)];
      redraw();
      ev.preventDefault();
    }, {passive:false});
    ref.addEventListener('touchmove', (ev)=>{
      if (!drawing) return;
      const t = ev.touches[0]; if (!t) return;
      const fake = {clientX: t.clientX, clientY: t.clientY};
      const p = toCanvasXY(fake, ref);
      const last = verts[verts.length-1];
      const dx = p[0]-last[0], dy = p[1]-last[1];
      if ((dx*dx + dy*dy) >= 4){
        verts.push(p);
        redraw();
      }
      ev.preventDefault();
    }, {passive:false});
    ref.addEventListener('touchend', async (ev)=>{
      if (!drawing) return;
      drawing = false;
      await finalizeAndPredict();
      ev.preventDefault();
    }, {passive:false});

    // Buttons
    document.getElementById('undoBtn').addEventListener('click', ()=>{
      if (verts.length > 0){ verts.pop(); }
      if (verts.length === 0){ drawRef(); drawTgt(); } else { redraw(); }
    });
    document.getElementById('clearBtn').addEventListener('click', ()=>{
      verts = [];
      drawRef(); drawTgt();
    });

    // Keyboard shortcuts: U / C
    window.addEventListener('keydown', (ev)=>{
      const k = (ev.key || '').toLowerCase();
      if (k === 'u'){
        if (verts.length > 0){ verts.pop(); }
        if (verts.length === 0){ drawRef(); drawTgt(); } else { redraw(); }
      } else if (k === 'c'){
        verts = [];
        drawRef(); drawTgt();
      }
    });
  })();
  </script>
</div>
""")
    html = html_tpl.substitute(IMG_SIZE=ctx.img_size, PANEL=panel_px, REF_URL=ref_url, TGT_URL=tgt_url, CB_NAME=cb_name)
    display(HTML(html))


# -------------------- Unified wrappers (public API you use) --------------------
def _colab_available() -> bool:
    try:
        from google.colab import output  # noqa: F401
        return True
    except Exception:
        return False

def demo_points(ctx: DemoCtx, store: Optional[Points] = None, *, in_jupyter: bool = True, panel_px: int = 520, **kwargs):
    if store is None: store = []
    if in_jupyter:
        return start_point_picker_live_predict(ctx, store=store, **kwargs)
    if _colab_available():
        return start_point_picker_live_predict_colab_canvas_click(ctx, store=store, panel_px=panel_px, **kwargs)
    return start_point_picker_live_predict(ctx, store=store, **kwargs)

def demo_box(ctx: DemoCtx, store: Optional[List[float]] = None, *, in_jupyter: bool = True, panel_px: int = 520, **kwargs):
    if store is None: store = []
    if in_jupyter:
        return start_box_picker_live_predict(ctx, store=store, **kwargs)
    if _colab_available():
        return start_box_picker_live_predict_colab_canvas_drag(ctx, store=store, panel_px=panel_px, **kwargs)
    return start_box_picker_live_predict(ctx, store=store, **kwargs)

def demo_scribble(ctx: DemoCtx, store: Optional[Points] = None, *, in_jupyter: bool = True, panel_px: int = 520, **kwargs):
    if store is None: store = []
    if in_jupyter:
        return start_scribble_drag_picker_live_predict(ctx, store=store, **kwargs)
    if _colab_available():
        return start_scribble_drag_picker_live_predict_colab_canvas_drag(ctx, store=store, panel_px=panel_px, **kwargs)
    return start_scribble_drag_picker_live_predict(ctx, store=store, **kwargs)

def demo_mask(ctx: DemoCtx, store: Optional[List] = None, *, in_jupyter: bool = True, panel_px: int = 520, **kwargs):
    if store is None: store = []
    if in_jupyter:
        return start_freehand_mask_picker_live_predict(ctx, store=store, **kwargs)
    if _colab_available():
        return start_freehand_mask_picker_live_predict_colab_canvas_lasso(ctx, store=store, panel_px=panel_px, **kwargs)
    return start_freehand_mask_picker_live_predict(ctx, store=store, **kwargs)

def demo_prompt(ctx: DemoCtx, store: Optional[list] = None, *, in_jupyter: bool = True, panel_px: int = 520, **kwargs):
    pt = (ctx.prompt_type or "").lower()
    if pt == "point":
        return demo_points(ctx, store=store, in_jupyter=in_jupyter, panel_px=panel_px, **kwargs)
    if pt == "box":
        return demo_box(ctx, store=store, in_jupyter=in_jupyter, panel_px=panel_px, **kwargs)
    if pt == "scribble":
        return demo_scribble(ctx, store=store, in_jupyter=in_jupyter, panel_px=panel_px, **kwargs)
    if pt == "mask":
        return demo_mask(ctx, store=store, in_jupyter=in_jupyter, panel_px=panel_px, **kwargs)
    raise ValueError(f"Unknown ctx.prompt_type='{ctx.prompt_type}'")

# Public API
__all__ = [
    "img_size", "show_ref_tgt", "demo_points", "demo_box", "demo_scribble", "demo_mask",
    "demo_prompt", "generate_scribble", "load_ref_mask", "DemoCtx", "predict_with_ctx",
]
