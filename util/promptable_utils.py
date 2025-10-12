import random
from typing import Dict, Tuple
from PIL import Image, ImageDraw
import torch
import numpy as np
from scipy.ndimage import label
import einops

from util.commons import rescale_points, resize_mask


PROMPT_CHOICES = ('mask', 'point', 'scribble', 'box')

def select_prompt(prompt: str) -> str:
    """
    Choose the prompt type for support frames.

    Returns:
        One of {"mask","point","scribble","box"} or a random choice if self.prompt == "multi".
    """
    if prompt == "multi":
        import random
        return random.choice(PROMPT_CHOICES)
    return prompt


def build_prompt_dict(masks: torch.Tensor, prompt_type: str, n_shots: int, train_mode: bool, device: torch.device):
    B = masks.shape[0]
    prompt_dict = {'shots': n_shots}

    for batch_idx in range(B):
        prompt_dict[batch_idx] = {}
        for support_idx in range(n_shots):
            prompt_dict[batch_idx][support_idx] = {}    
            
            prompt = select_prompt(prompt_type)
            gt_mask = masks[batch_idx:batch_idx+1, support_idx:support_idx+1]
            prompt_dict[batch_idx][support_idx]['prompt_type'] = prompt
            if prompt in ("point", "scribble", "box"):
                prompt_inputs = build_prompt_inputs(gt_mask, prompt, train_mode, device)
            else: # use mask
                prompt_inputs = gt_mask.to(device)
            prompt_dict[batch_idx][support_idx]['prompt'] = prompt_inputs
    
    return prompt_dict


def rescale_prompt(frame_prompt, prompt_type:str, orig_scale: Tuple[int, int], dest_scale: int):
    if prompt_type in ['point', 'box', 'scribble']:
        frame_prompt['point_coords'] = rescale_points(frame_prompt['point_coords'], orig_scale, (dest_scale, dest_scale))
    elif prompt_type == 'mask':
        frame_prompt = resize_mask(frame_prompt, dest_scale)
    else:
        raise NotImplementedError()
    
    return frame_prompt


def build_prompt_inputs(frame_gt: torch.Tensor, prompt: str, training: bool, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Build prompt inputs (points/scribble/box) from a binary GT mask at image size.

    Args:
        frame_gt: Boolean tensor [1,1,IMG,IMG].
        prompt:   "point" | "scribble" | "box".
        training: whether in training mode (affects sampling in helpers).

    Returns:
        Dict with keys:
        - "point_coords": [1, P, 2]
        - "point_labels": [1, P]
    """
    if prompt != 'box':
        if prompt == 'point':
            support_mask = get_point_mask(frame_gt, training)
        else:  # 'scribble'
            support_mask = get_scribble_mask(frame_gt, training)

        if support_mask.sum().item() > 0:
            pos = torch.nonzero(support_mask[0, 0] == 1, as_tuple=False)[:, [1, 0]]  # (y,x)->(x,y)
            point_coords = pos.unsqueeze(0).float()  # [1, P, 2]
            point_labels = torch.ones(pos.shape[0], dtype=torch.int32).unsqueeze(0)  # [1, P]
        else:
            point_coords = torch.zeros(1, 1, 2)
            point_labels = torch.ones(1, 1, dtype=torch.int32)

        return {"point_coords": point_coords.to(device), "point_labels": point_labels.to(device)}

    # box
    boxes = get_bounding_boxes(frame_gt)  # list of [x1,y1,x2,y2]
    n_boxes = len(boxes)
    box_coords = einops.rearrange(boxes, 'n (p1 p2) -> (n p1) p2', p1=2, p2=2).unsqueeze(0)  # [1, 2n, 2]
    box_labels = torch.tensor([2, 3], dtype=torch.int32, device=device).repeat((1, n_boxes))
    return {"point_coords": box_coords.to(device), "point_labels": box_labels.to(device)}


def get_bounding_boxes(mask):
    """
    Returns:
        Boxes: tight bounding boxes around bitmasks.
        If a mask is empty, it's bounding box will be all zero.
    """
    mask_np = mask[0, 0].cpu().numpy()
    labeled_mask, num_features = label(mask_np)
    boxes = []

    # Loop through each labeled region and compute bounding boxes
    for region_label in range(1, num_features + 1):  # Skip background (label 0)
        region = labeled_mask == region_label
        y_coords, x_coords = np.where(region)
        x1, x2 = x_coords.min(), x_coords.max() + 1
        y1, y2 = y_coords.min(), y_coords.max() + 1
        boxes.append([x1, y1, x2, y2])

    def boxes_overlap(box1, box2):
        return not (box1[2] <= box2[0] or box2[2] <= box1[0] or box1[3] <= box2[1] or box2[3] <= box1[1])

    # Function to merge two boxes
    def merge_boxes(box1, box2):
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])
        return [x1, y1, x2, y2]

    # Select the bounding box with the maximum area
    def compute_area(box):
        width = box[2] - box[0]
        height = box[3] - box[1]
        return width * height

    merged_boxes = []
    for box in boxes:
        merged = False
        for i, existing_box in enumerate(merged_boxes):
            if boxes_overlap(box, existing_box):
                merged_boxes[i] = merge_boxes(box, existing_box)
                merged = True
                break
        if not merged:
            merged_boxes.append(box)

    if len(merged_boxes) > 0:
        max_area_box = max(merged_boxes, key=compute_area)
        boxes_tensor = torch.tensor(max_area_box, dtype=torch.float32).unsqueeze(0)
    else:
        boxes_tensor = torch.tensor([0., 0., 0., 0.], dtype=torch.float32).unsqueeze(0)
    return boxes_tensor


def get_point_mask(mask, training, max_points=20):
    """
    Returns:
        Point_mask: random 20 point for train and test.
        If a mask is empty, it's Point_mask will be all zero.
    """
    h, w = mask.shape[-2:]
    if mask.sum().item() == 0:
        return torch.zeros(1, 1, h, w).to(mask.device)
    max_points = min(max_points, mask.sum().item())
    if training:
        num_points = random.Random().randint(1, max_points)  # get a random number of points
    else:
        num_points = max_points

    view_mask = mask[0,0].view(-1)
    non_zero_idx = view_mask.nonzero()[:, 0]  # get non-zero index of mask
    selected_idx = torch.randperm(len(non_zero_idx))[:num_points]  # select id
    non_zero_idx = non_zero_idx[selected_idx]  # select non-zero index
    rand_mask = torch.zeros(view_mask.shape).to(mask.device)  # init rand mask
    rand_mask[non_zero_idx] = 1  # get one place to zero
    rand_mask = rand_mask.reshape(h, w).unsqueeze(0).unsqueeze(0)
    return rand_mask


def get_scribble_mask(mask, training, stroke_preset=['rand_curve', 'rand_curve_small'], stroke_prob=[0.5, 0.5]):
    """
    Returns:
        Scribble_mask: random 20 point for train and test.
        If a mask is empty, it's Scribble_mask will be all zero.
    """
    h, w = mask.shape[-2:]
    if mask.sum().item() == 0:
        return torch.zeros(1, 1, h, w).to(mask.device)
    if training:
        stroke_preset_name = random.Random().choices(stroke_preset, weights=stroke_prob, k=1)[0]
        nStroke = random.Random().randint(1, min(20, mask.sum().item()))
    else:
        stroke_preset_name = random.Random(321).choices(stroke_preset, weights=stroke_prob, k=1)[0]
        nStroke = random.Random(321).randint(1, min(20, mask.sum().item()))
    preset = get_stroke_preset(stroke_preset_name)

    points = get_random_points_from_mask(mask[0,0].bool(), n=nStroke)
    rand_mask = get_mask_by_input_strokes(init_points=points, imageWidth=w, imageHeight=h,
                                              nStroke=min(nStroke, len(points)), **preset)
    rand_mask = (~torch.from_numpy(rand_mask)) * mask[0,0].bool().cpu()
    rand_mask = rand_mask.float().unsqueeze(0).unsqueeze(0)
    return rand_mask.to(mask.device)


# --------------------------------------------------------
# References:
# https://github.com/syp2ysy/VRP-SAM
# ---------

def get_stroke_preset(stroke_preset):
        if stroke_preset == 'rand_curve':
            return {
                "nVertexBound": [10, 30],
                "maxHeadSpeed": 20,
                "maxHeadAcceleration": (15, 0.5),
                "brushWidthBound": (3, 10),
                "nMovePointRatio": 0.5,
                "maxPiontMove": 3,
                "maxLineAcceleration": (5, 0.5),
                "boarderGap": None,
                "maxInitSpeed": 6
            }
        elif stroke_preset == 'rand_curve_small':
            return {
                "nVertexBound": [6, 22],
                "maxHeadSpeed": 12,
                "maxHeadAcceleration": (8, 0.5),
                "brushWidthBound": (2.5, 5),
                "nMovePointRatio": 0.5,
                "maxPiontMove": 1.5,
                "maxLineAcceleration": (3, 0.5),
                "boarderGap": None,
                "maxInitSpeed": 3
            }
        else:
            raise NotImplementedError(f'The stroke presetting "{stroke_preset}" does not exist.')


def get_random_points_from_mask(mask, n=5):
    h, w = mask.shape
    view_mask = mask.reshape(h * w)
    non_zero_idx = view_mask.nonzero()[:, 0]
    selected_idx = torch.randperm(len(non_zero_idx))[:n]
    non_zero_idx = non_zero_idx[selected_idx]
    # import pdb; pdb.set_trace()
    y = torch.div(non_zero_idx, w)
    x = (non_zero_idx % w)
    return torch.cat((x[:, None], y[:, None]), dim=1).cpu().numpy()


def get_mask_by_input_strokes(
    init_points, imageWidth=320, imageHeight=180, nStroke=5,
    nVertexBound=[10, 30], maxHeadSpeed=15, maxHeadAcceleration=(15, 0.5),
    brushWidthBound=(5, 20), boarderGap=None, nMovePointRatio=0.5, maxPiontMove=10,
    maxLineAcceleration=5, maxInitSpeed=5
):
    '''
    Get video masks by random strokes which move randomly between each
    frame, including the whole stroke and its control points

    Parameters
    ----------
        imageWidth: Image width
        imageHeight: Image height
        nStroke: Number of drawed lines
        nVertexBound: Lower/upper bound of number of control points for each line
        maxHeadSpeed: Max head speed when creating control points
        maxHeadAcceleration: Max acceleration applying on the current head point (
            a head point and its velosity decides the next point)
        brushWidthBound (min, max): Bound of width for each stroke
        boarderGap: The minimum gap between image boarder and drawed lines
        nMovePointRatio: The ratio of control points to move for next frames
        maxPiontMove: The magnitude of movement for control points for next frames
        maxLineAcceleration: The magnitude of acceleration for the whole line

    Examples
    ----------
        object_like_setting = {
            "nVertexBound": [5, 20],
            "maxHeadSpeed": 15,
            "maxHeadAcceleration": (15, 3.14),
            "brushWidthBound": (30, 50),
            "nMovePointRatio": 0.5,
            "maxPiontMove": 10,
            "maxLineAcceleration": (5, 0.5),
            "boarderGap": 20,
            "maxInitSpeed": 10,
        }
        rand_curve_setting = {
            "nVertexBound": [10, 30],
            "maxHeadSpeed": 20,
            "maxHeadAcceleration": (15, 0.5),
            "brushWidthBound": (3, 10),
            "nMovePointRatio": 0.5,
            "maxPiontMove": 3,
            "maxLineAcceleration": (5, 0.5),
            "boarderGap": 20,
            "maxInitSpeed": 6
        }
        get_video_masks_by_moving_random_stroke(video_len=5, nStroke=3, **object_like_setting)
    '''
    # Initilize a set of control points to draw the first mask
    # import pdb; pdb.set_trace()
    mask = Image.new(mode='1', size=(imageWidth, imageHeight), color=1)
    control_points_set = []
    for i in range(nStroke):
        brushWidth = np.random.randint(brushWidthBound[0], brushWidthBound[1])
        Xs, Ys, velocity = get_random_stroke_control_points(
            init_point=init_points[i],
            imageWidth=imageWidth, imageHeight=imageHeight,
            nVertexBound=nVertexBound, maxHeadSpeed=maxHeadSpeed,
            maxHeadAcceleration=maxHeadAcceleration, boarderGap=boarderGap,
            maxInitSpeed=maxInitSpeed
        )
        control_points_set.append((Xs, Ys, velocity, brushWidth))
        draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=0)

    # Generate the following masks by randomly move strokes and their control points
    mask = Image.new(mode='1', size=(imageWidth, imageHeight), color=1)
    for j in range(len(control_points_set)):
        Xs, Ys, velocity, brushWidth = control_points_set[j]
        new_Xs, new_Ys = random_move_control_points(
            Xs, Ys, velocity, nMovePointRatio, maxPiontMove,
            maxLineAcceleration, boarderGap
        )
        control_points_set[j] = (new_Xs, new_Ys, velocity, brushWidth)
    for Xs, Ys, velocity, brushWidth in control_points_set:
        draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=0)

    return np.array(mask)


def random_move_control_points(Xs, Ys, lineVelocity, nMovePointRatio, maxPiontMove, maxLineAcceleration, boarderGap=15):
    new_Xs = Xs.copy()
    new_Ys = Ys.copy()

    # move the whole line and accelerate
    speed, angle = lineVelocity
    new_Xs += int(speed * np.cos(angle))
    new_Ys += int(speed * np.sin(angle))

    # choose points to move
    chosen = np.arange(len(Xs))
    np.random.shuffle(chosen)
    chosen = chosen[:int(len(Xs) * nMovePointRatio)]
    for i in chosen:
        new_Xs[i] += np.random.randint(-maxPiontMove, maxPiontMove)
        new_Ys[i] += np.random.randint(-maxPiontMove, maxPiontMove)
    return new_Xs, new_Ys


def random_accelerate(velocity, maxAcceleration, dist='uniform'):
    speed, angle = velocity
    d_speed, d_angle = maxAcceleration

    if dist == 'uniform':
        speed += np.random.uniform(-d_speed, d_speed)
        angle += np.random.uniform(-d_angle, d_angle)
    elif dist == 'guassian':
        speed += np.random.normal(0, d_speed / 2)
        angle += np.random.normal(0, d_angle / 2)
    else:
        raise NotImplementedError(f'Distribution type {dist} is not supported.')

    return (speed, angle)


def draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=255):
    radius = brushWidth // 2 - 1
    for i in range(1, len(Xs)):
        draw = ImageDraw.Draw(mask)
        startX, startY = Xs[i - 1], Ys[i - 1]
        nextX, nextY = Xs[i], Ys[i]
        draw.line((startX, startY) + (nextX, nextY), fill=fill, width=brushWidth)
    for x, y in zip(Xs, Ys):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    return mask

def get_random_stroke_control_points(
    init_point,
    imageWidth, imageHeight,
    nVertexBound=(10, 30), maxHeadSpeed=10, maxHeadAcceleration=(5, 0.5), boarderGap=20,
    maxInitSpeed=10
):
    '''
    Implementation the free-form training masks generating algorithm
    proposed by JIAHUI YU et al. in "Free-Form Image Inpainting with Gated Convolution"
    '''
    startX = init_point[0]
    startY = init_point[1]

    Xs = [init_point[0]]
    Ys = [init_point[1]]

    numVertex = np.random.randint(nVertexBound[0], nVertexBound[1])

    angle = np.random.uniform(0, 2 * np.pi)
    speed = np.random.uniform(0, maxHeadSpeed)

    for i in range(numVertex):
        speed, angle = random_accelerate((speed, angle), maxHeadAcceleration)
        speed = np.clip(speed, 0, maxHeadSpeed)

        nextX = startX + speed * np.sin(angle)
        nextY = startY + speed * np.cos(angle)

        if boarderGap is not None:
            nextX = np.clip(nextX, boarderGap, imageWidth - boarderGap)
            nextY = np.clip(nextY, boarderGap, imageHeight - boarderGap)

        startX, startY = nextX, nextY
        Xs.append(nextX)
        Ys.append(nextY)

    velocity = get_random_velocity(maxInitSpeed, dist='guassian')

    return np.array(Xs), np.array(Ys), velocity


def get_random_velocity(max_speed, dist='uniform'):
    if dist == 'uniform':
        speed = np.random.uniform(max_speed)
    elif dist == 'guassian':
        speed = np.abs(np.random.normal(0, max_speed / 2))
    else:
        raise NotImplementedError(f'Distribution type {dist} is not supported.')

    angle = np.random.uniform(0, 2 * np.pi)
    return (speed, angle)
