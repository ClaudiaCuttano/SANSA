import random
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util


class CustomConcatDataset(Dataset):
    def __init__(self, dataset_list, dataset_ratio=None, samples_per_epoch=160000):
        self.dataset_list = dataset_list
        if dataset_ratio is not None :
            assert len(dataset_ratio) == len(dataset_list)
        else :
            dataset_ratio = [1] * len(dataset_list)
        self.dataset_ratio = dataset_ratio
        self.samples_per_epoch = samples_per_epoch

    def __len__(self,):
        return self.samples_per_epoch

    def __getitem__(self, index):
        dataset_idx = random.choices(list(range(len(self.dataset_ratio))), weights=self.dataset_ratio, k=1)[0]
        dataset = self.dataset_list[dataset_idx]
        index = random.randint(0, len(dataset) - 1)
        return dataset[index]
    
    
def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8') # np
    mask = mask > 0.5

    # origin_img[mask] = origin_img[mask] * 0.25 + color * 0.75
    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img = Image.fromarray(origin_img)
    return origin_img


def polygons_to_bitmask(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    if len(polygons) == 0:
        # COCOAPI does not support empty polygons
        return np.zeros((height, width)).astype(bool)
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(bool)
    
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    
    
def denormalize(tens):
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).to(tens.device)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).to(tens.device)

    return (tens*std)+mean