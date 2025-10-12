r""" ADE20k semantic segmentation dataset """
import json
import os
import random
from collections import defaultdict
from os.path import join

import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class SemADE(Dataset):
    def __init__(self, base_image_dir, transform, is_train=True, dataset_name='ade20k', shots=1, is_semseg=False, ext='png', is_meta=False):
        self.transform = transform
        self.max_inst = 1 if is_train else 150
        self.ext = ext
        self.shots = shots
        self.is_train = is_train
        self.split = split = 'training' if is_train else 'validation'
        self.is_semseg = is_semseg
        self.zero_start = False
        self.is_meta = is_meta
        base_image_dir = join(base_image_dir, 'ADEChallengeData2016')
        with open(f"{base_image_dir}/utils/{dataset_name.replace('sd_', '')}_classes.json", "r") as f:
            classes_name_list = json.load(f)
        if dataset_name in ('ade20k'):
            self.ignore_idx = 255
            self.img_root = join(base_image_dir, "images", split)
            self.anno_root = join(base_image_dir, "annotations", split)
            classes_name_list = ['none'] + classes_name_list
        elif dataset_name in ('sd_ade20k'):
            self.ignore_idx = 255
            self.img_root = join(base_image_dir, "images_detectron2", split)
            self.anno_root = join(base_image_dir, "annotations_detectron2", split)
            classes_name_list = ['none'] + classes_name_list
        elif dataset_name == 'cocostuff':
            self.img_root = join(base_image_dir, "images", split)
            self.anno_root = join(base_image_dir, "annotations", split)
            classes_name_list = [x.split(':')[-1] for x in classes_name_list]
        elif dataset_name == 'ade847':
            self.ignore_idx = 65535
            self.img_root = join(base_image_dir, "images_detectron2", split)
            self.anno_root = join(base_image_dir, "annotations_detectron2", split)
            classes_name_list = ['none'] + classes_name_list
        elif dataset_name == 'sd_ade847':
            self.zero_start = False
            self.ignore_idx = 65535
            self.img_root = join(base_image_dir, "images_detectron2", split)
            self.anno_root = join(base_image_dir, "annotations_detectron2", split)
            classes_name_list = ['none'] + classes_name_list
        elif dataset_name == 'pc459':
            assert not is_train
            self.zero_start = True
            self.split = split = 'training' if is_train else 'validation'
            self.img_root = os.path.join(base_image_dir, 'JPEGImages')
            self.anno_root = os.path.join(base_image_dir, 'annotations_detectron2/pc459_val')
        else :
            raise NotImplementedError

        self.cid_to_cat = np.array(classes_name_list)
        self.all_cls = set(list(range(len(self.cid_to_cat)))) 
        if not self.zero_start:
            self.all_cls = self.all_cls- {0}
        self.class_ids = self.get_class_ids()
        file_names = sorted(
            os.listdir(self.img_root)
        )
        
        gt_names = os.listdir(self.anno_root)
        if len(file_names) != len(gt_names):
            print('warning, not equal')
            file_names = [x[:-len('.jpg')] for x in file_names]
            gt_names = [x[:-(len(self.ext)+1)] for x in gt_names]
            intersect = list(set(file_names) & set(gt_names))
            file_names = [x+'.jpg' for x in intersect]
            
        image_ids = []
        for x in file_names:
            if x.endswith(".jpg"):
                image_ids.append(x[:-4])
        self.image_ids = []

        meta_path = f"{base_image_dir}/utils/{'train' if is_train else 'val'}_{dataset_name}_icl.pth"
        if not os.path.exists(meta_path):
            meta_info = defaultdict(list)
            for img_id in tqdm(image_ids) :
                anno_path = os.path.join(self.anno_root, '{}.{}'.format(img_id, self.ext))
                label = Image.open(anno_path)
                if self.ext != 'tif':
                    label = label.convert('L')
                uni_cids = np.unique(np.asarray(label))
                if not self.zero_start:
                    if uni_cids[0] == 0:
                        uni_cids = uni_cids[1:]
                    if uni_cids[-1] == self.ignore_idx:
                        uni_cids = uni_cids[:-1]
                # if len(uni_cids) >= 1 and self.zero_start or len(uni_cids) > 1:
                if len(uni_cids) >= 1 :
                    self.image_ids.append(img_id)
                for cid in uni_cids:
                    if cid in self.class_ids:
                        meta_info[cid].append(img_id)
            self.meta_info = meta_info
            torch.save([self.meta_info, self.image_ids], meta_path)
        else :
            self.meta_info, self.image_ids = torch.load(meta_path, weights_only=False)

    def get_meta(self, idx):
        cid = self.class_ids[idx]
        ref_img_id = self._get_ref_cid(cid, None)
        if ref_img_id is not None:
            ref_id = self.image_ids.index(ref_img_id)
        else :
            ref_id = 3
        return ref_id

    def get_class_ids(self,):
        return np.array(sorted(list(self.all_cls)))

    def get_class_names(self,):
        cls_ids = self.get_class_ids()
        return [self.cid_to_cat[x] for x in cls_ids]

    def _get_info(self, img_ids, cats_list=None, ret_uni_cids=None, sample_max_inst=True):
        # print('img', img_id)
        ims, m_list = [], []
        for img_id in img_ids:
            try:
                image = Image.open(join(self.img_root, '{}.jpg'.format(img_id))).convert('RGB')
            except:
                a = 1
            masks = Image.open(join(self.anno_root, '{}.{}'.format(img_id, self.ext)))
            # if self.ext != 'tif':
            #     masks = masks.convert('L')
            masks = np.array(masks)
            uni_cids = np.unique(masks)
            if not self.zero_start:
                if uni_cids[0] == 0:
                    uni_cids = uni_cids[1:]
                if uni_cids[-1] == self.ignore_idx:
                    uni_cids = uni_cids[:-1]

            if cats_list is None :
                if sample_max_inst and len(uni_cids) > self.max_inst :
                    cats_list = np.random.choice(
                        uni_cids, size=self.max_inst, replace=False
                    ).tolist()
                else :
                    cats_list = uni_cids.tolist()
            else :
                uni_cids = np.array(cats_list)

            masks_list = []
            for cid in cats_list :
                masks_list.append(masks==cid)

            masks = np.stack(masks_list)
            # image = to_tensor(image)
            masks = torch.tensor(masks).float()

            m_list.append(masks)
            ims.append(image)
        if ret_uni_cids :
            return ims, m_list, cats_list, uni_cids
        return ims, m_list, cats_list

    def __len__(self,):
        if self.is_meta:
            return len(self.class_ids)
        return len(self.image_ids)

    def _get_ref_cid(self, cid, index):
        idx_list = self.meta_info[cid]
        ref_index = [index]
        
        if not len(idx_list):
            return None

        if len(idx_list) > 1 or ref_index is None:
            while index in ref_index:
                ref_index = random.sample(idx_list, self.shots)
        return ref_index

    def __getitem__(self, index, cat_id=None) :
        if self.is_meta :
            cat_id = [self.class_ids[index]]
            index = self.get_meta(index)

        img_id = self.image_ids[index]
        image, masks, cats_list, uni_cids = self._get_info([img_id], cat_id, ret_uni_cids=True)
        image = self.transform(image[0])
        image_ref_list, masks_ref_list = [], []
        masks = F.interpolate(masks[0].unsqueeze(0).float(), image.size()[-2:], mode='nearest').squeeze()

        sample = {'image': image,
                'label': masks
                }
        
        if self.is_meta:
            sample.update(class_id=cat_id[0])

        if cat_id is None :
            for cid in cats_list:
                ref_img_id = self._get_ref_cid(cid, img_id)
                image_ref, masks_ref, _ = self._get_info(ref_img_id, [cid])
                image_ref = torch.stack([self.transform(im_r) for im_r in image_ref])
                masks_ref = torch.stack([F.interpolate(m_r.unsqueeze(0).float(), image.size()[-2:], mode='nearest').squeeze() for m_r in masks_ref])
                image_ref_list.append(image_ref)
                masks_ref_list.append(masks_ref)

            sample.update({
                'image_dual': torch.stack(image_ref_list).squeeze(0), # remove empty dim if it was 1 instance
                'label_dual': torch.stack(masks_ref_list).squeeze(0),
            })
            if not self.is_train:
                sample['neg_class_names'] = [self.cid_to_cat[cid] for cid in (self.all_cls - set(uni_cids.tolist()))]


        if self.is_semseg and not self.is_meta:
            _, masks, cat_ids = self._get_info([img_id], sample_max_inst=False)
            all_cats = np.array(self.get_class_ids())
            cat_ids = np.array(cat_ids)
            all_cats_this = cat_ids[None] == all_cats[:, None]

            semmask = masks
            semmask = torch.stack([F.interpolate(semmask_i[None].float(), sample['image'].size()[-2:], mode='nearest')[0] for semmask_i in semmask])
            sample['origin_semmask'] = semmask
            sample['valid_cids'] = all_cats_this.sum(-1)>0

        sample['query_img'] = sample['image']
        sample['support_imgs'] = sample['image_dual']
        sample['support_masks'] = sample['label_dual']
        sample['query_mask'] = sample['label']
        sample.pop('image')
        sample.pop('image_dual')
        sample.pop('label')
        sample.pop('label_dual')

        return sample


def build(image_set, args):
    img_size = 640
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = SemADE(base_image_dir=args.data_root, transform=transform, is_train=(image_set=='train'),
                     is_semseg=(image_set!='train'), shots=args.shots)

    return dataset
