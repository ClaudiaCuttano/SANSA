r""" COCO-20i few-shot semantic segmentation dataset """
import os
import pickle
from os.path import join

import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


class DatasetCOCO(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.test_episodes= 1000
        self.benchmark = 'coco'
        self.shot = shot
        self.split_coco = split if split == 'val2014' else 'train2014'
        self.base_path = join(datapath, 'COCO2014')
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize

        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else self.test_episodes

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize, base_query, base_supports = self.load_frame()

        query_img = self.transform(query_img)

        query_mask = query_mask.float()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
            base_query = F.interpolate(base_query.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            base_supports[midx] = F.interpolate(base_supports[midx].unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:],
                                            mode='nearest').squeeze()
        support_masks = torch.stack(support_masks)
        base_supports = torch.stack(base_supports)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'base_masks':[base_supports,base_query]
        }
        return batch

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        if self.fold != -1:
            class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
            class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        else:
            # fold is set to '-1' when training the generalist model, i.e. train on all folds.
            class_ids_val = list(range(self.nclass))            
            class_ids_trn = list(range(self.nclass))
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val
        return class_ids

    @staticmethod
    def load_pickle(pickle_path):
        with open(pickle_path, 'rb') as fp:
            meta = pickle.load(fp)
        return meta

    def build_img_metadata_classwise(self):
        if self.fold != -1:
            with open(f'{self.base_path}/splits/{self.split}/fold{self.fold}.pkl', 'rb') as f:
                img_metadata_classwise = pickle.load(f)
        else:
            split_meta_path = [f'{self.base_path}/splits/{self.split}/fold{str(idx)}.pkl' for idx in range(4)]
            # split_1 = f'{self.base_path}/splits/{self.split}/fold1.pkl'
            
            if self.split == 'trn':
                # collect all classes metadata; it's enough to merge 0 and 1, since fold 0 has all classes
                # except those % 4 == 0, which are present in 1
                metadata_0 = self.load_pickle(split_meta_path[0])
                metadata_1 = self.load_pickle(split_meta_path[1])
                for i_c, class_meta in metadata_0.items():
                    if len(class_meta) == 0:
                        metadata_0[i_c] = metadata_1[i_c]
                img_metadata_classwise = metadata_0
            else:
                # when validating, each pickle contains only a portion of classes
                # so we need to read them all
                split_meta = [self.load_pickle(meta_path) for meta_path in split_meta_path]
                img_metadata_classwise = {k:[] for k,v  in split_meta[0].items()}
                for meta in split_meta:
                    for class_id in meta:
                        if len(img_metadata_classwise[class_id]) == 0 and len(meta[class_id]) > 0:    
                            img_metadata_classwise[class_id] = meta[class_id] 
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = torch.tensor(np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png')))
        return mask

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        query_mask = self.read_mask(query_name)

        org_qry_imsize = query_img.size
        base_query_mask = query_mask.clone()
        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        all_classes = torch.unique(base_query_mask).tolist()
        for cl_i in all_classes:
            if (cl_i - 1) not in self.class_ids:
                base_query_mask[base_query_mask == cl_i] = 0

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        support_imgs, support_masks, base_support_masks = [], [], []
        for support_name in support_names:
            support_imgs.append(Image.open(os.path.join(self.base_path, support_name)).convert('RGB'))
            support_mask = self.read_mask(support_name)
            base_support_mask = support_mask.clone()
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1

            all_classes = torch.unique(base_support_mask).tolist()
            for cl_i in all_classes:
                if (cl_i - 1) not in self.class_ids:
                    base_support_mask[base_support_mask == cl_i] = 0

            support_masks.append(support_mask)
            base_support_masks.append(base_support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize, base_query_mask, base_support_masks


def build(image_set, args):
    img_size = 640
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = DatasetCOCO(datapath=args.data_root, fold=args.fold, transform=transform,
                 shot=args.shots, use_original_imgsize=False, split=image_set)

    return dataset
