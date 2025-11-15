r""" ISIC few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from torchvision import transforms


# 1:1867 2:519 3:208
class DatasetISIC(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num=600):
        self.split = 'test' if split in ['val', 'test'] else 'trn'
        self.benchmark = 'isic'
        self.shot = shot
        self.nclass = 3
        if self.split == 'test':
            self.num = num
            self.categories = ['1','2','3']
        else:
            self.num = num
            self.categories = ['1','2','3']
            
        self.base_path = os.path.join(datapath, 'ISIC')
        self.img_path = os.path.join(self.base_path, 'ISIC2018_Task1-2_Training_Input')
        self.ann_path = os.path.join(self.base_path, 'ISIC2018_Task1_Training_GroundTruth')
        self.transform = transform

        self.class_ids = range(0, 3)
        self.img_metadata_classwise = self.build_img_metadata_classwise()       

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)

        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)
        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,

                 'class_id': torch.tensor(class_sample)}

        return batch

    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_id = query_name.split('/')[-1].split('.')[0]
        query_name = os.path.join(self.ann_path, query_id) + '_segmentation.png'
        support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
        support_names = [os.path.join(self.ann_path, sid) + '_segmentation.png' for name, sid in zip(support_names, support_ids)]

        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        class_id = idx % len(self.class_ids)
        class_sample = self.categories[class_id]

        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_id

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for cat in self.categories:
            img_metadata_classwise[cat] = []

        build_path = self.img_path

        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(build_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata_classwise[cat] += [img_path]
        print('Total (%s) %s images are : %d' % (self.split, self.benchmark, self.__len__()))
        return img_metadata_classwise
    

def build(image_set, args):
    img_size = 518
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = DatasetISIC(datapath=args.data_root, fold=args.fold, transform=transform,
                 shot=args.shots, split=image_set)

    return dataset
