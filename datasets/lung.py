r""" Chest X-ray few-shot semantic segmentation dataset """
import os
import glob

from os.path import join
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from torchvision import transforms


# test: 704
class DatasetLung(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num=600):
        self.split = 'test' if split in ['val', 'test'] else 'trn'
        self.nclass = 1
        self.benchmark = 'lung'
        self.shot = shot
        self.num = num

        self.base_path = join(datapath, 'LungSegmentation')
        self.img_path = join(self.base_path, 'CXR_png')
        self.ann_path = join(self.base_path, 'masks')
        self.transform = transform

        self.categories = ['1']

        self.class_ids = range(0, 1)
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
        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]
        if query_name.find('MCUCXR')!=-1:
            query_id = query_name
        else:
            query_id = query_name[:-9] + '.png'

        query_img = Image.open(os.path.join(self.img_path, os.path.basename(query_id))).convert('RGB')
        support_ids = []
        for name in support_names:
            if name.find('MCUCXR')!=-1:
                support_ids.append(os.path.basename(name))
            else:
                support_ids.append(os.path.basename(name)[:-9] + '.png')

        support_names = [os.path.join(self.img_path, sid) for sid in support_ids]
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

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

        if self.split == 'test':
            build_path = self.ann_path
        elif self.split == 'aux':
            build_path = self.aux_path
        
        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % build_path)])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
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
    dataset = DatasetLung(datapath=args.data_root, fold=args.fold, transform=transform,
                 shot=args.shots, split=image_set)

    return dataset
