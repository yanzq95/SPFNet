import numpy as np
import os
import random
 
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from scipy.ndimage import gaussian_filter

def get_patch(img, lr, gt, seg,ns, scale, patch_size=16):
    th, tw = img.shape[:2]  ## HR image

    tp = round(patch_size)

    tx = random.randrange(0, (tw - tp))
    ty = random.randrange(0, (th - tp))
    lr_tx = tx // scale
    lr_ty = ty // scale
    lr_tp = tp // scale

    return img[ty:ty + tp, tx:tx + tp], lr[lr_ty:lr_ty + lr_tp, lr_tx:lr_tx + lr_tp]\
        , gt[ty:ty + tp, tx:tx + tp], seg[ty:ty + tp, tx:tx + tp], ns[ty:ty + tp, tx:tx + tp]


class TOFDSR_Dataset(Dataset):

    def __init__(self, root_dir="/opt/data/private/dataset/", scale=4, downsample='real', train=True, txt_file='./TOFDSR_Filled_Train.txt' ,
                 transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        self.downsample = downsample
        self.train = train
        self.image_list = txt_file
        with open(self.image_list, 'r') as f:
            self.filename = f.readlines()

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, idx):

        sample_path = self.filename[idx].strip('\n')
        sample_path_ = sample_path.split(',')
        rgb_path = sample_path_[0]
        gt_path = sample_path_[1]
        lr_path = sample_path_[2]
        normal_path = sample_path_[3]
        seg_path = sample_path_[4]
        name = gt_path[20:-4]
        rgb_path = os.path.join(self.root_dir, rgb_path)
        gt_path = os.path.join(self.root_dir, gt_path)
        lr_path = os.path.join(self.root_dir, lr_path)
        normal_path = os.path.join(self.root_dir, normal_path)
        seg_path = os.path.join(self.root_dir, seg_path)

        if self.downsample == 'real':
            image = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32)
            gt = np.array(Image.open(gt_path)).astype(np.float32)
            seg = np.array(Image.open(seg_path)).astype(np.float32)
            ns = np.array(Image.open(normal_path)).astype(np.float32)

            h, w = gt.shape
            s = self.scale
            lr = np.array(Image.open(lr_path).resize((w // s, h // s), Image.BICUBIC)).astype(np.float32)

        else:
            image = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32)
            gt = Image.open(gt_path)
            seg = np.array(Image.open(seg_path)).astype(np.float32)
            ns = np.array(Image.open(normal_path)).astype(np.float32)

            w, h = gt.size
            s = self.scale
            lr = np.array(gt.resize((w // s, h // s), Image.BICUBIC)).astype(np.float32)
            gt = np.array(gt).astype(np.float32)

        max_out = 5000.0
        min_out = 0.0


        image_max = np.max(image)
        image_min = np.min(image)
        image = (image - image_min) / (image_max - image_min)

        seg_max = np.max(seg)
        seg_min = np.min(seg)
        seg = (seg - seg_min) / (seg_max - seg_min)

        ns_max = np.max(ns)
        ns_min = np.min(ns)
        ns = (ns - ns_min) / (ns_max - ns_min)

        if self.train:
            gt = (gt - min_out) / (max_out - min_out)
            lr = (lr - min_out) / (max_out - min_out)
            image, lr, gt, seg, ns = get_patch(image, lr, gt, seg, ns, scale=self.scale, patch_size=256)
        else:
            lr = (lr - min_out) / (max_out - min_out)
            
        if self.transform:
            image = self.transform(image).float()
            seg = self.transform(np.expand_dims(seg, 2)).float()
            ns = self.transform(ns).float()
            gt = self.transform(np.expand_dims(gt, 2)).float()
            lr = self.transform(np.expand_dims(lr, 2)).float()

        sample = {'guidance': image, 'lr': lr, 'gt': gt, 'seg': seg, 'ns': ns, 'max': max_out, 'min': min_out,'name': name}

        return sample
