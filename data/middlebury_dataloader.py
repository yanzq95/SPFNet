from torchvision import transforms
import numpy as np
import os
import random

from torch.utils.data import Dataset, DataLoader
from PIL import Image

def modcrop(image, modulo):
    h, w = image.shape[0], image.shape[1]
    h = h - h % modulo
    w = w - w % modulo

    return image[:h,:w]

class Middlebury_dataset(Dataset):
    """RGB-D-D Dataset."""

    def __init__(self, root_dir, scale=8, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.transform = transform
        self.scale = scale

        self.GTs = []
        self.RGBs = []
        self.Seg = []
        self.NormalS = []
        
        list_dir = os.listdir(root_dir)
        for name in list_dir:
            if name.find('output_color') > -1:
                self.RGBs.append('%s/%s' % (root_dir, name))
            elif name.find('output_depth') > -1:
                self.GTs.append('%s/%s' % (root_dir, name))
            elif name.find('output_segment') > -1:
                self.Seg.append('%s/%s' % (root_dir, name))
            elif name.find('output_normal') > -1:
                self.NormalS.append('%s/%s' % (root_dir, name))
        self.RGBs.sort()
        self.GTs.sort()
        self.Seg.sort()
        self.NormalS.sort()

    def __len__(self):
        return len(self.GTs)

    def __getitem__(self, idx):
        
        image = np.array(Image.open(self.RGBs[idx]))
        gt = np.array(Image.open(self.GTs[idx]))
        seg = np.array(Image.open(self.Seg[idx]))
        ns = np.array(Image.open(self.NormalS[idx]))

        assert gt.shape[0] == image.shape[0] and gt.shape[1] == image.shape[1]
        s = self.scale  
        image = modcrop(image, s)
        gt = modcrop(gt, s)
        seg = modcrop(seg, s)
        ns = modcrop(ns, s)

        h, w = gt.shape[0], gt.shape[1]
        s = self.scale
        
        gt_max = 255.0
        gt_min = 0.0
        gt = (gt - gt_min) / (gt_max - gt_min)

        lr = np.array(Image.fromarray(gt).resize((w//s,h//s),Image.BICUBIC)).astype(np.float32)

        image_max = np.max(image)
        image_min = np.min(image)
        image = (image-image_min)/(image_max-image_min)

        seg_max = np.max(seg)
        seg_min = np.min(seg)
        seg = (seg - seg_min) / (seg_max - seg_min)

        ns_max = np.max(ns)
        ns_min = np.min(ns)
        ns = (ns - ns_min) / (ns_max - ns_min)
        

        if self.transform:
            image = self.transform(image).float()
            seg = self.transform(np.expand_dims(seg, 2)).float()
            ns = self.transform(ns).float()
            gt = self.transform(np.expand_dims(gt,2)).float()
            lr = self.transform(np.expand_dims(lr,2)).float()

        sample = {'guidance': image, 'lr': lr, 'gt': gt, 'seg': seg, 'ns': ns, 'max':gt_max, 'min': gt_min}
        return sample
