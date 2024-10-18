import numpy as np
import os
import random

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


def get_patch(img, lr, gt, seg,ns, scale, patch_size=16):
    th, tw = img.shape[:2]  ## HR image

    tp = round(patch_size)

    tx = random.randrange(0, (tw - tp))
    ty = random.randrange(0, (th - tp))
    lr_tx = tx // scale
    lr_ty = ty // scale
    lr_tp = tp // scale

    return img[ty:ty + tp, tx:tx + tp], lr[lr_ty:lr_ty + lr_tp, lr_tx:lr_tx + lr_tp], gt[ty:ty + tp, tx:tx + tp], seg[
                                                                                                                  ty:ty + tp,
                                                                                                                  tx:tx + tp], ns[
                                                                                                                  ty:ty + tp,
                                                                                                                  tx:tx + tp]


class RGBDD_Dataset(Dataset):
    """RGB-D-D Dataset."""

    def __init__(self, root_dir="/opt/data/private/dataset/RGB-D-D/", scale=4, downsample='real', train=True,
                 transform=None, isNearest=False, isNoisyLR=False, isNoisyLRRGB=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            downsample (str): kernel type of downsample, real mean use real LR and HR data
            train (bool): train or test
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        self.downsample = downsample
        self.train = train
        self.isNoisyLR = isNoisyLR
        self.isNoisyLRRGB = isNoisyLRRGB
        self.isNearest = isNearest

        if isNoisyLRRGB:
            RGBName = "RGBDD_RGB_Noisy"
            SegName = "RGBDD_RGB_Segment_Noisy"
            NorName = "RGBDD_RGB_Normal_Noisy"
        else:
            RGBName = "RGBDD_RGB"
            SegName = "RGBDD_RGB_Segment"
            NorName = "RGBDD_RGB_Normal/RGBDD_Test_normal"

        if train:
            if self.downsample == 'real':
                self.GTs = []
                self.LRs = []
                self.RGBs = []
                self.Seg = []
                self.NormalS = []
                list_dir = os.listdir('%s/%s/%s/' % (root_dir, "Train", "RGBDD_RGB"))
                for n in list_dir:
                    self.RGBs.append('%s/%s/%s/%s' % (root_dir, "Train", "RGBDD_RGB", n))
                    self.GTs.append('%s/%s/%s/%s_HR_gt.png' % (root_dir, "Train", "RGBDD_GT", n[:-8]))
                    self.LRs.append('%s/%s/%s/%s_LR_fill_depth.png' % (root_dir, "Train", "RGBDD_LR", n[:-8]))
                    self.Seg.append('%s/%s/%s/%s_RGB_segment.png' % (root_dir, "Train", "RGBDD_RGB_Segment", n[:-8]))
                    self.NormalS.append('%s/%s/%s/%s_RGB_normal.png' % (
                    root_dir, "Train", "RGBDD_RGB_Normal/RGBDD_Train_normal", n[:-8]))
            else:
                self.GTs = []
                self.RGBs = []
                self.Seg = []
                self.NormalS = []
                list_dir = os.listdir('%s/%s/%s/' % (root_dir, "Train", "RGBDD_RGB"))
                for n in list_dir:
                    self.RGBs.append('%s/%s/%s/%s' % (root_dir, "Train", "RGBDD_RGB", n))
                    self.GTs.append('%s/%s/%s/%s_HR_gt.png' % (root_dir, "Train", "RGBDD_GT", n[:-8]))
                    self.Seg.append('%s/%s/%s/%s_RGB_segment.png' % (root_dir, "Train", "RGBDD_RGB_Segment", n[:-8]))
                    self.NormalS.append('%s/%s/%s/%s_RGB_normal.png' % (
                        root_dir, "Train", "RGBDD_RGB_Normal/RGBDD_Train_normal", n[:-8]))

        else:
            if self.downsample == 'real':
                self.GTs = []
                self.LRs = []
                self.RGBs = []
                self.Seg = []
                self.NormalS = []
                list_dir = os.listdir('%s/%s/%s/' % (root_dir, "Test", "RGBDD_RGB"))
                for n in list_dir:
                    self.RGBs.append('%s/%s/%s/%s' % (root_dir, "Test", "RGBDD_RGB", n))
                    self.GTs.append('%s/%s/%s/%s_HR_gt.png' % (root_dir, "Test", "RGBDD_GT", n[:-8]))
                    self.LRs.append('%s/%s/%s/%s_LR_fill_depth.png' % (root_dir, "Test", "RGBDD_LR", n[:-8]))
                    self.Seg.append('%s/%s/%s/%s_RGB_segment.png' % (root_dir, "Test", "RGBDD_RGB_Segment", n[:-8]))
                    self.NormalS.append(
                        '%s/%s/%s/%s_RGB_normal.png' % (root_dir, "Test", "RGBDD_RGB_Normal/RGBDD_Test_normal", n[:-8]))
            else:
                self.GTs = []
                self.RGBs = []
                self.Seg = []
                self.NormalS = []
                list_dir = os.listdir('%s/%s/%s/' % (root_dir, "Test", RGBName))
                for n in list_dir:
                    self.RGBs.append('%s/%s/%s/%s' % (root_dir, "Test", RGBName, n))
                    self.GTs.append('%s/%s/%s/%s_HR_gt.png' % (root_dir, "Test", "RGBDD_GT", n[:-8]))
                    self.Seg.append('%s/%s/%s/%s_RGB_segment.png' % (root_dir, "Test", SegName, n[:-8]))
                    self.NormalS.append(
                        '%s/%s/%s/%s_RGB_normal.png' % (root_dir, "Test", NorName, n[:-8]))

    def __len__(self):
        return len(self.GTs)

    def __getitem__(self, idx):
        if self.downsample == 'real':
            image = np.array(Image.open(self.RGBs[idx]).convert("RGB")).astype(np.float32)
            name = self.RGBs[idx][-22:-8]
            gt = np.array(Image.open(self.GTs[idx])).astype(np.float32)
            seg = np.array(Image.open(self.Seg[idx])).astype(np.float32)
            ns = np.array(Image.open(self.NormalS[idx])).astype(np.float32)
            h, w = gt.shape
            s = self.scale
            lr = np.array(Image.open(self.LRs[idx]).resize((w // s, h // s), Image.BICUBIC)).astype(np.float32)

        else:
            image = Image.open(self.RGBs[idx]).convert("RGB")
            name = self.RGBs[idx][-22:-8]
            image = np.array(image).astype(np.float32)
            gt = Image.open(self.GTs[idx])
            seg = np.array(Image.open(self.Seg[idx])).astype(np.float32)
            ns = np.array(Image.open(self.NormalS[idx])).astype(np.float32)
            w, h = gt.size
            s = self.scale
            if self.isNearest:
                lr = np.array(gt.resize((w // s, h // s), Image.NEAREST)).astype(np.float32)
            else:
                lr = np.array(gt.resize((w // s, h // s), Image.BICUBIC)).astype(np.float32)
            gt = np.array(gt).astype(np.float32)

        # normalization
        max_out = np.max(lr)
        min_out = np.min(lr)
        
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
            max_gt = np.max(gt)
            min_gt = np.min(gt)
            gt = (gt - min_gt) / (max_gt - min_gt)
            lr = (lr - min_gt) / (max_gt - min_gt)
            image, lr, gt, seg, ns = get_patch(image, lr, gt, seg, ns, scale=self.scale, patch_size=256)
        else:
            lr = (lr - min_out) / (max_out - min_out)

        if self.isNoisyLR or self.isNoisyLRRGB:
            lr_minn = np.min(lr)
            lr_maxx = np.max(lr)
            np.random.seed(42)
            gaussian_noise = np.random.normal(0, 0.07, lr.shape)
            lr = lr + gaussian_noise
            lr = np.clip(lr, lr_minn, lr_maxx)

        if self.transform:
            image = self.transform(image).float()
            seg = self.transform(np.expand_dims(seg, 2)).float()
            ns = self.transform(ns).float()
            gt = self.transform(np.expand_dims(gt, 2)).float()
            lr = self.transform(np.expand_dims(lr, 2)).float()

        sample = {'guidance': image, 'lr': lr, 'gt': gt, 'seg': seg, 'ns': ns, 'max': max_out, 'min': min_out, 'name':name}

        return sample
