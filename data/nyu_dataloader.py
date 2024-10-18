from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from data.common import get_patch,arugment
 
def modcrop(image, modulo):
    h, w = image.shape[0], image.shape[1]
    h = h - h % modulo
    w = w - w % modulo

    return image[:h,:w]

class NYU_v2_datset(Dataset):

    def __init__(self, root_dir, scale=8, train=True, transform=None, isNearest=False, isNoisyLR=False, isNoisyLRRGB=False):
        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        self.train = train
        self.isNoisyLR = isNoisyLR
        self.isNoisyLRRGB = isNoisyLRRGB
        self.isNearest = isNearest

        if isNoisyLRRGB:
            if train:
                self.depths = np.load('%s/train_depth_split.npy' % root_dir)
                self.images = np.load('%s/train_images_Noisy.npy' % root_dir)
                self.segs = np.load('%s/train_Seg_Noisy.npy' % root_dir)
                self.norms = np.load('%s/train_Normal_Noisy.npy' % root_dir)
            else:
                self.depths = np.load('%s/test_depth.npy' % root_dir)
                self.images = np.load('%s/test_images_Noisy.npy' % root_dir)
                self.segs = np.load('%s/test_Seg_Noisy.npy' % root_dir)
                self.norms = np.load('%s/test_Normal_Noisy.npy' % root_dir)
        else:
            if train:
                self.depths = np.load('%s/train_depth_split.npy' % root_dir)
                self.images = np.load('%s/train_images_split.npy' % root_dir)
                self.segs = np.load('%s/train_Seg.npy' % root_dir)
                self.norms = np.load('%s/train_Normal.npy' % root_dir)
            else:
                self.depths = np.load('%s/test_depth.npy' % root_dir)
                self.images = np.load('%s/test_images_v2.npy' % root_dir)
                self.segs = np.load('%s/test_Seg.npy' % root_dir)
                self.norms = np.load('%s/test_Normal.npy' % root_dir)

    def __len__(self):
        return self.depths.shape[0]

    def __getitem__(self, idx):
        depth = self.depths[idx]
        image = self.images[idx]
        seg = self.segs[idx]
        ns = self.norms[idx]
        
        s = self.scale
        if s== 64:           
          image = modcrop(image, s)
          depth = modcrop(depth, s)
          seg = modcrop(seg, s)
          ns = modcrop(ns, s)
        
        if self.train:
            image, depth,seg, ns = get_patch(img=image, gt=np.expand_dims(depth, 2), seg=np.expand_dims(seg, 2), ns=ns,
                                          patch_size=256)
            image, depth,seg, ns = arugment(img=image, gt=depth, seg=seg, ns=ns)
        h, w = depth.shape[:2]
        s = self.scale

        if self.isNearest:
            lr = np.array(Image.fromarray(depth.squeeze()).resize((w // s, h // s), Image.NEAREST))
        else:
            lr = np.array(Image.fromarray(depth.squeeze()).resize((w // s, h // s), Image.BICUBIC))

        if not self.train:
            np.random.seed(42)

        if self.isNoisyLR or self.isNoisyLRRGB:
            lr_minn = np.min(lr)
            lr_maxx = np.max(lr)
            #np.random.seed(42)
            gaussian_noise = np.random.normal(0, 0.07, lr.shape)
            lr = lr + gaussian_noise
            lr = np.clip(lr, lr_minn, lr_maxx)
            
        if self.transform:
            image = self.transform(image).float()
            seg = self.transform(seg).float()
            ns = self.transform(ns).float()
            depth = self.transform(depth).float()
            lr = self.transform(np.expand_dims(lr, 2)).float()

        sample = {'guidance': image, 'lr': lr, 'gt': depth, 'seg': seg, 'ns': ns}

        return sample
