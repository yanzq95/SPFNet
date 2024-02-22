import random

def arugment(img,gt,seg,ns, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    if hflip: 
        img = img[:, ::-1, :].copy()
        gt = gt[:, ::-1, :].copy()
        seg = seg[:, ::-1, :].copy()
        ns = ns[:, ::-1, :].copy()
    if vflip: 
        img = img[::-1, :, :].copy()
        gt = gt[::-1, :, :].copy()
        seg = seg[::-1, :, :].copy()
        ns = ns[::-1, :, :].copy()
    if rot90:
        img = img.transpose(1, 0, 2).copy()
        gt = gt.transpose(1, 0, 2).copy()
        seg = seg.transpose(1, 0, 2).copy()
        ns = ns.transpose(1, 0, 2).copy()

    return img, gt, seg, ns

def get_patch(img, gt, seg,ns, patch_size=16):
    th, tw = img.shape[:2]

    tp = round(patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))
 
    return img[ty:ty + tp, tx:tx + tp, :], gt[ty:ty + tp, tx:tx + tp, :], seg[ty:ty + tp, tx:tx + tp, :], ns[ty:ty + tp, tx:tx + tp, :]