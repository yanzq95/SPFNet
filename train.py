import argparse
from model.SPFNet import *
from model.common import *

from data.nyu_dataloader import *
from data.rgbdd_dataloader import *
from data.tofdsr_dataloader import *
from utils import calc_rmse, rgbdd_calc_rmse, tofdsr_calc_rmse

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import logging
from datetime import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=4, help='scale factor')
parser.add_argument('--lr',  default='0.0001', type=float, help='learning rate')
parser.add_argument('--result',  default='experiment', help='learning rate')
parser.add_argument('--epoch',  default=200, type=int, help='max epoch')
parser.add_argument('--device',  default="0", type=str, help='which gpu use')
parser.add_argument("--decay_iterations", type=list, default=[5e4, 1e5, 2e5], help="steps to start lr decay")
parser.add_argument("--num_feats", type=int, default=42, help="channel number of the middle hidden layer")
parser.add_argument("--gamma", type=float, default=0.2, help="decay rate of learning rate")
parser.add_argument("--root_dir", type=str, default='./dataset/NYU-v2', help="root dir of dataset")
parser.add_argument("--batchsize", type=int, default=1, help="batchsize of training dataloader")
parser.add_argument('--tiny_model', action='store_true', help='tiny model')

opt = parser.parse_args()
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

s = datetime.now().strftime('%Y%m%d%H%M%S')
dataset_name = opt.root_dir.split('/')[-1]
result_root = '%s/%s-lr_%s-s_%s-%s-b_%s' % (opt.result, s, opt.lr, opt.scale, dataset_name, opt.batchsize)
if not os.path.exists(result_root):
    os.mkdir(result_root)

logging.basicConfig(filename='%s/train.log' % result_root, format='%(asctime)s %(message)s', level=logging.INFO)
logging.info(opt)

net = SPFNet(num_feats=opt.num_feats, kernel_size=3, scale=opt.scale, reduction=4, tiny_model=opt.tiny_model).cuda()
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=opt.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.decay_iterations, gamma=opt.gamma)
net.train()

data_transform = transforms.Compose([transforms.ToTensor()])
up = nn.Upsample(scale_factor=opt.scale, mode='bicubic')

if dataset_name == 'NYU-v2':
    test_minmax = np.load('%s/test_minmax.npy' % opt.root_dir)
    train_dataset = NYU_v2_datset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform, train=True)
    test_dataset = NYU_v2_datset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform, train=False)
if dataset_name == 'RGB-D-D':
    train_dataset = RGBDD_Dataset(root_dir=opt.root_dir, scale=opt.scale, downsample='real', train=True, transform=data_transform)
    test_dataset = RGBDD_Dataset(root_dir=opt.root_dir, scale=opt.scale, downsample='real', train=False, transform=data_transform)
if dataset_name == 'TOFDSR':
    train_dataset = TOFDSR_Dataset(root_dir=opt.root_dir, scale=opt.scale, downsample='real', train=True,
                                  txt_file="./data/TOFDSR_Filled_Train.txt", transform=data_transform)
    test_dataset = TOFDSR_Dataset(root_dir=opt.root_dir, scale=opt.scale, downsample='real', train=False,
                                 txt_file="./data/TOFDSR_Filled_Test.txt", transform=data_transform)


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=8)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

max_epoch = opt.epoch
num_train = len(train_dataloader)
best_rmse = 10.0
best_epoch = 0
for epoch in range(max_epoch):
    # ---------
    # Training
    # ---------
    net.train()
    running_loss = 0.0

    t = tqdm(iter(train_dataloader), leave=True, total=len(train_dataloader))

    for idx, data in enumerate(t):
        batches_done = num_train * epoch + idx
        optimizer.zero_grad()
        guidance, lr, gt, seg, ns = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda(), data[
            'seg'].cuda(), data['ns'].cuda()

        out = net((guidance, lr, seg,ns))

        if dataset_name == 'TOFDSR':
            mask = (gt >= 0.02) & (gt <= 1)
            gt = gt[mask]
            out = out[mask]

        loss = criterion(out, gt)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.data.item()
        running_loss_50 = running_loss

        if idx % 50 == 0:
            running_loss_50 /= 50
            t.set_description('[train epoch:%d] loss: %.8f' % (epoch + 1, running_loss_50))
            t.refresh()

    logging.info('epoch:%d iteration:%d running_loss:%.10f' % (epoch + 1, batches_done + 1, running_loss / num_train))

    # -----------
    # Validating
    # -----------
    if (epoch % 2 == 0) and (epoch < 30):
        with torch.no_grad():

            net.eval()
            if dataset_name == 'NYU-v2_Our':
                rmse = np.zeros(449)
            if dataset_name == 'RGB-D-D':
                rmse = np.zeros(405)
            if dataset_name == 'TOFDSR':
                rmse = np.zeros(560)
            t = tqdm(iter(test_dataloader), leave=True, total=len(test_dataloader))

            for idx, data in enumerate(t):
                if dataset_name == 'NYU-v2_Our':
                    guidance, lr, gt, seg, ns = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda(), data[
                        'seg'].cuda(), data['ns'].cuda()

                    out = net((guidance, lr, seg, ns))
                    minmax = test_minmax[:, idx]
                    minmax = torch.from_numpy(minmax).cuda()
                    rmse[idx] = calc_rmse(gt[0, 0], out[0, 0], minmax)
                if dataset_name == 'RGB-D-D':
                    guidance, lr, gt, seg, ns, max, min = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda(), data['seg'].cuda(), data['ns'].cuda(), data[
                        'max'].cuda(), data['min'].cuda()
                    out = net((guidance, lr, seg, ns))
                    minmax = [max, min]
                    rmse[idx] = rgbdd_calc_rmse(gt[0, 0], out[0, 0], minmax)
                if dataset_name == 'TOFDSR':
                    guidance, lr, gt, seg, ns, max, min = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda(), \
                    data['seg'].cuda(), data['ns'].cuda(), data[
                        'max'].cuda(), data['min'].cuda()
                    out = net((guidance, lr, seg, ns))
                    minmax = [max, min]
                    rmse[idx] = tofdsr_calc_rmse(gt[0, 0], out[0, 0], minmax)

                    t.set_description('[validate] rmse: %f' % rmse[:idx + 1].mean())
                    t.refresh()
            r_mean = rmse.mean()
            if r_mean < best_rmse:
                best_rmse = r_mean
                best_epoch = epoch
                torch.save(net.state_dict(),
                           os.path.join(result_root, "modelbest%f_8%d.pth" % (best_rmse, best_epoch + 1)))
            logging.info(
                '---------------------------------------------------------------------------------------------------------------------------')
            logging.info('epoch:%d lr:%f-------mean_rmse:%f (BEST: %f @epoch%d)' % (
                epoch + 1, scheduler.get_last_lr()[0], r_mean, best_rmse, best_epoch + 1))
            logging.info(
                '---------------------------------------------------------------------------------------------------------------------------')
    elif epoch >= 30:
        with torch.no_grad():

            net.eval()
            if dataset_name == 'NYU-v2_Our':
                rmse = np.zeros(449)
            elif dataset_name == 'RGB-D-D':
                rmse = np.zeros(405)
            else:
                rmse = np.zeros(560)
            t = tqdm(iter(test_dataloader), leave=True, total=len(test_dataloader))

            for idx, data in enumerate(t):
                if dataset_name == 'NYU-v2_Our':
                    guidance, lr, gt, seg, ns = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda(), data[
                        'seg'].cuda(), data['ns'].cuda()

                    out = net((guidance, lr, seg, ns))
                    minmax = test_minmax[:, idx]
                    minmax = torch.from_numpy(minmax).cuda()
                    rmse[idx] = calc_rmse(gt[0, 0], out[0, 0], minmax)
                elif dataset_name == 'RGB-D-D':
                    guidance, lr, gt, seg, ns, max, min = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda(), data['seg'].cuda(), data['ns'].cuda(), data[
                        'max'].cuda(), data['min'].cuda()
                    out = net((guidance, lr, seg, ns))
                    minmax = [max, min]
                    rmse[idx] = rgbdd_calc_rmse(gt[0, 0], out[0, 0], minmax)
                else:
                    guidance, lr, gt, seg, ns, max, min = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda(), \
                    data['seg'].cuda(), data['ns'].cuda(), data[
                        'max'].cuda(), data['min'].cuda()
                    out = net((guidance, lr, seg, ns))
                    minmax = [max, min]
                    rmse[idx] = tofdsr_calc_rmse(gt[0, 0], out[0, 0], minmax)

                    t.set_description('[validate] rmse: %f' % rmse[:idx + 1].mean())
                    t.refresh()
            r_mean = rmse.mean()
            if r_mean < best_rmse:
                best_rmse = r_mean
                best_epoch = epoch
            torch.save(net.state_dict(),
                       os.path.join(result_root, "modelRmse%f_8%d.pth" % (r_mean, epoch + 1)))
            logging.info(
                '---------------------------------------------------------------------------------------------------------------------------')
            logging.info('epoch:%d lr:%f-------mean_rmse:%f (BEST: %f @epoch%d)' % (
                epoch + 1, scheduler.get_last_lr()[0], r_mean, best_rmse, best_epoch + 1))
            logging.info(
                '---------------------------------------------------------------------------------------------------------------------------')
