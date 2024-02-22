import torch.nn as nn
import torch

import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False,
                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class DownBlock(nn.Module):
    def __init__(self, scale, in_channels=None, out_channels=None):
        super(DownBlock, self).__init__()
        down_m = []
        for _ in range(scale):
            down_m.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.PReLU()
                )
            )
        self.downModule = nn.Sequential(*down_m)

    def forward(self, x):
        x = self.downModule(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, scale, in_channels=None, out_channels=None):
        super(UpBlock, self).__init__()
        up_m = []
        for _ in range(scale):
            up_m.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True),
                    nn.PReLU()
                )
            )
        self.downModule = nn.Sequential(*up_m)

    def forward(self, x):
        x = self.downModule(x)
        return x


class MGF(nn.Module):
    def __init__(self, num_feature, kernel_size, reduction):
        super(MGF, self).__init__()
        
        self.reduction = reduction
        self.num = kernel_size * kernel_size

        self.aff_scale_const = nn.Parameter(0.5 * self.num * torch.ones(1))

        self.d1 = default_conv(num_feature, num_feature, 3)
        self.g1 = default_conv(num_feature, num_feature, 3)

        self.depth_kernel = nn.Sequential(
            default_conv(num_feature, num_feature, 1),
            nn.ReLU(True),
            default_conv(num_feature, kernel_size ** 2, 1)
        )

        self.guide_kernel = nn.Sequential(
            default_conv(num_feature, num_feature, 1),
            nn.ReLU(True),
            default_conv(num_feature, kernel_size ** 2, 1)
        )

        self.d2 = default_conv(kernel_size ** 2, kernel_size ** 2, 1)
        self.g2 = default_conv(kernel_size ** 2, kernel_size ** 2, 1)

        self.d3 = default_conv(num_feature, num_feature, 3)
        self.g3 = default_conv(num_feature, num_feature, 3)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=1)
        self.inputs_conv = ResidualGroup(default_conv, num_feature, 3, reduction=self.reduction, n_resblocks=1)

        self.guide_conv = ResidualGroup(default_conv, num_feature, 3, reduction=self.reduction, n_resblocks=1)

    def getKernel(self, input_kernel):

        fuse_kernel = torch.tanh(input_kernel) / (
                    self.aff_scale_const + 1e-8)
        abs_kernel = torch.abs(fuse_kernel)
        abs_kernel_sum = torch.sum(abs_kernel, dim=1, keepdim=True) + 1e-4
        abs_kernel_sum[abs_kernel_sum < 1.0] = 1.0
        fuse_kernel = fuse_kernel / abs_kernel_sum

        return fuse_kernel

    def forward(self, depth, guide, S):
        b, c, h, w = depth.size()

        depth = self.d1(depth)
        guide = self.g1(guide)

        inputs_depth = self.inputs_conv(depth)
        guide_kernel = self.guide_kernel(guide)
        guide_kernel = self.g2(guide_kernel * S + guide_kernel)
        guide_kernel = self.getKernel(guide_kernel)
        unfold_inputs_depth = self.unfold(inputs_depth).view(b, c, -1, h, w)
        w_depth = torch.einsum('bkhw, bckhw->bchw', [guide_kernel, unfold_inputs_depth]) + inputs_depth

        inputs_guide = self.guide_conv(guide)
        depth_kernel = self.depth_kernel(w_depth)
        depth_kernel = self.d2(depth_kernel * S + depth_kernel)
        depth_kernel = self.getKernel(depth_kernel)
        unfold_inputs_guide = self.unfold(inputs_guide).view(b, c, -1, h, w)
        w_guide = torch.einsum('bkhw, bckhw->bchw', [depth_kernel, unfold_inputs_guide]) + inputs_guide

        out_depth = self.d3(w_depth)
        out_guide = self.g3(w_guide)

        return out_depth, out_guide

class OPE(nn.Module):

    def __init__(self, input_planes, weight_planes,reduction):
        super().__init__()
        self.reduction = reduction
        self.w1 = MGF(weight_planes, 3,self.reduction)
        self.w2 = MGF(weight_planes, 3,self.reduction)
        self.w3 = MGF(weight_planes, 3,self.reduction)
        self.conv11 = default_conv(weight_planes,weight_planes,3)
        self.conv12 = default_conv(weight_planes,weight_planes,3)
        self.conv21 = default_conv(weight_planes,weight_planes,3)
        self.conv22 = default_conv(3*weight_planes,weight_planes,1)
        self.conv31 = default_conv(weight_planes, input_planes, 3)

        self.c1 = default_conv(weight_planes,weight_planes,3)
        self.c2 = default_conv(weight_planes,weight_planes,3)
        self.c3 = default_conv(weight_planes,weight_planes,3)
        self.c4 = default_conv(weight_planes,weight_planes,3)

    def forward(self, depth, image, seg, ns, S_seg, S_ns, S_rgb): # depth?? torch.Size([1, 40, 36, 36]), image?? torch.Size([1, 40, 36, 36])??seg: torch.Size([1, 40, 36, 36]), ns?? torch.Size([1, 40, 36, 36])
        depth = self.c1(depth)
        image = self.c2(image)
        seg = self.c3(seg)
        ns = self.c4(ns)

        out_drsn, out_ns = self.w2(depth, ns, S_ns)#.view(B, Ci, -1)
        out_drsn = self.conv12(out_drsn + depth)
        out_drs, out_seg = self.w3(out_drsn, seg, S_seg)#.view(B, Ci, -1)
        out_drs = self.conv21(out_drs + out_drsn)
        out_dr, out_rgb = self.w1(out_drs, image,S_rgb)#.view(B, Ci, -1)
        out_dr = self.conv11(out_dr + out_drs)

        cat1 = torch.cat([out_dr,out_drsn,out_drs],dim=1)
        out0 = self.conv22(cat1) + depth
        out = self.conv31(out0)
        return out, out_rgb, out_seg, out_ns

class APP(nn.Module):
    def __init__(self, input_planes, weight_planes, scale_factor,reduction, tiny_model=False):
        super(APP, self).__init__()
        
        self.tiny_model = tiny_model
        
        if self.tiny_model:
            self.red1 = reduction
            self.red2 = reduction
            self.red3 = reduction
        else:
            self.red1 = reduction
            self.red2 = 2*reduction
            self.red3 = 4*reduction

        self.d1 = nn.Sequential(
            default_conv(input_planes, weight_planes, 1),
            ResidualGroup(default_conv, weight_planes, 3, reduction=self.red1, n_resblocks=2)
        )
        self.r1 = ResidualGroup(default_conv, weight_planes, 3, reduction=self.red2, n_resblocks=2)
        self.n1 = ResidualGroup(default_conv, weight_planes, 3, reduction=self.red2, n_resblocks=2)
        self.s1 = ResidualGroup(default_conv, weight_planes, 3, reduction=self.red2, n_resblocks=2)
        self.c2 = nn.Sequential(
            default_conv(input_planes, input_planes + weight_planes, 1),
            ResidualGroup(default_conv, input_planes + weight_planes, 3, reduction=self.red3, n_resblocks=4)
        )
        self.rd = nn.Sequential(
            default_conv(input_planes + 3 * weight_planes, input_planes, 1),
            ResidualGroup(default_conv, input_planes, 3, reduction=self.red3, n_resblocks=4)
        )

        self.rgb_down = DownBlock(scale_factor, weight_planes, weight_planes)
        self.ns_down = DownBlock(scale_factor, weight_planes, weight_planes)
        self.seg_down = DownBlock(scale_factor, weight_planes, weight_planes)

        self.rgb_up = UpBlock(scale_factor, weight_planes, weight_planes)
        self.ns_up = UpBlock(scale_factor, weight_planes, weight_planes)
        self.seg_up = UpBlock(scale_factor, weight_planes, weight_planes)

        self.d_g = OPE(input_planes, weight_planes,reduction=self.red1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.cc1 = default_conv(weight_planes, weight_planes, 1)
        self.cc2 = default_conv(weight_planes, weight_planes, 1)
        self.cc3 = default_conv(weight_planes, weight_planes, 1)

    def forward(self, depth, rgb, seg, ns):
        rgb = self.rgb_down(rgb)
        ns = self.ns_down(ns)
        seg = self.seg_down(seg)

        depth1 = self.d1(depth)
        rgb = self.r1(rgb)
        ns = self.n1(ns)
        seg = self.s1(seg)

        depth_unfold = F.unfold(depth1, kernel_size=(3, 3), padding=1)
        rgb_unfold = F.unfold(rgb, kernel_size=(3, 3), padding=1)
        rgb_unfold_p = rgb_unfold.permute(0, 2, 1)
        ns_unfold = F.unfold(ns, kernel_size=(3, 3), padding=1)
        ns_unfold_p = ns_unfold.permute(0, 2, 1)
        seg_unfold = F.unfold(seg, kernel_size=(3, 3), padding=1)
        seg_unfold_p = seg_unfold.permute(0, 2, 1)

        rgb_unfold_p = F.normalize(rgb_unfold_p, dim=2)
        ns_unfold_p = F.normalize(ns_unfold_p, dim=2)
        seg_unfold_p = F.normalize(seg_unfold_p, dim=2)
        depth_unfold_p = F.normalize(depth_unfold, dim=1)

        R_r = torch.bmm(rgb_unfold_p, depth_unfold_p)
        Score_dr = torch.diagonal(R_r,dim1=-2,dim2=-1)
        R_dn = torch.bmm(ns_unfold_p, depth_unfold_p)
        Score_dn = torch.diagonal(R_dn,dim1=-2,dim2=-1)
        R_ds = torch.bmm(seg_unfold_p, depth_unfold_p)
        Score_ds = torch.diagonal(R_ds,dim1=-2,dim2=-1)

        S_rgb0 = Score_dr.view(Score_dr.size(0), 1, depth.size(2), depth.size(3))
        S_ns = Score_dn.view(Score_dn.size(0), 1, depth.size(2), depth.size(3))
        S_seg = Score_ds.view(Score_ds.size(0), 1, depth.size(2), depth.size(3))
        S_rgb = S_rgb0 + self.gamma1 * S_ns + self.gamma2 * S_seg

        cat_ns = self.cc1(ns * S_ns + ns)
        cat_seg = self.cc2(seg * S_seg + seg)
        cat_rgb = self.cc3(rgb * S_rgb + rgb)

        out_g, out_rgb, out_seg, out_ns = self.d_g(depth1, cat_rgb, cat_seg, cat_ns, S_seg, S_ns, S_rgb)
        res_depth = self.rd(torch.cat([out_g, out_rgb, out_seg, out_ns], 1))

        res = self.c2(res_depth + depth)

        return res, self.rgb_up(out_rgb), self.seg_up(out_seg), self.ns_up(out_ns)
