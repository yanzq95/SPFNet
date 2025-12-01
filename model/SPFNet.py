import numpy as np
import math
from model.common import *
import torch.nn as nn

 
class SPFNet(nn.Module):
    def __init__(self, num_feats, kernel_size, scale,reduction, tiny_model=False):
        super(SPFNet, self).__init__()
        self.reduction = reduction
        self.tiny_model = tiny_model
        
        if self.tiny_model:
            self.red1 = self.reduction
            self.red2 = self.reduction
        else:
            self.red1 = 2*self.reduction
            self.red2 = 4*self.reduction
        
        self.scale_factor = int(math.log(scale,2))
        
        self.conv_rgb1 = nn.Conv2d(in_channels=3, out_channels=num_feats,
                                   kernel_size=kernel_size, padding=1)
        self.rgb_rb2 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=self.red1, n_resblocks=1)
        self.rgb_rb3 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=self.red1, n_resblocks=1)
        self.rgb_rb4 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=self.red1, n_resblocks=1)
        self.conv_rgb2 = default_conv(num_feats,num_feats*4,1)

        self.rgb_rb5 = ResidualGroup(default_conv, num_feats*4, kernel_size, reduction=self.red1, n_resblocks=1)

        self.conv_ns1 = nn.Conv2d(in_channels=3, out_channels=num_feats,
                                   kernel_size=kernel_size, padding=1)
        self.ns_rb2 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=self.red1, n_resblocks=1)
        self.ns_rb3 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=self.red1, n_resblocks=1)
        self.ns_rb4 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=self.red1, n_resblocks=1)
        self.conv_ns2 = default_conv(num_feats,num_feats*4,1)
        self.ns_rb5 = ResidualGroup(default_conv, num_feats*4, kernel_size, reduction=self.red1, n_resblocks=1)

        self.conv_seg1 = nn.Conv2d(in_channels=1, out_channels=num_feats,
                                   kernel_size=kernel_size, padding=1)
        self.seg_rb2 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=self.red1, n_resblocks=1)
        self.seg_rb3 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=self.red1, n_resblocks=1)
        self.seg_rb4 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=self.red1, n_resblocks=1)
        self.conv_seg2 = default_conv(num_feats,num_feats*4,1)
        self.seg_rb5 = ResidualGroup(default_conv, num_feats*4, kernel_size, reduction=self.red1, n_resblocks=1)

        self.conv_dp1 = nn.Conv2d(in_channels=1, out_channels=num_feats,
                                  kernel_size=kernel_size, padding=1)
        self.dp_rg1 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=self.red1, n_resblocks=4)
        self.dp_rg2 = ResidualGroup(default_conv, num_feats*2, kernel_size, reduction=self.red2, n_resblocks=4)
        self.dp_rg3 = ResidualGroup(default_conv, num_feats*3, kernel_size, reduction=self.red2, n_resblocks=4)
        self.dp_rg4 = ResidualGroup(default_conv, num_feats*4, kernel_size, reduction=self.red2, n_resblocks=4)

        self.f1 = nn.Sequential(
            ResidualGroup(default_conv, num_feats*4, kernel_size, reduction=self.red2, n_resblocks=4),
            default_conv(num_feats*4,num_feats*3,1)
        )
        self.f2 = nn.Sequential(
            ResidualGroup(default_conv, num_feats*3, kernel_size, reduction=self.red2, n_resblocks=4),
            default_conv(num_feats*3,num_feats*2,1)
        )
        self.f3 = nn.Sequential(
            ResidualGroup(default_conv, num_feats*2, kernel_size, reduction=self.red2, n_resblocks=4),
            default_conv(num_feats*2,num_feats,1)
        )

        self.f4 = nn.Sequential(
            ResidualGroup(default_conv, num_feats, kernel_size, reduction=self.red1, n_resblocks=4),
            default_conv(num_feats,num_feats*4,1)
        )

        self.bridge1 = APP(input_planes=num_feats, weight_planes=num_feats, scale_factor=self.scale_factor, reduction=self.reduction, tiny_model=self.tiny_model)
        self.bridge2 = APP(input_planes=num_feats*2, weight_planes=num_feats, scale_factor=self.scale_factor, reduction=self.reduction, tiny_model=self.tiny_model)
        self.bridge3 = APP(input_planes=num_feats*3, weight_planes=num_feats, scale_factor=self.scale_factor, reduction=self.reduction, tiny_model=self.tiny_model)

        my_tail = [
            ResidualGroup(
                default_conv, num_feats*4, kernel_size, reduction=self.red2, n_resblocks=8),
            ResidualGroup(
                default_conv, num_feats*4, kernel_size, reduction=self.red2, n_resblocks=8)
        ]
        self.tail = nn.Sequential(*my_tail)

        self.up1 = UpBlock(self.scale_factor,num_feats*4,num_feats*4)
        self.up2 = UpBlock(self.scale_factor,num_feats*3,num_feats*3)
        self.up3 = UpBlock(self.scale_factor,num_feats*2,num_feats*2)
        self.up4 = UpBlock(self.scale_factor,num_feats,num_feats)
        last_conv = [
            default_conv(num_feats*4, num_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(num_feats, 1, kernel_size=3, bias=True)
        ]
        self.last_conv = nn.Sequential(*last_conv)
        self.bicubic = nn.Upsample(scale_factor=scale, mode='bicubic')

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        image, depth,seg,ns = x

        dp_in = self.act(self.conv_dp1(depth))
        dp1 = self.dp_rg1(dp_in)

        rgb1 = self.act(self.conv_rgb1(image))
        rgb2 = self.rgb_rb2(rgb1)

        seg1 = self.act(self.conv_seg1(seg))
        seg2 = self.seg_rb2(seg1)

        ns1 = self.act(self.conv_ns1(ns))
        ns2 = self.ns_rb2(ns1)

        ca1_in, out_rgb1, out_seg1, out_ns1 = self.bridge1(dp1, rgb2,seg2,ns2)

        dp2 = self.dp_rg2(ca1_in)
        rgb3 = self.rgb_rb3(rgb2 + out_rgb1)
        seg3 = self.seg_rb3(seg2 + out_seg1)
        ns3 = self.ns_rb3(ns2 + out_ns1)

        ca2_in, out_rgb2, out_seg2, out_ns2 = self.bridge2(dp2, rgb3,seg3,ns3)

        dp3 = self.dp_rg3(ca2_in)
        rgb4 = self.rgb_rb4(rgb3 + out_rgb2)
        seg4 = self.seg_rb4(seg3 + out_seg2)
        ns4 = self.ns_rb4(ns3 + out_ns2)

        ca3_in, out_rgb3, out_seg3, out_ns3 = self.bridge3(dp3, rgb4,seg4,ns4)
        dp4 = self.dp_rg4(ca3_in)
        rgb5 = self.rgb_rb5(self.conv_rgb2(rgb4 + out_rgb3))
        seg5 = self.seg_rb5(self.conv_seg2(seg4 + out_seg3))
        ns5 = self.ns_rb5(self.conv_ns2(ns4 + out_ns3))

        rgb5 = rgb5 + seg5 + ns5

        y1 = self.f1(rgb5 + self.up1(dp4)) # 96
        y2 = self.f2(y1 + self.up2(dp3)) # 64
        y3 = self.f3(y2 + self.up3(dp2)) #32
        y4 = self.f4(y3 + self.up4(dp1))

        out = self.last_conv(self.tail(y4))

        out = out + self.bicubic(depth)

        return out
