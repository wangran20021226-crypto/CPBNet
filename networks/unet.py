# -*- coding: utf-8 -*-
"""
UNet implementation (v9 patch)
Original: https://github.com/HiLab-git/PyMIC

v8 change:
    Decoder.forward gains 'with_dec_feat' parameter.
    UNet.forward return signature:
        old: (out1, out2, emb1, emb2)
        new: (out1, out2, emb1, emb2, dec_feat1, dec_feat2)

v9 change (boundary fix):
    dec_feat 从 up3_feat [B, 32, 128×128] 升级为 up4_feat [B, 16, 256×256]。

    原因：up3_feat 之后还有 up4（两层 ConvBlock），会完全覆写
    BADFC/BTNA 在 up3_feat 上强制的边界结构，导致 DecoderBoundaryModule 无效。
    up4_feat 是 out_conv 的直接输入，约束在此层才能直接影响分割输出。
    同时分辨率从 128→256 与 SDF 对齐，边界法向精度提升。
"""
from __future__ import division, print_function

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upsampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels,
                 dropout_p, mode_upsampling=1):
        super(UpBlock, self).__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling == 0:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        elif mode_upsampling == 1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                                  align_corners=True)
        elif mode_upsampling == 2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        elif mode_upsampling == 3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic',
                                  align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params  = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert len(self.ft_chns) == 5

        self.in_conv = ConvBlock(self.in_chns,    self.ft_chns[0], self.dropout[0])
        self.down1   = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2   = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3   = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4   = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params  = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert len(self.ft_chns) == 5

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3],
                           dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2],
                           dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1],
                           dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0],
                           dropout_p=0.0, mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature, with_feature=False, with_dec_feat=False):
        """
        Args:
            feature       : list of encoder features [x0..x4]
            with_feature  : if True, also return bottleneck x4 (for CPAM)
            with_dec_feat : if True, also return up4_feat [B, 16, 256, 256]
                            (for DecoderBoundaryModule)

                            v9: 暴露 up4_feat（out_conv 直接输入），
                            替换 v8 的 up3_feat [B, 32, 128, 128]。
                            up4_feat 分辨率与分割输出一致（256×256），
                            BADFC/BTNA 约束不再被后续卷积覆写。

        Return signatures:
            with_feature=F, with_dec_feat=F  ->  output
            with_feature=T, with_dec_feat=F  ->  (output, x4)
            with_feature=F, with_dec_feat=T  ->  (output, up4_feat)
            with_feature=T, with_dec_feat=T  ->  (output, x4, up4_feat)
        """
        x0, x1, x2, x3, x4 = feature

        x        = self.up1(x4, x3)
        x        = self.up2(x,  x2)
        x        = self.up3(x,  x1)
        up4_feat = self.up4(x,  x0)   # [B, ft_chns[0]=16, 256, 256]
        output   = self.out_conv(up4_feat)

        if with_feature and with_dec_feat:
            return output, x4, up4_feat
        if with_dec_feat:
            return output, up4_feat
        if with_feature:
            return output, x4
        return output


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params1 = {
            'in_chns':      in_chns,
            'feature_chns': [16, 32, 64, 128, 256],
            'dropout':      [0.05, 0.1, 0.2, 0.3, 0.5],
            'class_num':    class_num,
            'up_type':      1,
            'acti_func':    'relu',
        }
        params2 = {
            'in_chns':      in_chns,
            'feature_chns': [16, 32, 64, 128, 256],
            'dropout':      [0.05, 0.1, 0.2, 0.3, 0.5],
            'class_num':    class_num,
            'up_type':      0,
            'acti_func':    'relu',
        }

        self.encoder1 = Encoder(params1)
        self.encoder2 = Encoder(params2)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)

    def forward(self, x):
        """
        Returns: (out_seg1, out_seg2, emb1, emb2, dec_feat1, dec_feat2)

            out_seg1/2  : [B, C, 256, 256]  segmentation output
            emb1/2      : [B, 256, 16, 16]  bottleneck features (for CPAM)
            dec_feat1/2 : [B, 16, 256, 256] up4 decoder features (for DecoderBoundaryModule)
                          v9: 从 v8 的 [B,32,128,128] up3_feat 升级为 [B,16,256,256] up4_feat
        """
        feature1 = self.encoder1(x)
        feature2 = self.encoder2(x)

        out_seg1, emb1, dec_feat1 = self.decoder1(
            feature1, with_feature=True, with_dec_feat=True)
        out_seg2, emb2, dec_feat2 = self.decoder2(
            feature2, with_feature=True, with_dec_feat=True)

        return out_seg1, out_seg2, emb1, emb2, dec_feat1, dec_feat2