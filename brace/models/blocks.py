import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def kaiming_init(module, scale=1.0):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            m.weight.data *= scale
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


class ResidualBlockNoBN(nn.Module):

    def __init__(self, channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        kaiming_init(self.conv1, scale=0.1)
        kaiming_init(self.conv2, scale=0.1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x))) * self.res_scale


class ResidualBlocksWithInputConv(nn.Module):

    def __init__(self, in_channels, out_channels=64, num_blocks=5):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            *[ResidualBlockNoBN(out_channels) for _ in range(num_blocks)],
        )

    def forward(self, x):
        return self.main(x)


class PixelShufflePack(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=1, upsample_kernel=3):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample_conv = nn.Conv2d(
            in_channels,
            out_channels * scale_factor * scale_factor,
            upsample_kernel,
            padding=(upsample_kernel - 1) // 2,
        )
        kaiming_init(self.upsample_conv)

    def forward(self, x):
        x = self.upsample_conv(x)
        if self.scale_factor > 1:
            x = F.pixel_shuffle(x, self.scale_factor)
        return x


def random_augment(*tensors, allow_transpose=True):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    transpose = allow_transpose and random.random() < 0.5
    out = []
    for t in tensors:
        if hflip:
            t = torch.flip(t, dims=[-1])
        if vflip:
            t = torch.flip(t, dims=[-2])
        if transpose:
            t = t.transpose(-1, -2)
        out.append(t)
    return out
