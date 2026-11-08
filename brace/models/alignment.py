import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops


def flow_warp(x, flow, interpolation="bilinear", padding_mode="zeros", align_corners=True):
    """Warp ``x`` (n, c, h, w) with ``flow`` (n, h, w, 2) given in pixels."""
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError("spatial sizes of input {} and flow {} differ".format(x.size()[-2:], flow.size()[1:3]))
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=flow.device, dtype=x.dtype),
        torch.arange(0, w, device=flow.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), 2)  # (h, w, 2)
    grid = grid.unsqueeze(0).expand(flow.size(0), -1, -1, -1)
    grid = grid + flow
    norm_grid = torch.empty_like(grid)
    norm_grid[..., 0] = 2 * grid[..., 0] / max(w - 1, 1) - 1
    norm_grid[..., 1] = 2 * grid[..., 1] / max(h - 1, 1) - 1
    return F.grid_sample(x, norm_grid, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)


class SPyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        n, _, h, w = ref.size()
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref.append(F.avg_pool2d(ref[-1], kernel_size=2, stride=2, count_include_pad=False))
            supp.append(F.avg_pool2d(supp[-1], kernel_size=2, stride=2, count_include_pad=False))
        ref, supp = ref[::-1], supp[::-1]

        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            flow = flow_up + self.basic_module[level](
                torch.cat(
                    [
                        ref[level],
                        flow_warp(supp[level], flow_up.permute(0, 2, 3, 1), padding_mode="border"),
                        flow_up,
                    ],
                    1,
                )
            )
        return flow

    def forward(self, ref, supp):
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(self.compute_flow(ref, supp), size=(h, w), mode="bilinear", align_corners=False)
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)
        return flow


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            nn.Conv2d(8, 32, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 7, 1, 3),
        )

    def forward(self, x):
        return self.basic_module(x)


def load_spynet(net, path):
    state_dict = torch.load(path, map_location="cpu")
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    net_state = net.state_dict()
    loaded = 0
    for name, param in state_dict.items():
        name = name.replace("basic_module.0.conv", "basic_module.0")
        name = name.replace("basic_module.1.conv", "basic_module.2")
        name = name.replace("basic_module.2.conv", "basic_module.4")
        name = name.replace("basic_module.3.conv", "basic_module.6")
        name = name.replace("basic_module.4.conv", "basic_module.8")
        if name in net_state and net_state[name].shape == param.shape:
            net_state[name].copy_(param)
            loaded += 1
    net.load_state_dict(net_state)
    if loaded == 0:
        raise RuntimeError("no SPyNet weights were loaded from %s" % path)
    for p in net.parameters():
        p.requires_grad = False
    return net


class DeformableAlignment(nn.Module):
    """Flow-guided deformable convolution used to refine the alignment."""

    def __init__(self, channels=64, kernel=3, deform_groups=8, max_residue_magnitude=10):
        super().__init__()
        self.max_residue_magnitude = max_residue_magnitude
        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * channels + 2, channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, 27 * deform_groups, 3, 1, 1),
        )
        self.deform_conv = ops.DeformConv2d(
            channels,
            channels,
            kernel_size=kernel,
            stride=1,
            padding=(kernel - 1) // 2,
            dilation=1,
            bias=True,
            groups=deform_groups,
        )
        nn.init.constant_(self.conv_offset[-1].weight, 0.0)
        nn.init.constant_(self.conv_offset[-1].bias, 0.0)

    def forward(self, cur_feat, ref_feat, warped_feat, flow):
        extra_feat = torch.cat([warped_feat, ref_feat, flow], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        mask = torch.sigmoid(mask)
        return self.deform_conv(cur_feat, offset, mask=mask)
