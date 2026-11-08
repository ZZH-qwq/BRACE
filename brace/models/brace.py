"""BRACE: Bracketed RAW Flicker-Banding Removal"""

import os

import torch
import torch.nn as nn

from .alignment import DeformableAlignment, SPyNet, flow_warp, load_spynet
from .blocks import PixelShufflePack, ResidualBlocksWithInputConv, kaiming_init
from .fabp import FABP
from .mscam import MSCAM


class BRACE(nn.Module):
    def __init__(
        self,
        num_frames=5,
        mid_channels=64,
        ref_index=None,
        num_blocks=(5, 16, 24, 5),
        fabp_kwargs=None,
        mscam_kwargs=None,
        spynet_weights=None,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.mid_channels = mid_channels
        self.ref_index = num_frames // 2 if ref_index is None else ref_index

        # exposure normalisation to the reference frame
        expo = 4.0 ** (self.ref_index - torch.arange(num_frames).float())
        self.register_buffer("expo", expo.view(1, -1, 1, 1, 1))

        order, d = [self.ref_index], 1
        while len(order) < num_frames:
            for idx in (self.ref_index - d, self.ref_index + d):
                if 0 <= idx < num_frames:
                    order.append(idx)
            d += 1
        self.register_buffer("order", torch.tensor(order, dtype=torch.long), persistent=False)

        self.fabp = FABP(**(fabp_kwargs or {}))
        self.mscam = MSCAM(channels=mid_channels, **(mscam_kwargs or {}))

        self.spynet = SPyNet()
        if spynet_weights is not None:
            load_spynet(self.spynet, spynet_weights)

        self.align = DeformableAlignment(mid_channels)

        n_feat, n_agg, n_refine, n_rec = num_blocks
        self.feat_extract = ResidualBlocksWithInputConv(2 * 4, mid_channels, n_feat)
        self.aggregate = ResidualBlocksWithInputConv(3 * mid_channels, mid_channels, n_agg)
        self.refine = nn.ModuleList(
            [ResidualBlocksWithInputConv(3 * mid_channels, mid_channels, n_refine) for _ in range(num_frames)]
        )
        self.refine_weight = nn.Parameter(torch.zeros(1, num_frames, mid_channels, 1, 1))

        self.reconstruct = ResidualBlocksWithInputConv(2 * mid_channels, mid_channels, n_rec)
        self.skip = PixelShufflePack(4, mid_channels, scale_factor=1, upsample_kernel=3)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, 4, 3, 1, 1)
        kaiming_init(self.conv_hr)
        kaiming_init(self.conv_last)

    @staticmethod
    def _interleave_raw(x):
        """RAW (n,t,4,h,w) -> (n,t,8,h,w) with [raw_i, raw_i^(1/2.2)] interleaved."""
        x = torch.clamp(x, min=0.0)
        g = torch.pow(x, 1.0 / 2.2)
        return torch.stack([x, g], dim=3).flatten(2, 3)

    @staticmethod
    def _flow_input(x):
        """RAW (n,t,4,h,w) -> gamma-encoded 3-channel images for SPyNet."""
        rgb = torch.stack((x[:, :, 0], x[:, :, 1:3].mean(dim=2), x[:, :, 3]), dim=2)
        return torch.pow(torch.clamp(rgb, 0.0, 1.0), 1.0 / 2.2)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, raws, prior_frames=None):
        n, t, _, h, w = raws.shape
        if t != self.num_frames:
            raise ValueError("model expects %d frames, got %d" % (self.num_frames, t))

        x = raws * self.expo
        prior = self.fabp(x if prior_frames is None else prior_frames * self.expo)

        order = self.order.to(x.device)
        x_ord = x[:, order]

        feats = self.feat_extract(self._interleave_raw(x_ord).flatten(0, 1))
        feats = feats.view(n, t, -1, h, w)

        flow_img = self._flow_input(x_ord).flatten(0, 1).view(n, t, 3, h, w)
        ref_img = flow_img[:, :1].expand(-1, t - 1, -1, -1, -1).reshape(-1, 3, h, w)
        oth_img = flow_img[:, 1:].reshape(-1, 3, h, w)
        flow = self.spynet(ref_img, oth_img)

        ref_feat = feats[:, :1].expand(-1, t - 1, -1, -1, -1).reshape(-1, self.mid_channels, h, w)
        oth_feat = feats[:, 1:].reshape(-1, self.mid_channels, h, w)
        warped = flow_warp(oth_feat, flow.permute(0, 2, 3, 1))
        oth_feat = self.align(oth_feat, ref_feat, warped, flow)

        aligned = torch.cat([feats[:, :1], oth_feat.view(n, t - 1, -1, h, w)], dim=1)
        aligned = self.mscam(aligned, prior["token"])  # (n, t, c, h, w)

        base = aligned[:, 0]
        state = torch.zeros_like(base)
        for i in range(t):
            current = aligned[:, i]
            inp = torch.cat([base, current, state], dim=1)
            state = state + self.aggregate(inp)
            state = state + self.refine[i](torch.cat([base, current, state], dim=1)) * self.refine_weight[:, i]

        hr = torch.cat([base, state], dim=1)
        hr = self.reconstruct(hr) + self.skip(x_ord[:, 0])
        return self.conv_last(self.lrelu(self.conv_hr(hr)))


def build_model(cfg):
    """Build BRACE from the ``model`` section of a config dict."""
    kwargs = dict(cfg or {})
    fabp = kwargs.pop("fabp", None)
    mscam = kwargs.pop("mscam", None)
    if fabp is not None:
        kwargs["fabp_kwargs"] = fabp
    if mscam is not None:
        kwargs["mscam_kwargs"] = mscam
    if kwargs.get("spynet_weights") is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        kwargs["spynet_weights"] = os.path.join(root, "weights", "spynet.pth")
    return BRACE(**kwargs)
