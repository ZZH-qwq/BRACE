"""Multi-scale Spatial Cross-Attention Modulator (MSCAM)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import kaiming_init


class MSCAM(nn.Module):
    def __init__(self, channels=64, prior_dim=64, attn_dim=64, scales=(1, 2, 4), mod_scale=0.1):
        super().__init__()
        self.scales = tuple(scales)
        self.attn_dim = attn_dim
        self.mod_scale = mod_scale

        self.frame_proj = nn.ModuleList([nn.Linear(channels, attn_dim) for _ in self.scales])
        self.prior_proj = nn.Linear(prior_dim, attn_dim)
        self.qkv = nn.Linear(attn_dim, 3 * attn_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(attn_dim, 2 * channels),
        )
        kaiming_init(self.frame_proj)
        kaiming_init(self.prior_proj)
        kaiming_init(self.qkv)
        kaiming_init(self.out_proj)

    def _scale_context(self, feats, prior_token, scale, proj):
        n, t, c, h, w = feats.shape
        pooled = F.adaptive_avg_pool2d(feats.flatten(0, 1), (scale, scale))
        tokens = pooled.flatten(2).transpose(1, 2)  # (n*t, s*s, c)
        tokens = proj(tokens).view(n, t, scale * scale, self.attn_dim)
        tokens = tokens + prior_token[:, None, None, :]  # add the banding prior

        q, k, v = self.qkv(tokens).chunk(3, dim=-1)
        q = q[:, 0]  # reference frame
        k = k.flatten(1, 2)
        v = v.flatten(1, 2)

        attn = torch.softmax(q @ k.transpose(-2, -1) / self.attn_dim**0.5, dim=-1)
        context = attn @ v  # (n, s*s, a)
        return context.mean(dim=1)  # (n, a)

    def forward(self, feats, prior_token):
        """feats: (n, t, c, h, w) aligned to the reference frame (reference first)."""
        prior = self.prior_proj(prior_token)
        contexts = [self._scale_context(feats, prior, s, proj) for s, proj in zip(self.scales, self.frame_proj)]
        context = torch.stack(contexts, dim=1).mean(dim=1)

        gamma, beta = self.out_proj(context).chunk(2, dim=-1)
        gamma = torch.tanh(gamma).view(-1, gamma.shape[1], 1, 1)
        beta = beta.view(-1, beta.shape[1], 1, 1)

        return feats * (1.0 + self.mod_scale * gamma[:, None]) + self.mod_scale * beta[:, None]
