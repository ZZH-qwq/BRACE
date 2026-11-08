"""Multi-level wavelet (MWL) frequency loss."""

import torch.nn as nn
import torch.nn.functional as F


class WaveletFrequencyLoss(nn.Module):
    def __init__(self, levels=2, ll_weight=1.0, detail_weight=0.1):
        super().__init__()
        self.levels = levels
        self.ll_weight = ll_weight
        self.detail_weight = detail_weight

    @staticmethod
    def _haar_dwt(x):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 + x01 - x10 + x11) * 0.5
        hl = (-x00 - x01 + x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh

    def forward(self, pred, gt):
        loss = 0.0
        p, g = pred, gt
        for _ in range(self.levels):
            _, _, h, w = p.shape
            pad_h, pad_w = h % 2, w % 2
            if pad_h or pad_w:
                p = F.pad(p, (0, pad_w, 0, pad_h), mode='reflect')
                g = F.pad(g, (0, pad_w, 0, pad_h), mode='reflect')
            p_ll, p_lh, p_hl, p_hh = self._haar_dwt(p)
            g_ll, g_lh, g_hl, g_hh = self._haar_dwt(g)
            loss = loss + self.ll_weight * F.l1_loss(p_ll, g_ll)
            loss = loss + self.detail_weight * (
                F.l1_loss(p_lh, g_lh) + F.l1_loss(p_hl, g_hl) + F.l1_loss(p_hh, g_hh)
            ) / 3.0
            p, g = p_ll, g_ll
        return loss / float(self.levels)
