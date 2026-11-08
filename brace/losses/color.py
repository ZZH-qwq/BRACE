import torch
import torch.nn as nn

from .msswd import MSSWD


class ColorLoss(nn.Module):
    def __init__(self, num_scale=3, num_proj=64):
        super().__init__()
        self.msswd = MSSWD(num_scale=num_scale, num_proj=num_proj)

    @staticmethod
    def render(raw, wb, cam2rgb):
        """raw: (n, 4, h, w) in [0, 1]; wb: (n, 4); cam2rgb: (n, 3, 3)."""
        x = torch.clamp(raw, 0.0, 1.0) * wb.view(-1, 4, 1, 1)
        x = torch.clamp(x, 0.0, 1.0)
        rgb = torch.stack([x[:, 0], 0.5 * (x[:, 1] + x[:, 2]), x[:, 3]], dim=1)
        rgb = torch.clamp(torch.einsum('bij,bjhw->bihw', cam2rgb, rgb), 0.0, 1.0)
        rgb = torch.clamp(rgb, min=1e-8)
        return torch.log1p(5000.0 * rgb) / float(torch.log1p(torch.tensor(5000.0)))

    def forward(self, pred, gt, wb, cam2rgb):
        return self.msswd(self.render(pred, wb, cam2rgb),
                          self.render(gt, wb, cam2rgb))
