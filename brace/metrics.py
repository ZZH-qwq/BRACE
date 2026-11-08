"""Full-reference image quality metrics on 8-bit sRGB PNGs."""

DATA_RANGE = 255.0

import numpy as np
import torch

_LPIPS = None
_IQA = {}


def psnr(pred, gt):
    mse = np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2)
    if mse <= 1e-12:
        return float('inf')
    return float(-10.0 * np.log10(mse / DATA_RANGE ** 2))


def ssim(pred, gt):
    from skimage.metrics import structural_similarity
    h, w = pred.shape[:2]
    win = min(11, min(h, w) if min(h, w) % 2 == 1 else min(h, w) - 1)
    win = max(3, win)
    return float(structural_similarity(pred, gt, win_size=win, data_range=DATA_RANGE,
                                       channel_axis=2, gaussian_weights=True))


def _to_tensor(img, device):
    """(H, W, 3) uint8 -> (1, 3, H, W) float in [0, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32))
    return (t / DATA_RANGE).unsqueeze(0).to(device)


def _load_models(device):
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net='alex').to(device).eval()
    if 'dists' not in _IQA:
        from pyiqa.default_model_configs import DEFAULT_CONFIGS
        from pyiqa.archs import build_network

        class _Inference(torch.nn.Module):
            def __init__(self, name):
                super().__init__()
                opts = dict(DEFAULT_CONFIGS[name]['metric_opts'])
                self.net = build_network(opts).to(device).eval()

            def forward(self, x, y):
                return self.net(x, y)

        _IQA['dists'] = _Inference('dists')
        _IQA['msswd'] = _Inference('msswd')
    return _LPIPS, _IQA


@torch.no_grad()
def deep_metrics(pred, gt, device):
    """LPIPS / DISTS / MSSWD for one RGB pair."""
    loss_fn, iqa = _load_models(device)
    pred_t = _to_tensor(pred, device)
    gt_t = _to_tensor(gt, device)
    lpips_score = float(loss_fn(pred_t * 2 - 1, gt_t * 2 - 1).item())
    dists_score = float(iqa['dists'](pred_t, gt_t).item())
    msswd_score = float(iqa['msswd'](pred_t, gt_t).item())
    return lpips_score, dists_score, msswd_score


def evaluate_pair(pred, gt, device):
    """All metrics for one predicted / ground-truth RGB pair."""
    lp, di, ms = deep_metrics(pred, gt, device)
    return {
        'psnr': psnr(pred, gt),
        'ssim': ssim(pred, gt),
        'lpips': lp,
        'dists': di,
        'msswd': ms,
    }
