"""
Evaluation of BRACE on Bricker
"""

import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from . import metrics as M
from .data.bricker import build_loader
from .isp import render_image
from .sfc import SFC


def isp_params(batch, index=0):
    """Rebuild the renderer parameter dict from a collated batch."""
    return {
        'as_shot_neutral': batch['as_shot_neutral'][index].numpy(),
        'cfa': [0, 1, 1, 2],
        'color_matrix_1': batch['color_matrix_1'][index].numpy(),
    }


def get_valid_crop(cfg, subset):
    crop_cfg = cfg.get('eval', {}).get('valid_crop', {'syn': 5, 'real': 0})
    if isinstance(crop_cfg, dict):
        return int(crop_cfg.get(subset, 0))
    return int(crop_cfg)


@torch.no_grad()
def evaluate(model, cfg, subset, split='Test', device='cuda:0', max_images=None,
             full_metrics=True, save_dir=None, compute_sfc=None, log=None):
    data_cfg = dict(cfg['data'])
    data_cfg['subset'] = subset
    data_cfg['batch_size'] = 1
    data_cfg['num_workers'] = min(4, data_cfg.get('num_workers', 4))
    _, loader = build_loader({'data': data_cfg}, split)

    crop = get_valid_crop(cfg, subset)
    if compute_sfc is None:
        compute_sfc = subset == 'real'

    model.eval()
    sfc_metric = SFC().to(device) if compute_sfc else None
    rows = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i, batch in enumerate(tqdm(loader, desc='eval %s' % subset, disable=log is None)):
        if max_images is not None and i >= max_images:
            break
        parts = batch['name'][0].split('/')          # subset/split/scene/sample
        sample_dir = os.path.join(cfg['data']['root'], subset, split, parts[2], parts[3])

        raws = batch['raws'].to(device)
        prior = batch['prior'].to(device)
        pred = model(raws, prior).clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
        if crop > 0:
            pred = pred[crop:-crop, crop:-crop]

        params = isp_params(batch)
        pred_rgb = render_image(pred, params, subset, device=device)

        gt_png = np.ascontiguousarray(cv2.imread(os.path.join(sample_dir, 'rgb_gt.png'), -1)[..., ::-1])
        gt_rgb = gt_png[2 * crop:gt_png.shape[0] - 2 * crop, 2 * crop:gt_png.shape[1] - 2 * crop] \
            if crop > 0 else gt_png
        if pred_rgb.shape != gt_rgb.shape:
            raise RuntimeError('shape mismatch for %s: pred %s vs gt %s'
                               % (parts[3], pred_rgb.shape, gt_rgb.shape))

        row = {'name': parts[3]}
        row.update(M.evaluate_pair(pred_rgb, gt_rgb, device) if full_metrics
                   else {'psnr': M.psnr(pred_rgb, gt_rgb), 'ssim': M.ssim(pred_rgb, gt_rgb)})

        if sfc_metric is not None:
            lq_raw = batch['raws'][0, 0].permute(1, 2, 0).cpu().numpy()
            if crop > 0:
                lq_raw = lq_raw[crop:-crop, crop:-crop]
            lq_rgb = render_image(lq_raw, params, subset, device=device)
            as_tensor = lambda img: torch.from_numpy(
                np.ascontiguousarray(img.transpose(2, 0, 1))).float().unsqueeze(0).to(device) / 255.0
            row['sfc'] = sfc_metric(as_tensor(pred_rgb), as_tensor(gt_rgb), as_tensor(lq_rgb))['sfc']
        rows.append(row)

        if save_dir:
            cv2.imwrite(os.path.join(save_dir, parts[3] + '.png'), pred_rgb[..., ::-1])

    keys = [k for k in rows[0] if k != 'name']
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary['num_images'] = len(rows)
    if log is not None:
        log.info('%s/%s  ' % (subset, split) + '  '.join('%s=%.4f' % (k, summary[k]) for k in keys))
    return summary, rows
