"""
RAW -> sRGB rendering used for evaluation
"""

import os

import numpy as np
import torch


# ----------------
# demosaic network
# ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DEMOSAIC_WEIGHTS = os.path.join(_HERE, '..', 'weights', 'demosaicnet.bin')
XYZ2SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                     [-0.9692660, 1.8760108, 0.0415560],
                     [0.0556434, -0.2040259, 1.0572252]], dtype=np.float32)

_net_cache = {}


def _demosaic_net(device):
    key = str(device)
    if key not in _net_cache:
        from .third_party import demosaicnet as demosaic_bayer
        _net_cache[key] = demosaic_bayer.get_demosaic_net_model(
            pretrained=DEMOSAIC_WEIGHTS, device=device, cfa='bayer', state_dict=True)
    return _net_cache[key]


def camera_params(meta):
    """Extract the parameters the renderer needs from a ``metadata.npy`` dict."""
    asn = np.asarray(meta['AsShotNeutral'], dtype=np.float32).reshape(-1)
    cfa = [int(v) for v in np.asarray(meta['CFAPattern']).reshape(-1)]
    xyz2cam = np.asarray(meta['ColorMatrix1'], dtype=np.float32).reshape(3, 3)
    xyz2cam = xyz2cam / xyz2cam.sum(axis=1, keepdims=True)
    cam2xyz = np.linalg.inv(xyz2cam)
    xyz2srgb = XYZ2SRGB / XYZ2SRGB.sum(axis=1, keepdims=True)
    cam2rgb = xyz2srgb @ cam2xyz
    # white-balance gains per packed channel (R, G1, G2, B)
    wb = np.array([1.0 / asn[c] for c in cfa], dtype=np.float32)
    return {
        'as_shot_neutral': asn,
        'cfa': cfa,
        'color_matrix_1': xyz2cam.reshape(-1),
        'orientation': int(np.asarray(meta['Orientation']).reshape(-1)[0]),
        'cam2rgb': cam2rgb.astype(np.float32),
        'wb': wb,
    }


def render_srgb(raw4, params, device='cpu'):
    """Linear sRGB (H*2, W*2, 3) in [0, 1] from a packed RAW frame."""
    from .third_party import demosaicnet as demosaic_bayer

    raw4 = np.asarray(raw4, dtype=np.float32)
    bayer = np.zeros((2 * raw4.shape[0], 2 * raw4.shape[1]), dtype=np.float32)
    bayer[0::2, 0::2] = raw4[..., 0]
    bayer[0::2, 1::2] = raw4[..., 1]
    bayer[1::2, 0::2] = raw4[..., 2]
    bayer[1::2, 1::2] = raw4[..., 3]

    neutral = params['as_shot_neutral']
    for (dy, dx), c in zip(((0, 0), (0, 1), (1, 0), (1, 1)), params['cfa']):
        bayer[dy::2, dx::2] = np.clip(bayer[dy::2, dx::2] / neutral[c], 0.0, 1.0)

    cfa_str = ''.join('RGB'[i] for i in params['cfa'])
    net_input = np.power(np.clip(bayer, 0.0, 1.0), 1.0 / 2.2)
    with torch.no_grad():
        rgb = demosaic_bayer.demosaic_by_demosaic_net(
            bayer=net_input, cfa=cfa_str, demosaic_net=_demosaic_net(device), device=device)
    rgb = np.power(np.clip(rgb, 0.0, 1.0), 2.2)

    xyz = np.einsum('ij,hwj->hwi', np.linalg.inv(
        np.asarray(params['color_matrix_1'], dtype=np.float32).reshape(3, 3)), rgb)
    srgb = np.einsum('ij,hwj->hwi', XYZ2SRGB / XYZ2SRGB.sum(axis=1, keepdims=True), np.clip(xyz, 0, 1))
    return np.clip(srgb, 0.0, 1.0)


def to_display(rgb, subset):
    """Linear sRGB -> 8-bit display image."""
    if subset == 'real':
        out = np.log1p(5000.0 * np.clip(rgb, 0.0, None)) / np.log1p(5000.0)
    else:
        out = np.power(np.clip(rgb, 0.0, 1.0), 1.0 / 2.2)
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def render_image(raw4, params, subset, device='cpu'):
    """Packed RAW frame in [0, 1] -> 8-bit RGB uint8 image."""
    return to_display(render_srgb(np.clip(raw4, 0.0, 1.0), params, device=device), subset)
