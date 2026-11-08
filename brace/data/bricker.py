import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ..isp import camera_params
from ..models.blocks import random_augment

FRAME_KEYS = ('I1', 'I2', 'I3', 'I4', 'I5')
WHITE_LEVEL = 1023.0
PRIOR_SIDE = 128.0  # longest side of the FABP analysis image


def list_samples(root, subset, split):
    base = os.path.join(root, subset, split)
    if not os.path.isdir(base):
        raise FileNotFoundError('missing split directory: %s' % base)
    samples = []
    for scene in sorted(os.listdir(base)):
        scene_dir = os.path.join(base, scene)
        if not os.path.isdir(scene_dir):
            continue
        for sample in sorted(os.listdir(scene_dir)):
            sample_dir = os.path.join(scene_dir, sample)
            if os.path.isfile(os.path.join(sample_dir, 'metadata.npy')):
                samples.append(('%s/%s/%s/%s' % (subset, split, scene, sample), sample_dir))
    if len(samples) == 0:
        raise RuntimeError('no samples found under %s' % base)
    return samples


def _prior_image(frames, target_side):
    h, w = frames.shape[1:3]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return _mean_pool(frames[:, top:top + side, left:left + side], target_side)


def _mean_pool(frames, target_side):
    h, w = frames.shape[1:3]
    k = max(1, int(round(max(h, w) / float(target_side))))
    if k == 1:
        return frames
    pad_h = (-h) % k
    pad_w = (-w) % k
    if pad_h or pad_w:
        frames = np.pad(frames, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    t, h, w, c = frames.shape
    return frames.reshape(t, h // k, k, w // k, k, c).mean(axis=(2, 4))


def _to_tensor(array, target_dim):
    array = np.moveaxis(array, -1, target_dim)
    return torch.from_numpy(np.ascontiguousarray(array))


class BrickerDataset(Dataset):
    def __init__(self, root, subset, split, patch_size=None, augment=None,
                 max_samples=None, seed=0):
        self.root = root
        self.subset = subset
        self.split = split
        self.patch_size = patch_size
        self.augment = (split == 'Train') if augment is None else augment
        self.samples = list_samples(root, subset, split)
        if max_samples:
            self.samples = self.samples[:max_samples]
        self.epoch = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, sample_dir = self.samples[index]
        raws = np.stack([np.load(os.path.join(sample_dir, 'raw_%s.npy' % k)).astype(np.float32) / WHITE_LEVEL
                         for k in FRAME_KEYS])
        gt = np.load(os.path.join(sample_dir, 'raw_gt.npy')).astype(np.float32)
        meta = np.load(os.path.join(sample_dir, 'metadata.npy'), allow_pickle=True).item()
        params = camera_params(meta)

        raws_t = _to_tensor(raws, 1)          # (5, 4, H, W)
        gt_t = _to_tensor(gt, 0)              # (4, H, W)

        if self.patch_size is not None:
            size = self.patch_size
            top = np.random.randint(0, raws_t.shape[-2] - size + 1)
            left = np.random.randint(0, raws_t.shape[-1] - size + 1)
            raws_t = raws_t[..., top:top + size, left:left + size]
            gt_t = gt_t[..., top:top + size, left:left + size]

        prior_t = _to_tensor(_prior_image(raws, PRIOR_SIDE), 1)       # (5, 4, h, w)
        if self.augment:
            raws_t, gt_t, prior_t = random_augment(raws_t, gt_t, prior_t)

        item = {
            'raws': raws_t,
            'gt': gt_t,
            'prior': prior_t,
            'wb': torch.from_numpy(params['wb']),
            'cam2rgb': torch.from_numpy(params['cam2rgb']),
            'as_shot_neutral': torch.from_numpy(np.asarray(params['as_shot_neutral'], dtype=np.float32)),
            'color_matrix_1': torch.from_numpy(np.asarray(params['color_matrix_1'], dtype=np.float32)),
            'name': name,
        }
        return item


def _worker_init(worker_id):
    seed = (torch.initial_seed() + worker_id) % (2 ** 31 - 1)
    np.random.seed(seed)


def build_loader(cfg, split, distributed=False, batch_size=None):
    data_cfg = dict(cfg['data'])
    patch_size = data_cfg.get('patch_size') if split == 'Train' else None
    dataset = BrickerDataset(
        root=data_cfg['root'],
        subset=data_cfg['subset'],
        split=split,
        patch_size=patch_size,
    )
    sampler = None
    shuffle = split == 'Train'
    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle, drop_last=shuffle)
        shuffle = False
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=(batch_size or data_cfg.get('batch_size', 1)) if split == 'Train' else 1,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=data_cfg.get('num_workers', 8),
        worker_init_fn=_worker_init,
        pin_memory=True,
        drop_last=(split == 'Train'),
        persistent_workers=data_cfg.get('num_workers', 8) > 0,
    )
    return dataset, loader
