"""Config, seeding, logging and checkpoint helpers."""

import json
import os
import random
import sys

import numpy as np
import torch
import yaml


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split('=', 1)
        node = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse(value)
    return cfg


def _parse(value):
    """Parse a ``--set key=value`` override (numbers, booleans, strings)."""
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        return value
    if isinstance(parsed, str):
        # YAML 1.1 does not recognise "3e-5" as a float; be forgiving about it
        for cast in (int, float):
            try:
                return cast(parsed)
            except ValueError:
                continue
    return parsed


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """Return (local_rank, world_size, distributed) from the torchrun env."""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size > 1:
        torch.distributed.init_process_group(backend='nccl')
    return local_rank, world_size, world_size > 1


def is_main():
    return int(os.environ.get('RANK', '0')) == 0


def get_logger(output_dir, name='train'):
    if not is_main():
        class _Silent:
            def info(self, *a, **k):
                pass
        return _Silent()
    os.makedirs(output_dir, exist_ok=True)
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S')
    for handler in (logging.FileHandler(os.path.join(output_dir, '%s.log' % name)),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def save_checkpoint(path, model, optimizer, epoch, step, cfg):
    state = model.state_dict() if not hasattr(model, 'module') else model.module.state_dict()
    torch.save({
        'model': state,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'config': cfg,
    }, path)


def load_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    if 'model' in ckpt:
        return ckpt['model'], ckpt
    return ckpt, {}


def dump_results(results, path):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
