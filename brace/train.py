"""
Training entry point for BRACE.

Stage one (synthetic pre-training):

    torchrun --nproc_per_node=2 -m brace.train --config configs/syn_stage1.yaml

Stage two (real fine-tuning, initialised from stage one):

    torchrun --nproc_per_node=2 -m brace.train --config configs/real_stage2.yaml \
        --init-from outputs/syn_stage1/epoch_50.pth
"""

import argparse
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from .data.bricker import build_loader
from .evaluate import evaluate
from .losses.color import ColorLoss
from .losses.wavelet import WaveletFrequencyLoss
from .models.brace import build_model
from .utils import (get_logger, is_main, load_checkpoint, load_config, save_checkpoint,
                    seed_everything, setup_distributed)


def mu_tonemap(x, mu=5000.0):
    return torch.log1p(mu * x) / torch.log1p(torch.tensor(mu, device=x.device))


def infinite_batches(loader, sampler):
    epoch = 0
    while True:
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def lr_factor(step, total, warmup, min_ratio):
    if step < warmup:
        return max(min_ratio, (step + 1) / float(max(1, warmup)))
    t = (step - warmup) / float(max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


def main():
    parser = argparse.ArgumentParser(description='Train BRACE')
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', nargs='*', default=[], help='config overrides, key=value')
    parser.add_argument('--resume', default='', help='resume model/optimizer/step')
    parser.add_argument('--init-from', default='', help='initialise model weights only')
    parser.add_argument('--wandb-mode', default=None, choices=[None, 'online', 'offline', 'disabled'])
    parser.add_argument('--name', default='', help='wandb run name (default: output dir name)')
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    tcfg = cfg['train']
    local_rank, world_size, distributed = setup_distributed()
    device = torch.device('cuda', local_rank)
    seed_everything(tcfg.get('seed', 0) + 17 * local_rank)
    torch.backends.cudnn.benchmark = True

    output_dir = tcfg['output_dir']
    logger = get_logger(output_dir)
    if is_main():
        logger.info('config: %s' % args.config)
        logger.info('world size: %d  device: %s' % (world_size, str(device)))

    global_batch = int(tcfg.get('batch_size', 16))
    if global_batch % world_size != 0:
        raise ValueError('global batch %d is not divisible by world size %d' % (global_batch, world_size))
    per_rank_batch = global_batch // world_size
    dataset, loader = build_loader(cfg, 'Train', distributed=distributed,
                                   batch_size=per_rank_batch)
    if is_main():
        logger.info('training samples: %d  batches per pass: %d  global batch: %d (%d per GPU)'
                    % (len(dataset), len(loader), global_batch, per_rank_batch))

    model = build_model(cfg['model']).to(device)
    if is_main():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('trainable parameters: %.2f M' % (n_params / 1e6))

    if args.init_from:
        state, _ = load_checkpoint(args.init_from, device='cpu')
        model.load_state_dict(state, strict=True)
        if is_main():
            logger.info('initialised from %s' % args.init_from)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg['lr'],
        betas=tuple(tcfg.get('betas', (0.9, 0.999))),
        weight_decay=tcfg.get('weight_decay', 0.01),
    )

    start_step = 0
    if args.resume:
        state, meta = load_checkpoint(args.resume, device='cpu')
        model.load_state_dict(state, strict=True)
        if 'optimizer' in meta:
            optimizer.load_state_dict(meta['optimizer'])
        start_step = int(meta.get('step', 0))
        if is_main():
            logger.info('resumed from %s at step %d' % (args.resume, start_step))

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    wavelet = WaveletFrequencyLoss(levels=tcfg.get('wavelet_levels', 2),
                                   ll_weight=tcfg.get('wavelet_ll_weight', 1.0),
                                   detail_weight=tcfg.get('wavelet_detail_weight', 0.1)).to(device)
    color = None
    if tcfg.get('color_weight', 0.0) > 0:
        color = ColorLoss(num_scale=tcfg.get('color_num_scale', 3),
                          num_proj=tcfg.get('color_num_proj', 64)).to(device)

    wandb = None
    if is_main() and tcfg.get('wandb', True):
        try:
            import wandb as _wandb
            mode = args.wandb_mode or tcfg.get('wandb_mode', 'online')
            if mode != 'disabled':
                _wandb.init(project=tcfg.get('wandb_project', 'BRACE'),
                            entity=tcfg.get('wandb_entity') or None,
                            name=args.name or os.path.basename(output_dir.rstrip('/')),
                            config=cfg, dir=output_dir, mode=mode)
                wandb = _wandb
        except Exception as exc:  # pragma: no cover - wandb is optional
            logger.info('wandb disabled: %s' % exc)

    iters_per_epoch = int(tcfg['iters_per_epoch'])
    epochs = int(tcfg['epochs'])
    total_iters = epochs * iters_per_epoch
    warmup = int(tcfg.get('warmup_iters', 500))
    min_ratio = float(tcfg.get('min_lr_ratio', 0.0))
    log_every = int(tcfg.get('log_every', 50))
    eval_every = int(tcfg.get('eval_every', 0))
    eval_images = int(tcfg.get('eval_images', 16))
    save_every = int(tcfg.get('save_every', 10))

    batches = infinite_batches(loader, loader.sampler)
    model.train()
    logger.info('training for %d steps (%d epochs x %d iters)' % (total_iters, epochs, iters_per_epoch))

    meters = {}
    t0 = time.time()
    for step in range(start_step, total_iters):
        batch = next(batches)
        raws = batch['raws'].to(device, non_blocking=True)
        gt = batch['gt'].to(device, non_blocking=True)
        prior = batch['prior'].to(device, non_blocking=True)

        lr = tcfg['lr'] * lr_factor(step, total_iters, warmup, min_ratio)
        for group in optimizer.param_groups:
            group['lr'] = lr

        pred = model(raws, prior)

        if cfg['data']['subset'] == 'real':
            wb = batch['wb'].to(device)
            cam2rgb = batch['cam2rgb'].to(device)
            pred_d = ColorLoss.render(pred, wb, cam2rgb)
            gt_d = ColorLoss.render(gt, wb, cam2rgb)
        else:
            pred_d = mu_tonemap(torch.clamp(pred, 0.0, 1.0))
            gt_d = mu_tonemap(torch.clamp(gt, 0.0, 1.0))

        loss_l1 = F.l1_loss(pred_d, gt_d)
        loss_wavelet = wavelet(pred_d, gt_d)
        loss = loss_l1 + tcfg['wavelet_weight'] * loss_wavelet
        logs = {'loss': loss, 'l1': loss_l1, 'wavelet': loss_wavelet}
        if color is not None:
            loss_color = color(pred, gt, batch['wb'].to(device), batch['cam2rgb'].to(device)).mean()
            loss = loss + tcfg['color_weight'] * loss_color
            logs['color'] = loss_color
            logs['loss'] = loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        for key, value in logs.items():
            meters.setdefault(key, []).append(float(value.detach()))
        meters.setdefault('lr', []).append(lr)

        if is_main() and (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            msg = 'step %d/%d  ' % (step + 1, total_iters)
            msg += '  '.join('%s=%.5f' % (k, sum(v) / len(v)) for k, v in meters.items())
            msg += '  %.3f s/iter' % (elapsed / len(meters['lr']))
            logger.info(msg)
            if wandb is not None:
                wandb.log({('train/' + k): sum(v) / len(v) for k, v in meters.items()},
                          step=step + 1)
            meters = {}
            t0 = time.time()

        epoch_done = (step + 1) % iters_per_epoch == 0
        if epoch_done:
            epoch = (step + 1) // iters_per_epoch
            if eval_every and epoch % eval_every == 0:
                if is_main():
                    model.eval()
                    summary = {}
                    for subset in tcfg.get('eval_subsets', ['syn']):
                        s, _ = evaluate(model.module if distributed else model, cfg, subset,
                                        device=device, max_images=eval_images,
                                        full_metrics=False, log=logger)
                        summary.update({('%s_%s' % (subset, k)): v for k, v in s.items()
                                        if k != 'num_images'})
                    if wandb is not None:
                        wandb.log({'val/' + k: v for k, v in summary.items()}, step=step + 1)
                    model.train()
                if distributed:
                    dist.barrier()

            if is_main() and (epoch % save_every == 0 or epoch == epochs):
                path = os.path.join(output_dir, 'epoch_%d.pth' % epoch)
                save_checkpoint(path, model, optimizer, epoch, step + 1, cfg)
                logger.info('saved %s' % path)

    if is_main():
        save_checkpoint(os.path.join(output_dir, 'last.pth'), model, optimizer,
                        total_iters // iters_per_epoch, total_iters, cfg)
        if wandb is not None:
            wandb.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
