"""
Testing entry point: full-frame inference + metrics.

    python -m brace.test --config configs/syn_stage1.yaml \
        --checkpoint outputs/syn_stage1/epoch_50.pth --subset syn

Writes the predicted PNGs (optional) and a JSON/txt summary under
``<checkpoint dir>/eval_<subset>/``.
"""

import argparse
import os

import torch

from .evaluate import evaluate
from .models.brace import build_model
from .utils import dump_results, load_checkpoint, load_config


def main():
    parser = argparse.ArgumentParser(description='Evaluate BRACE on Bricker')
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', nargs='*', default=[])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--subset', default=None, choices=['syn', 'real'])
    parser.add_argument('--split', default='Test')
    parser.add_argument('--device', default='0')
    parser.add_argument('--output', default='', help='output directory (default: next to the checkpoint)')
    parser.add_argument('--save-images', action='store_true')
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--sfc', default=None, choices=['auto', 'on', 'off'])
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    subset = args.subset or cfg['data']['subset']
    device = torch.device('cuda:%s' % args.device if torch.cuda.is_available() else 'cpu')

    model = build_model(cfg['model']).to(device)
    state, _ = load_checkpoint(args.checkpoint, device='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()

    output = args.output or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)),
                                         'eval_%s' % subset)
    os.makedirs(output, exist_ok=True)
    save_dir = os.path.join(output, 'pred') if args.save_images else None

    compute_sfc = None if args.sfc in (None, 'auto') else args.sfc == 'on'
    summary, rows = evaluate(model, cfg, subset, split=args.split, device=device,
                             max_images=args.max_images, full_metrics=True,
                             save_dir=save_dir, compute_sfc=compute_sfc)

    print('\n%s/%s (%d images)' % (subset, args.split, summary['num_images']))
    print('  ' + '  '.join('%s=%.4f' % (k, v) for k, v in summary.items() if k != 'num_images'))

    dump_results({'summary': summary, 'per_image': rows}, os.path.join(output, 'metrics.json'))
    with open(os.path.join(output, 'metrics.txt'), 'w') as f:
        f.write('%s/%s (%d images)\n' % (subset, args.split, summary['num_images']))
        for key, value in summary.items():
            if key != 'num_images':
                f.write('  %-8s %.4f\n' % (key, value))
    print('\nwritten to %s' % output)


if __name__ == '__main__':
    with torch.no_grad():
        main()
