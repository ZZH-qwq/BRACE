#!/bin/bash

set -e
cd "$(dirname "$0")/.."
CKPT=${1:-outputs/real_stage2/epoch_20.pth}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python -m brace.test --config configs/real_stage2.yaml --checkpoint "$CKPT" \
    --subset real --save-images
