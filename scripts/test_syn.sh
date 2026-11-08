#!/bin/bash

set -e
cd "$(dirname "$0")/.."
CKPT=${1:-outputs/syn_stage1/epoch_50.pth}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python -m brace.test --config configs/syn_stage1.yaml --checkpoint "$CKPT" \
    --subset syn --save-images
