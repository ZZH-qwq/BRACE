#!/bin/bash
# Stage 1: synthetic pre-training on Bricker-Syn.

set -e
cd "$(dirname "$0")/.."
NPROC=${NPROC:-2}
PORT=${PORT:-29520}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
torchrun --nproc_per_node=$NPROC --master_port=$PORT -m brace.train \
    --config configs/syn_stage1.yaml "$@"
