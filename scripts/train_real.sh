#!/bin/bash
# Stage 2: real-world fine-tuning, initialised from a stage-1 checkpoint.

set -e
cd "$(dirname "$0")/.."

CKPT=${CKPT:-outputs/syn_stage1/epoch_50.pth}
NGPU=${NGPU:-2}
GPUS=${GPUS:-0,1}
PORT=${PORT:-29521}

export CUDA_VISIBLE_DEVICES=$GPUS
torchrun --nproc_per_node=$NGPU --master_port=$PORT -m brace.train \
    --config configs/real_stage2.yaml "$@" --init-from "$CKPT"
