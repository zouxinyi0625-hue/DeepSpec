#!/usr/bin/env bash
set -euo pipefail

# Lightweight acceptance evaluation for a small Gemma4 DSpark checkpoint.
# This intentionally evaluates only a few samples so it can be used as a smoke
# test after small-data training.

TARGET_NAME_OR_PATH=${TARGET_NAME_OR_PATH:-google/gemma-4-12B-it}
DRAFT_NAME_OR_PATH=${DRAFT_NAME_OR_PATH:-${HOME}/checkpoints/deepspec_small/dspark_block5_gemma4_12b_1k/step_latest}
TASKS=${TASKS:-gsm8k:32,mt-bench:16,alpaca:32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0.0}
DATASET_ROOT=${DATASET_ROOT:-./eval_datasets}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/train_small_gemma4/eval_small.py \
  --target_name_or_path "${TARGET_NAME_OR_PATH}" \
  --draft_name_or_path "${DRAFT_NAME_OR_PATH}" \
  --tasks "${TASKS}" \
  --dataset-root "${DATASET_ROOT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --confidence-threshold "${CONFIDENCE_THRESHOLD}"
