#!/usr/bin/env bash
set -euo pipefail

# Train DSpark on the 1k Gemma4-12B target cache prepared by prepare_1k_cache.sh.
#
# Usage:
#   TARGET_CACHE_DIR=~/.cache/deepspec/gemma4_12b_target_cache_1k \
#   bash scripts/train_small_gemma4/train_dspark_1k.sh
#
# Optional env vars:
#   CONFIG_PATH       default: config/dspark/dspark_gemma4_12b_small.py
#   TARGET_CACHE_DIR  default: ~/.cache/deepspec/gemma4_12b_target_cache_1k
#   CUDA_VISIBLE_DEVICES, MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
#   LOCAL_BATCH_SIZE  default: value from config unless set here
#   GLOBAL_BATCH_SIZE default: value from config unless set here
#   MAX_TRAIN_STEPS   default: value from config unless set here

CONFIG_PATH=${CONFIG_PATH:-config/dspark/dspark_gemma4_12b_small.py}
TARGET_CACHE_DIR=${TARGET_CACHE_DIR:-${HOME}/.cache/deepspec/gemma4_12b_target_cache_1k}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -d "${TARGET_CACHE_DIR}" ]]; then
  cat >&2 <<EOF
ERROR: TARGET_CACHE_DIR not found: ${TARGET_CACHE_DIR}

Run first:
  TRAIN_JSONL=/path/to/gemma4_12b_regen.jsonl bash scripts/train_small_gemma4/prepare_1k_cache.sh
EOF
  exit 1
fi

opts=(--opts "data.target_cache_path=${TARGET_CACHE_DIR}")
if [[ -n "${LOCAL_BATCH_SIZE:-}" ]]; then
  opts+=(--opts "train.local_batch_size=${LOCAL_BATCH_SIZE}")
fi
if [[ -n "${GLOBAL_BATCH_SIZE:-}" ]]; then
  opts+=(--opts "train.global_batch_size=${GLOBAL_BATCH_SIZE}")
fi
if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
  opts+=(--opts "train.max_train_steps=${MAX_TRAIN_STEPS}")
fi

python train.py \
  --config "${CONFIG_PATH}" \
  "${opts[@]}"

cat <<'EOF'

Training complete or stopped. Checkpoints are under:
  ~/checkpoints/deepspec_small/dspark_block5_gemma4_12b_1k/

TensorBoard logs are under:
  ~/tensorboard/deepspec_small/dspark_block5_gemma4_12b_1k/

Example eval after a checkpoint exists:
  TARGET_NAME_OR_PATH=google/gemma-4-12B-it \
  DRAFT_NAME_OR_PATH=~/checkpoints/deepspec_small/dspark_block5_gemma4_12b_1k/step_latest \
  bash scripts/train_small_gemma4/eval_dspark_1k.sh
EOF
