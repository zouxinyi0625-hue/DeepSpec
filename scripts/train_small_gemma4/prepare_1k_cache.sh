#!/usr/bin/env bash
set -euo pipefail

# Prepare a 1k-sample Gemma4-12B target cache for DSpark smoke training.
#
# Prerequisite: a regenerated training JSONL whose answers were produced by
# the same target model. If you only have prompt-only data, first run
# scripts/data/generate_train_data.py with an OpenAI-compatible Gemma4-12B
# server (see scripts/data/README.md).
#
# Usage:
#   TRAIN_JSONL=train_datasets/gemma4_12b/perfectblend_train_regen.jsonl \
#   bash scripts/train_small_gemma4/prepare_1k_cache.sh
#
# Optional env vars:
#   CONFIG_PATH       default: config/dspark/dspark_gemma4_12b_small.py
#   TRAIN_JSONL       required unless the default path exists
#   SAMPLE_JSONL      default: train_datasets/gemma4_12b/perfectblend_train_regen_1k.jsonl
#   TARGET_CACHE_DIR  default: ~/.cache/deepspec/gemma4_12b_target_cache_1k
#   NUM_SAMPLES       default: 1000
#   CUDA_VISIBLE_DEVICES, MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
#   LOCAL_BATCH_SIZE  default: 2
#   MIN_LOSS_TOKENS   default: 8
#   MAX_SHARD_BYTES   default: 8589934592 (8 GiB)

CONFIG_PATH=${CONFIG_PATH:-config/dspark/dspark_gemma4_12b_small.py}
TRAIN_JSONL=${TRAIN_JSONL:-train_datasets/gemma4_12b/perfectblend_train_regen.jsonl}
SAMPLE_JSONL=${SAMPLE_JSONL:-train_datasets/gemma4_12b/perfectblend_train_regen_1k.jsonl}
TARGET_CACHE_DIR=${TARGET_CACHE_DIR:-${HOME}/.cache/deepspec/gemma4_12b_target_cache_1k}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-2}
MIN_LOSS_TOKENS=${MIN_LOSS_TOKENS:-8}
MAX_SHARD_BYTES=${MAX_SHARD_BYTES:-8589934592}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  cat >&2 <<EOF
ERROR: TRAIN_JSONL not found: ${TRAIN_JSONL}

Provide a regenerated Gemma4-12B training JSONL, for example:
  TRAIN_JSONL=/path/to/perfectblend_train_regen.jsonl bash $0

If you only have prompt data, first generate target answers with:
  python scripts/data/generate_train_data.py --model google/gemma-4-12B-it ...
EOF
  exit 1
fi

mkdir -p "$(dirname "${SAMPLE_JSONL}")"
python - <<PY
from pathlib import Path
src = Path(${TRAIN_JSONL@Q})
dst = Path(${SAMPLE_JSONL@Q})
n = int(${NUM_SAMPLES@Q})
if src.resolve() == dst.resolve():
    count = sum(1 for line in src.open('r', encoding='utf-8') if line.strip())
    print(f"Reusing existing sample file {dst} with {count} non-empty rows")
else:
    count = 0
    with src.open('r', encoding='utf-8') as fin, dst.open('w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(line)
            count += 1
            if count >= n:
                break
    print(f"Wrote {count} samples to {dst}")
if count == 0:
    raise SystemExit(f"No non-empty samples found in {src}")
PY

python scripts/data/prepare_target_cache.py \
  --config "${CONFIG_PATH}" \
  --train-data-path "${SAMPLE_JSONL}" \
  --output-dir "${TARGET_CACHE_DIR}" \
  --local-batch-size "${LOCAL_BATCH_SIZE}" \
  --min-loss-tokens "${MIN_LOSS_TOKENS}" \
  --max-shard-bytes "${MAX_SHARD_BYTES}"

cat <<EOF

Prepared target cache:
  ${TARGET_CACHE_DIR}

Next:
  TARGET_CACHE_DIR=${TARGET_CACHE_DIR} bash scripts/train_small_gemma4/train_dspark_1k.sh
EOF
