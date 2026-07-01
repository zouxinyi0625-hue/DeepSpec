#!/usr/bin/env bash
set -euo pipefail

# End-to-end 1k Gemma4-12B DSpark data preparation:
#   1. Download/sample Open-PerfectBlend prompt data if needed.
#   2. Regenerate answers with a Gemma4-12B OpenAI-compatible server.
#   3. Build the target cache consumed by train_dspark_1k.sh.
#
# Minimal usage when a server is already running on localhost:30000:
#   SERVER_ADDRESS=127.0.0.1:30000 bash scripts/train_small_gemma4/prepare_end_to_end_1k.sh
#
# Multiple servers are supported:
#   SERVER_ADDRESS="127.0.0.1:30000 127.0.0.1:30001" bash ...
#
# If REGEN_JSONL already exists, generation is skipped and only cache prep runs.

CONFIG_PATH=${CONFIG_PATH:-config/dspark/dspark_gemma4_12b_small.py}
MODEL_NAME=${MODEL_NAME:-google/gemma-4-12B-it}
DATASET_NAME=${DATASET_NAME:-mlabonne/open-perfectblend}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
# Download slightly more rows because download_and_split holds out TEST_SIZE.
SOURCE_SAMPLE_SIZE=${SOURCE_SAMPLE_SIZE:-1200}
TEST_SIZE=${TEST_SIZE:-0.05}

PROMPT_JSONL=${PROMPT_JSONL:-train_datasets/gemma4_12b/perfectblend_train_prompt_1k_source.jsonl}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-eval_datasets}
EVAL_OUTPUT_NAME=${EVAL_OUTPUT_NAME:-perfectblend_small_gemma4.jsonl}
REGEN_JSONL=${REGEN_JSONL:-train_datasets/gemma4_12b/perfectblend_train_regen_1k.jsonl}
TARGET_CACHE_DIR=${TARGET_CACHE_DIR:-${HOME}/.cache/deepspec/gemma4_12b_target_cache_1k}

SERVER_ADDRESS=${SERVER_ADDRESS:-}
GEN_CONCURRENCY=${GEN_CONCURRENCY:-8}
GEN_TEMPERATURE=${GEN_TEMPERATURE:-1.0}
GEN_TOP_P=${GEN_TOP_P:-0.95}
GEN_MAX_TOKENS=${GEN_MAX_TOKENS:-2048}

LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-2}
MIN_LOSS_TOKENS=${MIN_LOSS_TOKENS:-8}
MAX_SHARD_BYTES=${MAX_SHARD_BYTES:-8589934592}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

mkdir -p "$(dirname "${PROMPT_JSONL}")" "$(dirname "${REGEN_JSONL}")" "${EVAL_OUTPUT_DIR}"

if [[ ! -f "${PROMPT_JSONL}" ]]; then
  echo "[1/3] Downloading/sampling ${DATASET_NAME} -> ${PROMPT_JSONL}"
  python scripts/data/download_and_split.py \
    --dataset-name "${DATASET_NAME}" \
    --sample-size "${SOURCE_SAMPLE_SIZE}" \
    --test-size "${TEST_SIZE}" \
    --train-output-path "${PROMPT_JSONL}" \
    --test-output-dir "${EVAL_OUTPUT_DIR}" \
    --test-output-name "${EVAL_OUTPUT_NAME}"
else
  echo "[1/3] Reusing existing prompt JSONL: ${PROMPT_JSONL}"
fi

if [[ ! -f "${REGEN_JSONL}" ]]; then
  if [[ -z "${SERVER_ADDRESS}" ]]; then
    cat >&2 <<EOF
ERROR: REGEN_JSONL does not exist and SERVER_ADDRESS is empty.

Start a Gemma4-12B OpenAI-compatible server first, then run for example:
  SERVER_ADDRESS=127.0.0.1:30000 bash $0

If you already have regenerated answers, pass:
  REGEN_JSONL=/path/to/gemma4_12b_regen_1k.jsonl bash $0
EOF
    exit 1
  fi

  echo "[2/3] Regenerating answers with ${MODEL_NAME} -> ${REGEN_JSONL}"
  # shellcheck disable=SC2206
  server_args=(${SERVER_ADDRESS})
  python scripts/data/generate_train_data.py \
    --model "${MODEL_NAME}" \
    --server-address "${server_args[@]}" \
    --concurrency "${GEN_CONCURRENCY}" \
    --temperature "${GEN_TEMPERATURE}" \
    --top-p "${GEN_TOP_P}" \
    --max-tokens "${GEN_MAX_TOKENS}" \
    --num-samples "${NUM_SAMPLES}" \
    --input-file-path "${PROMPT_JSONL}" \
    --output-file-path "${REGEN_JSONL}" \
    --resume
else
  echo "[2/3] Reusing existing regenerated JSONL: ${REGEN_JSONL}"
fi

actual_count=$(python - <<PY
from pathlib import Path
p = Path(${REGEN_JSONL@Q})
print(sum(1 for line in p.open('r', encoding='utf-8') if line.strip()))
PY
)
echo "Regenerated sample count: ${actual_count}"
if [[ "${actual_count}" -le 0 ]]; then
  echo "ERROR: regenerated JSONL is empty: ${REGEN_JSONL}" >&2
  exit 1
fi

# Reuse the lower-level cache-prep script so the validation and cache commands stay in one place.
echo "[3/3] Building target cache -> ${TARGET_CACHE_DIR}"
TRAIN_JSONL="${REGEN_JSONL}" \
SAMPLE_JSONL="${REGEN_JSONL}" \
TARGET_CACHE_DIR="${TARGET_CACHE_DIR}" \
NUM_SAMPLES="${NUM_SAMPLES}" \
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE}" \
MIN_LOSS_TOKENS="${MIN_LOSS_TOKENS}" \
MAX_SHARD_BYTES="${MAX_SHARD_BYTES}" \
CONFIG_PATH="${CONFIG_PATH}" \
bash scripts/train_small_gemma4/prepare_1k_cache.sh

cat <<EOF

End-to-end data preparation done.
Prompt JSONL:      ${PROMPT_JSONL}
Regenerated JSONL: ${REGEN_JSONL}
Target cache:      ${TARGET_CACHE_DIR}

Next:
  TARGET_CACHE_DIR=${TARGET_CACHE_DIR} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
  bash scripts/train_small_gemma4/train_dspark_1k.sh
EOF
