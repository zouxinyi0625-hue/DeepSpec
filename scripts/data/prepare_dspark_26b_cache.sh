#!/usr/bin/env bash
set -euo pipefail

# Generate the DSpark 26B target cache (multi-layer hidden states) for the
# Gemma4-26B-A4B dense-draft run. Runs the 26B target forward over the regen
# split and writes a sharded on-mount cache. NOT the MTP last_hidden+KV cache.
#
# New mount layout (2026): everything under the ukwdata mount. The mount env var
# name differs between sessions; override MOUNT if yours is named differently
# (e.g. AZURE_ML_INPUT_UKDATA vs AZURE_ML_INPUT_ukwdata).

# --- Resolve the data mount (handle both spellings/casings) -----------------
MOUNT="${MOUNT:-${AZURE_ML_INPUT_UKDATA:-${AZURE_ML_INPUT_ukwdata:-${AZURE_ML_INPUT_UKWDATA:-}}}}"
if [[ -z "${MOUNT}" ]]; then
    echo "ERROR: no data mount found. Set MOUNT=/path/to/ukwdata explicitly." >&2
    exit 1
fi

CONFIG_PATH=${CONFIG_PATH:-config/dspark/dspark_gemma4_26b.py}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-${MOUNT}/maiprofile/models/text_only}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-${MOUNT}/maiprofile/mtp_26b/split/train_maiprofile_26b.jsonl}

# New cache output dir on the new mount.
OUTPUT_DIR=${OUTPUT_DIR:-${MOUNT}/maiprofile/dspark_26b/target_cache}

MIN_LOSS_TOKENS=${MIN_LOSS_TOKENS:-14}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_SHARD_BYTES=${MAX_SHARD_BYTES:-$((64 * 1024 * 1024 * 1024))}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=${HF_HOME:-/home/aiscuser/.cache/huggingface}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

# --- Sanity checks before spending GPU hours --------------------------------
[[ -e "${TARGET_MODEL_PATH}" ]] || { echo "ERROR: target model not found: ${TARGET_MODEL_PATH}" >&2; exit 1; }
[[ -e "${TRAIN_DATA_PATH}"   ]] || { echo "ERROR: train data not found: ${TRAIN_DATA_PATH}" >&2; exit 1; }

# Create the cache dir (and parents) on the new mount.
mkdir -p "${OUTPUT_DIR}"

echo "Generating DSpark 26B target cache"
echo "  mount:        ${MOUNT}"
echo "  config:       ${CONFIG_PATH}"
echo "  target model: ${TARGET_MODEL_PATH}"
echo "  train data:   ${TRAIN_DATA_PATH}"
echo "  output dir:   ${OUTPUT_DIR}"
echo "  min_loss_tokens: ${MIN_LOSS_TOKENS}, local_bsz: ${LOCAL_BATCH_SIZE}"
echo "  target_layer_ids come from ${CONFIG_PATH} -> [3, 11, 19, 28]"

python scripts/data/prepare_target_cache.py \
  --config "${CONFIG_PATH}" \
  --opts "model.target_model_name_or_path=${TARGET_MODEL_PATH}" \
  --train-data-path "${TRAIN_DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --min-loss-tokens "${MIN_LOSS_TOKENS}" \
  --local-batch-size "${LOCAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-shard-bytes "${MAX_SHARD_BYTES}"

echo "Done. Cache at: ${OUTPUT_DIR}"
echo "Next: TARGET_CACHE_DIR=${OUTPUT_DIR} bash scripts/train_maiprofile/train_dspark_26b_sync.sh"
