#!/usr/bin/env bash
set -euo pipefail

# Train Gemma4-26B-A4B DSpark (DENSE draft over MoE target) on MAI Profile.
# Checkpoints + tensorboard are written to LOCAL disk only (no mount sync).
#
# PREREQ: the DSpark 26B target cache must already exist (multi-layer hidden
# states, NOT the MTP last_hidden+KV cache). Generate it with
# scripts/data/prepare_dspark_26b_cache.sh (writes to the ukwdata mount).

EXP_NAME=${EXP_NAME:-dspark_block7_gemma4_26b_dense_maiprofile_4096ctx}
CONFIG_PATH=${CONFIG_PATH:-config/dspark/dspark_gemma4_26b.py}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-5000}
CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-500}
# 0 = single-process dataloader. Multi-worker deadlocks (futex_wait) when the
# cache is mmap-read from the mount, so keep 0 unless the cache is local.
NUM_WORKERS=${NUM_WORKERS:-0}

# Data mount (target model + target cache live here). Handle spelling variants.
MOUNT="${MOUNT:-${AZURE_ML_INPUT_UKWDATA:-${AZURE_ML_INPUT_UKDATA:-${AZURE_ML_INPUT_ukwdata:-}}}}"
if [[ -z "${MOUNT}" ]]; then
    echo "ERROR: no data mount found. Set MOUNT=/path/to/ukwdata explicitly." >&2
    exit 1
fi
export TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-${MOUNT}/maiprofile/models/text_only}

# Target cache dir must match the cache-gen run (same CACHE_TAG).
CACHE_TAG=${CACHE_TAG:-v2}
TARGET_CACHE_DIR=${TARGET_CACHE_DIR:-${MOUNT}/maiprofile/dspark_26b/target_cache_${CACHE_TAG}}

# LOCAL-only artifacts. NOTE: the REAL checkpoint/tb paths are decided by the
# config's finalize_cfg (BASE_CKPT_DIR/project/exp = ~/checkpoints/deepspec/EXP).
# These vars mirror that for the mkdir + the echo below; they are display/prep
# only and are not passed to train.py.
LOCAL_CKPT_DIR=${LOCAL_CKPT_DIR:-${HOME}/checkpoints/deepspec/${EXP_NAME}}
LOCAL_TB_DIR=${LOCAL_TB_DIR:-${HOME}/tensorboard/deepspec/${EXP_NAME}}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

export HF_HOME=${HF_HOME:-/home/aiscuser/.cache/huggingface}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

mkdir -p "${LOCAL_CKPT_DIR}" "${LOCAL_TB_DIR}"

# Sanity-check the cache is actually ready before spending GPU hours.
if [[ ! -f "${TARGET_CACHE_DIR}/manifest.json" ]]; then
    echo "ERROR: target cache manifest not found: ${TARGET_CACHE_DIR}/manifest.json" >&2
    echo "  Generate it first: bash scripts/data/prepare_dspark_26b_cache.sh" >&2
    exit 1
fi

echo "Training Gemma4-26B DSpark (dense draft, local checkpoints only)"
echo "  exp_name:          ${EXP_NAME}"
echo "  target model:      ${TARGET_MODEL_PATH}"
echo "  target_cache:      ${TARGET_CACHE_DIR}"
echo "  local checkpoint:  ${LOCAL_CKPT_DIR}"
echo "  local tensorboard: ${LOCAL_TB_DIR}"
echo "  steps:             ${MAX_TRAIN_STEPS}, ckpt every ${CHECKPOINTING_STEPS}"

python train.py \
  --config "${CONFIG_PATH}" \
  --opts "exp_name=${EXP_NAME}" \
  --opts "model.target_model_name_or_path=${TARGET_MODEL_PATH}" \
  --opts "data.target_cache_path=${TARGET_CACHE_DIR}" \
  --opts "data.num_workers=${NUM_WORKERS}" \
  --opts "train.max_train_steps=${MAX_TRAIN_STEPS}" \
  --opts "logging.checkpointing_steps=${CHECKPOINTING_STEPS}"

echo "Done. Checkpoints in: ${LOCAL_CKPT_DIR}"
