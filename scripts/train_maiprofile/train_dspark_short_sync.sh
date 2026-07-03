#!/usr/bin/env bash
set -euo pipefail

# Train MAI Profile DSpark while writing checkpoints locally for speed, then
# periodically sync local checkpoints/tensorboard logs to the MSN.DnI mount.
#
# Rationale: writing FSDP checkpoints directly to the ADLS/Cosmos mount is slow.
# This wrapper keeps the training hot path on local disk and mirrors artifacts
# to $AZURE_ML_INPUT_msndni so they survive VM/job cleanup.

EXP_NAME=${EXP_NAME:-dspark_block5_gemma4_12b_maiprofile_short_1024ctx_10k}
CONFIG_PATH=${CONFIG_PATH:-config/dspark/dspark_gemma4_12b_small.py}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-3000}
CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-250}
SYNC_INTERVAL_SECS=${SYNC_INTERVAL_SECS:-300}
SYNC_DELETE=${SYNC_DELETE:-0}
RUN_SYNC_LOOP=${RUN_SYNC_LOOP:-1}
DATE=${DATE:-20260615}

msndni_mount=${AZURE_ML_INPUT_msndni:?AZURE_ML_INPUT_msndni is not set}
BASE_DIR=${BASE_DIR:-${msndni_mount}/shares/users/zxy/maiprofile}
TARGET_CACHE_DIR=${TARGET_CACHE_DIR:-${BASE_DIR}/target_cache/${DATE}/gemma4_12b_maiprofile_short_layers}

LOCAL_CKPT_DIR=${LOCAL_CKPT_DIR:-/home/aiscuser/checkpoints/deepspec_small/${EXP_NAME}}
LOCAL_TB_DIR=${LOCAL_TB_DIR:-/home/aiscuser/tensorboard/deepspec_small/${EXP_NAME}}
REMOTE_CKPT_DIR=${REMOTE_CKPT_DIR:-${BASE_DIR}/checkpoints/deepspec_small/${EXP_NAME}}
REMOTE_TB_DIR=${REMOTE_TB_DIR:-${BASE_DIR}/tensorboard/deepspec_small/${EXP_NAME}}
SYNC_LOG=${SYNC_LOG:-${BASE_DIR}/logs/${EXP_NAME}_sync.log}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

# Keep HF cache stable even if callers override HOME elsewhere.
export HF_HOME=${HF_HOME:-/home/aiscuser/.cache/huggingface}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

mkdir -p "$(dirname "${LOCAL_CKPT_DIR}")" "$(dirname "${LOCAL_TB_DIR}")"
mkdir -p "${REMOTE_CKPT_DIR}" "${REMOTE_TB_DIR}" "$(dirname "${SYNC_LOG}")"

sync_once() {
    local rsync_flags=(-a)
    if [[ "${SYNC_DELETE}" == "1" || "${SYNC_DELETE}" == "true" ]]; then
        rsync_flags+=(--delete)
    fi
    {
        echo "[$(date --iso-8601=seconds)] sync start"
        if [[ -d "${LOCAL_CKPT_DIR}" ]]; then
            mkdir -p "${REMOTE_CKPT_DIR}"
            rsync "${rsync_flags[@]}" "${LOCAL_CKPT_DIR}/" "${REMOTE_CKPT_DIR}/"
            du -sh "${LOCAL_CKPT_DIR}" "${REMOTE_CKPT_DIR}" 2>/dev/null || true
        else
            echo "checkpoint dir not created yet: ${LOCAL_CKPT_DIR}"
        fi
        if [[ -d "${LOCAL_TB_DIR}" ]]; then
            mkdir -p "${REMOTE_TB_DIR}"
            rsync "${rsync_flags[@]}" "${LOCAL_TB_DIR}/" "${REMOTE_TB_DIR}/"
        else
            echo "tensorboard dir not created yet: ${LOCAL_TB_DIR}"
        fi
        echo "[$(date --iso-8601=seconds)] sync end"
    } >> "${SYNC_LOG}" 2>&1
}

sync_loop() {
    while true; do
        sync_once || true
        sleep "${SYNC_INTERVAL_SECS}"
    done
}

SYNC_PID=""
cleanup() {
    local exit_code=$?
    if [[ -n "${SYNC_PID}" ]]; then
        kill "${SYNC_PID}" 2>/dev/null || true
        wait "${SYNC_PID}" 2>/dev/null || true
    fi
    echo "Final artifact sync to MSN.DnI mount..."
    sync_once || true
    echo "Local checkpoint dir:  ${LOCAL_CKPT_DIR}"
    echo "Remote checkpoint dir: ${REMOTE_CKPT_DIR}"
    echo "Sync log:              ${SYNC_LOG}"
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

if [[ "${RUN_SYNC_LOOP}" == "1" || "${RUN_SYNC_LOOP}" == "true" ]]; then
    sync_loop &
    SYNC_PID=$!
    echo "Started background artifact sync loop: pid=${SYNC_PID}, interval=${SYNC_INTERVAL_SECS}s"
    echo "Sync log: ${SYNC_LOG}"
fi

echo "Training MAI Profile DSpark"
echo "  exp_name:          ${EXP_NAME}"
echo "  target_cache:      ${TARGET_CACHE_DIR}"
echo "  local checkpoint:  ${LOCAL_CKPT_DIR}"
echo "  remote checkpoint: ${REMOTE_CKPT_DIR}"
echo "  local tensorboard: ${LOCAL_TB_DIR}"
echo "  remote tensorboard:${REMOTE_TB_DIR}"
echo "  steps:             ${MAX_TRAIN_STEPS}"
echo "  ckpt every:        ${CHECKPOINTING_STEPS}"

python train.py \
  --config "${CONFIG_PATH}" \
  --opts "exp_name=${EXP_NAME}" \
  --opts "data.target_cache_path=${TARGET_CACHE_DIR}" \
  --opts "train.max_train_steps=${MAX_TRAIN_STEPS}" \
  --opts "logging.checkpointing_steps=${CHECKPOINTING_STEPS}"
