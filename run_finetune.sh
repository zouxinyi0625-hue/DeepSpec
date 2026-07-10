#!/usr/bin/env bash
set -euo pipefail

# Finetune DSpark on maiprofile, WARM-STARTED from a pretrained DSpark checkpoint
# (instead of training from scratch). Uses the CORRECT 4096-context cache
# (5 layers balanced, 35,800/36,103 valid). Trains block7 and block5 back-to-back.
#
# Warm-start is via model.pretrained_draft_path (loads backbone+heads weights,
# fresh optimizer/step 0). This is NOT resume -- resume needs per-rank state files
# which a public checkpoint doesn't have.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=1
export PYTHONPATH=$PWD

# checkpoint / tensorboard -> MOUNT
export HOME=${AZURE_ML_INPUT_msndni}/shares/users/zxy/maiprofile

export HF_HOME=/home/aiscuser/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

BASE=${AZURE_ML_INPUT_msndni}/shares/users/zxy/maiprofile

# ---- maiprofile data (identical to the from-scratch runs) ----
CACHE_4096=${BASE}/target_cache/20260615/gemma4_12b_maiprofile_short_layers_4096
EVAL_ROOT=${BASE}/prepared_prompts/20260615/short_layers/eval_datasets
EVAL_TASKS="maiprofile_layer1_actual:200,maiprofile_layer1_intent:200,maiprofile_layer2_temporal:200,maiprofile_layer3_seasonality:200,maiprofile_layer4_commercial_preference:200"
TARGET=google/gemma-4-12B-it
EVAL_MAX_NEW_TOKENS=512
EVAL_TEMPERATURE=1.0

# ---- pretrained DSpark checkpoint to warm-start from ----
# Set this to the local path or HF id of the public pretrained DSpark block7.
# NOTE: the pretrained checkpoint's block_size must match the block_size trained
# below (backbone shapes must line up). block7 pretrained -> train block7.
PRETRAINED_DRAFT=${PRETRAINED_DRAFT:-dspark_gemma4_12b_block7}

NUM_WORKERS=${NUM_WORKERS:-0}

CKPT_DIR=${HOME}/checkpoints
RESULTS_DIR=${BASE}/eval_results/20260615
mkdir -p "${HOME}/checkpoints" "${HOME}/tensorboard" "${RESULTS_DIR}"

if [[ ! -f "${CACHE_4096}/manifest.json" ]]; then
  echo "ERROR: 4096 cache manifest not found at ${CACHE_4096}/manifest.json" >&2
  exit 1
fi
if [[ ! -d "${EVAL_ROOT}" ]]; then
  echo "ERROR: eval dataset root not found: ${EVAL_ROOT}" >&2
  exit 1
fi

train_one() {
  local block=$1
  local exp_name="dspark_block${block}_gemma4_12b_maiprofile_finetune_4096ctx_36k"
  echo "=================================================================="
  echo ">>> TRAIN (warm-start) START ${exp_name} ($(date --iso-8601=seconds))"
  echo ">>>   pretrained_draft_path=${PRETRAINED_DRAFT}"
  echo "=================================================================="
  python train.py \
    --config config/dspark/dspark_gemma4_12b_small.py \
    --opts "exp_name=${exp_name}" \
    --opts "model.block_size=${block}" \
    --opts "model.pretrained_draft_path=${PRETRAINED_DRAFT}" \
    --opts "data.max_length=4096" \
    --opts "data.num_workers=${NUM_WORKERS}" \
    --opts "data.target_cache_path=${CACHE_4096}" \
    --opts "train.max_train_steps=2000" \
    --opts "logging.checkpointing_steps=500"
  echo ">>> TRAIN DONE ${exp_name} ($(date --iso-8601=seconds))"
}

eval_one() {
  local block=$1
  local exp_name="dspark_block${block}_gemma4_12b_maiprofile_finetune_4096ctx_36k"
  local draft="${CKPT_DIR}/deepspec/${exp_name}/step_latest"
  echo "=================================================================="
  echo ">>> EVAL START ${exp_name} ($(date --iso-8601=seconds))"
  echo "=================================================================="
  if [[ ! -e "${draft}" ]]; then
    echo "ERROR: checkpoint not found for eval: ${draft}" >&2
    return 1
  fi
  python eval.py \
    --target_name_or_path "${TARGET}" \
    --draft_name_or_path "${draft}" \
    --dataset-root "${EVAL_ROOT}" \
    --tasks "${EVAL_TASKS}" \
    --max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
    --temperature "${EVAL_TEMPERATURE}"
  echo ">>> EVAL DONE ${exp_name} ($(date --iso-8601=seconds))"
}

run_one() {
  local block=$1
  train_one "${block}"
  eval_one "${block}"
}

main() {
  echo "###### DSpark maiprofile FINETUNE run START $(date --iso-8601=seconds) ######"
  echo "###### warm-start from: ${PRETRAINED_DRAFT} ######"
  run_one 7
  run_one 5
  echo "###### ALL FINETUNE RUNS + EVALS COMPLETE $(date --iso-8601=seconds) ######"
}

LOG=${RESULTS_DIR}/dspark_finetune_4096_run.log
main 2>&1 | tee -a "${LOG}"
