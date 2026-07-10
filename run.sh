#!/usr/bin/env bash
set -euo pipefail

# EAGLE-3 on the CORRECT 4096-context maiprofile cache (5 layers balanced,
# 35,800/36,103 valid). For each ttt_length in {7,5}: train then eval, all on
# maiprofile data (same cache / same eval layers as the DSpark runs).
#
# ttt_length is EAGLE-3's block_size equivalent: ttt7 <-> DSpark block7,
# ttt5 <-> DSpark block5. The two runs use different exp_name so their
# checkpoints/tensorboard never collide.
#
# 2000 train steps each, checkpoint every 500. All stdout+stderr is shown in
# the terminal AND written to a log file on the mount (via tee at the end).

# Run from wherever this script lives (the DeepSpec repo root). No hard-coded
# job path -- works on any machine as long as you `bash run.sh` inside DeepSpec.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=1
export PYTHONPATH=$PWD

# checkpoint / tensorboard -> MOUNT (same HOME trick as the DSpark run)
export HOME=${AZURE_ML_INPUT_msndni}/shares/users/zxy/maiprofile

# HF cache stays on local disk
export HF_HOME=/home/aiscuser/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

BASE=${AZURE_ML_INPUT_msndni}/shares/users/zxy/maiprofile

# ---- maiprofile data (identical to the DSpark runs) ----
CACHE_4096=${BASE}/target_cache/20260615/gemma4_12b_maiprofile_short_layers_4096
EVAL_ROOT=${BASE}/prepared_prompts/20260615/short_layers/eval_datasets
EVAL_TASKS="maiprofile_layer1_actual:200,maiprofile_layer1_intent:200,maiprofile_layer2_temporal:200,maiprofile_layer3_seasonality:200,maiprofile_layer4_commercial_preference:200"
TARGET=google/gemma-4-12B-it

# eval knobs must match the DSpark eval for a fair comparison
EVAL_MAX_NEW_TOKENS=512
EVAL_TEMPERATURE=1.0

# Dataloader workers. eagle3 config defaults to 4 which deadlocked on this mount
# (futex_wait, read_bytes=0). DSpark ran fine at 2. Set 0 to fully disable
# worker subprocesses if 2 still hangs. Override: NUM_WORKERS=0 bash run.sh
NUM_WORKERS=${NUM_WORKERS:-2}

CKPT_DIR=${HOME}/checkpoints            # BASE_CKPT_DIR (HOME/checkpoints)
RESULTS_DIR=${BASE}/eval_results/20260615
mkdir -p "${HOME}/checkpoints" "${HOME}/tensorboard" "${RESULTS_DIR}"

# ---- sanity checks: fail fast if the maiprofile data isn't where we expect ----
if [[ ! -f "${CACHE_4096}/manifest.json" ]]; then
  echo "ERROR: 4096 cache manifest not found at ${CACHE_4096}/manifest.json" >&2
  exit 1
fi
if [[ ! -d "${EVAL_ROOT}" ]]; then
  echo "ERROR: eval dataset root not found: ${EVAL_ROOT}" >&2
  exit 1
fi

train_one() {
  local ttt=$1
  local exp_name="eagle3_ttt${ttt}_gemma4_12b_maiprofile_short_4096ctx_36k"
  echo "=================================================================="
  echo ">>> TRAIN START ${exp_name} ($(date --iso-8601=seconds))"
  echo "=================================================================="
  python train.py \
    --config config/eagle3/eagle3_gemma4_12b.py \
    --opts "exp_name=${exp_name}" \
    --opts "model.ttt_length=${ttt}" \
    --opts "data.max_length=4096" \
    --opts "data.num_workers=${NUM_WORKERS}" \
    --opts "data.target_cache_path=${CACHE_4096}" \
    --opts "train.max_train_steps=2000" \
    --opts "logging.checkpointing_steps=500"
  echo ">>> TRAIN DONE ${exp_name} ($(date --iso-8601=seconds))"
}

eval_one() {
  local ttt=$1
  local exp_name="eagle3_ttt${ttt}_gemma4_12b_maiprofile_short_4096ctx_36k"
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
  local ttt=$1
  train_one "${ttt}"
  eval_one "${ttt}"
}

main() {
  echo "###### EAGLE-3 maiprofile run START $(date --iso-8601=seconds) ######"
  run_one 7
  run_one 5
  echo "###### ALL EAGLE-3 RUNS + EVALS COMPLETE $(date --iso-8601=seconds) ######"
}

# Run everything, streaming to terminal AND appending to a log file on the mount.
LOG=${RESULTS_DIR}/eagle3_4096_run.log
main 2>&1 | tee -a "${LOG}"
