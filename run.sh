#!/usr/bin/env bash
set -euo pipefail

# EAGLE-3 training on the CORRECT 4096-context maiprofile cache (5 layers balanced,
# 35,800/36,103 valid). Runs two configs back-to-back: ttt_length=7 then =5,
# matching DSpark block7 / block5 for apples-to-apples comparison.
#
# 2000 steps each, checkpoint every 500. At ~34s/step (ttt7) / ~24s/step (ttt5)
# this is roughly 19h + 13h = ~32h, well inside a 72h window.

cd /scratch/azureml/cr/j/62762bfeddfd4c1b8e0df81ac7b09742/exe/wd/DeepSpec

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

CACHE_4096=${AZURE_ML_INPUT_msndni}/shares/users/zxy/maiprofile/target_cache/20260615/gemma4_12b_maiprofile_short_layers_4096

mkdir -p ${HOME}/checkpoints ${HOME}/tensorboard

# sanity: correct cache must exist and be 4096
if [[ ! -f "${CACHE_4096}/manifest.json" ]]; then
  echo "ERROR: 4096 cache manifest not found at ${CACHE_4096}/manifest.json" >&2
  exit 1
fi

train_one() {
  local ttt=$1
  local exp_name="eagle3_ttt${ttt}_gemma4_12b_maiprofile_short_4096ctx_36k"
  echo "=================================================================="
  echo ">>> START ${exp_name} ($(date --iso-8601=seconds))"
  echo "=================================================================="
  python train.py \
    --config config/eagle3/eagle3_gemma4_12b.py \
    --opts "exp_name=${exp_name}" \
    --opts "model.ttt_length=${ttt}" \
    --opts "data.max_length=4096" \
    --opts "data.target_cache_path=${CACHE_4096}" \
    --opts "train.max_train_steps=2000" \
    --opts "logging.checkpointing_steps=500"
  echo ">>> DONE ${exp_name} ($(date --iso-8601=seconds))"
}

train_one 7
train_one 5

echo "ALL EAGLE-3 RUNS COMPLETE ($(date --iso-8601=seconds))"
