#!/usr/bin/env bash
set -euo pipefail

model_path=google/gemma-4-12B-it
config_path=config/dspark/dspark_gemma4_12b.py

dataset_name=mlabonne/open-perfectblend
test_size=0.05
train_split_path=train_datasets/perfectblend_train.jsonl
train_split_small_path=train_datasets/perfectblend_train_small_1k.jsonl
eval_data_dir=eval_datasets

train_data_path=train_datasets/gemma4_12b/perfectblend_train_regen_small.jsonl
cache_dir=${HOME}/.cache/deepspec/gemma4_12b_target_cache_small

max_samples=1000

server_address="https://fabricrouter-azureglobalprivate.ingress-dlis.ingress.cus.microsoft-falcon.net/dlis-coreranker.chrona-gemma4-optimized-single/v1"
concurrency=32
temperature=0.7
top_p=0.8
top_k=20
min_p=0
max_tokens=4096

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

echo "Step 1/3: downloading and splitting ${dataset_name}"
python scripts/data/download_and_split.py \
    --dataset-name "${dataset_name}" \
    --test-size "${test_size}" \
    --train-output-path "${train_split_path}" \
    --test-output-dir "${eval_data_dir}" \
    --skip-existing

echo "Truncating to ${max_samples} samples: ${train_split_small_path}"
head -n "${max_samples}" "${train_split_path}" > "${train_split_small_path}"

mkdir -p "$(dirname "${train_data_path}")"

echo "Step 2/3: generating gemma4-12b train data (${max_samples} samples): ${train_data_path}"
python scripts/data/generate_train_data.py \
    --model "${model_path}" \
    --server-address "${server_address}" \
    --concurrency "${concurrency}" \
    --temperature "${temperature}" \
    --top-p "${top_p}" \
    --top-k "${top_k}" \
    --min-p "${min_p}" \
    --max-tokens "${max_tokens}" \
    --disable-thinking \
    --resume \
    --input-file-path "${train_split_small_path}" \
    --output-file-path "${train_data_path}"

echo "Step 3/3: preparing gemma4-12b target cache (small): ${cache_dir}"
python scripts/data/prepare_target_cache.py \
    --config "${config_path}" \
    --train-data-path "${train_data_path}" \
    --output-dir "${cache_dir}" \
    --local-batch-size 16
