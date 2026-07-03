#!/usr/bin/env bash
set -euo pipefail

model_path=${model_path:-google/gemma-4-12B-it}
config_path=${config_path:-config/dspark/dspark_gemma4_12b_small.py}
date=${date:-20260615}
layers=${layers:-short}
eval_size=${eval_size:-200}
max_train_samples=${max_train_samples:-}

msndni_mount=${AZURE_ML_INPUT_msndni:?AZURE_ML_INPUT_msndni is not set}
base_dir=${base_dir:-${msndni_mount}/shares/users/zxy/maiprofile}
raw_dir=${raw_dir:-${base_dir}/raw_data/${date}}
prepared_dir=${prepared_dir:-${base_dir}/prepared_prompts/${date}/short_layers}
train_split_path=${train_split_path:-${prepared_dir}/train_maiprofile_short_layers.jsonl}
train_data_path=${train_data_path:-${base_dir}/regenerated/${date}/maiprofile_short_layers_regen.jsonl}
cache_dir=${cache_dir:-${base_dir}/target_cache/${date}/gemma4_12b_maiprofile_short_layers}

server_host=${server_host:-127.0.0.1}
num_workers=${num_workers:-8}
start_port=${start_port:-30000}
concurrency=${concurrency:-32}
temperature=${temperature:-1.0}
top_p=${top_p:-0.95}
top_k=${top_k:-20}
min_p=${min_p:-0}
max_tokens=${max_tokens:-2048}
num_samples=${num_samples:-}
local_batch_size=${local_batch_size:-2}
stop_sglang_after_generation=${stop_sglang_after_generation:-1}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

server_addresses=()
for ((worker_id = 0; worker_id < num_workers; worker_id++)); do
    server_addresses+=("${server_host}:$((start_port + worker_id))")
done

stop_sglang_workers() {
    local port pids pid
    echo "Stopping SGLang workers for model=${model_path}, ports ${start_port}..$((start_port + num_workers - 1))"
    for ((worker_id = 0; worker_id < num_workers; worker_id++)); do
        port=$((start_port + worker_id))
        pids=$(pgrep -f "sglang serve.*--model-path ${model_path}.*--port ${port}" || true)
        if [[ -z "${pids}" ]]; then
            pids=$(pgrep -f "sglang.*--model-path ${model_path}.*--port ${port}" || true)
        fi
        if [[ -z "${pids}" ]]; then
            echo "  port=${port}: no matching SGLang process found"
            continue
        fi
        for pid in ${pids}; do
            echo "  port=${port}: terminate pid=${pid}"
            kill "${pid}" 2>/dev/null || true
        done
    done
    sleep 5
}

echo "Step 1/3: preparing MAI Profile train/eval splits"
split_args=(
    --input-dir "${raw_dir}"
    --output-dir "${prepared_dir}"
    --layers "${layers}"
    --eval-size "${eval_size}"
    --date "${date}"
)
if [[ -n "${max_train_samples}" ]]; then
    split_args+=(--max-train-samples "${max_train_samples}")
fi
python scripts/data/prepare_maiprofile_splits.py "${split_args[@]}"

mkdir -p "$(dirname "${train_data_path}")"

echo "Step 2/3: generating ${model_path} MAI Profile train data"
echo "  input:  ${train_split_path}"
echo "  output: ${train_data_path}"
echo "Start SGLang first with scripts/data/launch_sglang_server.sh"
gen_args=(
    --model "${model_path}"
    --server-address "${server_addresses[@]}"
    --concurrency "${concurrency}"
    --temperature "${temperature}"
    --top-p "${top_p}"
    --top-k "${top_k}"
    --min-p "${min_p}"
    --max-tokens "${max_tokens}"
    --disable-thinking
    --resume
    --input-file-path "${train_split_path}"
    --output-file-path "${train_data_path}"
)
if [[ -n "${num_samples}" ]]; then
    gen_args+=(--num-samples "${num_samples}")
fi
python scripts/data/generate_train_data.py "${gen_args[@]}"

if [[ "${stop_sglang_after_generation}" == "1" || "${stop_sglang_after_generation}" == "true" ]]; then
    stop_sglang_workers
else
    echo "Stop SGLang before Step 3 if it is using the same GPUs."
fi

echo "Step 3/3: preparing target cache"
echo "  regen: ${train_data_path}"
echo "  cache: ${cache_dir}"
python scripts/data/prepare_target_cache.py \
    --config "${config_path}" \
    --train-data-path "${train_data_path}" \
    --output-dir "${cache_dir}" \
    --local-batch-size "${local_batch_size}"

echo "Done."
echo "Prepared prompts: ${prepared_dir}"
echo "Eval datasets: ${prepared_dir}/eval_datasets"
echo "Regenerated train data: ${train_data_path}"
echo "Target cache: ${cache_dir}"
