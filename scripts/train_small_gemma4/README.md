# Gemma4-12B DSpark Small-Data Reproduction

Goal: **follow the repository's existing pipeline first**, using only a small number of prompts, so we can verify the project runs end to end before using custom data or adding innovations.

The repository's native data pipeline is:

```text
scripts/data/download_and_split.py
  -> scripts/data/launch_sglang_server.sh
  -> scripts/data/generate_train_data.py
  -> scripts/data/prepare_target_cache.py
  -> train.py
```

This branch keeps that flow. The default Qwen3 behavior of `scripts/data/launch_sglang_server.sh` and `scripts/data/prepare_data.sh` is unchanged, but they now accept environment-variable overrides so we can run a Gemma4-12B small-data smoke test.

## Small Gemma4 config

```text
config/dspark/dspark_gemma4_12b_small.py
```

Key differences from the full Gemma4 config:

```text
block_size=5
num_anchors=128
data.max_length=1024
train.global_batch_size=64
train.max_train_steps=200
```

## Step 0: install dependencies

```bash
cd /path/to/DeepSpec
python -m pip install -r requirements.txt
# SGLang is intentionally not in requirements.txt in this repo.
python -m pip install "sglang[all]"
```

## Step 1: launch Gemma4-12B through the repo's SGLang launcher

Use the project script, only overriding env vars:

```bash
cd /path/to/DeepSpec

model_path=google/gemma-4-12B-it \
num_workers=1 \
start_port=30000 \
log_dir=logs/sglang_gemma4_12b_small \
mem_frac=0.9 \
bash scripts/data/launch_sglang_server.sh
```

For multiple GPUs/workers:

```bash
model_path=google/gemma-4-12B-it \
num_workers=4 \
start_port=30000 \
log_dir=logs/sglang_gemma4_12b_small \
bash scripts/data/launch_sglang_server.sh
```

Leave this terminal running. The script prints worker URLs like:

```text
http://<host-ip>:30000
```

The data script below uses `server_host=127.0.0.1` by default, so on the same node it will call `127.0.0.1:30000`.

## Step 2: run the repo's data pipeline on a small prompt sample

In a second terminal:

```bash
cd /path/to/DeepSpec

model_path=google/gemma-4-12B-it \
config_path=config/dspark/dspark_gemma4_12b_small.py \
sample_size=1100 \
num_samples=1000 \
train_split_path=train_datasets/gemma4_12b/perfectblend_train_prompt_small.jsonl \
train_data_path=train_datasets/gemma4_12b/perfectblend_train_regen_1k.jsonl \
cache_dir=${HOME}/.cache/deepspec/gemma4_12b_target_cache_1k \
num_workers=1 \
start_port=30000 \
concurrency=8 \
temperature=1.0 \
top_p=0.95 \
top_k=20 \
min_p=0 \
max_tokens=2048 \
local_batch_size=2 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/data/prepare_data.sh
```

What this does, using repo scripts:

1. `download_and_split.py` downloads `mlabonne/open-perfectblend`, samples `sample_size=1100`, and writes prompts to `train_split_path`.
2. `generate_train_data.py` calls the SGLang server launched in Step 1 and writes 1000 regenerated Gemma4 samples to `train_data_path`.
3. `prepare_target_cache.py` builds the target cache at `cache_dir` with `config/dspark/dspark_gemma4_12b_small.py`.

Outputs:

```text
train_datasets/gemma4_12b/perfectblend_train_prompt_small.jsonl
train_datasets/gemma4_12b/perfectblend_train_regen_1k.jsonl
~/.cache/deepspec/gemma4_12b_target_cache_1k
```

If SGLang is using the same GPU needed for target-cache preparation, stop the SGLang launcher after generation completes and before Step 3 in `prepare_data.sh`. The script prints this reminder before cache preparation.

## Step 3: train DSpark on the small cache

```bash
TARGET_CACHE_DIR=${HOME}/.cache/deepspec/gemma4_12b_target_cache_1k \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_small_gemma4/train_dspark_1k.sh
```

Optional quick smoke test:

```bash
TARGET_CACHE_DIR=${HOME}/.cache/deepspec/gemma4_12b_target_cache_1k \
MAX_TRAIN_STEPS=50 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_small_gemma4/train_dspark_1k.sh
```

Checkpoints:

```text
~/checkpoints/deepspec_small/dspark_block5_gemma4_12b_1k/
```

TensorBoard:

```text
~/tensorboard/deepspec_small/dspark_block5_gemma4_12b_1k/
```

## Step 4: small eval smoke test

```bash
TARGET_NAME_OR_PATH=google/gemma-4-12B-it \
DRAFT_NAME_OR_PATH=${HOME}/checkpoints/deepspec_small/dspark_block5_gemma4_12b_1k/step_latest \
TASKS=gsm8k:32,mt-bench:16,alpaca:32 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_small_gemma4/eval_dspark_1k.sh
```

## Notes

- This is only to get the project running with a small sample. After that we can switch to custom data and add architectural changes.
- The open-source Gemma4 DSpark implementation is dense-only for the draft backbone:

```python
assert not bool(config.enable_moe_block), "Gemma4 DSpark prototype does not support Gemma4 MoE blocks yet."
```

- The native `prepare_data.sh` still defaults to the original Qwen3 pipeline if no env vars are provided.
