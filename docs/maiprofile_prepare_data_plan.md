# MAI Profile DSpark Data Preparation Plan

This plan covers the first DSpark data pipeline for MAI Profile prompt data. It intentionally starts with shorter-input layers only.

## Scope

Raw data lives under:

```text
$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/raw_data/20260615
```

Initial short-layer set:

```text
layer1_actual
layer1_intent
layer2_temporal
layer3_seasonality
layer4_commercial_preference
```

These layers were selected because their prompt lengths are mostly within or near 4k tokens, unlike `layer1_delta` and other long-context layers.

## Outputs

The prepare step writes all derived data under the same mounted MSN.DnI tree:

```text
$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/prepared_prompts/20260615/short_layers/
$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/regenerated/20260615/
$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/target_cache/20260615/
```

Split artifacts:

```text
prepared_prompts/20260615/short_layers/train_maiprofile_short_layers.jsonl
prepared_prompts/20260615/short_layers/train_<layer>.jsonl
prepared_prompts/20260615/short_layers/eval_datasets/maiprofile_<layer>.jsonl
prepared_prompts/20260615/short_layers/split_summary.json
```

Regenerated training data:

```text
regenerated/20260615/maiprofile_short_layers_regen.jsonl
```

Target cache:

```text
target_cache/20260615/gemma4_12b_maiprofile_short_layers
```

## Split Format

Training prompt JSONL uses the same schema expected by `scripts/data/generate_train_data.py`:

```json
{
  "id": "layer1_actual:<prompt_hash>",
  "source_layer": "layer1_actual",
  "source_file": "layer1_actual.jsonl",
  "source_line": 123,
  "user_id": "...",
  "prompt_hash": "...",
  "conversations": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

Eval JSONL uses the format expected by `deepspec/eval/base_evaluator.py`. It preserves the original message roles via `messages` and also writes a legacy single-turn fallback in `turns`:

```json
{
  "id": "layer1_actual:<prompt_hash>",
  "source_layer": "layer1_actual",
  "user_id": "...",
  "prompt_hash": "...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "turns": ["<system prompt>\n\n<user prompt>"]
}
```

The evaluator now prefers `messages` when present, so MAI Profile eval uses the same `system + user` structure as SGLang generation. `turns` is retained only for compatibility with the original eval dataset format.

## Step 1: Start SGLang

Example with Gemma4-12B:

```bash
model_path=google/gemma-4-12B-it \
num_workers=8 \
start_port=30000 \
start_nccl_port=32000 \
log_dir=logs/sglang_gemma4_12b_maiprofile \
stream_logs=1 \
bash scripts/data/launch_sglang_server.sh
```

Use a non-conflicting `start_nccl_port` if `31000-31007` are occupied.

## Step 2: Prepare splits, regenerate responses, and build cache

```bash
PYTHONPATH=$PWD \
model_path=google/gemma-4-12B-it \
config_path=config/dspark/dspark_gemma4_12b_small.py \
layers=short \
eval_size=200 \
max_tokens=2048 \
local_batch_size=2 \
stop_sglang_after_generation=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/data/prepare_maiprofile_data.sh
```

Useful pilot options:

```bash
max_train_samples=1000
num_samples=1000
```

`num_samples` limits SGLang generation after the train split has been created. `max_train_samples` caps the train split itself.

## Step 3: Train

Training command will be finalized after the first target cache is built and inspected. Use an experiment name that identifies the source data and context length.

## Step 4: Eval on held-out MAI Profile prompts

The split script writes eval files named `maiprofile_<layer>.jsonl`. Use `DATASET_ROOT` to point eval at the mounted eval directory.

Example:

```bash
DATASET_ROOT=$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/prepared_prompts/20260615/short_layers/eval_datasets \
TASKS=maiprofile_layer1_actual:200,maiprofile_layer1_intent:200,maiprofile_layer2_temporal:200,maiprofile_layer3_seasonality:200,maiprofile_layer4_commercial_preference:200 \
TARGET_NAME_OR_PATH=google/gemma-4-12B-it \
DRAFT_NAME_OR_PATH=<checkpoint-path> \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/train_small_gemma4/eval_dspark_1k.sh
```

## Notes

- The first pipeline deliberately excludes long-context layers such as `layer1_delta` and `layer3_commercial_interests`.
- The held-out eval split is for acceptance-rate metrics only; it does not need target-generated assistant labels.
- Training data still requires target-generated assistant responses before target-cache generation.
