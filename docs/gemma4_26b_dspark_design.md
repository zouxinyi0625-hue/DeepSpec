# Gemma4-26B-A4B DSpark 设计（MoE target, DENSE draft）

> 决策：**方案 A — dense draft over MoE target**，对齐 Google 官方 26B MTP assistant。
> 状态：代码就绪，**待服务器验证结构 + 生成 target cache 后方可训练**。未训练。

## Master 表：状态总览

| 项 | 结论 | 状态 | 依据 |
|---|---|---|---|
| target 模型 | `gemma-4-26B-A4B-it-text-only`（MoE，激活4B） | ✅ 定 | `26b_e011_mtp.json` |
| target 路径 | `$AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only` | ✅ 定 | 用户提供 |
| **draft 结构** | **DENSE**（`enable_moe_block=False`） | ✅ 定（方案A） | 官方 26B MTP 是 dense |
| draft backbone 层数 | `num_draft_layers=4`（官方 MTP=4层） | ⚠️ 待探测确认 | `gemma4_mtp.py` |
| target 层数 / `target_layer_ids` | 自动从 target `num_hidden_layers` 派生 | ⚠️ **待 probe** | 未知具体层数 |
| block_size | 7（沿用 best-result 配方） | ✅ 定 | 12B block7 最优 |
| 训练数据（regen） | `mtp_26b/split/train_maiprofile_26b.jsonl`（29.3万条） | ✅ 已存在，不重跑 | `gemma4-mtp-trainer/docs/DATA.md` |
| **DSpark target cache** | 多层 hidden 拼接，**≠ MTP cache** | ❌ **未生成，必须先跑** | `prepare_target_cache.py` |
| smoke（结构+前向） | tiny MoE 版已过 | ✅ 服务器已验证 | `smoke_dspark_gemma4_moe.py` |

## 关键设计判断：为什么 draft 用 dense

Google 自己给 26B **MoE** target 配的官方 draft（`gemma-4-26B-A4B-it-assistant`，
见 `vllm-msn/vllm/model_executor/models/gemma4_mtp.py`）是 **dense** 的：

- 4 层 decoder，FFN 用 `Gemma4MLP`（dense），**无 router / experts**
- Q-only attention + KV-shared（读 target KV，draft 自己不算 K/V）
- `pre_projection: Linear(2×backbone_hidden → draft_hidden)`

即：**target 是 MoE 不代表 draft 也要 MoE**。draft 的使命是轻、快、够准就行。
issue #60 往 DSpark draft 加 MoE 是另一条路（方案 B），26B 这里我们先走 A 对齐官方
baseline。issue #60 的 MoE 代码保留但通过 `enable_moe_block=False` 不启用，将来做 A/B 时再开。

## 数据流

```
raw prompts ($MNT/data/raw_data/20260616)
  → 26B regen ($MNT/regen_26b/20260616)                    [已完成，不重跑]
  → 90/10 split ($MNT/mtp_26b/split/{train,eval}_maiprofile_26b.jsonl)  [已完成]
  → DSpark target cache (prepare_target_cache.py)          [❌ 待生成]
  → train (train.py --config dspark_gemma4_26b.py)         [待 cache 后]
```

`$MNT = $AZURE_ML_INPUT_UKWDATA/maiprofile`。regen row schema：
`{id, source_layer, conversations:[sys/user/assistant], status}`，
`ConversationCollator(chat_template="gemma4")` 直接吃 `conversations`。

## ⚠️ DSpark cache ≠ MTP cache（重点）

MTP trainer 的 cache 存 `last_hidden` + `{full,sliding} shared_kv_states`（单层 + KV）。
DSpark 需要的是 **多个 `target_layer_ids` 层的 hidden 拼接**（`prepare_target_cache.py`
用 forward hook 抓 `target_layer_ids` 各层输出，dim=-1 concat）+ `last_hidden`。
两者**不通用**，26B DSpark cache 必须用 `prepare_target_cache.py` 重新生成。

## 待服务器执行的步骤（顺序）

### 1. 探测 target 结构（只读，无副作用）
```bash
python scripts/probe_gemma4_target.py
```
确认：`num_hidden_layers`、`enable_moe_block=True`、`num_experts`/`top_k_experts`、
`hidden_size`、以及自动派生的 `target_layer_ids`。**用输出的层数回填/确认 config。**

### 2. 生成 DSpark 26B target cache
```bash
python scripts/data/prepare_target_cache.py \
  --config config/dspark/dspark_gemma4_26b.py \
  --train-data-path $AZURE_ML_INPUT_UKWDATA/maiprofile/mtp_26b/split/train_maiprofile_26b.jsonl \
  --output-dir <cache_out_dir> \
  --min-loss-tokens 14
```
⚠️ 潜在坑：`prepare_target_cache.py:57,68` 只识别 `model_type in (gemma4, gemma4_unified)`
取 `language_model` / `text_config.hidden_size`。若 26B text-only 顶层 `model_type` 是
`gemma4_text`（直接文本模型），会走 else 分支取 `target_model.model`——**需 probe 确认
model_type 后验证这条路径**，可能要小补丁。

### 3. 训练
```bash
TARGET_CACHE_DIR=<cache_out_dir> bash scripts/train_maiprofile/train_dspark_26b_sync.sh
```

## 待定 / 需你拍板的开放项

| # | 问题 | 默认 | 需确认 |
|---|---|---|---|
| 1 | draft 层数 4 还是沿用 5？ | 4（对齐官方 MTP） | probe 后定 |
| 2 | `target_layer_ids` 具体层号 | 自动均匀派生 | probe 出层数后可手工调优 |
| 3 | `prepare_target_cache.py` 对 26B model_type 的兼容 | 假设 gemma4 顶层 | probe 后验证，可能补丁 |
| 4 | warm-start：12B block7 权重能否热启 26B draft？ | 否（hidden/vocab 不同） | 大概率不行，from scratch |
| 5 | 是否要做 A/B（dense vs MoE draft） | 先只做 A | 你定 |

## 已改代码（本分支 dev/maiprofile_moe）

| 文件 | 改动 |
|---|---|
| `deepspec/modeling/dspark/gemma4/config.py` | 加 `model.enable_moe_block` 覆盖，允许 MoE target 上建 dense draft |
| `config/dspark/dspark_gemma4_26b.py` | 26B dense draft 训练 config（原 `dspark_gemma4_moe.py` 改名） |
| `scripts/train_maiprofile/train_dspark_26b_sync.sh` | 26B launcher（原 `train_dspark_moe_sync.sh` 改名） |
| `scripts/probe_gemma4_target.py` | 只读 target 结构探测 |
| `deepspec/modeling/dspark/gemma4/modeling.py` | issue #60 MoE 块支持（保留，A 方案下不启用） |
| `scripts/smoke_dspark_gemma4_moe.py` | MoE 结构 smoke（✅ 已过） |
