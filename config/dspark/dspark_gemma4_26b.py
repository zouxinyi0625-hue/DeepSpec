import os

from deepspec.trainer import Gemma4DSparkTrainer


BASE_TB_DIR = os.path.expanduser("~/tensorboard")
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")
project_name = "deepspec"
exp_name = "dspark_block7_gemma4_26b_dense_maiprofile"
seed = 42

# Gemma4-26B-A4B MoE-target DSpark config — DENSE draft (design decision "A").
#
# The 26B target (google/gemma-4-26B-A4B-it-text-only) IS a MoE model, but the
# draft is kept DENSE to mirror Google's own 26B MTP assistant (a dense, Q-only
# draft over a MoE target). model.enable_moe_block=False forces the draft dense
# even though the target text_config has enable_moe_block=True; build_draft_config
# honors this override.
#
# Target model path resolves from $AZURE_ML_INPUT_UKWDATA at launch:
#   $AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only
# Override with env TARGET_MODEL_PATH.
#
# NOTE: target_layer_ids default is auto-derived from the target's
# num_hidden_layers at finalize (env TARGET_LAYER_IDS overrides). Run
# scripts/probe_gemma4_target.py first to confirm the real layer count.
_UKW = os.environ.get("AZURE_ML_INPUT_UKWDATA", "")
_DEFAULT_TARGET = (
    os.path.join(_UKW, "maiprofile/models/text_only")
    if _UKW
    else "google/gemma-4-26B-A4B-it-text-only"
)
TARGET_MODEL_PATH = os.environ.get("TARGET_MODEL_PATH", _DEFAULT_TARGET)

model = dict(
    target_model_name_or_path=TARGET_MODEL_PATH,
    block_size=7,
    num_draft_layers=4,
    # Pinned from probe (2026: 26B has 30 layers, hidden 2816). Uniform spread
    # ending at N-2. Override via env TARGET_LAYER_IDS or --opts if retuning.
    target_layer_ids=[3, 11, 19, 28],
    mask_token_id=4,
    num_anchors=512,
    pretrained_draft_path=None,

    # DENSE draft over a MoE target (design "A"). Set True to build a MoE draft.
    enable_moe_block=False,

    # markov head
    markov_rank=256,
    markov_head_type="vanilla",

    # confidence head
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    # loss
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    trainer_cls=Gemma4DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
)

data = dict(
    target_cache_path=None,
    chat_template="gemma4",
    max_length=4096,
    # num_workers=0 (single-process loading). Even with os.pread (not mmap),
    # multi-worker DataLoaders on the Azure mount deadlock: one rank's worker
    # stalls on mount I/O, that rank never reaches the first collective, and the
    # other 7 ranks spin-wait (100% util / ~120W) forever. Confirmed twice.
    # Keep 0 for mount caches; only raise if the cache is on fast local disk.
    num_workers=0,
)


def _resolve_target_layer_ids(target_path, num_draft_layers):
    """Uniformly sample `num_draft_layers` target layers, last = N-2.

    Priority: env TARGET_LAYER_IDS > read target config num_hidden_layers.
    """
    env_ids = os.environ.get("TARGET_LAYER_IDS")
    if env_ids:
        return [int(x) for x in env_ids.split(",") if x.strip() != ""]

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(target_path)
    text_cfg = getattr(cfg, "text_config", cfg)
    num_layers = int(text_cfg.num_hidden_layers)

    k = int(num_draft_layers)
    assert 1 <= k <= num_layers, f"num_draft_layers={k} out of range for {num_layers} layers"
    last = num_layers - 2
    if k == 1:
        return [last]
    # Evenly spread the first k-1 picks over [start, last), then append last.
    start = max(1, num_layers // (2 * k))
    step = max(1, (last - start) // (k - 1))
    ids = [start + i * step for i in range(k - 1)]
    ids.append(last)
    # Dedup + clamp while preserving order.
    seen, out = set(), []
    for v in ids:
        v = min(max(0, v), num_layers - 1)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def finalize_cfg(cfg):
    model_cfg = dict(cfg["model"])
    if model_cfg.get("target_layer_ids") is None:
        model_cfg["target_layer_ids"] = _resolve_target_layer_ids(
            model_cfg["target_model_name_or_path"],
            model_cfg["num_draft_layers"],
        )
    cfg["model"] = model_cfg

    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg["project_name"])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg
    return cfg
