import os

from deepspec.trainer import Gemma4DSparkTrainer


BASE_TB_DIR = os.path.expanduser("~/tensorboard")
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")
project_name = "deepspec_small"
exp_name = "dspark_block5_gemma4_12b_1k"
seed = 42

# Small-data smoke/reproduction config for Gemma4-12B DSpark training.
# This keeps the same model family as config/dspark/dspark_gemma4_12b.py,
# but reduces sequence length, block size, sampled anchors, and train steps so
# the full pipeline can be tested on a 1k-sample subset before scaling up.
model = dict(
    target_model_name_or_path="google/gemma-4-12B-it",
    block_size=5,
    num_draft_layers=5,
    target_layer_ids=[5, 17, 29, 41, 46],
    mask_token_id=4,
    num_anchors=128,
    # Optional warm-start: local path or HF id of a pretrained DSpark draft
    # checkpoint. None = train draft from scratch. Overridable via
    # --opts "model.pretrained_draft_path=/path/to/ckpt".
    pretrained_draft_path=None,

    # Markov head: DSpark = DFlash-style backbone + lightweight transition bias.
    markov_rank=256,
    markov_head_type="vanilla",

    # Confidence head is trained so offline threshold sweeps can be tested.
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    # Loss mirrors the paper/default DSpark objective.
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
    global_batch_size=64,
    num_train_epochs=1,
    max_train_steps=200,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=False,
)

logging = dict(
    logging_steps=5,
    checkpointing_steps=50,
)

data = dict(
    target_cache_path=None,
    chat_template="gemma4",
    max_length=1024,
    num_workers=2,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg["project_name"])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(
        BASE_CKPT_DIR,
        project_name,
        exp_name,
    )
    logging_cfg["tensorboard_dir"] = os.path.join(
        BASE_TB_DIR,
        project_name,
        exp_name,
    )
    cfg["logging"] = logging_cfg
    return cfg
