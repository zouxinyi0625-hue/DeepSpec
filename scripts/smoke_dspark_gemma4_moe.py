"""Offline structural smoke test for Gemma4 MoE DSpark draft.

Builds a *tiny* Gemma4 MoE text config, constructs Gemma4DSparkModel, and runs a
forward pass on random inputs — no real weights, no target checkpoint, CPU-only.
Verifies that the MoE branch (router/experts + the 3 extra RMSNorms) instantiates
and the forward path is shape-consistent.

Run on the server (where transformers is importable):
    python scripts/smoke_dspark_gemma4_moe.py
"""

import os
import sys

import torch

# Make the repo root importable regardless of the current working directory
# (the server launches this from an arbitrary exe/wd path).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from deepspec.modeling.dspark.gemma4.modeling import Gemma4DSparkModel


def build_tiny_moe_config() -> Gemma4TextConfig:
    hidden = 64
    cfg = Gemma4TextConfig(
        vocab_size=256,
        hidden_size=hidden,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        # MoE on
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=128,
    )
    # DSpark-specific extra fields the draft model/config expect.
    cfg.attention_k_eq_v = False
    cfg.num_global_key_value_heads = 2
    cfg.global_head_dim = 16
    cfg.attention_bias = False
    cfg.attention_dropout = 0.0
    cfg.hidden_size_per_layer_input = 0
    cfg.num_kv_shared_layers = 0
    cfg.use_double_wide_mlp = False
    cfg.final_logit_softcapping = None
    # DSpark draft head config
    cfg.target_layer_ids = [0, 1]
    cfg.mask_token_id = 255
    cfg.num_anchors = 2
    cfg.enable_confidence_head = False
    cfg.markov_rank = 0
    cfg.block_size = 4
    cfg.tie_word_embeddings = False
    cfg._attn_implementation = "flex_attention"
    cfg.layer_types = ["full_attention"] * cfg.num_hidden_layers
    return cfg


def main() -> None:
    torch.manual_seed(0)
    cfg = build_tiny_moe_config()
    model = Gemma4DSparkModel(cfg).eval()

    # Confirm MoE modules exist on each decoder layer.
    for i, layer in enumerate(model.layers):
        assert layer.enable_moe_block, f"layer {i} MoE not enabled"
        assert hasattr(layer, "router") and hasattr(layer, "experts")
        assert hasattr(layer, "pre_feedforward_layernorm_2")
        assert hasattr(layer, "post_feedforward_layernorm_1")
        assert hasattr(layer, "post_feedforward_layernorm_2")
    print("[ok] MoE modules instantiated on all layers")

    bsz, seq_len = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))
    hidden = cfg.hidden_size
    target_hidden = torch.randn(bsz, seq_len, len(cfg.target_layer_ids) * hidden)
    loss_mask = torch.ones(bsz, seq_len, dtype=torch.long)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            target_hidden_states=target_hidden,
            loss_mask=loss_mask,
        )
    print(f"[ok] forward pass ran; output type={type(out).__name__}")
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
