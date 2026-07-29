"""Read-only probe of the Gemma4-26B-A4B MoE target config.

Prints everything the DSpark 26B config needs to be pinned correctly:
model_type, layer count, hidden size, MoE params, and the auto-derived
target_layer_ids. NO model weights are loaded, NO GPU needed.

Usage (server):
    python scripts/probe_gemma4_target.py
    # or point at a specific path:
    TARGET_MODEL_PATH=$AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only \
        python scripts/probe_gemma4_target.py
"""

import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import AutoConfig


def _resolve_target_path() -> str:
    p = os.environ.get("TARGET_MODEL_PATH")
    if p:
        return p
    ukw = os.environ.get("AZURE_ML_INPUT_UKWDATA", "")
    if ukw:
        return os.path.join(ukw, "maiprofile/models/text_only")
    return "google/gemma-4-26B-A4B-it-text-only"


def _derive_layer_ids(num_layers: int, k: int) -> list[int]:
    last = num_layers - 2
    if k == 1:
        return [last]
    start = max(1, num_layers // (2 * k))
    step = max(1, (last - start) // (k - 1))
    ids = [start + i * step for i in range(k - 1)] + [last]
    seen, out = set(), []
    for v in ids:
        v = min(max(0, v), num_layers - 1)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def main() -> None:
    path = _resolve_target_path()
    print(f"Target path: {path}\n")
    cfg = AutoConfig.from_pretrained(path)
    top_type = getattr(cfg, "model_type", None)
    text = getattr(cfg, "text_config", cfg)
    text_type = getattr(text, "model_type", None)

    fields = [
        "num_hidden_layers", "hidden_size", "intermediate_size",
        "num_attention_heads", "num_key_value_heads",
        "num_global_key_value_heads", "global_head_dim", "head_dim",
        "vocab_size", "max_position_embeddings",
        "enable_moe_block", "num_experts", "top_k_experts",
        "moe_intermediate_size", "num_kv_shared_layers",
        "hidden_size_per_layer_input", "use_double_wide_mlp",
        "attention_k_eq_v", "rms_norm_eps", "final_logit_softcapping",
    ]
    print(f"top-level model_type : {top_type}")
    print(f"text   model_type    : {text_type}")
    print("-" * 48)
    summary = {}
    for f in fields:
        v = getattr(text, f, "<MISSING>")
        summary[f] = v
        print(f"{f:28s}: {v}")

    n_layers = getattr(text, "num_hidden_layers", None)
    print("-" * 48)
    if isinstance(n_layers, int):
        for k in (4, 5):
            print(f"auto target_layer_ids (num_draft_layers={k}): "
                  f"{_derive_layer_ids(n_layers, k)}")

    print("\nMoE?  ->", "YES (MoE target)" if getattr(text, "enable_moe_block", False)
          else "no (dense target)")
    print("\nJSON:")
    print(json.dumps({k: (v if not hasattr(v, "__dict__") else str(v))
                      for k, v in summary.items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
