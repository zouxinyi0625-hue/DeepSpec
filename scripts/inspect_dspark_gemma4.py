from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.gemma4.config import build_draft_config
from deepspec.utils import CustomJSONEncoder, load_config, parse_opts_to_config


def num_params(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def num_trainable_params(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def module_summary(name: str, module: torch.nn.Module | None) -> dict[str, Any] | None:
    if module is None:
        return None
    return {
        "name": name,
        "class": module.__class__.__name__,
        "params": num_params(module),
        "trainable_params": num_trainable_params(module),
    }


def parameter_shapes(model: torch.nn.Module, patterns: list[str]) -> list[dict[str, Any]]:
    rows = []
    for name, parameter in model.named_parameters():
        if patterns and not any(pattern in name for pattern in patterns):
            continue
        rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "requires_grad": bool(parameter.requires_grad),
                "dtype": str(parameter.dtype),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect Gemma4 DSpark target/draft config, module sizes, and important parameter shapes."
    )
    parser.add_argument(
        "--config",
        default="config/dspark/dspark_gemma4_12b_small.py",
        help="DeepSpec config path.",
    )
    parser.add_argument(
        "--opts",
        action="append",
        default=[],
        help="Config override, same dotted syntax as train.py. Repeatable.",
    )
    parser.add_argument(
        "--shape-pattern",
        action="append",
        default=["fc", "markov", "confidence", "embed_tokens", "lm_head"],
        help="Only print parameter shapes whose names contain this substring. Repeatable. Use empty string to print all.",
    )
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    cfg = parse_opts_to_config(args.opts, load_config(str(config_path)))

    target_config = AutoConfig.from_pretrained(cfg.model.target_model_name_or_path)
    draft_config = build_draft_config(target_config=target_config, model_args=cfg.model)
    draft_model = Gemma4DSparkModel(draft_config)

    target_text_config = getattr(target_config, "text_config", target_config)
    payload: dict[str, Any] = {
        "config": str(config_path),
        "target": {
            "model_name_or_path": str(cfg.model.target_model_name_or_path),
            "model_type": getattr(target_config, "model_type", None),
            "text_model_type": getattr(target_text_config, "model_type", None),
            "hidden_size": getattr(target_text_config, "hidden_size", None),
            "num_hidden_layers": getattr(target_text_config, "num_hidden_layers", None),
            "vocab_size": getattr(target_text_config, "vocab_size", None),
            "num_attention_heads": getattr(target_text_config, "num_attention_heads", None),
            "num_key_value_heads": getattr(target_text_config, "num_key_value_heads", None),
            "num_global_key_value_heads": getattr(target_text_config, "num_global_key_value_heads", None),
            "global_head_dim": getattr(target_text_config, "global_head_dim", None),
            "enable_moe_block": getattr(target_text_config, "enable_moe_block", None),
            "hidden_size_per_layer_input": getattr(target_text_config, "hidden_size_per_layer_input", None),
        },
        "draft": {
            "architectures": getattr(draft_config, "architectures", None),
            "hidden_size": draft_config.hidden_size,
            "num_hidden_layers": draft_config.num_hidden_layers,
            "vocab_size": draft_config.vocab_size,
            "block_size": draft_config.block_size,
            "num_anchors": draft_config.num_anchors,
            "target_layer_ids": list(draft_config.target_layer_ids),
            "num_target_layers": draft_config.num_target_layers,
            "markov_rank": draft_config.markov_rank,
            "markov_head_type": getattr(draft_config, "markov_head_type", None),
            "enable_confidence_head": draft_config.enable_confidence_head,
            "confidence_head_with_markov": getattr(draft_config, "confidence_head_with_markov", None),
            "mask_token_id": draft_config.mask_token_id,
            "total_params": num_params(draft_model),
            "trainable_params": num_trainable_params(draft_model),
        },
        "modules": {
            "embed_tokens": module_summary("embed_tokens", draft_model.embed_tokens),
            "layers": module_summary("layers", draft_model.layers),
            "fc": module_summary("fc", draft_model.fc),
            "hidden_norm": module_summary("hidden_norm", draft_model.hidden_norm),
            "lm_head": module_summary("lm_head", draft_model.lm_head),
            "markov_head": module_summary("markov_head", draft_model.markov_head),
            "confidence_head": module_summary("confidence_head", draft_model.confidence_head),
        },
        "parameter_shapes": parameter_shapes(
            draft_model,
            [] if args.shape_pattern == [""] else list(args.shape_pattern),
        ),
    }

    hidden_size = int(draft_config.hidden_size)
    num_target_layers = len(draft_config.target_layer_ids)
    block_size = int(draft_config.block_size)
    num_anchors = int(draft_config.num_anchors)
    payload["derived_shapes"] = {
        "target_hidden_states_last_dim": num_target_layers * hidden_size,
        "projected_target_hidden_states_last_dim": hidden_size,
        "training_output_hidden_shape_per_batch": [
            "batch_size",
            num_anchors * block_size,
            hidden_size,
        ],
        "training_output_hidden_4d_shape_per_batch": [
            "batch_size",
            num_anchors,
            block_size,
            hidden_size,
        ],
        "training_draft_logits_shape_per_batch": [
            "batch_size",
            num_anchors,
            block_size,
            int(draft_config.vocab_size),
        ],
        "confidence_feature_dim": hidden_size
        + (int(draft_config.markov_rank) if getattr(draft_config, "confidence_head_with_markov", False) else 0),
    }

    if args.format == "json":
        print(json.dumps(payload, indent=2, cls=CustomJSONEncoder))
        return

    print("=== Target config ===")
    for key, value in payload["target"].items():
        print(f"{key}: {value}")
    print("\n=== Draft config ===")
    for key, value in payload["draft"].items():
        print(f"{key}: {value}")
    print("\n=== Derived shapes ===")
    for key, value in payload["derived_shapes"].items():
        print(f"{key}: {value}")
    print("\n=== Module parameter counts ===")
    for key, value in payload["modules"].items():
        print(f"{key}: {value}")
    print("\n=== Selected parameter shapes ===")
    for row in payload["parameter_shapes"]:
        print(
            f"{row['name']}: shape={row['shape']} numel={row['numel']} "
            f"requires_grad={row['requires_grad']} dtype={row['dtype']}"
        )


if __name__ == "__main__":
    main()
