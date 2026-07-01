from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from deepspec.data.target_cache_dataset import CacheCollator, CacheDataset
from deepspec.modeling.dspark.common import (
    build_eval_mask,
    create_dspark_attention_mask,
    create_noise_embed,
    create_position_ids,
    sample_anchor_positions,
)
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.gemma4.config import build_draft_config
from deepspec.utils import load_config, parse_opts_to_config


def shape_of(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
    return repr(value)


def print_tensor(name: str, value: torch.Tensor) -> None:
    print(f"{name}: {shape_of(value)}")
    if value.numel() <= 20:
        print(f"  values={value.detach().cpu().tolist()}")


def load_batch(cache_dir: str, indices: list[int]) -> tuple[CacheDataset, dict[str, torch.Tensor]]:
    dataset = CacheDataset(cache_dir)
    items = [dataset[index] for index in indices]
    batch = CacheCollator()(items)
    return dataset, batch


def trace_sampling(batch: dict[str, torch.Tensor], draft_config) -> dict[str, torch.Tensor]:
    input_ids = batch["input_ids"]
    loss_mask = batch["loss_mask"]
    bsz, seq_len = input_ids.shape
    device = input_ids.device
    block_size = int(draft_config.block_size)
    num_anchors = int(draft_config.num_anchors)

    print("\n=== Train batch tensors ===")
    for key, value in batch.items():
        print_tensor(key, value)
    print(f"batch_size={bsz} padded_seq_len={seq_len}")
    print(f"loss_mask.sum per sample={loss_mask.sum(dim=1).tolist()}")

    anchor_positions, block_keep_mask = sample_anchor_positions(
        seq_len=seq_len,
        loss_mask=loss_mask,
        num_anchors=num_anchors,
        device=device,
    )
    print("\n=== Anchor sampling ===")
    print_tensor("anchor_positions", anchor_positions)
    print_tensor("block_keep_mask", block_keep_mask)
    print(f"valid blocks per sample={block_keep_mask.sum(dim=1).tolist()}")
    if anchor_positions.numel() > 0:
        preview = anchor_positions[0, : min(16, anchor_positions.shape[1])].tolist()
        print(f"first sample anchor preview={preview}")

    noise_embedding = create_noise_embed(
        # The caller may replace this with the real embedding for run-forward.
        torch.nn.Embedding(int(draft_config.vocab_size), int(draft_config.hidden_size), device=device),
        input_ids,
        anchor_positions,
        block_keep_mask,
        mask_token_id=int(draft_config.mask_token_id),
        block_size=block_size,
    )
    print("\n=== Draft block construction (dummy embedding for shape) ===")
    print_tensor("noise_embedding", noise_embedding)
    print(
        "draft block tokens per sample = num_anchors * block_size = "
        f"{num_anchors} * {block_size} = {num_anchors * block_size}"
    )

    context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
    draft_position_ids = create_position_ids(anchor_positions, block_size)
    full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
    print_tensor("context_position_ids", context_position_ids)
    print_tensor("draft_position_ids", draft_position_ids)
    print_tensor("full_position_ids", full_position_ids)

    label_offsets = torch.arange(1, block_size + 1, device=device).view(1, 1, -1)
    label_indices = anchor_positions.unsqueeze(-1) + label_offsets
    safe_label_indices = label_indices.clamp(max=seq_len - 1)
    safe_label_indices = torch.where(
        block_keep_mask.unsqueeze(-1),
        safe_label_indices,
        torch.zeros_like(safe_label_indices),
    )
    target_ids = torch.gather(
        input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
        2,
        safe_label_indices,
    )
    eval_mask = build_eval_mask(
        seq_len=seq_len,
        loss_mask=loss_mask,
        label_indices=label_indices,
        safe_label_indices=safe_label_indices,
        block_keep_mask=block_keep_mask,
    )
    anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)
    prev_token_ids = torch.cat([anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1)
    print("\n=== Labels / teacher forcing ===")
    print_tensor("label_indices", label_indices)
    print_tensor("target_ids", target_ids)
    print_tensor("eval_mask", eval_mask)
    print_tensor("prev_token_ids", prev_token_ids)
    print(f"eval_mask supervised draft positions={int(eval_mask.sum().item())}")
    if target_ids.numel() > 0:
        print(f"first block target_ids={target_ids[0, 0].tolist()}")
        print(f"first block prev_token_ids={prev_token_ids[0, 0].tolist()}")

    # Constructing the flex_attention block mask can be heavy/noisy, but useful to
    # verify the query/key lengths and masking semantics.
    print("\n=== DSpark attention mask dimensions ===")
    print(f"Q_LEN = num_anchors * block_size = {num_anchors * block_size}")
    print(f"KV_LEN = seq_len + num_anchors * block_size = {seq_len + num_anchors * block_size}")
    try:
        mask = create_dspark_attention_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len,
            block_size=block_size,
            device=device,
        )
        print(f"dspark_attn_mask class={mask.__class__.__name__}")
    except Exception as exc:
        print(f"dspark_attn_mask construction skipped/failed: {exc!r}")

    return {
        "anchor_positions": anchor_positions,
        "block_keep_mask": block_keep_mask,
        "target_ids": target_ids,
        "eval_mask": eval_mask,
        "prev_token_ids": prev_token_ids,
    }


def run_forward_once(args, cfg, draft_config, batch: dict[str, torch.Tensor]) -> None:
    print("\n=== Optional real DSpark forward ===")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    model = Gemma4DSparkModel(draft_config).to(device=device, dtype=dtype).eval()

    if args.init_target_embeddings:
        print("Loading target model on CPU to initialize/freeze embedding + lm_head...")
        target_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.target_model_name_or_path,
            dtype=dtype,
        ).to(device="cpu").eval()
        model.initialize_embeddings_and_head(
            embed_tokens=target_model.get_input_embeddings(),
            lm_head=target_model.get_output_embeddings(),
            freeze=True,
        )
        del target_model

    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        outputs = model(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"].to(dtype=dtype),
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=batch["target_last_hidden_states"].to(dtype=dtype),
        )
    print_tensor("outputs.draft_logits", outputs.draft_logits)
    print_tensor("outputs.target_ids", outputs.target_ids)
    print_tensor("outputs.eval_mask", outputs.eval_mask)
    print_tensor("outputs.block_keep_mask", outputs.block_keep_mask)
    if outputs.confidence_pred is not None:
        print_tensor("outputs.confidence_pred", outputs.confidence_pred)
    if outputs.aligned_target_logits is not None:
        print_tensor("outputs.aligned_target_logits", outputs.aligned_target_logits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace DSpark Gemma4 training tensors from cache sample to draft forward.")
    parser.add_argument("--config", default="config/dspark/dspark_gemma4_12b_small.py")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--indices", default="0", help="Comma-separated cache indices, e.g. 0,1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--run-forward", action="store_true", help="Instantiate Gemma4DSparkModel and run one forward pass.")
    parser.add_argument("--init-target-embeddings", action="store_true", help="Copy/freeze target embedding/lm_head before run-forward.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = parse_opts_to_config([], load_config(args.config))
    target_config = AutoConfig.from_pretrained(cfg.model.target_model_name_or_path)
    draft_config = build_draft_config(target_config=target_config, model_args=cfg.model)
    indices = [int(item) for item in args.indices.split(",") if item.strip()]

    dataset, batch = load_batch(args.cache_dir, indices)
    print("=== Cache metadata ===")
    print(f"cache_dir={Path(args.cache_dir)}")
    print(f"dataset_len={len(dataset)}")
    print(f"target_layer_ids={dataset.target_layer_ids}")
    print(f"hidden_size={dataset.hidden_size}")
    print(f"num_target_layers={dataset.num_target_layers}")
    print(f"indices={indices}")

    trace_sampling(batch, draft_config)
    if args.run_forward:
        run_forward_once(args, cfg, draft_config, batch)
    dataset.close()


if __name__ == "__main__":
    main()
