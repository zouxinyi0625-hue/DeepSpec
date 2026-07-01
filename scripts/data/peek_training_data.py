from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from deepspec.data.parser import preprocess_record
from deepspec.data.target_cache_dataset import CacheDataset
from deepspec.utils import load_config, parse_opts_to_config


def load_jsonl_record(path: Path, index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise IndexError(f"JSONL index {index} out of range for {path}")


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... <truncated {len(text) - max_chars} chars>"


def describe_conversation(record: dict[str, Any], max_chars: int) -> None:
    print("=== JSONL record ===")
    print(f"id: {record.get('id')}")
    print(f"status: {record.get('status')}")
    conversations = record.get("conversations") or []
    print(f"num_messages: {len(conversations)}")
    for i, message in enumerate(conversations):
        role = message.get("role") or message.get("from")
        content = message.get("content") if "content" in message else message.get("value")
        if not isinstance(content, str):
            content = repr(content)
        print(f"\n--- message[{i}] role={role} chars={len(content)} words={len(content.split())} ---")
        print(truncate_text(content, max_chars))


def describe_processed(record: dict[str, Any], *, tokenizer, chat_template: str, max_length: int, window: int) -> None:
    print("\n=== Tokenized training view ===")
    processed = preprocess_record(
        record=record,
        tokenizer=tokenizer,
        chat_template=chat_template,
        max_length=max_length,
    )
    input_ids = processed["input_ids"]
    attention_mask = processed["attention_mask"]
    loss_mask = processed["loss_mask"]
    print(f"input_ids.shape: {tuple(input_ids.shape)}")
    print(f"attention_mask.sum: {int(attention_mask.sum().item())}")
    print(f"loss_mask.sum: {int(loss_mask.sum().item())}")
    loss_positions = torch.nonzero(loss_mask, as_tuple=False).view(-1).tolist()
    if not loss_positions:
        print("No supervised/loss tokens for this record.")
        return
    print(f"first_loss_pos: {loss_positions[0]} last_loss_pos: {loss_positions[-1]}")

    start = max(0, loss_positions[0] - window)
    end = min(len(input_ids), loss_positions[0] + window)
    print(f"\nDecoded token window around first supervised token [{start}:{end}]:")
    print(tokenizer.decode(input_ids[start:end].tolist(), skip_special_tokens=False))

    supervised_ids = input_ids[loss_mask.bool()]
    print("\nFirst supervised tokens decoded:")
    print(tokenizer.decode(supervised_ids[: min(128, supervised_ids.numel())].tolist(), skip_special_tokens=False))


def describe_cache(cache_dir: Path, index: int) -> None:
    print("\n=== Target cache sample ===")
    dataset = CacheDataset(str(cache_dir))
    print(f"cache_dir: {cache_dir}")
    print(f"num_samples: {len(dataset)}")
    print(f"target_layer_ids: {dataset.target_layer_ids}")
    print(f"hidden_size: {dataset.hidden_size}")
    print(f"num_target_layers: {dataset.num_target_layers}")
    item = dataset[index]
    for key, value in item.items():
        print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
    seq_len = item["input_ids"].shape[0]
    target_hidden_bytes = item["target_hidden_states"].numel() * 2
    target_last_bytes = item["target_last_hidden_states"].numel() * 2
    print(f"seq_len: {seq_len}")
    print(f"target_hidden_states bytes (bf16): {target_hidden_bytes:,}")
    print(f"target_last_hidden_states bytes (bf16): {target_last_bytes:,}")
    print(f"loss_mask.sum: {int(item['loss_mask'].sum().item())}")
    dataset.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Peek at DeepSpec regenerated JSONL and target-cache samples.")
    parser.add_argument("--jsonl", required=True, help="Regenerated training JSONL path.")
    parser.add_argument("--cache-dir", help="Optional target cache directory to inspect.")
    parser.add_argument("--index", type=int, default=0, help="Record/cache sample index to inspect.")
    parser.add_argument("--config", default="config/dspark/dspark_gemma4_12b_small.py")
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--token-window", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = parse_opts_to_config([], load_config(args.config))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.target_model_name_or_path)
    record = load_jsonl_record(Path(args.jsonl), args.index)
    describe_conversation(record, max_chars=args.max_chars)
    describe_processed(
        record,
        tokenizer=tokenizer,
        chat_template=str(cfg.data.chat_template),
        max_length=int(cfg.data.max_length),
        window=int(args.token_window),
    )
    if args.cache_dir:
        describe_cache(Path(args.cache_dir), args.index)


if __name__ == "__main__":
    main()
