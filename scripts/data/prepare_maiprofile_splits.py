from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

DEFAULT_LAYERS = [
    "layer1_actual",
    "layer1_intent",
    "layer2_temporal",
    "layer3_seasonality",
    "layer4_commercial_preference",
]


def get_msndni_mount() -> Path:
    mount = os.environ.get("AZURE_ML_INPUT_msndni")
    if not mount:
        raise SystemExit(
            "AZURE_ML_INPUT_msndni is not set. Export it or pass --input-dir/--output-dir explicitly."
        )
    return Path(mount)


def default_input_dir() -> Path:
    return get_msndni_mount() / "shares/users/zxy/maiprofile/raw_data/20260615"


def default_output_dir(date: str) -> Path:
    return get_msndni_mount() / "shares/users/zxy/maiprofile/prepared_prompts" / date / "short_layers"


def normalize_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("prompt_messages") or record.get("conversations") or []
    if not isinstance(messages, list):
        return []
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str):
            continue
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        if role not in {"system", "user", "assistant"}:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def prompt_text_for_eval(messages: list[dict[str, str]]) -> str:
    # Existing evaluator supports a single user turn. Preserve the full MAI
    # Profile input by folding system text into the first user prompt.
    system_parts = []
    user_parts = []
    for message in messages:
        if message["role"] == "system":
            system_parts.append(message["content"])
        elif message["role"] == "user":
            user_parts.append(message["content"])
    parts = []
    if system_parts:
        parts.append("\n\n".join(system_parts).strip())
    if user_parts:
        parts.append("\n\n".join(user_parts).strip())
    return "\n\n".join(part for part in parts if part)


def read_layer_records(input_dir: Path, layer: str):
    path = input_dir / f"{layer}.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = normalize_messages(record)
            if not messages:
                continue
            if not any(message["role"] == "user" for message in messages):
                continue
            yield {
                "id": f"{layer}:{record.get('prompt_hash') or line_number}",
                "source_layer": layer,
                "source_file": path.name,
                "source_line": line_number,
                "user_id": record.get("user_id"),
                "prompt_hash": record.get("prompt_hash"),
                "conversations": messages,
            }


def parse_layers(raw: str) -> list[str]:
    if raw.strip().lower() in {"short", "short_layers"}:
        return list(DEFAULT_LAYERS)
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MAI Profile prompt train/eval split for DSpark generation and eval.")
    parser.add_argument("--input-dir", default=None, help="Raw MAI Profile JSONL dir. Defaults to $AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/raw_data/20260615.")
    parser.add_argument("--output-dir", default=None, help="Output dir. Defaults to $AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/prepared_prompts/20260615/short_layers.")
    parser.add_argument("--layers", default="short", help="Comma-separated layer names or 'short'.")
    parser.add_argument("--eval-size", type=int, default=200, help="Eval samples per layer before train split.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap on total train samples after eval split.")
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument("--date", default="20260615")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else default_input_dir().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(args.date).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = output_dir / "eval_datasets"
    eval_dir.mkdir(parents=True, exist_ok=True)

    layers = parse_layers(args.layers)
    rng = random.Random(args.seed)
    train_records = []
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "layers": layers,
        "eval_size_per_layer": args.eval_size,
        "seed": args.seed,
        "by_layer": {},
    }

    layer_iter = tqdm(layers, desc="layers", unit="layer") if tqdm is not None else layers
    for layer in layer_iter:
        records = list(read_layer_records(input_dir, layer) or [])
        rng.shuffle(records)
        eval_records = records[: args.eval_size]
        train_layer_records = records[args.eval_size :]
        train_records.extend(train_layer_records)

        train_layer_path = output_dir / f"train_{layer}.jsonl"
        eval_layer_path = eval_dir / f"maiprofile_{layer}.jsonl"
        with train_layer_path.open("w", encoding="utf-8") as handle:
            for record in train_layer_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        with eval_layer_path.open("w", encoding="utf-8") as handle:
            for record in eval_records:
                handle.write(json.dumps({
                    "id": record["id"],
                    "source_layer": record["source_layer"],
                    "user_id": record.get("user_id"),
                    "prompt_hash": record.get("prompt_hash"),
                    "turns": [prompt_text_for_eval(record["conversations"])],
                }, ensure_ascii=False) + "\n")
        summary["by_layer"][layer] = {
            "raw_records": len(records),
            "train_records": len(train_layer_records),
            "eval_records": len(eval_records),
            "train_file": str(train_layer_path),
            "eval_file": str(eval_layer_path),
        }
        print(
            f"[{layer}] raw={len(records)} train={len(train_layer_records)} eval={len(eval_records)}",
            flush=True,
        )

    rng.shuffle(train_records)
    if args.max_train_samples is not None:
        train_records = train_records[: args.max_train_samples]
    train_all_path = output_dir / "train_maiprofile_short_layers.jsonl"
    with train_all_path.open("w", encoding="utf-8") as handle:
        for record in train_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary["train_all_file"] = str(train_all_path)
    summary["train_all_records"] = len(train_records)
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "output_dir": str(output_dir),
        "train_all_file": str(train_all_path),
        "train_all_records": len(train_records),
        "eval_dir": str(eval_dir),
        "summary": str(summary_path),
    }, indent=2))


if __name__ == "__main__":
    main()
