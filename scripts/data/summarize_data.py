from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from collections.abc import Sequence
from typing import Any


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def conversation_stats(record: dict[str, Any]) -> tuple[int, int, int, int, int]:
    conversations = record.get("conversations") or []
    if not isinstance(conversations, list):
        return 0, 0, 0, 0, 0
    messages = 0
    user_messages = 0
    assistant_messages = 0
    chars = 0
    words = 0
    for msg in conversations:
        if not isinstance(msg, dict):
            continue
        messages += 1
        role = msg.get("role") or msg.get("from")
        if role in {"user", "human"}:
            user_messages += 1
        if role in {"assistant", "gpt", "chatgpt", "bing", "bard"}:
            assistant_messages += 1
        text = text_from_content(msg.get("content") if "content" in msg else msg.get("value"))
        chars += len(text)
        words += len(text.split())
    return messages, user_messages, assistant_messages, chars, words


def summarize_jsonl(path: Path, max_rows: int | None = None) -> dict[str, Any]:
    total_rows = 0
    parse_errors = 0
    statuses: dict[str, int] = {}
    msg_counts: list[int] = []
    user_counts: list[int] = []
    assistant_counts: list[int] = []
    char_counts: list[int] = []
    word_counts: list[int] = []
    file_size = path.stat().st_size

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if max_rows is not None and total_rows >= max_rows:
                break
            total_rows += 1
            try:
                record = json.loads(line)
            except Exception:
                parse_errors += 1
                continue
            status = record.get("status")
            if status is not None:
                statuses[str(status)] = statuses.get(str(status), 0) + 1
            messages, users, assistants, chars, words = conversation_stats(record)
            msg_counts.append(messages)
            user_counts.append(users)
            assistant_counts.append(assistants)
            char_counts.append(chars)
            word_counts.append(words)

    effective_rows = len(msg_counts)
    return {
        "path": str(path),
        "file_size": file_size,
        "rows": total_rows if max_rows is None else min(total_rows, max_rows),
        "parsed_rows": effective_rows,
        "parse_errors": parse_errors,
        "statuses": statuses,
        "messages": describe_numeric(msg_counts),
        "user_messages": describe_numeric(user_counts),
        "assistant_messages": describe_numeric(assistant_counts),
        "chars_per_record": describe_numeric(char_counts),
        "words_per_record": describe_numeric(word_counts),
        "rough_tokens_per_record": describe_numeric([c / 4.0 for c in char_counts]),
    }


def describe_numeric(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"min": 0, "p50": 0, "mean": 0, "p95": 0, "max": 0, "sum": 0}
    return {
        "min": min(values),
        "p50": percentile(values, 0.50),
        "mean": statistics.fmean(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "sum": sum(values),
    }


def summarize_cache(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json under cache dir: {cache_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_files = []
    shard_total_bytes = 0
    for shard in manifest.get("shards", []):
        name = shard.get("file_name")
        if not name:
            continue
        shard_path = cache_dir / name
        size = shard_path.stat().st_size if shard_path.exists() else 0
        shard_total_bytes += size
        shard_files.append({"file_name": name, "size": size})
    idx_path = cache_dir / "samples.idx"
    idx_size = idx_path.stat().st_size if idx_path.exists() else 0
    return {
        "path": str(cache_dir),
        "num_samples": manifest.get("num_samples"),
        "num_shards": manifest.get("num_shards"),
        "target_layer_ids": manifest.get("target_layer_ids"),
        "hidden_size": manifest.get("hidden_size"),
        "target_model_name_or_path": manifest.get("target_model_name_or_path"),
        "source_jsonl_paths": manifest.get("source_jsonl_paths"),
        "max_length": manifest.get("max_length"),
        "min_loss_tokens": manifest.get("min_loss_tokens"),
        "index_size": idx_size,
        "shard_total_size": shard_total_bytes,
        "total_size": idx_size + shard_total_bytes,
        "shards": shard_files,
    }


def print_numeric(label: str, stats: dict[str, float]) -> None:
    print(
        f"  {label}: min={stats['min']:.1f} p50={stats['p50']:.1f} "
        f"mean={stats['mean']:.1f} p95={stats['p95']:.1f} max={stats['max']:.1f} sum={stats['sum']:.1f}"
    )


def print_jsonl_summary(summary: dict[str, Any]) -> None:
    print(f"\nJSONL: {summary['path']}")
    print(f"  size: {human_bytes(summary['file_size'])}")
    print(f"  rows: {summary['rows']} parsed={summary['parsed_rows']} parse_errors={summary['parse_errors']}")
    if summary["statuses"]:
        print(f"  statuses: {summary['statuses']}")
    print_numeric("messages/record", summary["messages"])
    print_numeric("user_messages/record", summary["user_messages"])
    print_numeric("assistant_messages/record", summary["assistant_messages"])
    print_numeric("chars/record", summary["chars_per_record"])
    print_numeric("words/record", summary["words_per_record"])
    print_numeric("rough_tokens/record(chars/4)", summary["rough_tokens_per_record"])


def print_cache_summary(summary: dict[str, Any]) -> None:
    print(f"\nTARGET CACHE: {summary['path']}")
    print(f"  num_samples: {summary['num_samples']}")
    print(f"  num_shards: {summary['num_shards']}")
    print(f"  target_model: {summary['target_model_name_or_path']}")
    print(f"  source_jsonl_paths: {summary['source_jsonl_paths']}")
    print(f"  target_layer_ids: {summary['target_layer_ids']}")
    print(f"  hidden_size: {summary['hidden_size']}")
    print(f"  max_length: {summary['max_length']} min_loss_tokens: {summary['min_loss_tokens']}")
    print(f"  index_size: {human_bytes(summary['index_size'])}")
    print(f"  shard_total_size: {human_bytes(summary['shard_total_size'])}")
    print(f"  total_size: {human_bytes(summary['total_size'])}")
    for shard in summary["shards"][:10]:
        print(f"    {shard['file_name']}: {human_bytes(shard['size'])}")
    if len(summary["shards"]) > 10:
        print(f"    ... {len(summary['shards']) - 10} more shards")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DeepSpec JSONL data and target caches.")
    parser.add_argument("--jsonl", action="append", default=[], help="JSONL file to summarize. Repeatable.")
    parser.add_argument("--cache-dir", action="append", default=[], help="Target cache directory to summarize. Repeatable.")
    parser.add_argument("--max-rows", type=int, default=None, help="Read at most N rows per JSONL for quick estimates.")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {"jsonl": [], "cache_dir": []}
    for item in args.jsonl:
        payload["jsonl"].append(summarize_jsonl(Path(item), max_rows=args.max_rows))
    for item in args.cache_dir:
        payload["cache_dir"].append(summarize_cache(Path(item)))
    if args.format == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    for summary in payload["jsonl"]:
        print_jsonl_summary(summary)
    for summary in payload["cache_dir"]:
        print_cache_summary(summary)


if __name__ == "__main__":
    main()
