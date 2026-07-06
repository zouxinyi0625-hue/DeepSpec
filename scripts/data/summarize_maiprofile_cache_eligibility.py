from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from deepspec.data.parser import GeneralParser, TEMPLATE_REGISTRY, render_chat_messages
from deepspec.utils import load_config, parse_opts_to_config

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def describe(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None, "mean": None, "sum": 0}
    values = sorted(values)
    def pct(q: float) -> int:
        return values[int(round((len(values) - 1) * q))]
    return {
        "min": values[0],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": values[-1],
        "mean": round(sum(values) / len(values), 2),
        "sum": sum(values),
    }


def get_msndni_mount() -> Path:
    mount = os.environ.get("AZURE_ML_INPUT_msndni")
    if not mount:
        raise SystemExit("AZURE_ML_INPUT_msndni is not set. Pass paths explicitly or export it.")
    return Path(mount)


def default_base_dir() -> Path:
    return get_msndni_mount() / "shares/users/zxy/maiprofile"


def render_untruncated_text(parser: GeneralParser, conversations: list[dict[str, Any]], tokenizer) -> str:
    messages = []
    conversation = list(conversations)
    if conversation and conversation[0].get("role") == "system":
        messages.append({"role": "system", "content": conversation[0].get("content", "")})
        conversation = conversation[1:]
    elif parser.system_prompt:
        messages.append({"role": "system", "content": parser.system_prompt})
    for message in conversation:
        messages.append(message)
    render_messages = parser._prepare_render_messages(messages)
    return render_chat_messages(tokenizer, render_messages, add_generation_prompt=False)


def analyze(path: Path, *, tokenizer, chat_template: str, max_length: int, min_loss_tokens: int, progress_every: int) -> dict[str, Any]:
    template = TEMPLATE_REGISTRY.get(chat_template)
    parser = GeneralParser(tokenizer=tokenizer, chat_template=template)
    rows = 0
    parse_errors = 0
    status_counter = Counter()
    by_layer = defaultdict(lambda: {
        "rows": 0,
        "success": 0,
        "errors": 0,
        "valid": 0,
        "invalid": 0,
        "over_max_length": 0,
        "truncated_to_max_length": 0,
        "untruncated_seq_len": [],
        "truncated_seq_len": [],
        "loss_tokens_after_truncation": [],
        "assistant_tokens_untruncated": [],
    })
    error_counter = Counter()

    total_lines = sum(1 for line in path.open("r", encoding="utf-8") if line.strip())
    print(f"Analyzing cache eligibility: {path} ({total_lines} rows)", file=sys.stderr, flush=True)
    iterator = path.open("r", encoding="utf-8")
    if tqdm is not None:
        iterator = tqdm(iterator, total=total_lines, desc="eligibility", unit="rows", dynamic_ncols=True)
    try:
        for line in iterator:
            if not line.strip():
                continue
            rows += 1
            try:
                record = json.loads(line)
            except Exception:
                parse_errors += 1
                continue
            layer = record.get("source_layer") or "<missing>"
            layer_stats = by_layer[layer]
            layer_stats["rows"] += 1
            status = record.get("status") or "<missing>"
            status_counter[status] += 1
            if status != "success":
                layer_stats["errors"] += 1
                error_counter[str(record.get("error") or "<no error>")[:300]] += 1
                continue
            layer_stats["success"] += 1
            conversations = record.get("conversations") or []
            if not conversations:
                layer_stats["invalid"] += 1
                continue
            try:
                untruncated_text = render_untruncated_text(parser, conversations, tokenizer)
                untruncated_ids = tokenizer.encode(untruncated_text, add_special_tokens=False)
                untruncated_len = len(untruncated_ids)
                processed = parser.parse(conversations, max_length=max_length)
            except Exception as exc:
                layer_stats["invalid"] += 1
                error_counter[f"parse:{type(exc).__name__}:{str(exc)[:200]}"] += 1
                continue
            truncated_len = int(processed["input_ids"].numel())
            loss_tokens = int(processed["loss_mask"].sum().item())
            layer_stats["untruncated_seq_len"].append(untruncated_len)
            layer_stats["truncated_seq_len"].append(truncated_len)
            layer_stats["loss_tokens_after_truncation"].append(loss_tokens)
            if untruncated_len > max_length:
                layer_stats["over_max_length"] += 1
            if truncated_len >= max_length and untruncated_len > max_length:
                layer_stats["truncated_to_max_length"] += 1
            assistant_text = "\n".join(
                message.get("content") or ""
                for message in conversations
                if isinstance(message, dict) and message.get("role") == "assistant"
            )
            layer_stats["assistant_tokens_untruncated"].append(len(tokenizer.encode(assistant_text, add_special_tokens=False)))
            if loss_tokens >= min_loss_tokens:
                layer_stats["valid"] += 1
            else:
                layer_stats["invalid"] += 1
            if tqdm is None and progress_every > 0 and rows % progress_every == 0:
                print(f"[eligibility] rows={rows}/{total_lines} layer={layer}", file=sys.stderr, flush=True)
    finally:
        close = getattr(iterator, "close", None)
        if close:
            close()

    out_by_layer = {}
    for layer, stats in sorted(by_layer.items()):
        success = int(stats["success"])
        valid = int(stats["valid"])
        rows_layer = int(stats["rows"])
        over = int(stats["over_max_length"])
        out_by_layer[layer] = {
            "rows": rows_layer,
            "success": success,
            "errors": int(stats["errors"]),
            "valid": valid,
            "invalid": int(stats["invalid"]),
            "valid_ratio_of_success": round(valid / success, 6) if success else None,
            "valid_ratio_of_rows": round(valid / rows_layer, 6) if rows_layer else None,
            "over_max_length": over,
            "over_max_length_ratio_of_success": round(over / success, 6) if success else None,
            "truncated_to_max_length": int(stats["truncated_to_max_length"]),
            "untruncated_seq_len": describe(stats["untruncated_seq_len"]),
            "truncated_seq_len": describe(stats["truncated_seq_len"]),
            "loss_tokens_after_truncation": describe(stats["loss_tokens_after_truncation"]),
            "assistant_tokens_untruncated": describe(stats["assistant_tokens_untruncated"]),
        }
    return {
        "path": str(path),
        "rows": rows,
        "parse_errors": parse_errors,
        "status": dict(status_counter),
        "max_length": max_length,
        "min_loss_tokens": min_loss_tokens,
        "errors_top10": error_counter.most_common(10),
        "by_layer": out_by_layer,
    }


def write_md(summary: dict[str, Any], path: Path) -> None:
    lines = ["# MAI Profile Cache Eligibility by Layer", ""]
    lines.append(f"- Regenerated JSONL: `{summary['path']}`")
    lines.append(f"- Rows: **{summary['rows']}**")
    lines.append(f"- Status: `{summary['status']}`")
    lines.append(f"- Cache max_length: **{summary['max_length']}**")
    lines.append(f"- Min loss tokens: **{summary['min_loss_tokens']}**")
    lines.append("")
    lines.append("| layer | rows | success | valid | valid % | >max_length | >max % | untrunc p50 | untrunc p95 | trunc p50 | loss p50 | loss p95 | assistant p50 | assistant p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for layer, item in summary["by_layer"].items():
        lines.append(
            f"| {layer} | {item['rows']} | {item['success']} | {item['valid']} | "
            f"{(item['valid_ratio_of_success'] or 0):.1%} | {item['over_max_length']} | "
            f"{(item['over_max_length_ratio_of_success'] or 0):.1%} | "
            f"{item['untruncated_seq_len']['p50']} | {item['untruncated_seq_len']['p95']} | "
            f"{item['truncated_seq_len']['p50']} | {item['loss_tokens_after_truncation']['p50']} | "
            f"{item['loss_tokens_after_truncation']['p95']} | {item['assistant_tokens_untruncated']['p50']} | "
            f"{item['assistant_tokens_untruncated']['p95']} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MAI Profile target-cache eligibility by source layer.")
    parser.add_argument("--config", default="config/dspark/dspark_gemma4_12b_small.py")
    parser.add_argument("--opts", action="append", default=[])
    parser.add_argument("--regen-path", default=None)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-loss-tokens", type=int, default=14)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--date", default="20260615")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = parse_opts_to_config(args.opts, load_config(args.config))
    base = get_msndni_mount() / "shares/users/zxy/maiprofile"
    regen_path = Path(args.regen_path).expanduser().resolve() if args.regen_path else base / "regenerated" / args.date / "maiprofile_short_layers_regen.jsonl"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else base / "reports" / args.date / "cache_eligibility"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading tokenizer from {args.tokenizer!r} local_files_only={args.local_files_only}", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=args.local_files_only)
    summary = analyze(
        regen_path,
        tokenizer=tokenizer,
        chat_template=str(cfg.data.chat_template),
        max_length=int(cfg.data.max_length),
        min_loss_tokens=int(args.min_loss_tokens),
        progress_every=int(args.progress_every),
    )
    json_path = output_dir / "cache_eligibility_by_layer.json"
    md_path = output_dir / "cache_eligibility_by_layer.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(summary, md_path)
    print(json.dumps({"ok": True, "json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
