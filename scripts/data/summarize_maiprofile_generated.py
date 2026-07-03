from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def get_msndni_mount() -> Path:
    mount = os.environ.get("AZURE_ML_INPUT_msndni")
    if not mount:
        raise SystemExit("AZURE_ML_INPUT_msndni is not set. Pass paths explicitly or export it.")
    return Path(mount)


def default_base_dir() -> Path:
    return get_msndni_mount() / "shares/users/zxy/maiprofile"


def describe(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None, "mean": None, "sum": 0}
    values = sorted(values)
    def pct(q: float) -> int:
        return values[int(round((len(values)-1)*q))]
    return {
        "min": values[0],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": values[-1],
        "mean": round(sum(values)/len(values), 2),
        "sum": sum(values),
    }


def count_jsonl(path: Path) -> tuple[int, int]:
    rows = 0
    parse_errors = 0
    if not path.exists():
        return rows, parse_errors
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            try:
                json.loads(line)
            except Exception:
                parse_errors += 1
    return rows, parse_errors


def message_token_count(tokenizer, messages: list[dict[str, str]]) -> int:
    text = "\n\n".join(f"[{m.get('role')}]\n{m.get('content','')}" for m in messages)
    return len(tokenizer.encode(text, add_special_tokens=False))


def analyze_regen(path: Path, tokenizer, *, progress_every: int = 1000) -> dict[str, Any]:
    status_counter = Counter()
    layer_counter = Counter()
    layer_status = defaultdict(Counter)
    assistant_tokens_by_layer = defaultdict(list)
    assistant_chars_by_layer = defaultdict(list)
    prompt_tokens_by_layer = defaultdict(list)
    errors = Counter()
    rows = 0
    parse_errors = 0

    total_lines = count_jsonl(path)[0]
    print(f"Analyzing regenerated data: {path} ({total_lines} rows)", file=sys.stderr, flush=True)
    iterator = path.open("r", encoding="utf-8")
    progress = None
    if tqdm is not None:
        progress = tqdm(iterator, total=total_lines, desc="regen", unit="rows", dynamic_ncols=True)
        iterator = progress
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
            status = record.get("status") or "<missing>"
            status_counter[status] += 1
            layer_counter[layer] += 1
            layer_status[layer][status] += 1
            if status != "success":
                errors[str(record.get("error") or "<no error>")[:300]] += 1
                continue
            conversations = record.get("conversations") or []
            prompt_messages = [m for m in conversations if isinstance(m, dict) and m.get("role") in {"system", "user"}]
            assistant_messages = [m for m in conversations if isinstance(m, dict) and m.get("role") == "assistant"]
            if prompt_messages:
                prompt_tokens_by_layer[layer].append(message_token_count(tokenizer, prompt_messages))
            assistant_text = "\n".join(m.get("content") or "" for m in assistant_messages)
            assistant_chars_by_layer[layer].append(len(assistant_text))
            assistant_tokens_by_layer[layer].append(len(tokenizer.encode(assistant_text, add_special_tokens=False)))
            if tqdm is None and progress_every > 0 and rows % progress_every == 0:
                print(f"[regen] rows={rows}/{total_lines} layer={layer}", file=sys.stderr, flush=True)
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()

    by_layer = {}
    for layer in sorted(layer_counter):
        by_layer[layer] = {
            "rows": layer_counter[layer],
            "status": dict(layer_status[layer]),
            "prompt_tokens": describe(prompt_tokens_by_layer[layer]),
            "assistant_tokens": describe(assistant_tokens_by_layer[layer]),
            "assistant_chars": describe(assistant_chars_by_layer[layer]),
        }
    return {
        "path": str(path),
        "rows": rows,
        "parse_errors": parse_errors,
        "status": dict(status_counter),
        "errors_top10": errors.most_common(10),
        "by_layer": by_layer,
    }


def summarize_cache(cache_dir: Path) -> dict[str, Any]:
    if not cache_dir.exists():
        return {"path": str(cache_dir), "exists": False}
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    files = [p for p in cache_dir.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    return {
        "path": str(cache_dir),
        "exists": True,
        "manifest": manifest,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / 1024**3, 3),
        "top_files": [
            {"path": str(p.relative_to(cache_dir)), "bytes": p.stat().st_size}
            for p in sorted(files, key=lambda item: item.stat().st_size, reverse=True)[:20]
        ],
    }


def write_md(summary: dict[str, Any], path: Path) -> None:
    lines = ["# MAI Profile Generated Data / Cache Summary", ""]
    split = summary["split"]
    regen = summary["regen"]
    cache = summary["cache"]
    lines += [
        "## Split",
        "",
        f"- Prepared dir: `{split['prepared_dir']}`",
        f"- Train rows: **{split['train_rows']}**",
        f"- Eval dir: `{split['eval_dir']}`",
        "",
        "## Regenerated Training Data",
        "",
        f"- Path: `{regen['path']}`",
        f"- Rows: **{regen['rows']}**",
        f"- Status: `{regen['status']}`",
        "",
        "| layer | rows | success | assistant p50 toks | assistant p95 toks | assistant max toks | prompt p50 toks | prompt p95 toks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, item in regen["by_layer"].items():
        status = item["status"]
        at = item["assistant_tokens"]
        pt = item["prompt_tokens"]
        lines.append(
            f"| {layer} | {item['rows']} | {status.get('success', 0)} | "
            f"{at['p50']} | {at['p95']} | {at['max']} | {pt['p50']} | {pt['p95']} |"
        )
    lines += ["", "## Target Cache", ""]
    lines.append(f"- Path: `{cache['path']}`")
    lines.append(f"- Exists: `{cache.get('exists')}`")
    if cache.get("exists"):
        manifest = cache.get("manifest") or {}
        lines.append(f"- Total size: **{cache['total_gib']} GiB**")
        lines.append(f"- Num samples: **{manifest.get('num_samples')}**")
        lines.append(f"- Target layers: `{manifest.get('target_layer_ids')}`")
        lines.append(f"- Hidden size: `{manifest.get('hidden_size')}`")
        lines.append(f"- Max length: `{manifest.get('max_length')}`")
        lines.append(f"- Min loss tokens: `{manifest.get('min_loss_tokens')}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MAI Profile regenerated data and target cache.")
    parser.add_argument("--date", default="20260615")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--regen-path", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir).expanduser().resolve() if args.base_dir else default_base_dir().resolve()
    prepared_dir = Path(args.prepared_dir).expanduser().resolve() if args.prepared_dir else base / "prepared_prompts" / args.date / "short_layers"
    regen_path = Path(args.regen_path).expanduser().resolve() if args.regen_path else base / "regenerated" / args.date / "maiprofile_short_layers_regen.jsonl"
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else base / "target_cache" / args.date / "gemma4_12b_maiprofile_short_layers"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else base / "reports" / args.date / "generated_cache_stats"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading tokenizer from {args.tokenizer!r} (local_files_only={args.local_files_only})...",
        file=sys.stderr,
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=args.local_files_only)
    print("Counting split rows...", file=sys.stderr, flush=True)
    train_rows, train_parse_errors = count_jsonl(prepared_dir / "train_maiprofile_short_layers.jsonl")
    eval_dir = prepared_dir / "eval_datasets"
    eval_counts = {}
    if eval_dir.exists():
        for p in sorted(eval_dir.glob("*.jsonl")):
            eval_counts[p.name] = count_jsonl(p)[0]
    regen_summary = analyze_regen(regen_path, tokenizer, progress_every=args.progress_every) if regen_path.exists() else {"path": str(regen_path), "exists": False}
    print("Summarizing cache files...", file=sys.stderr, flush=True)
    cache_summary = summarize_cache(cache_dir)
    summary = {
        "split": {
            "prepared_dir": str(prepared_dir),
            "train_rows": train_rows,
            "train_parse_errors": train_parse_errors,
            "eval_dir": str(eval_dir),
            "eval_counts": eval_counts,
        },
        "regen": regen_summary,
        "cache": cache_summary,
    }
    json_path = output_dir / "generated_cache_stats.json"
    md_path = output_dir / "generated_cache_stats.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(summary, md_path)
    print(json.dumps({"ok": True, "json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
