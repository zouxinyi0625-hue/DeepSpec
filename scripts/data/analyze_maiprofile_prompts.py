from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional for this helper
    tqdm = None


def percentile(sorted_values: list[int], q: float) -> int | None:
    if not sorted_values:
        return None
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    idx = int(round((len(sorted_values) - 1) * q))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


def describe(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
            "sum": 0,
        }
    sorted_values = sorted(values)
    return {
        "min": sorted_values[0],
        "p25": percentile(sorted_values, 0.25),
        "p50": percentile(sorted_values, 0.50),
        "p75": percentile(sorted_values, 0.75),
        "p90": percentile(sorted_values, 0.90),
        "p95": percentile(sorted_values, 0.95),
        "p99": percentile(sorted_values, 0.99),
        "max": sorted_values[-1],
        "mean": round(sum(sorted_values) / len(sorted_values), 2),
        "sum": sum(sorted_values),
    }


def infer_layer_name(path: Path) -> str:
    name = path.name
    if name.endswith(".jsonl"):
        name = name[:-6]
    return name


def normalize_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("prompt_messages")
    if messages is None:
        messages = record.get("conversations")
    if not isinstance(messages, list):
        return []
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str):
            role = "<missing>"
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        normalized.append({"role": role, "content": content})
    return normalized


def render_gemma4_prompt_for_stats(messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    """Render prompt-only MAI Profile messages for length statistics.

    The Gemma4 tokenizer/chat-template combination used in some environments can
    return a nearly empty sequence for `system + user` prompt-only messages. For
    statistics we need a stable approximation of what is sent to the target
    service, so we render the prompt explicitly with the DeepSpec Gemma4 turn
    tokens and merge any system text into the first user turn.
    """
    system_parts: list[str] = []
    rendered: list[str] = []
    pending_system_prefix = ""

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "system":
            system_parts.append(content)
            continue
        if role == "user":
            if system_parts:
                pending_system_prefix = "\n\n".join(system_parts).strip()
                system_parts = []
            if pending_system_prefix:
                content = f"{pending_system_prefix}\n\n{content}"
                pending_system_prefix = ""
            rendered.append(f"<|turn>user\n{content}<turn|>\n")
        elif role == "assistant":
            rendered.append(f"<|turn>model\n{content}<turn|>\n")
        else:
            rendered.append(f"<|turn>user\n[{role}]\n{content}<turn|>\n")

    if system_parts:
        system_text = "\n\n".join(system_parts)
        rendered.insert(0, f"<|turn>user\n{system_text}<turn|>\n")
    if add_generation_prompt:
        rendered.append("<|turn>model\n")
    return "".join(rendered)


def count_tokens(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> int:
    text = render_gemma4_prompt_for_stats(
        messages,
        add_generation_prompt=add_generation_prompt,
    )
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_jsonl_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line), None
            except Exception as exc:  # pragma: no cover - diagnostic path
                yield line_number, None, str(exc)


def analyze_file(
    path: Path,
    *,
    tokenizer,
    max_rows: int | None,
    length_thresholds: list[int],
    sample_examples: int,
    progress_every: int,
) -> dict[str, Any]:
    layer = infer_layer_name(path)
    rows = 0
    parse_errors = 0
    message_counts: list[int] = []
    char_counts: list[int] = []
    token_counts: list[int] = []
    role_counter = Counter()
    key_counter = Counter()
    assistant_records = 0
    examples: list[dict[str, Any]] = []
    longest: list[dict[str, Any]] = []

    iterator = load_jsonl_records(path)
    progress = None
    if tqdm is not None:
        progress = tqdm(
            iterator,
            desc=f"analyze {layer}",
            unit="rows",
            total=max_rows,
            dynamic_ncols=True,
        )
        iterator = progress

    for line_number, record, error in iterator:
        if max_rows is not None and rows >= max_rows:
            break
        if error is not None:
            parse_errors += 1
            continue
        assert record is not None
        rows += 1
        if isinstance(record, dict):
            key_counter.update(record.keys())
        messages = normalize_messages(record)
        message_counts.append(len(messages))
        role_counter.update(message["role"] for message in messages)
        if any(message["role"] == "assistant" for message in messages):
            assistant_records += 1
        chars = sum(len(message["content"]) for message in messages)
        char_counts.append(chars)
        tokens = count_tokens(tokenizer, messages, add_generation_prompt=True)
        token_counts.append(tokens)

        item_summary = {
            "line_number": line_number,
            "user_id": record.get("user_id") if isinstance(record, dict) else None,
            "prompt_hash": record.get("prompt_hash") if isinstance(record, dict) else None,
            "message_count": len(messages),
            "chars": chars,
            "tokens": tokens,
            "roles": [message["role"] for message in messages],
            "preview": "\n".join(
                f"[{message['role']}] {message['content'][:500]}"
                for message in messages[:3]
            ),
        }
        if len(examples) < sample_examples:
            examples.append(item_summary)
        longest.append(item_summary)
        longest = sorted(longest, key=lambda item: int(item["tokens"]), reverse=True)[:sample_examples]
        if max_rows and rows >= max_rows:
            break
        if tqdm is None and progress_every > 0 and rows % progress_every == 0:
            print(
                f"[analyze {layer}] rows={rows} last_line={line_number} "
                f"last_tokens={tokens} last_chars={chars}",
                file=sys.stderr,
                flush=True,
            )

    if progress is not None:
        progress.close()
    threshold_stats = {}
    for threshold in length_thresholds:
        over = sum(1 for value in token_counts if value > threshold)
        threshold_stats[str(threshold)] = {
            "over_count": over,
            "over_ratio": round(over / rows, 6) if rows else None,
            "within_count": rows - over,
            "within_ratio": round((rows - over) / rows, 6) if rows else None,
        }

    return {
        "layer": layer,
        "file": str(path),
        "bytes": path.stat().st_size,
        "mib": round(path.stat().st_size / 1024**2, 3),
        "rows": rows,
        "parse_errors": parse_errors,
        "top_level_keys": dict(key_counter),
        "roles": dict(role_counter),
        "assistant_records": assistant_records,
        "message_count": describe(message_counts),
        "chars_per_record": describe(char_counts),
        "prompt_tokens_with_generation_prompt": describe(token_counts),
        "length_thresholds": threshold_stats,
        "examples": examples,
        "longest_examples": longest,
    }


def write_markdown_report(summary: dict[str, Any], output_path: Path) -> None:
    layers = summary["layers"]
    thresholds = summary["length_thresholds"]
    lines = []
    lines.append("# MAI Profile Prompt Data Statistics")
    lines.append("")
    lines.append(f"- Input dir: `{summary['input_dir']}`")
    lines.append(f"- Tokenizer: `{summary['tokenizer']}`")
    lines.append(f"- Total rows: **{summary['total_rows']}**")
    lines.append(f"- Total bytes: **{summary['total_bytes']}** ({summary['total_mib']:.2f} MiB)")
    lines.append("")
    lines.append("## Per-layer summary")
    lines.append("")
    lines.append("| layer | rows | MiB | p50 tokens | p95 tokens | p99 tokens | max tokens | assistant records |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for layer in layers:
        token_desc = layer["prompt_tokens_with_generation_prompt"]
        lines.append(
            f"| {layer['layer']} | {layer['rows']} | {layer['mib']:.2f} | "
            f"{token_desc['p50']} | {token_desc['p95']} | {token_desc['p99']} | {token_desc['max']} | "
            f"{layer['assistant_records']} |"
        )
    lines.append("")
    lines.append("## Truncation risk by max_length")
    lines.append("")
    header = "| layer | rows | " + " | ".join(f">{t}" for t in thresholds) + " |"
    sep = "|---|---:|" + "---:|" * len(thresholds)
    lines.append(header)
    lines.append(sep)
    for layer in layers:
        cells = []
        for threshold in thresholds:
            stat = layer["length_thresholds"][str(threshold)]
            if stat["over_ratio"] is None:
                cells.append(f"{stat['over_count']} (n/a)")
            else:
                cells.append(f"{stat['over_count']} ({stat['over_ratio']:.1%})")
        lines.append(f"| {layer['layer']} | {layer['rows']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `prompt_tokens_with_generation_prompt` is computed from `prompt_messages` / `conversations` using an explicit DeepSpec Gemma4-style turn rendering (`<|turn>user`, `<turn|>`, `<|turn>model`) with `add_generation_prompt=True`.")
    lines.append("- These files are prompt-only if `assistant_records` is zero; target-model response generation is still required before target-cache creation.")
    lines.append("- Use truncation-risk rows to decide whether a DSpark `max_length` is realistic for each layer.")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MAI Profile prompt JSONL files for DSpark data planning.")
    parser.add_argument(
        "--input-dir",
        default=None,
        help=(
            "Directory containing MAI Profile layer *.jsonl files. Defaults to "
            "$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/raw_data/20260615."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to "
            "$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/reports/<date>/prompt_stats."
        ),
    )
    parser.add_argument("--tokenizer", default="google/gemma-4-12B-it", help="Tokenizer/model name or local path.")
    parser.add_argument("--file-glob", default="*.jsonl")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional per-file row cap for quick profiling.")
    parser.add_argument("--thresholds", default="2048,4096,8192,16384,32768", help="Comma-separated max_length thresholds for truncation-risk reporting.")
    parser.add_argument("--sample-examples", type=int, default=3, help="Number of first/longest examples to keep per layer in JSON output.")
    parser.add_argument("--progress-every", type=int, default=1000, help="Fallback progress print interval when tqdm is unavailable.")
    parser.add_argument(
        "--include-empty-files",
        action="store_true",
        help="Include 0-byte or 0-row JSONL files in the summary. Defaults to skipping them.",
    )
    return parser.parse_args()


def get_msndni_mount() -> Path:
    mount = os.environ.get("AZURE_ML_INPUT_msndni")
    if not mount:
        raise SystemExit(
            "AZURE_ML_INPUT_msndni is not set. Export it or pass --input-dir and --output-dir explicitly."
        )
    return Path(mount)


def resolve_input_dir(args: argparse.Namespace) -> Path:
    if args.input_dir:
        return Path(args.input_dir).expanduser().resolve()
    return (get_msndni_mount() / "shares/users/zxy/maiprofile/raw_data/20260615").resolve()


def resolve_output_dir(args: argparse.Namespace, input_dir: Path) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    date_name = input_dir.name
    return (
        get_msndni_mount()
        / "shares/users/zxy/maiprofile/reports"
        / date_name
        / "prompt_stats"
    ).resolve()


def main() -> None:
    args = parse_args()
    input_dir = resolve_input_dir(args)
    if not input_dir.is_dir():
        raise SystemExit(f"input dir not found: {input_dir}")
    thresholds = [int(item) for item in args.thresholds.split(",") if item.strip()]
    output_dir = resolve_output_dir(args, input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    files = sorted(path for path in input_dir.glob(args.file_glob) if path.is_file())
    if not args.include_empty_files:
        files = [path for path in files if path.stat().st_size > 0]
    if not files:
        raise SystemExit(f"no non-empty files matched {args.file_glob!r} under {input_dir}")

    layers = []
    skipped_empty = []
    file_iter = files
    if tqdm is not None:
        file_iter = tqdm(files, desc="files", unit="file", dynamic_ncols=True)
    print(
        f"Analyzing {len(files)} files from {input_dir}; outputs -> {output_dir}",
        file=sys.stderr,
        flush=True,
    )
    for file_idx, path in enumerate(file_iter, start=1):
        print(f"[file {file_idx}/{len(files)}] start {path.name}", file=sys.stderr, flush=True)
        layer_summary = analyze_file(
            path,
            tokenizer=tokenizer,
            max_rows=args.max_rows,
            length_thresholds=thresholds,
            sample_examples=args.sample_examples,
            progress_every=args.progress_every,
        )
        if not args.include_empty_files and int(layer_summary["rows"]) == 0:
            skipped_empty.append(str(path))
            print(f"[file {file_idx}/{len(files)}] skip empty {path.name}", file=sys.stderr, flush=True)
            continue
        token_desc = layer_summary["prompt_tokens_with_generation_prompt"]
        print(
            f"[file {file_idx}/{len(files)}] done {path.name}: rows={layer_summary['rows']} "
            f"p50_tokens={token_desc['p50']} p95_tokens={token_desc['p95']} "
            f"max_tokens={token_desc['max']}",
            file=sys.stderr,
            flush=True,
        )
        layers.append(layer_summary)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "tokenizer": args.tokenizer,
        "file_glob": args.file_glob,
        "max_rows": args.max_rows,
        "length_thresholds": thresholds,
        "total_rows": sum(layer["rows"] for layer in layers),
        "total_bytes": sum(layer["bytes"] for layer in layers),
        "total_mib": round(sum(layer["bytes"] for layer in layers) / 1024**2, 3),
        "skipped_empty_files": skipped_empty,
        "layers": layers,
    }

    json_path = output_dir / "prompt_stats.json"
    md_path = output_dir / "prompt_stats.md"
    csv_path = output_dir / "prompt_stats_layers.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(summary, md_path)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "layer",
            "rows",
            "mib",
            "tokens_p50",
            "tokens_p95",
            "tokens_p99",
            "tokens_max",
            "assistant_records",
        ] + [f"over_{threshold}" for threshold in thresholds]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for layer in layers:
            token_desc = layer["prompt_tokens_with_generation_prompt"]
            row = {
                "layer": layer["layer"],
                "rows": layer["rows"],
                "mib": layer["mib"],
                "tokens_p50": token_desc["p50"],
                "tokens_p95": token_desc["p95"],
                "tokens_p99": token_desc["p99"],
                "tokens_max": token_desc["max"],
                "assistant_records": layer["assistant_records"],
            }
            for threshold in thresholds:
                row[f"over_{threshold}"] = layer["length_thresholds"][str(threshold)]["over_count"]
            writer.writerow(row)

    print(json.dumps({
        "ok": True,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "json": str(json_path),
        "markdown": str(md_path),
        "csv": str(csv_path),
        "total_rows": summary["total_rows"],
        "total_mib": summary["total_mib"],
        "skipped_empty_files": summary["skipped_empty_files"],
    }, indent=2))


if __name__ == "__main__":
    main()
