"""Filter a merged maiprofile split jsonl by source_layer into a smaller jsonl.

The 26B DSpark target cache is ~14TB because it covers ALL layers. To train /
iterate on a single layer (e.g. layer1) you only need that layer's rows, so
filter them out first, then generate a much smaller per-layer cache with
scripts/data/prepare_dspark_26b_cache.sh (TRAIN_DATA_PATH=<filtered jsonl>).

Usage:
    python scripts/data/filter_layer_jsonl.py \
        --input  $AZURE_ML_INPUT_UKWDATA/maiprofile/mtp_26b/split/train_maiprofile_26b.jsonl \
        --output $AZURE_ML_INPUT_UKWDATA/maiprofile/mtp_26b/split/train_layer1_26b.jsonl \
        --layers layer1_actual,layer1_intent

    # single layer:
    ... --layers layer1_actual
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Merged split jsonl (has per-row source_layer).")
    ap.add_argument("--output", required=True, help="Filtered jsonl path.")
    ap.add_argument(
        "--layers",
        required=True,
        help="Comma-separated source_layer names to keep (e.g. layer1_actual,layer1_intent).",
    )
    ap.add_argument(
        "--prefix",
        action="store_true",
        help="Match by prefix instead of exact (e.g. --layers layer1 --prefix keeps all layer1_*).",
    )
    args = ap.parse_args()

    keep = [s.strip() for s in args.layers.split(",") if s.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    def matches(layer: str) -> bool:
        if args.prefix:
            return any(layer.startswith(k) for k in keep)
        return layer in keep

    total = 0
    kept = 0
    per_layer = {}
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            layer = rec.get("source_layer", "")
            if matches(layer):
                fout.write(line + "\n")
                kept += 1
                per_layer[layer] = per_layer.get(layer, 0) + 1

    print(f"input rows:  {total}")
    print(f"kept rows:   {kept}  ({100.0 * kept / max(total, 1):.1f}%)")
    print("kept by layer:")
    for layer in sorted(per_layer):
        print(f"  {layer}: {per_layer[layer]}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
