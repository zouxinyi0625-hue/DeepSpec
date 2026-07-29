"""Time how long it takes to (1) read the target embed/lm_head weights and
(2) open + sample-read the DSpark target cache. Single process, no GPU, no DDP.

Usage (server):
    python scripts/time_load_model_and_cache.py \
        --target $AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only \
        --cache  $AZURE_ML_INPUT_UKWDATA/maiprofile/dspark_26b/target_cache_v2

Both default to the ukwdata mount layout if flags are omitted.
"""

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from deepspec.trainer.base_trainer import _load_target_embed_and_lm_head
from deepspec.data.target_cache_dataset import CacheDataset


def _mount():
    for var in ("AZURE_ML_INPUT_UKWDATA", "AZURE_ML_INPUT_UKDATA", "AZURE_ML_INPUT_ukwdata"):
        v = os.environ.get(var)
        if v:
            return v
    return ""


def main():
    m = _mount()
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=os.path.join(m, "maiprofile/models/text_only") if m else None)
    ap.add_argument("--cache", default=os.path.join(m, "maiprofile/dspark_26b/target_cache_v2") if m else None)
    ap.add_argument("--n-samples", type=int, default=20, help="how many cache samples to read")
    args = ap.parse_args()

    print("=" * 60)
    print(f"target: {args.target}")
    print(f"cache:  {args.cache}")
    print("=" * 60)

    # ---- 1. target embed/lm_head read -------------------------------------
    if args.target:
        print("\n[1] Reading target embed_tokens.weight + lm_head.weight ...")
        t0 = time.perf_counter()
        embed_w, lm_w = _load_target_embed_and_lm_head(args.target, dtype=torch.bfloat16)
        dt = time.perf_counter() - t0
        gib = (embed_w.numel() * embed_w.element_size()
               + lm_w.numel() * lm_w.element_size()) / 1024**3
        print(f"    embed shape: {tuple(embed_w.shape)}  lm_head shape: {tuple(lm_w.shape)}")
        print(f"    read {gib:.2f} GiB in {dt:.2f}s  ({gib/dt:.2f} GiB/s)")
        del embed_w, lm_w
    else:
        print("\n[1] skipped (no --target)")

    # ---- 2. cache open + sample reads -------------------------------------
    if args.cache:
        print("\n[2] Opening target cache (manifest + index) ...")
        t0 = time.perf_counter()
        ds = CacheDataset(args.cache)
        dt_open = time.perf_counter() - t0
        n = len(ds)
        print(f"    open took {dt_open:.2f}s;  cache has {n} samples")

        k = min(args.n_samples, n)
        print(f"\n[3] Reading {k} samples (first, middle, random-ish) ...")
        idxs = [0, n // 2, n - 1] + list(range(1, k))
        idxs = list(dict.fromkeys(i for i in idxs if 0 <= i < n))[:k]
        t0 = time.perf_counter()
        total_bytes = 0
        for i in idxs:
            sample = ds[i]
            for v in sample.values():
                if torch.is_tensor(v):
                    total_bytes += v.numel() * v.element_size()
        dt = time.perf_counter() - t0
        mib = total_bytes / 1024**2
        print(f"    read {len(idxs)} samples ({mib:.1f} MiB) in {dt:.2f}s"
              f"  ({dt/len(idxs)*1000:.1f} ms/sample, {mib/dt:.1f} MiB/s)")
        print(f"    sample keys: {list(sample.keys())}")
        for kk, vv in sample.items():
            if torch.is_tensor(vv):
                print(f"      {kk}: shape={tuple(vv.shape)} dtype={vv.dtype}")
    else:
        print("\n[2] skipped (no --cache)")

    print("\nDone.")


if __name__ == "__main__":
    main()
