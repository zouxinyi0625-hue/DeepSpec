from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import AutoConfig

from deepspec.eval.dspark import Gemma4DSparkEvaluator, Qwen3DSparkEvaluator
from deepspec.eval.eagle3 import Gemma4Eagle3Evaluator, Qwen3Eagle3Evaluator
from deepspec.utils import CustomJSONEncoder


EVALUATORS = {
    "Qwen3DSparkModel": Qwen3DSparkEvaluator,
    "Gemma4DSparkModel": Gemma4DSparkEvaluator,
    "Qwen3Eagle3Model": Qwen3Eagle3Evaluator,
    "Gemma4Eagle3Model": Gemma4Eagle3Evaluator,
    "Eagle3DraftModel": Qwen3Eagle3Evaluator,
}


def _parse_tasks(raw: str) -> list[tuple[str, int]]:
    tasks: list[tuple[str, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Task spec must be name:max_samples, got {item!r}")
        name, max_samples = item.split(":", 1)
        tasks.append((name.strip(), int(max_samples)))
    if not tasks:
        raise ValueError("At least one eval task is required.")
    return tasks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name_or_path", type=str, required=True)
    parser.add_argument("--draft_name_or_path", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Confidence-head early-stop threshold. Metrics are collected only when this is 0.0.",
    )
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument(
        "--tasks",
        type=str,
        default="gsm8k:32,mt-bench:16,alpaca:32",
        help="Comma-separated task:max_samples list, e.g. gsm8k:32,mt-bench:16.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./eval_datasets",
        help="Directory containing eval JSONL files with `turns` fields.",
    )
    args = parser.parse_args()
    args.tasks = _parse_tasks(args.tasks)
    return args


def main(local_rank: int, args):
    start_time = time.perf_counter()
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
        print(f"[eval rank0] loading draft config from {args.draft_name_or_path}", flush=True)
    draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)
    assert draft_config.architectures, "Draft checkpoint config must define architectures."
    architecture = draft_config.architectures[0]
    evaluator_cls = EVALUATORS[architecture]
    if local_rank == 0:
        print(
            f"[eval rank0] architecture={architecture} evaluator={evaluator_cls.__name__}",
            flush=True,
        )
    evaluator = evaluator_cls(local_rank, args)
    if local_rank == 0:
        print(f"[eval rank0] evaluator initialized in {time.perf_counter() - start_time:.1f}s", flush=True)
    evaluator.evaluate()
    if local_rank == 0:
        print(f"[eval rank0] evaluate complete in {time.perf_counter() - start_time:.1f}s", flush=True)
    evaluator.clean_up()


if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(
        main,
        args=(args,),
        nprocs=torch.cuda.device_count(),
    )
