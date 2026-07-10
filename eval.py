from __future__ import annotations
import argparse
import json
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

TASKS = [
    ("gsm8k", 500),
    ("math500", 500),
    ("aime25",30),
    ("humaneval", 164),
    ("mbpp", 256),
    ("livecodebench", 500),
    ("mt-bench", 80),
    ("alpaca", 500),
    ("arena-hard-v2", 500),
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name_or_path", type=str, required=True)
    parser.add_argument("--draft_name_or_path",type=str,required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=("Confidence-head early-stop threshold. Confidence calibration metrics are collected only when this is 0.0."),
    )
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--step", type=int, default=None,help=("step for tensorboard logging"),)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help=(
            "Directory holding <task>.jsonl eval files. Defaults to "
            "./eval_datasets. Point this at the maiprofile eval_datasets dir "
            "on the mount to eval on maiprofile."
        ),
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help=(
            "Comma-separated task spec 'name:num_samples' to override the "
            "default public benchmark suite. Example: "
            "'maiprofile_layer3_seasonality:200'. num_samples optional "
            "(defaults to all rows in the file)."
        ),
    )
    args = parser.parse_args()
    if args.tasks:
        parsed_tasks = []
        for item in args.tasks.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                name, count = item.rsplit(":", 1)
                parsed_tasks.append((name.strip(), int(count)))
            else:
                parsed_tasks.append((item, None))
        args.tasks = parsed_tasks
    else:
        args.tasks = list(TASKS)
    return args


def main(local_rank: int, args):
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)
    evaluator_cls = EVALUATORS[draft_config.architectures[0]]
    evaluator = evaluator_cls(local_rank, args)
    evaluator.evaluate()
    evaluator.clean_up()

if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(
        main,
        args=(args,),
        nprocs=torch.cuda.device_count(),
    )
