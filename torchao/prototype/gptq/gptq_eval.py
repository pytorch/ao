# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone evaluation script for quantized model checkpoints.

Usage:
    python -m torchao.prototype.gptq.gptq_eval --checkpoint-dir <path_to_checkpoint>
    python -m torchao.prototype.gptq.gptq_eval --checkpoint-dir dir1 dir2 dir3
    python -m torchao.prototype.gptq.gptq_eval --checkpoint-dir Llama-3.1-8B-Instruct_mxfp8-*
"""

import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run lm_eval on quantized model checkpoints"
    )
    parser.add_argument(
        "checkpoint_dirs",
        nargs="+",
        help="One or more checkpoint directories to evaluate",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_challenge,arc_easy,hellaswag,piqa,winogrande",
        help="Comma-separated list of lm_eval tasks",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default="auto",
        help="Batch size for evaluation",
    )
    return parser.parse_args()


def run_lm_eval(checkpoint_dir: str, tasks: str, num_fewshot: int, batch_size: str):
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {checkpoint_dir}")
    print(f"{'=' * 60}\n")

    lm_eval_cmd = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={checkpoint_dir}",
        "--tasks",
        tasks,
        "--num_fewshot",
        str(num_fewshot),
        "--batch_size",
        batch_size,
    ]

    print(f"Running: {' '.join(lm_eval_cmd)}")
    try:
        subprocess.run(lm_eval_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"lm_eval failed for {checkpoint_dir}: {e}")
        return False
    except FileNotFoundError:
        print("lm_eval not found. Install with: pip install lm-eval")
        sys.exit(1)
    return True


def main():
    args = parse_args()

    results = {}
    for checkpoint_dir in args.checkpoint_dirs:
        success = run_lm_eval(
            checkpoint_dir, args.tasks, args.num_fewshot, args.batch_size
        )
        results[checkpoint_dir] = "OK" if success else "FAILED"

    if len(args.checkpoint_dirs) > 1:
        print(f"\n{'=' * 60}")
        print("Summary:")
        print(f"{'=' * 60}")
        for checkpoint_dir, status in results.items():
            print(f"  {status}: {checkpoint_dir}")


if __name__ == "__main__":
    main()
