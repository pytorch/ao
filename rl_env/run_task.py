#!/usr/bin/env python3
"""
Run and score a single evaluation task against a model.

Usage:
    python run_task.py tasks/pr_1234.json
    python run_task.py tasks/pr_1234.json --model anthropic/claude-sonnet-4-6
    python run_task.py tasks/pr_1234.json --dry-run   # print prompt and exit

The script:
  1. Creates a git worktree checked out at the task's base_commit.
  2. Writes the reference test + benchmark files into the worktree.
  3. Runs mini-swe-agent on the task prompt inside the worktree.
  4. Scores by running pytest and the benchmarks; prints a score in [0, 1].
  5. Saves the full result to --out (default: results/<task_id>.json).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.resolve()


# ---------------------------------------------------------------------------
# Worktree helpers
# ---------------------------------------------------------------------------

def create_worktree(base_commit: str, worktree_dir: Path) -> None:
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree_dir), base_commit],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )


def remove_worktree(worktree_dir: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree_dir)],
        cwd=REPO_ROOT, check=False, capture_output=True,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def run_cmd(cmd: str, cwd: Path, timeout: int = 300) -> tuple[int, str]:
    """Run a shell command, return (returncode, combined output)."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return result.returncode, result.stdout.decode(errors="replace")


def parse_pytest_score(output: str) -> float:
    """
    Parse pytest output and return fraction of tests passed.
    Looks for the summary line: '5 passed, 2 failed' or '7 passed'.
    Returns 0.0 if no tests were collected.
    """
    # pytest -q summary line: "3 passed, 1 failed in 2.34s"
    passed = sum(int(m) for m in re.findall(r"(\d+) passed", output))
    failed = sum(int(m) for m in re.findall(r"(\d+) failed", output))
    error  = sum(int(m) for m in re.findall(r"(\d+) error", output))
    total = passed + failed + error
    if total == 0:
        return 0.0
    return passed / total


def score_task(task: dict, worktree: Path) -> dict:
    """Run scoring commands and return a result dict."""
    test_cmd  = task["score_commands"]["tests"]
    bench_cmd = task["score_commands"]["benchmarks"]

    print(f"\n{'─'*60}")
    print(f"Running tests:  {test_cmd}")
    try:
        test_rc, test_out = run_cmd(test_cmd, worktree)
    except subprocess.TimeoutExpired:
        test_rc, test_out = 1, "TIMEOUT"

    test_score = parse_pytest_score(test_out)
    print(test_out[-3000:] if len(test_out) > 3000 else test_out)
    print(f"Test score: {test_score:.2f}  (exit {test_rc})")

    print(f"\nRunning benchmarks: {bench_cmd}")
    try:
        bench_rc, bench_out = run_cmd(bench_cmd, worktree, timeout=120)
    except subprocess.TimeoutExpired:
        bench_rc, bench_out = 1, "TIMEOUT"

    bench_ok = bench_rc == 0
    print(bench_out[-1000:] if len(bench_out) > 1000 else bench_out)
    print(f"Benchmark: {'OK' if bench_ok else 'FAILED'}  (exit {bench_rc})")

    # Combined score: 80% tests, 20% benchmark
    score = 0.8 * test_score + 0.2 * float(bench_ok)
    print(f"\n{'─'*60}")
    print(f"Final score: {score:.3f}")

    return {
        "score": score,
        "test_score": test_score,
        "benchmark_ok": bench_ok,
        "test_output": test_out,
        "benchmark_output": bench_out,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", help="Path to task JSON file (e.g. tasks/pr_1234.json)")
    parser.add_argument("--model", "-m", default=os.environ.get("MSWEA_MODEL_NAME", "anthropic/claude-sonnet-4-6"),
                        help="Model name passed to mini-swe-agent")
    parser.add_argument("--out", default=None,
                        help="Directory to save result JSON (default: results/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the prompt and environment setup, then exit")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip the agent; just score whatever is in the worktree")
    parser.add_argument("--keep-worktree", action="store_true",
                        help="Don't delete the worktree after running (useful for debugging)")
    args = parser.parse_args()

    task_path = Path(args.task)
    if not task_path.exists():
        sys.exit(f"Task file not found: {task_path}")

    task = json.loads(task_path.read_text())
    task_id     = task["id"]
    base_commit = task["base_commit"]

    print(f"Task:         {task_id}")
    print(f"PR:           {task['pr_url']}")
    print(f"Title:        {task['pr_title']}")
    print(f"Base commit:  {base_commit}")
    print(f"Impl files:   {task['implementation_files']}")
    print(f"Test files:   {task['test_files']}")
    print(f"Bench files:  {task['benchmark_files']}")

    if args.dry_run:
        print(f"\n{'─'*60}  PROMPT  {'─'*60}")
        print(task["prompt"])
        return

    # ── 1. Create worktree ────────────────────────────────────────────────
    worktree_base = REPO_ROOT / ".worktrees"
    worktree_base.mkdir(exist_ok=True)
    worktree = worktree_base / task_id

    if worktree.exists():
        print(f"\nWorktree already exists at {worktree} — reusing it.")
        print("Delete it first if you want a clean run: git worktree remove --force .worktrees/" + task_id)
    else:
        print(f"\nCreating worktree at {worktree} (base commit {base_commit[:12]})...")
        create_worktree(base_commit, worktree)

    try:
        # ── 2. Write reference files into worktree ────────────────────────
        ref_contents = task.get("reference_file_contents", {})
        for filename, contents in ref_contents.items():
            dest = worktree / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(contents)
            print(f"  Wrote reference file: {filename}")

        if not args.score_only:
            # ── 3. Run mini-swe-agent ─────────────────────────────────────
            # Write prompt to a temp file to avoid shell quoting issues
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(task["prompt"])
                prompt_file = f.name

            try:
                prompt_arg = Path(prompt_file).read_text()
                print(f"\nRunning mini-swe-agent (model: {args.model})...")
                print(f"Working directory: {worktree}")
                agent_result = subprocess.run(
                    ["mini", "--model", args.model, "--yolo",
                     "--exit-immediately", "--task", prompt_arg],
                    cwd=worktree,
                )
                print(f"\nmini-swe-agent exited with code {agent_result.returncode}")
            finally:
                os.unlink(prompt_file)

        # ── 4. Score ──────────────────────────────────────────────────────
        result = score_task(task, worktree)

    finally:
        if not args.keep_worktree:
            print(f"\nCleaning up worktree {worktree}...")
            remove_worktree(worktree)
        else:
            print(f"\nWorktree kept at: {worktree}")

    # ── 5. Save result ────────────────────────────────────────────────────
    out_dir = Path(args.out) if args.out else REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task_id}.json"
    out_path.write_text(json.dumps({
        "task_id":    task_id,
        "pr_number":  task["pr_number"],
        "pr_url":     task["pr_url"],
        "model":      args.model,
        "base_commit": base_commit,
        **result,
    }, indent=2))
    print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()
