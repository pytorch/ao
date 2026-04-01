#!/usr/bin/env python3
"""
Run and score a single evaluation task against a model.

Usage:
    python run_task.py tasks/pr_1234.json
    python run_task.py tasks/pr_1234.json tasks/pr_5678.json
    python run_task.py tasks/
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

try:
    from google import genai as _genai
    _genai_available = True
except ImportError:
    _genai_available = False


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


_BENCHMARK_SCORE_PROMPT = """\
You are scoring a coding agent's implementation of a performance-critical ML kernel \
by comparing benchmark output tables.

Hardware: NVIDIA H100 SXM5
  HBM bandwidth : 3.35 TB/s
  FP16 compute  : 989 TFLOPS
  FP8 compute   : 1979 TFLOPS

The benchmark produces a table of results across different problem shapes and \
configurations (e.g. matrix sizes, dtypes, scaling modes). Compare the two tables \
holistically — consider whether the agent's numbers are close to the reference across \
all rows, not just the best or worst case.

Reference output (from the merged PR — this is the target):
<reference>
{reference_output}
</reference>

Agent output:
<agent>
{agent_output}
</agent>

Score the agent from 0.0 to 1.0:
  1.0 = matches or exceeds reference across all configurations
  0.7 = mostly matches, minor regressions on some shapes
  0.5 = roughly half the reference performance on average
  0.2 = runs but produces clearly wrong or very poor numbers
  0.0 = benchmark did not run or produced no output

Respond with JSON only, no prose:
{{"score": <float 0-1>, "reason": "<one sentence summarising the comparison>"}}
"""


def score_benchmark(agent_output: str, reference_output: str,
                    bench_model: str) -> tuple[float, str]:
    """
    Score agent benchmark output vs reference using LLM table comparison.
    Returns (score 0-1, reason string).
    Falls back to binary pass/fail if LLM is unavailable.
    """
    if not _genai_available:
        return 0.5, "google-genai not installed; defaulting to 0.5"
    try:
        client = _genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        prompt = _BENCHMARK_SCORE_PROMPT.format(
            reference_output=reference_output[-3000:],
            agent_output=agent_output[-3000:],
        )
        response = client.models.generate_content(model=bench_model, contents=prompt)
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        return float(parsed["score"]), parsed.get("reason", "")
    except Exception as e:
        print(f"  LLM benchmark scoring error: {e}", file=sys.stderr)
        return 0.5, f"LLM error: {e}"


def get_impl_diff(task: dict, worktree: Path) -> str:
    """Return the git diff of implementation files the agent modified."""
    result = subprocess.run(
        ["git", "diff", "HEAD", "--"] + task.get("implementation_files", []),
        cwd=worktree, capture_output=True,
    )
    return result.stdout.decode(errors="replace")


_REVIEW_PROMPT = """\
You are auditing a coding agent's solution to a software engineering task.

## Task
The agent was asked to implement a feature in the torchao library (PyTorch quantization/optimization).
PR title: {pr_title}

## Test files the agent must pass (these were given to the agent as the spec)
{test_files_content}

## Implementation diff (what the agent wrote)
```diff
{impl_diff}
```

## What to look for
Determine whether the implementation is GENUINE or TRIVIAL/CHEATING.

A GENUINE implementation:
- Adds real algorithmic logic (kernels, quantization math, data transformations)
- The code would work correctly for inputs beyond those in the tests
- Changes are proportional to the complexity of the task

A TRIVIAL/CHEATING implementation:
- Hardcodes return values that match exactly what the tests expect
- Mocks or stubs out the real logic (e.g. `return torch.ones(...)`, `pass`)
- Patches or monkeypatches functions to bypass the real implementation
- Special-cases specific input values from the tests
- Modifies sys.path or imports to swap in a pre-existing implementation

Respond with JSON only, no prose:
{{
  "genuine": true or false,
  "confidence": 0.0 to 1.0,
  "score": 0.0 to 1.0,
  "reason": "one sentence"
}}

Where `score` is a multiplier to apply to the test/benchmark score:
  1.0 = clearly genuine implementation
  0.5 = suspicious but could be legitimate
  0.0 = clearly trivial or cheating
"""


def llm_review_implementation(task: dict, worktree: Path,
                               model: str = "gemini-3-flash-preview") -> tuple[float, str]:
    """
    Ask an LLM to review the agent's implementation diff for reward hacking.
    Returns (score_multiplier 0-1, reason).
    Falls back to 1.0 (no penalty) on error so a broken review never kills a good run.
    """
    if not _genai_available:
        return 1.0, "google-genai not installed; skipping review"

    impl_diff = get_impl_diff(task, worktree)
    if not impl_diff.strip():
        return 0.0, "agent made no changes to implementation files"

    # Include test file contents so the LLM can spot hardcoded values
    test_sections = []
    for filename, contents in task.get("reference_file_contents", {}).items():
        if "test" in filename:
            excerpt = contents[:2000] + ("\n# ...(truncated)" if len(contents) > 2000 else "")
            test_sections.append(f"### {filename}\n```python\n{excerpt}\n```")
    test_files_content = "\n\n".join(test_sections) or "(not available)"

    # Truncate diff to avoid huge prompts
    diff_excerpt = impl_diff[:4000] + ("\n# ...(diff truncated)" if len(impl_diff) > 4000 else "")

    prompt = _REVIEW_PROMPT.format(
        pr_title=task.get("pr_title", ""),
        test_files_content=test_files_content,
        impl_diff=diff_excerpt,
    )

    try:
        client = _genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        response = client.models.generate_content(model=model, contents=prompt)
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        score = float(parsed["score"])
        reason = parsed.get("reason", "")
        genuine = parsed.get("genuine", score >= 0.5)
        confidence = parsed.get("confidence", 1.0)
        print(f"  Review: genuine={genuine} confidence={confidence:.2f} score={score:.2f} — {reason}")
        return score, reason
    except Exception as e:
        print(f"  Review error: {e}", file=sys.stderr)
        return 1.0, f"review error: {e}"


def verify_reference_files(task: dict, worktree: Path) -> list[str]:
    """
    Check that the agent did not delete or modify the reference test/benchmark
    files that were written into the worktree before the agent ran.
    Returns a list of violation descriptions (empty = all good).
    """
    violations = []
    for filename, expected in task.get("reference_file_contents", {}).items():
        dest = worktree / filename
        if not dest.exists():
            violations.append(f"DELETED: {filename}")
        else:
            actual = dest.read_text(errors="replace")
            if actual.strip() != expected.strip():
                violations.append(f"MODIFIED: {filename}")
    return violations


def score_task(task: dict, worktree: Path, bench_model: str = "gemini-3-flash-preview",
               review_model: str = "gemini-3-flash-preview", skip_review: bool = False) -> dict:
    """Run scoring commands and return a result dict."""
    test_cmd  = task["score_commands"]["tests"]
    bench_cmd = task["score_commands"]["benchmarks"]

    # ── 0. Verify reference files weren't tampered with ───────────────────
    print(f"\n{'─'*60}")
    violations = verify_reference_files(task, worktree)
    if violations:
        print(f"REFERENCE FILE TAMPERING DETECTED — score forced to 0:")
        for v in violations:
            print(f"  {v}")
        return {
            "score": 0.0, "raw_score": 0.0,
            "test_score": 0.0, "benchmark_score": None, "benchmark_method": None,
            "review_multiplier": 0.0, "review_reason": "reference file tampering: " + "; ".join(violations),
            "test_output": "", "benchmark_output": None,
            "tampering": violations,
        }

    print(f"\n{'─'*60}")
    print(f"Running tests:  {test_cmd}")
    try:
        test_rc, test_out = run_cmd(test_cmd, worktree)
    except subprocess.TimeoutExpired:
        test_rc, test_out = 1, "TIMEOUT"

    test_score = parse_pytest_score(test_out)
    print(test_out[-3000:] if len(test_out) > 3000 else test_out)
    print(f"Test score: {test_score:.2f}  (exit {test_rc})")

    if bench_cmd:
        ref_bench_outputs = task.get("reference_benchmark_output", {})
        bench_files = task.get("benchmark_files", [])

        # Run each benchmark file individually to get isolated per-file output,
        # which is needed for accurate per-file comparison against reference output.
        per_file_outputs: dict[str, tuple[int, str]] = {}
        for bench_file in bench_files:
            print(f"\nRunning benchmark: {bench_file}")
            try:
                rc, out = run_cmd(f"python {bench_file}", worktree, timeout=180)
            except subprocess.TimeoutExpired:
                rc, out = 1, "TIMEOUT"
            per_file_outputs[bench_file] = (rc, out)
            print(out[-1000:] if len(out) > 1000 else out)
            print(f"  exit {rc}")

        bench_out = "\n\n".join(out for _, out in per_file_outputs.values())

        bench_scores = []
        bench_methods = []
        for bench_file, (rc, out) in per_file_outputs.items():
            name = Path(bench_file).name
            ref_out = ref_bench_outputs.get(bench_file, "")
            if ref_out and rc == 0:
                s, method = score_benchmark(out, ref_out, bench_model)
                bench_scores.append(s)
                bench_methods.append(f"{name}: {method}")
                print(f"  {bench_methods[-1]} → {s:.3f}")
            else:
                reason = "no reference captured" if not ref_out else "benchmark failed"
                bench_methods.append(f"{name}: skipped ({reason})")
                print(f"  {bench_methods[-1]}")

        if bench_scores:
            bench_score = sum(bench_scores) / len(bench_scores)
            bench_method = "; ".join(bench_methods)
            print(f"Benchmark score: {bench_score:.3f}")
            score = 0.8 * test_score + 0.2 * bench_score
        else:
            # No scoreable benchmarks — don't penalise, use test score only
            bench_score = None
            bench_method = "; ".join(bench_methods)
            score = test_score
    else:
        bench_out = None
        bench_score = None
        bench_method = None
        score = test_score

    # ── LLM implementation review ─────────────────────────────────────────
    if skip_review:
        review_multiplier, review_reason = 1.0, "skipped"
    else:
        print(f"\n{'─'*60}")
        print("Running implementation review...")
        review_multiplier, review_reason = llm_review_implementation(
            task, worktree, model=review_model,
        )

    final_score = score * review_multiplier

    print(f"\n{'─'*60}")
    print(f"Raw score:    {score:.3f}")
    print(f"Review:       {review_multiplier:.3f}  ({review_reason})")
    print(f"Final score:  {final_score:.3f}")

    return {
        "score": final_score,
        "raw_score": score,
        "test_score": test_score,
        "benchmark_score": bench_score,
        "benchmark_method": bench_method,
        "review_multiplier": review_multiplier,
        "review_reason": review_reason,
        "test_output": test_out,
        "benchmark_output": bench_out,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_task_paths(inputs: list[str]) -> list[Path]:
    """Expand a mix of task JSON files and directories into a sorted list of task paths."""
    paths = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            paths.extend(sorted(p.glob("pr_*.json")))
        elif p.exists():
            paths.append(p)
        else:
            print(f"Warning: {p} not found, skipping.", file=sys.stderr)
    return paths


def run_one(task_path: Path, args) -> None:
    task = json.loads(task_path.read_text())
    task_id     = task["id"]
    base_commit = task["base_commit"]

    print(f"\n{'═'*60}")
    print(f"Task:         {task_id}")
    print(f"PR:           {task['pr_url']}")
    print(f"Title:        {task['pr_title']}")
    print(f"Base commit:  {base_commit}")
    print(f"Impl files:   {task['implementation_files']}")
    print(f"Test files:   {task['test_files']}")
    if task.get("desc_test_files"):
        print(f"Desc tests:   {task['desc_test_files']}")
    print(f"Bench files:  {task['benchmark_files']}")

    if args.dry_run:
        print(f"\n{'─'*60}  PROMPT  {'─'*60}")
        print(task["prompt"])
        return

    out_dir = Path(args.out) if args.out else REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task_id}.json"
    if out_path.exists() and not args.score_only:
        print(f"Result already exists at {out_path}, skipping.")
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
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(task["prompt"])
                prompt_file = f.name

            try:
                prompt_arg = Path(prompt_file).read_text()
                print(f"\nRunning mini-swe-agent (model: {args.model})...")
                print(f"Working directory: {worktree}")
                traj_path = out_dir / f"{task_id}.traj.json"
                cmd = ["mini", "--model", args.model, "--yolo",
                       "--exit-immediately", "--task", prompt_arg,
                       "-o", str(traj_path)]
                if args.cost_limit_per_task is not None:
                    cmd += ["--cost-limit", str(args.cost_limit_per_task)]
                agent_result = subprocess.run(cmd, cwd=worktree)
                print(f"\nmini-swe-agent exited with code {agent_result.returncode}")
            finally:
                os.unlink(prompt_file)

        # ── 4. Score ──────────────────────────────────────────────────────
        result = score_task(task, worktree, bench_model=args.benchmark_model,
                            review_model=args.review_model, skip_review=args.no_review)

    finally:
        if not args.keep_worktree:
            print(f"\nCleaning up worktree {worktree}...")
            remove_worktree(worktree)
        else:
            print(f"\nWorktree kept at: {worktree}")

    # ── 5. Save result ────────────────────────────────────────────────────
    out_path.write_text(json.dumps({
        "task_id":    task_id,
        "pr_number":  task["pr_number"],
        "pr_url":     task["pr_url"],
        "model":      args.model,
        "base_commit": base_commit,
        **result,
    }, indent=2))
    print(f"\nResult saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="+",
                        help="Task JSON file(s) or directory of tasks (e.g. tasks/ or tasks/pr_1234.json)")
    parser.add_argument("--model", "-m", default=os.environ.get("MSWEA_MODEL_NAME", "anthropic/claude-sonnet-4-6"),
                        help="Model name passed to mini-swe-agent")
    parser.add_argument("--cost-limit-per-task", type=float, default=None, metavar="USD",
                        help="Per-task cost limit in dollars passed to mini-swe-agent (e.g. 0.50)")
    parser.add_argument("--out", default=None,
                        help="Directory to save result JSON (default: results/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the prompt and environment setup, then exit")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip the agent; just score whatever is in the worktree")
    parser.add_argument("--keep-worktree", action="store_true",
                        help="Don't delete the worktree after running (useful for debugging)")
    parser.add_argument("--benchmark-model", default=os.environ.get("GEMINI_FILTER_MODEL", "gemini-3-flash-preview"),
                        help="Gemini model used to score benchmarks when regex parsing fails (default: gemini-3-flash-preview)")
    parser.add_argument("--review-model", default=os.environ.get("GEMINI_FILTER_MODEL", "gemini-3-flash-preview"),
                        help="Gemini model used to review implementation for reward hacking (default: gemini-3-flash-preview)")
    parser.add_argument("--no-review", action="store_true",
                        help="Skip the LLM implementation review (review multiplier defaults to 1.0)")
    args = parser.parse_args()

    task_paths = resolve_task_paths(args.tasks)
    if not task_paths:
        sys.exit("No task files found.")

    print(f"Running {len(task_paths)} task(s)...")
    for i, task_path in enumerate(task_paths, 1):
        print(f"\n[{i}/{len(task_paths)}] {task_path}")
        try:
            run_one(task_path, args)
        except Exception as e:
            print(f"\nERROR on {task_path}: {e}", file=sys.stderr)
            continue


if __name__ == "__main__":
    main()
