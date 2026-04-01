#!/usr/bin/env python3
"""
Generate evaluation tasks for the torchao benchmark.

Each task is derived from a merged PR that added:
  - at least one implementation file
  - at least one unit test
  - optionally one or more benchmarks (included when present)

Task starting state: repo checked out at the PR's base commit, with the
implementation files removed (tests + benchmarks are kept as the spec).

Scoring: run the PR's test files with pytest; score = fraction of tests passing.
The benchmark files must also execute without error for full credit.

Usage:
    python generate_tasks.py [--token TOKEN] [--limit N] [--out DIR]

Output:
    <out>/tasks.json          — list of all task descriptors
    <out>/tasks/<id>.json     — one file per task
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

try:
    from google import genai as _genai
    _genai_available = True
except ImportError:
    _genai_available = False


REPO = "pytorch/ao"
API_BASE = "https://api.github.com"
REPO_ROOT = Path(__file__).parent.parent.resolve()

# ---------------------------------------------------------------------------
# File classification helpers
# ---------------------------------------------------------------------------

def classify_file(filename: str) -> str:
    """Return 'test', 'benchmark', or 'implementation'."""
    parts = filename.split("/")
    basename = parts[-1]
    if "test" in parts[:-1] or basename.startswith("test_") or basename.endswith("_test.py"):
        return "test"
    if parts[0] == "benchmarks" or "benchmarks" in parts[:-1] or "benchmark" in basename.lower():
        return "benchmark"
    return "implementation"


def is_python(filename: str) -> bool:
    return filename.endswith(".py")


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def gh_get(path: str, token: Optional[str], params: dict = None) -> object:
    url = f"{API_BASE}{path}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"\nHTTP {e.code} for {url}: {body[:200]}", file=sys.stderr)
        raise


def paginate(path: str, token: Optional[str], per_page: int = 100) -> list:
    results = []
    page = 1
    while True:
        batch = gh_get(path, token, {"per_page": per_page, "page": page})
        if not batch:
            break
        results.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
        time.sleep(0.05)
    return results


def get_pr_files(pr_number: int, token: Optional[str]) -> list[dict]:
    return paginate(f"/repos/{REPO}/pulls/{pr_number}/files", token)


def get_merged_prs(limit: int, token: Optional[str]) -> list[dict]:
    """Fetch merged PRs newest-first."""
    prs = []
    page = 1
    while len(prs) < limit:
        batch = gh_get(
            f"/repos/{REPO}/pulls",
            token,
            {"state": "closed", "sort": "updated", "direction": "desc",
             "per_page": "100", "page": str(page)},
        )
        if not batch:
            break
        # Only keep actually merged PRs (closed but not merged have merged_at=None)
        prs.extend(pr for pr in batch if pr.get("merged_at"))
        if len(batch) < 100:
            break
        page += 1
        time.sleep(0.05)
    return prs[:limit]


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

_ISSUE_RE = re.compile(r"(?:closes|fixes|resolves)\s+#(\d+)", re.IGNORECASE)

# Matches Python test file paths mentioned in PR descriptions, e.g.:
#   pytest test/quantization/test_fp8.py
#   python -m pytest test/foo/test_bar.py::TestClass
#   ran `test/prototype/test_x.py` manually
_TEST_PATH_RE = re.compile(r"(?:[\w./\-]+/)?test[\w./\-]*\.py", re.IGNORECASE)


def extract_test_files_from_description(body: str) -> list[str]:
    """
    Parse a PR body for test file paths the author mentioned running.
    Returns deduplicated paths that look like repo-relative test files.
    """
    if not body:
        return []
    seen = set()
    results = []
    for match in _TEST_PATH_RE.findall(body):
        # Strip leading ./ and any pytest node ids (::ClassName::test_name)
        path = match.split("::")[0].lstrip("./")
        if path and path not in seen and Path(path).name != "test_distributed.py":
            seen.add(path)
            results.append(path)
    return results

def extract_linked_issues(body: str) -> list[int]:
    if not body:
        return []
    return [int(m) for m in _ISSUE_RE.findall(body)]


def fetch_issue_body(issue_number: int, token: Optional[str]) -> str:
    try:
        issue = gh_get(f"/repos/{REPO}/issues/{issue_number}", token)
        return issue.get("body") or ""
    except Exception:
        return ""


def fetch_file_at_commit(filename: str, commit_sha: str, token: Optional[str]) -> Optional[str]:
    """Fetch raw file contents at a specific commit via the Git Trees / Blobs API."""
    try:
        data = gh_get(f"/repos/{REPO}/contents/{filename}", token,
                      {"ref": commit_sha})
        if data.get("encoding") == "base64":
            import base64
            return base64.b64decode(data["content"]).decode(errors="replace")
        # Fall back to download_url for large files
        download_url = data.get("download_url")
        if download_url:
            req = urllib.request.Request(download_url)
            with urllib.request.urlopen(req) as resp:
                return resp.read().decode(errors="replace")
    except Exception:
        pass
    return None


def build_prompt(pr: dict, impl_files: list[str], test_files: list[str],
                 bench_files: list[str], reference_file_contents: dict[str, str],
                 token: Optional[str]) -> str:
    title = pr["title"]
    pr_body = (pr.get("body") or "").strip()

    # Fetch linked issue for extra context
    issue_context = ""
    for issue_num in extract_linked_issues(pr_body)[:1]:
        body = fetch_issue_body(issue_num, token)
        if body:
            issue_context = f"\n\nLinked issue #{issue_num}:\n{body[:1000]}"
            break

    impl_list  = "\n".join(f"  - {f}" for f in impl_files) or "  (see existing files)"
    bench_list = "\n".join(f"  - {f}" for f in bench_files)

    # Truncate PR body so prompt stays reasonable
    pr_body_excerpt = pr_body[:1500] + ("..." if len(pr_body) > 1500 else "")

    # Inline the reference test files so the agent has a concrete spec.
    # desc_test_files are pre-existing in the repo at base_sha, so their contents
    # are not in reference_file_contents — note this clearly rather than saying "unavailable".
    test_spec_sections = []
    for tf in test_files:
        contents = reference_file_contents.get(tf, "")
        if contents:
            excerpt = contents[:3000] + ("\n# ... (truncated)" if len(contents) > 3000 else "")
            test_spec_sections.append(f"#### `{tf}`\n```python\n{excerpt}\n```")
        else:
            test_spec_sections.append(f"#### `{tf}`\n(already present in the repo — read it directly)")
    test_spec = "\n\n".join(test_spec_sections)

    test_cmd = f"python -m pytest {' '.join(test_files)} -v"

    bench_section = ""
    if bench_files:
        bench_cmd = " && ".join(f"python {b}" for b in bench_files)
        bench_section = f"""
### Benchmarks (already placed in the repo for you)
The following benchmark(s) must run to completion without error:
{bench_list}

```bash
# Run benchmarks
{bench_cmd}
```
"""

    return f"""Implement the following feature in the torchao repository (pytorch/ao).

## Task: {title}

### Background
{pr_body_excerpt}{issue_context}

### What you need to do
Modify the relevant file(s) to implement this feature. The main files involved are:
{impl_list}

These files already exist in the repo — read them first to understand the current
structure before making changes.

### Acceptance tests (already placed in the repo for you)
The following test file(s) have been added to the repo as your spec.
Your implementation must make them pass:

{test_spec}
{bench_section}
### How to verify your work
```bash
# Run tests
{test_cmd}

# Be careful with distributed tests (names containing "distributed", "fsdp", "tp", "ep", "parallel").
# If the local environment lacks enough GPUs for them, skip those tests.
```
"""


# ---------------------------------------------------------------------------
# LLM-based PR quality filter
# ---------------------------------------------------------------------------

_llm_client = None

def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        if not _genai_available:
            raise ImportError("pip install google-genai to use LLM filtering")
        _llm_client = _genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _llm_client


_PR_DATA_BLOCK = """\
Here is the PR to evaluate:

Title: {title}

Body:
{body}

Changed files:
{files}

Respond with JSON only, no prose:
{{
  "accept": true or false,
  "reason": "one sentence"
}}
"""

LLM_FILTER_PROMPT = """\
You are helping curate a benchmark for ML systems programming. The benchmark uses \
merged PRs from torchao (a PyTorch quantization and optimization library) as tasks, \
where a model must re-implement what the PR does.
""" + _PR_DATA_BLOCK


def llm_filter(pr: dict, all_files: list[str], model: str = "gemini-3-flash-preview",
               filter_prompt: str = None) -> tuple[bool, str]:
    """
    Ask an LLM whether this PR makes a good benchmark task.
    Returns (accept, reason).
    Falls back to True if the google-generativeai client is unavailable.
    """
    if not _genai_available:
        return True, "LLM filter skipped (google-genai not installed)"

    body_excerpt = (pr.get("body") or "").strip()[:1000]
    files_excerpt = "\n".join(f"  {f}" for f in all_files[:40])
    if len(all_files) > 40:
        files_excerpt += f"\n  ... and {len(all_files) - 40} more"

    template = filter_prompt + "\n\n" + _PR_DATA_BLOCK if filter_prompt else LLM_FILTER_PROMPT
    prompt = template.format(
        title=pr["title"],
        body=body_excerpt or "(no description)",
        files=files_excerpt,
    )

    try:
        client = _get_llm_client()
        response = client.models.generate_content(model=model, contents=prompt)
        raw = response.text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        return bool(parsed["accept"]), parsed.get("reason", "")
    except Exception as e:
        # Don't let a filter failure drop a PR — log and let it through
        print(f"\n  LLM filter error for PR #{pr['number']}: {e}", file=sys.stderr)
        return True, f"filter error: {e}"


# ---------------------------------------------------------------------------
# Reference benchmark capture
# ---------------------------------------------------------------------------

def run_reference_benchmark(bench_files: list[str], merge_sha: str) -> dict[str, str]:
    """
    Create a temporary worktree at merge_sha, run each benchmark, and return
    a dict mapping filename -> stdout+stderr output.
    Requires a GPU on the generation machine.
    """
    outputs: dict[str, str] = {}
    worktree = REPO_ROOT / ".worktrees" / f"ref_{merge_sha[:12]}"
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), merge_sha],
            cwd=REPO_ROOT, check=True, capture_output=True,
        )
        for bench_file in bench_files:
            print(f"  Running reference benchmark: {bench_file}", file=sys.stderr)
            try:
                result = subprocess.run(
                    ["python", bench_file],
                    cwd=worktree,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    timeout=180,
                )
                outputs[bench_file] = result.stdout.decode(errors="replace")
            except subprocess.TimeoutExpired:
                outputs[bench_file] = "TIMEOUT"
                print(f"  Reference benchmark timed out: {bench_file}", file=sys.stderr)
            except Exception as e:
                print(f"  Reference benchmark error ({bench_file}): {e}", file=sys.stderr)
    except Exception as e:
        print(f"  Could not create reference worktree at {merge_sha[:12]}: {e}", file=sys.stderr)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=REPO_ROOT, check=False, capture_output=True,
        )
    return outputs


# ---------------------------------------------------------------------------
# Task assembly
# ---------------------------------------------------------------------------

def make_task(pr: dict, files: list[dict], token: Optional[str],
              use_llm_filter: bool = True, filter_model: str = "gemini-3-flash-preview",
              filter_prompt: str = None,
              title_include: list[str] = None, title_exclude: list[str] = None,
              capture_benchmarks: bool = False) -> Optional[dict]:
    """
    Return a task descriptor dict, or None if the PR doesn't qualify.

    Qualification criteria:
      - Has ≥1 Python implementation file added/modified
      - Has ≥1 Python test file added/modified
      - Benchmarks are included if present but are not required

    Environment setup (done by the task runner, not here):
      1. git checkout <base_commit>   — repo state before the PR
      2. Write the reference test and benchmark files (from <merge_commit>)
         into the working tree so the agent has a concrete spec.
      Implementation files remain at their pre-PR state; the agent must
      modify them to make the reference tests pass.
    """
    by_class: dict[str, list[dict]] = {"test": [], "benchmark": [], "implementation": []}
    for f in files:
        if not is_python(f["filename"]):
            continue
        cls = classify_file(f["filename"])
        by_class[cls].append(f)

    added_tests   = [f["filename"] for f in by_class["test"]
                     if f["status"] in ("added", "modified")
                     and Path(f["filename"]).name != "test_distributed.py"]
    added_benches = [f["filename"] for f in by_class["benchmark"]      if f["status"] in ("added", "modified")]
    impl_files    = [f["filename"] for f in by_class["implementation"] if f["status"] in ("added", "modified")]

    if not added_tests or not impl_files:
        missing = [name for name, lst in [("tests", added_tests), ("impl", impl_files)] if not lst]
        print(f"\n  Skipping PR #{pr['number']}: no {', '.join(missing)} files", file=sys.stderr)
        return None

    title_lower = pr["title"].lower()

    if title_exclude and any(w in title_lower for w in title_exclude):
        return None

    if title_include and not any(w in title_lower for w in title_include):
        return None
    
    # LLM quality filter (optional)
    if use_llm_filter:
        all_filenames = [f["filename"] for f in files]
        accept, reason = llm_filter(pr, all_filenames, model=filter_model, filter_prompt=filter_prompt)
        if not accept:
            print(f"\n  LLM rejected PR #{pr['number']}: {reason}", file=sys.stderr)
            return None
        print(f"\n  LLM accepted PR #{pr['number']}: {reason}", file=sys.stderr)

    pr_number = pr["number"]
    base_sha  = pr["base"]["sha"]
    merge_sha = pr["merge_commit_sha"]

    # Fetch the final contents of test and benchmark files at merge commit.
    # These are written into the task environment so the agent has a concrete spec.
    reference_file_contents: dict[str, str] = {}
    for filename in added_tests + added_benches:
        contents = fetch_file_at_commit(filename, merge_sha, token)
        if contents:
            reference_file_contents[filename] = contents
        time.sleep(0.05)

    # Parse PR description for pre-existing test files the author mentioned running.
    # Fetch their contents at base_sha and include in reference_file_contents so they
    # are explicitly written into the worktree and covered by the tampering check.
    desc_tests = []
    for path in extract_test_files_from_description(pr.get("body") or ""):
        if path in added_tests:
            continue  # already included
        contents = fetch_file_at_commit(path, base_sha, token)
        if contents is not None:
            desc_tests.append(path)
            reference_file_contents[path] = contents  # write + verify like any reference file
            print(f"  Found description-mentioned test: {path}", file=sys.stderr)
        time.sleep(0.05)

    all_test_files = added_tests + desc_tests

    # Optionally run benchmarks at merge_sha to capture reference output for scoring
    reference_benchmark_output: dict[str, str] = {}
    if capture_benchmarks and added_benches:
        reference_benchmark_output = run_reference_benchmark(added_benches, merge_sha)

    prompt = build_prompt(pr, impl_files, all_test_files, added_benches, reference_file_contents, token)

    return {
        "id": f"pr_{pr_number}",
        "pr_number": pr_number,
        "pr_url": pr["html_url"],
        "pr_title": pr["title"],
        "merged_at": pr["merged_at"],
        # ── Environment setup ─────────────────────────────────────────────
        # 1. git checkout base_commit
        # 2. write reference_file_contents into the working tree (tests + benchmarks
        #    from the merged PR — these don't exist yet at base_commit)
        "base_commit": base_sha,
        "merge_commit": merge_sha,
        # ── Files the agent must modify ───────────────────────────────────
        # These already exist at base_commit in their pre-PR state.
        "implementation_files": impl_files,
        # ── Oracle files (written into env before the agent runs) ─────────
        "test_files": added_tests,           # new/modified by the PR — written into worktree
        "desc_test_files": desc_tests,       # pre-existing, mentioned in PR description — also written + verified
        "benchmark_files": added_benches,
        "reference_file_contents": reference_file_contents,      # keyed by filename; includes both test_files and desc_test_files
        "reference_benchmark_output": reference_benchmark_output,  # keyed by bench filename; empty if not captured
        # ── Scoring commands ──────────────────────────────────────────────
        "score_commands": {
            "tests": f"python -m pytest {' '.join(all_test_files)} -v --tb=short --no-header -q",
            "benchmarks": " && ".join(f"python {b}" for b in added_benches) if added_benches else None,
        },
        # ── Prompt handed to mini-swe-agent ──────────────────────────────
        "prompt": prompt,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"),
                        help="GitHub token (or set GITHUB_TOKEN env var)")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max PRs to scan when searching for tasks (default: 500)")
    parser.add_argument("--tasks", type=int, default=100,
                        help="Number of tasks to generate (default: 100)")
    parser.add_argument("--out", default="tasks",
                        help="Output directory (default: ./tasks)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip PRs already saved in --out directory")
    parser.add_argument("--llm-filter", action="store_true",
                        help="Enable LLM quality filter to score PR suitability")
    parser.add_argument("--title-include", nargs="+", metavar="WORD", default=None,
                        help="Only include PRs whose title contains at least one of these words (unset = no filter)")
    parser.add_argument("--title-exclude", nargs="+", metavar="WORD", default=None,
                        help="Exclude PRs whose title contains any of these words (unset = no filter)")
    parser.add_argument("--filter-model", default=os.environ.get("GEMINI_FILTER_MODEL", "gemini-3-flash-preview"),
                        help="Gemini model for LLM quality filter (default: gemini-3-flash-preview, or GEMINI_FILTER_MODEL env var)")
    parser.add_argument("--filter-prompt", default=None, metavar="TEXT",
                        help="Custom instructions for the LLM filter (replaces the default preamble; "
                             "the PR title/body/files are always appended automatically)")
    parser.add_argument("--capture-benchmarks", action="store_true",
                        help="Run each benchmark at merge_sha to capture reference output for scoring (requires GPU)")
    args = parser.parse_args()

    if not args.token:
        print("Warning: no GITHUB_TOKEN set — unauthenticated requests are limited to 60/hr.",
              file=sys.stderr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load already-generated task IDs if resuming
    existing_ids: set[str] = set()
    if args.resume:
        existing_ids = {p.stem for p in out_dir.glob("*.json") if p.stem != "tasks"}
        print(f"Resume mode: {len(existing_ids)} tasks already saved.", file=sys.stderr)

    print(f"Fetching up to {args.limit} merged PRs from {REPO}...", file=sys.stderr)
    prs = get_merged_prs(args.limit, args.token)
    print(f"Fetched {len(prs)} merged PRs.", file=sys.stderr)

    tasks: list[dict] = []
    scanned = 0

    for pr in prs:
        if len(tasks) >= args.tasks:
            break

        pr_number = pr["number"]
        task_id = f"pr_{pr_number}"
        scanned += 1
        print(f"  [{scanned}/{len(prs)}] PR #{pr_number}: {pr['title'][:55]:<55} "
              f"tasks={len(tasks)}/{args.tasks}",
              end="\r", file=sys.stderr)

        if args.resume and task_id in existing_ids:
            # Re-load from disk to include in tasks.json
            saved = json.loads((out_dir / f"{task_id}.json").read_text())
            tasks.append(saved)
            continue

        try:
            files = get_pr_files(pr_number, args.token)
        except Exception as e:
            print(f"\n  Skipping PR #{pr_number}: {e}", file=sys.stderr)
            continue

        task = make_task(pr, files, args.token,
                         use_llm_filter=args.llm_filter or bool(args.filter_prompt),
                         filter_model=args.filter_model,
                         filter_prompt=args.filter_prompt,
                         title_include=args.title_include,
                         title_exclude=args.title_exclude,
                         capture_benchmarks=args.capture_benchmarks)
        if task is None:
            continue

        # Save individual task file
        task_path = out_dir / f"{task_id}.json"
        task_path.write_text(json.dumps(task, indent=2))
        tasks.append(task)

        time.sleep(0.1)  # be a polite API citizen

    print(f"\n\nGenerated {len(tasks)} tasks from {scanned} PRs scanned.", file=sys.stderr)

    # Write combined index
    index_path = out_dir / "tasks.json"
    index_path.write_text(json.dumps(tasks, indent=2))
    print(f"Wrote {index_path} and {len(tasks)} individual task files to {out_dir}/",
          file=sys.stderr)

    if len(tasks) < args.tasks:
        print(f"\nWarning: only found {len(tasks)} qualifying PRs out of {args.tasks} requested.",
              file=sys.stderr)
        print("Try increasing --limit to scan more PRs.", file=sys.stderr)

    # Print a quick summary table
    print(f"\n{'ID':<15} {'PR':<7} {'Tests':<7} {'Benches':<9} {'Title'}")
    print("-" * 90)
    for t in tasks[:20]:
        print(f"{t['id']:<15} #{t['pr_number']:<6} "
              f"{len(t['test_files']):<7} {len(t['benchmark_files']):<9} "
              f"{t['pr_title'][:45]}")
    if len(tasks) > 20:
        print(f"  ... and {len(tasks) - 20} more (see {index_path})")


if __name__ == "__main__":
    main()
