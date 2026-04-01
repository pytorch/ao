# torchao RL Environment

Coding-agent evaluation environment built on the [pytorch/ao](https://github.com/pytorch/ao) repository.
Each task is derived from a merged PR: the agent must re-implement the PR's changes and make the included tests pass.

## Setup

```bash
pip install -r rl_env/rl-requirements.txt
export GITHUB_TOKEN=<your token>       # required for task generation
export ANTHROPIC_API_KEY=<your key>    # required for run_task.py (agent)
export GEMINI_API_KEY=<your key>       # required for LLM filter / benchmark scoring / review
```

---

## generate_tasks.py

Scans merged PRs and produces task JSON files.

```bash
# Basic — scan 500 PRs, generate up to 100 tasks
python rl_env/generate_tasks.py --token $GITHUB_TOKEN

# Filter by title keywords
python rl_env/generate_tasks.py \
  --title-include fp8 float8 triton \
  --title-exclude cuda mxfp8

# Enable LLM quality filter with a custom prompt
python rl_env/generate_tasks.py \
  --filter-prompt "Only accept PRs that add new fp8 quantization methods runnable on a 1xH100." \
  --filter-model gemini-2.0-flash

# Also capture reference benchmark output for scoring (requires GPU)
python rl_env/generate_tasks.py --capture-benchmarks

# Resume a previous run (skip already-saved tasks)
python rl_env/generate_tasks.py --resume --out tasks/

# All options
python rl_env/generate_tasks.py \
  --token $GITHUB_TOKEN \
  --limit 1000 \
  --tasks 100 \
  --out tasks/ \
  --resume \
  --title-include fp8 float8 \
  --title-exclude cuda cutedsl mxfp8 \
  --filter-prompt "..." \
  --filter-model gemini-2.0-flash \
  --capture-benchmarks
```

**Output:** `tasks/tasks.json` (index) and `tasks/pr_<number>.json` (one file per task).

---

## run_task.py

Runs the agent on one or more tasks and scores the results.

```bash
# Single task
python rl_env/run_task.py tasks/pr_4069.json

# Multiple tasks
python rl_env/run_task.py tasks/pr_4069.json tasks/pr_4100.json

# All tasks in a directory (runs sequentially)
python rl_env/run_task.py tasks/

# Specify model and cost limit
python rl_env/run_task.py tasks/ \
  --model anthropic/claude-sonnet-4-6 \
  --cost-limit-per-task 0.50

# Print the prompt and exit without running the agent
python rl_env/run_task.py tasks/pr_4069.json --dry-run

# Skip the agent and score whatever is already in the worktree
python rl_env/run_task.py tasks/pr_4069.json --score-only --keep-worktree

# Save results to a custom directory
python rl_env/run_task.py tasks/ --out results/sonnet/

# Skip the LLM implementation review (faster, no anti-cheat)
python rl_env/run_task.py tasks/ --no-review

# Use a specific Gemini model for benchmark scoring and review
python rl_env/run_task.py tasks/ \
  --benchmark-model gemini-2.0-flash \
  --review-model gemini-2.0-flash
```

**Output:** `results/<task_id>.json` (scores) and `results/<task_id>.traj.json` (agent trajectory).

---

## Scoring

| Component | Weight | Method |
|---|---|---|
| Tests | 80% | `pytest` pass rate |
| Benchmarks | 20% | LLM table comparison vs reference output (requires `--capture-benchmarks` at generation time; falls back to no-op) |
| Review multiplier | ×0–1 | LLM checks implementation diff for reward hacking (hardcoding, mocking, file deletion) |

```
final_score = (0.8 × test_score + 0.2 × bench_score) × review_multiplier
```

If no reference benchmark output was captured, benchmarks are a no-op and `final_score = test_score × review_multiplier`.

---

## Docker

```bash
# Build
docker build -f rl_env/Dockerfile -t torchao-rl-env .

# Dry-run (no GPU needed)
docker run --rm \
  -v $(pwd)/tasks:/workspace/ao/tasks \
  torchao-rl-env \
  python rl_env/run_task.py tasks/pr_4069.json --dry-run

# Full run
docker run --rm --gpus all \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -v $(pwd)/tasks:/workspace/ao/tasks \
  -v $(pwd)/results:/workspace/ao/results \
  torchao-rl-env \
  python rl_env/run_task.py tasks/ --cost-limit-per-task 0.50
```
