# Float8 training benchmarking

The `float8_training_benchmark.sh` script in this directory can be used to launch a Llama3 8b training run with [torchtitan](https://github.com/pytorch/torchtitan) training run, and parse the logs to calculate the median tokens/sec and peak memory usage for you.

## Usage

Example: `TORCHTITAN_ROOT=${HOME}/torchtitan FLOAT8_RECIPE_WITH_BEST_SETTINGS=rowwise ./float8_training_benchmark.sh`

Training parameters can be configured via environment variables.

- Required:
    - `TORCHTITAN_ROOT`: Root directory of torchtitan in your local filesystem
- Optional:
    - `FLOAT8_RECIPE_WITH_BEST_SETTINGS`: "rowwise" or "tensorwise". Applies float8 training with the specified scaling recipe, as well as additional training configs which are optimal for that scaling recipe. See `float8_training_benchmark.sh` for more details.
    - `BATCH_SIZE`: Defaults to 1.
    - `STEPS`: Defaults to 100.

**NOTE**: `torch.compile` and FSDP2 are always used. Other forms of parallelism supported in torchtitan are not yet supported in this script.
