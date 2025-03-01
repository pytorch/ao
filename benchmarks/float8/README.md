# Float8 training benchmarking

The `float8_training_benchmark.sh` script in this directory can be used to launch a Llama3 8b training run with [torchtitan](https://github.com/pytorch/torchtitan) training run, and parse the logs to calculate the median tokens/sec and peak memory usage for you.

## Usage

Example: `TORCHTITAN_ROOT=${HOME}/torchtitan FLOAT8_RECIPE=rowwise ./float8_training_benchmark.sh`

Training parameters can be configured via environment variables.

- Required:
    - `TORCHTITAN_ROOT`
- Optional:
    - `RECIPE`: rowwise|tensorwise. defaults to tensorwise.
    - `BATCH_SIZE`: defaults to 1.
    - `STEPS`: defaults to 100.

**NOTE**: `torch.compile` and FSDP2 are always used. Other forms of parallelism supported in torchtitan are not yet supported in this script.
