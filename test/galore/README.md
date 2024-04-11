### GaLore Memory Profiler

Tests memory usage of `GaLore` optimizers.

Uses `torch.profiler` under the hood with additional options for `nsys`, [`torch.cuda.memory`](https://pytorch.org/docs/stable/torch_cuda_memory.html) analyses.

Runs an untrained Llama model with configs for various model sizes (see `configs`) from the original GaLore [repo](https://github.com/jiaweizzhao/GaLore/tree/master/configs) on a sample batch of data for a configurable set of iterations.

The results of the profiler are saved and can be analyzed using the provided notebook.

#### Examples

Run memory profiler with `torch.optim.AdamW`

```
python galore_mem_prof.py -t --optimizer=adamw
```

Run profiler with `GaLoreAdamW` reference implementation with rank 128

```
python galore_mem_prof.py -t --optimizer=galore_adamw --rank=128
```

More options

```
python test_memory_usage.py --help

usage: test_memory_usage.py [-h] [-t] [-m] [-ns] [--optimizer {adamw,galore_adamw}] [--rank RANK] [--update_proj_gap UPDATE_PROJ_GAP]
                            [--galore_scale GALORE_SCALE] [--wait_steps WAIT_STEPS] [--warmup_steps WARMUP_STEPS] [--profiler_steps PROFILER_STEPS]
                            [--max_steps MAX_STEPS] [--model_config MODEL_CONFIG] [--data_path DATA_PATH] [--output_dir OUTPUT_DIR] [-lr LEARNING_RATE]
                            [--weight_decay WEIGHT_DECAY] [--seed SEED]

options:
  -h, --help            show this help message and exit
  -t, --torch_profiler  Enable torch profiler (default: False)
  -m, --torch_memory_snapshot
                        Enable torch memory snapshot (default: False)
  -ns, --nsys_profiler  Enable nsys profiling context managerSurrounds training loop with cudaProfilerApi.{Start,Stop} (default: False)
  --optimizer {adamw,galore_adamw}
                        Which optimizer to use (default: adamw)
  --rank RANK
  --update_proj_gap UPDATE_PROJ_GAP
  --galore_scale GALORE_SCALE
  --wait_steps WAIT_STEPS
                        Number of steps to run before starting torch profiler (default: 0)
  --warmup_steps WARMUP_STEPS
                        Number of warmup steps for torch profiler (default: 0)
  --profiler_steps PROFILER_STEPS
                        Number of active steps for torch profiler (default: 5)
  --max_steps MAX_STEPS
                        Max number of train steps to run.Total train steps will be min of `max_steps` and the sum of torch profiler steps (`wait_steps` +
                        `warmup_steps` + `profiler_steps`). (default: 100)
  --model_config MODEL_CONFIG
                        Path to Llama config file see `https://github.com/jiaweizzhao/GaLore/tree/master/configs` (default: ./configs/llama_100m.json)
  --data_path DATA_PATH
                        Path to sample batch (default: ./data/sample_batch.pt)
  --output_dir OUTPUT_DIR
                        Directory for profiler outputs (default: profiler_out)
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate (default: 0.001)
  --weight_decay WEIGHT_DECAY
                        Weight decay for AdamW (default: 0.01)
  --seed SEED           Random seed for torch (default: 0)
```

#### Analysis

After running the `test_memory_usage`, the output directory (defaults to `profiler_out`) will have three types of files:

- `*.{json,html} - these are the memory trace exports of `torch.profiler`
  - the `html` contains the memory timeline plot
  - the `json` file contains the raw data for this plot, which can be analyzed to extract summary stats.
  - `galore_memory_analysis.py` along with `galore_memory_analysis_utils.py` demonstrate such analysis.
- `*.json.gz` - these are the complete `torch.profiler` traces which can be viewed using `perfetto`.

#### Preliminary Observations

- Memory Usage over Time

  - We can see a long delay between the first backwards step for `GaLoreAdamW` due to the calculation of the projection matrix (calls `torch.linalg.svd` on the `grad`).

- Memory Usage Stats (all in `MB`)
  - torch.optim.AdamW

|        | Parameter | Optimizer_State | Input | Temporary | Activation | Gradient | Autograd_Detail | Unknown |
| ------ | --------- | --------------- | ----- | --------- | ---------- | -------- | --------------- | ------- |
| mean   | 381.9     | 628.1           | 0.0   | 0.2       | 356.0      | 162.8    | 6.6             | 29.5    |
| min    | 381.9     | 0.0             | 0.0   | 0.0       | 0.0        | 0.0      | 0.0             | 0.0     |
| median | 381.9     | 769.6           | 0.0   | 0.0       | 337.8      | 171.7    | 3.1             | 16.3    |
| max    | 382.0     | 769.6           | 0.3   | 6.7       | 1,338.1    | 395.7    | 312.9           | 402.8   |

- GaLoreAdamW reference, rank 128

|        | Parameter | Optimizer_State | Input | Temporary | Activation | Gradient | Autograd_Detail | Unknown |
| ------ | --------- | --------------- | ----- | --------- | ---------- | -------- | --------------- | ------- |
| mean   | 491.4     | 349.0           | 0.0   | 0.2       | 226.6      | 246.1    | 4.2             | 18.7    |
| min    | 381.9     | 0.0             | 0.0   | 0.0       | 0.0        | 0.0      | 0.0             | 0.0     |
| median | 516.3     | 403.6           | 0.0   | 0.0       | 75.7       | 272.8    | 0.0             | 18.1    |
| max    | 595.0     | 403.6           | 0.3   | 6.6       | 1,336.0    | 395.3    | 312.9           | 173.6   |

- bitsandbytes AdamW8bit

|        | Parameter | Optimizer_State | Input | Temporary | Activation | Gradient | Autograd_Detail | Unknown |
| ------ | --------- | --------------- | ----- | --------- | ---------- | -------- | --------------- | ------- |
| mean   | 381.9     | 157.3           | 0.0   | 0.2       | 376.0      | 149.4    | 6.9             | 17.1    |
| min    | 381.9     | 0.0             | 0.0   | 0.0       | 0.0        | 0.0      | 0.0             | 0.0     |
| median | 381.9     | 196.8           | 0.0   | 0.0       | 367.5      | 157.4    | 3.1             | 16.3    |
| max    | 382.0     | 196.8           | 0.3   | 6.4       | 1,336.8    | 395.3    | 312.9           | 25.8    |

- GaLore AdamW8bit

|        | Parameter | Optimizer_State | Input | Temporary | Activation | Gradient | Autograd_Detail | Unknown |
| ------ | --------- | --------------- | ----- | --------- | ---------- | -------- | --------------- | ------- |
| mean   | 378.0     | 87.9            | 0.0   | 0.2       | 260.7      | 186.1    | 4.8             | 142.7   |
| min    | 156.8     | 0.0             | 0.0   | 0.0       | 0.0        | 0.0      | 0.0             | 0.0     |
| median | 383.9     | 102.9           | 0.0   | 0.0       | 164.2      | 210.5    | 0.0             | 157.4   |
| max    | 392.1     | 102.9           | 0.3   | 6.8       | 1,337.7    | 395.6    | 312.9           | 438.5   |

- The `optimizer state` is indeed smaller for the `GaLoreAdamW` optimizer. Interestingly, the `Parameter` sizes balloons in the `GaLore` optimizer, likely due to extra data copies. Admittedly, the implementation is only a reference (per original repo) and leaves much room for optimization.

**NOTE**: The `json` output of the torch profiler memory trace is unlabeled. However, we can infer -- and confirm -- the labels by comparing the plots of the parsed dataframe with that of the direct `html` export of the profiler.
