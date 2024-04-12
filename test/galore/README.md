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
python profile_memory_usage.py --help

usage: profile_memory_usage.py [-h] [-t] [-m] [-ns] [--optimizer {adamw,galore_adamw}] [--rank RANK] [--update_proj_gap UPDATE_PROJ_GAP]
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

After running the `profile_memory_usage`, the output directory (defaults to `profiler_out`) will have three types of files:

- `*.{json,html} - these are the memory trace exports of `torch.profiler`
  - the `html` contains the memory timeline plot
  - the `json` file contains the raw data for this plot, which can be analyzed to extract summary stats.
  - `galore_memory_analysis.py` along with `galore_memory_analysis_utils.py` demonstrate such analysis.
- `*.json.gz` - these are the complete `torch.profiler` traces which can be viewed using `perfetto`.

#### Preliminary Observations

- Memory Usage over Time

  - We can see a long delay between the first backwards step for `GaLoreAdamW` due to the calculation of the projection matrix (calls `torch.linalg.svd` on the `grad`).
  - To visualize, paste the following into a jupyter notebook (replacing the filenames with the those after running the profiler script):

  ```python
    adamW_html_trace = "./profiler_out/adamw_04-09-23.html"
    adamW8bit_html_trace = "./profiler_out/adamw8bit_04-11-01.html"
    galore_adamw_128_html_trace = "./profiler_out/galore_adamw-128-1.0-50_04-09-23.html"
    galore_adamw8bit_128_html_trace = "./profiler_out/galore_adamw8bit-128-1.0-50_04-11-01.html"

    plot_memory_timeline(adamW_html_trace)
    plot_memory_timeline(adamW8bit_html_trace)
    plot_memory_timeline(galore_adamw_128_html_trace)
    plot_memory_timeline(galore_adamw8bit_128_html_trace)
  ```

- Memory Usage Stats

  - Summary stats for memory usage by type as well as total across all types can be viewed by running the following in jupyter notebook, again replacing the respective filepaths:

  ```python
  adamW_trace = "./profiler_out/adamw_04-11-21-memory-timeline.json"
  adamW8bit_trace = "./profiler_out/adamw8bit_04-11-21-memory-timeline.json"
  galore_adamW_trace_128 = "./profiler_out/galore_adamw-128-1.0-50_04-11-21-memory-timeline.json"
  galore_adamW8bit_trace_128 = "./profiler_out/galore_adamw8bit-128-1.0-50_04-11-21-memory-timeline.json"

  adamW_df = create_mem_df(adamW_trace, units="MB")
  adamW8bit_df = create_mem_df(adamW8bit_trace, units="MB")
  galore_adamW_df_128 = create_mem_df(galore_adamW_trace_128, units="MB")
  galore_adamW8bit_df_128 = create_mem_df(galore_adamW8bit_trace_128, units="MB")

  show_memory_stats(adamW_df)
  show_memory_stats(adamW8bit_df)
  show_memory_stats(galore_adamW_df_128)
  show_memory_stats(galore_adamW8bit_df_128)
  ```

  The following are results from sample runs of `Llama1B` model config with the following optimizers (all units in MB):

- torch.optim.AdamW

  |        | Parameter | Optimizer_State | Input | Temporary | Activation | Gradient | Autograd_Detail | Unknown | Total    |
  | ------ | --------- | --------------- | ----- | --------- | ---------- | -------- | --------------- | ------- | -------- |
  | mean   | 5,108.2   | 8,330.3         | 0.0   | 0.6       | 2,249.5    | 2,113.8  | 19.0            | 197.3   | 18,018.8 |
  | min    | 5,108.2   | 0.0             | 0.0   | 0.0       | 0.0        | 0.0      | 0.0             | 0.0     | 5,108.2  |
  | median | 5,108.2   | 10,216.4        | 0.0   | 0.0       | 2,151.1    | 1,930.1  | 10.0            | 16.3    | 20,306.5 |
  | max    | 5,108.3   | 10,216.4        | 0.3   | 20.0      | 5,946.4    | 5,108.2  | 312.2           | 5,124.4 | 25,557.3 |

- GaLoreAdamW reference, rank 128

  |        | Parameter | Optimizer_State | Input | Temporary | Activation | Gradient | Autograd_Detail | Unknown | Total    |
  | ------ | --------- | --------------- | ----- | --------- | ---------- | -------- | --------------- | ------- | -------- |
  | mean   | 7,298.0   | 1,348.4         | 0.0   | 0.7       | 1,455.6    | 3,183.6  | 12.2            | 31.3    | 13,330.0 |
  | min    | 5,108.2   | 0.0             | 0.0   | 0.0       | 0.0        | 0.0      | 0.0             | 0.0     | 5,108.2  |
  | median | 7,796.2   | 1,576.7         | 0.0   | 0.0       | 545.4      | 3,898.2  | 0.0             | 26.2    | 14,422.8 |
  | max    | 8,047.2   | 1,576.7         | 0.3   | 42.7      | 5,960.0    | 5,108.2  | 312.2           | 518.2   | 15,349.2 |

- bitsandbytes AdamW8bit

  |        | Parameter | Optimizer_State | Input | Temporary | Activation | Gradient | Autograd_Detail | Unknown | Total    |
  | ------ | --------- | --------------- | ----- | --------- | ---------- | -------- | --------------- | ------- | -------- |
  | mean   | 5,108.2   | 2,047.4         | 0.0   | 0.7       | 2,390.0    | 1,925.2  | 20.1            | 20.3    | 11,511.9 |
  | min    | 5,108.2   | 0.0             | 0.0   | 0.0       | 0.0        | 0.0      | 0.0             | 0.0     | 5,108.2  |
  | median | 5,108.2   | 2,560.4         | 0.0   | 0.0       | 2,351.0    | 1,738.1  | 10.0            | 16.3    | 12,621.3 |
  | max    | 5,108.3   | 2,560.4         | 0.3   | 20.0      | 5,946.4    | 5,108.2  | 312.2           | 46.9    | 13,631.3 |

- GaLore AdamW8bit

  |        | Parameter | Optimizer_State | Input | Temporary | Activation | Gradient | Autograd_Detail | Unknown | Total    |
  | ------ | --------- | --------------- | ----- | --------- | ---------- | -------- | --------------- | ------- | -------- |
  | mean   | 4,971.0   | 334.7           | 0.1   | 0.8       | 1,644.0    | 2,130.9  | 13.8            | 2,360.3 | 11,455.6 |
  | min    | 500.4     | 0.0             | 0.0   | 0.0       | 0.0        | 0.0      | 0.0             | 0.0     | 5,108.2  |
  | median | 5,108.2   | 395.6           | 0.0   | 0.0       | 1,076.4    | 2,106.1  | 0.0             | 2,704.3 | 11,673.8 |
  | max    | 5,153.5   | 395.6           | 85.4  | 42.7      | 5,947.8    | 5,109.2  | 312.2           | 7,685.4 | 14,155.9 |

- The `optimizer state` is indeed smaller for the `GaLoreAdamW` optimizer.
- Interestingly, the `Parameter` sizes balloons in the `GaLore` optimizer, likely due to extra data copies. Admittedly, the implementation is only a reference (per original repo) and leaves much room for optimization.
- The memory usage is in terms of memory allocated, which we can confirm by printing the max cuda memory allocated vs reserved (which the profiler script prints automatically).
- The `Total` column shows the allocation stats across all categories across all sampled timepoints. (Should not be interpreted as the row-wise sums).

**NOTE**: The `json` output of the torch profiler memory trace is unlabeled. However, we can infer -- and confirm -- the labels by comparing the plots of the parsed dataframe with that of the direct `html` export of the profiler.

- For example, after creating the dataframes per above, the following will plot the raw data, which should roughly reproduce the direct `html` export from `torch.profiler`, albeit with different timescale:

```python
_ = adamW_df.plot(kind="area", stacked=True, ylabel="Memory (MB)" )
_ = adamW8bit_df.plot(kind="area", stacked=True, ylabel="Memory (MB)" )
_ = galore_adamW_df_128.plot(kind="area", stacked=True, ylabel="Memory (MB)" )
_ = galore_adamW8bit_df_128.plot(kind="area", stacked=True, ylabel="Memory (MB)" )
```
