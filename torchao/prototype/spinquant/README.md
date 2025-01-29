# SpinQuant

Re-implementation of SpinQuant based on the official code implementation (https://github.com/facebookresearch/SpinQuant).

## Usage

For optimal performance on CUDA GPUs, install the Fast Hadamard Transform package:

```shell
pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git
```

## Performance 

See https://github.com/pytorch/ao/pull/983 for Wikitext benchmark results.

Tested on:

- Llama-2-7b
- PyTorch 2.4.1
- NVIDIA A100
- CUDA 12.1

Without `torch.compile`:

| Configuration  | Average tokens/sec | Average Bandwidth (GB/s) | Peak Memory Usage (GB) | Model Size (GB) |
|----------------|--------------------|--------------------------|------------------------|-----------------|
| Baseline       | 27.33              | 361.21                   | 13.62                  | 13.21           |
| Spinquant (R4) | 23.01              | 304.10                   | 14.24                  | 13.22           |

With `torch.compile`:

| Configuration        | Average tokens/sec | Average Bandwidth (GB/s) | Peak Memory Usage (GB) | Model Size (GB) |
|----------------------|--------------------|--------------------------|------------------------|-----------------|
| Baseline             | 114.08             | 1507.58                  | 13.88                  | 13.21           |
| Spinquant (R4)       | 109.59             | 1448.61                  | 13.72                  | 13.22           |
| Spinquant (R1+R2+R4) | 109.64             | 1449.28                  | 14.90                  | 13.22           |


NB: R1 and R2 are fused into the linear weights before inference takes place, so it is expected that they do not lead to additional overhead at inference time.
