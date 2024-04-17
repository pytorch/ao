## Fused GaLore Adam (WIP)

### Various fused implementations of `Adam` update step per [Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)

This is an initial attempt at optimizing the update step of the `GaLore Adam` optimizer.

#### Overview

The `GaLore` `Adam` optimizer introduces additional ops to the traditional `adam` update step.

Specifically:

1.  `grad` is projected to low rank --> additional matmul
2.  `adam` states are updated with `grad` elementwise (same as `Adam` except in low-rank)
3.  normalized `grad` is projected to full rank --> additional matmul
4.  `params` are updated with the normalized full rank grad

#### Implementation

Various fusions were attempted across 2 kernel implementations:

- `Fused`
  - Steps 1 & 2 are fused: the `adam` state updates are loaded and updated (inplace) during the first `matmul`
  - Steps 3 & 4 are fused: the param update is folded as an epilogue into the second `matmul`
- `Hybrid`
  - Step 1 is performed using standard `torch matmul` (i.e., `cuBlas`)
  - Step 2 is fused as an elementwise kernel
  - Steps 3 & 4 per `Fused`

#### Performance

Below are benchmarks for various kernels:

- `torch` - reference `torch` implementation where each of the steps are implemented verbatim per above
- `hybrid` - see above
- `fused` - see above
- `compiled` - `torch` reference implementation compiled using `torch.compile` with `fullgraph=True` and `mode="max-autotune"`.

Configs for each benchmark are the `grad (param)` shape, `dtype` of `grad` and `adam` states, and `allow_tf32`, whether `torch` and `triton` matmuls are allowed to use `TF32` tensor cores (see `Discussion`).

`Grad shape`: `4096x4096`, `dtype`: `torch.float32`, `allow_tf32`: `False`

```
Median times (ms):
    rank     torch    hybrid     fused  compiled
0   32.0  0.560128  0.347136  0.505856  0.534528
1   64.0  0.627712  0.404480  0.600960  0.615424
2  128.0  0.825232  0.583168  0.985072  0.833536
3  256.0  1.378304  1.126400  1.489920  1.375232
4  512.0  2.286080  2.101760  2.969600  2.302976
```

`Grad shape`: `4096x4096`, `dtype`: `torch.float32`, `allow_tf32`: `True`

```
Median times (ms):
    rank     torch    hybrid     fused  compiled
0   32.0  0.540672  0.321536  0.316416  0.508928
1   64.0  0.612240  0.337728  0.345024  0.538624
2  128.0  0.640000  0.395264  0.393216  0.693248
3  256.0  0.777216  0.489472  0.548784  1.102848
4  512.0  1.216512  0.864256  0.960512  1.968128
```

`Grad shape`: `4096x11008`, `dtype`: `torch.float32`, `allow_tf32`: `False`

```
Median times (ms):
    rank     torch    hybrid     fused  compiled
0   32.0  1.538672  0.915456  0.835584  1.364032
1   64.0  1.546240  0.940032  1.022976  1.486848
2  128.0  2.116608  1.498112  1.613312  2.098176
3  256.0  3.423744  2.719744  2.881536  3.227136
4  512.0  5.499904  5.036544  5.450752  5.508096
```

`Grad shape`: `4096x11008`, `dtype`: `torch.float32`, `allow_tf32`: `True`

```
Median times (ms):
    rank     torch    hybrid     fused  compiled
0   32.0  1.413120  0.871424  0.817152  1.353184
1   64.0  1.489920  0.916480  0.854016  1.389568
2  128.0  1.679360  0.996352  1.005568  1.563648
3  256.0  2.152448  1.415168  1.470464  2.185216
4  512.0  3.210240  2.460672  2.580480  3.477504
```

##### Accuracy

Comparison to reference `torch` implementation:

```
Running with 4096 x 4096 grad (param) shape, GaLore orthogonal matrix [128, 4096], dtype torch.float32, and allow_tf32 True
Kernel: hybrid
Accuracy:
-> adam state - running grad mean:
  Max err: 0.000000 Relative err: 0.000001
-> adam state - running grad var:
  Max err: 0.000002 Relative err: 0.000002
-> params (after update):
  Max err: 0.000000 Relative err: 0.000001
```

```
Running with 4096 x 4096 grad (param) shape, GaLore orthogonal matrix [128, 4096], dtype torch.float32 and allow_tf32 False
Kernel: hybrid
Accuracy:
-> adam state - running grad mean:
  Max err: 0.000000 Relative err: 0.000000
-> adam state - running grad var:
  Max err: 0.000002 Relative err: 0.000002
-> params (after update):
  Max err: 0.000000 Relative err: 0.000000
```

```
Running with 4096 x 4096 grad (param) shape, GaLore orthogonal matrix [128, 4096], dtype torch.float32 and allow_tf32 True
Kernel: fused
Accuracy:
-> adam state - running grad mean:
  Max err: 0.000845 Relative err: 0.001152
-> adam state - running grad var:
  Max err: 0.000162 Relative err: 0.000161
-> params (after update):
  Max err: 0.000000 Relative err: 0.000001
```

```
Running with 4096 x 4096 grad (param) shape, GaLore orthogonal matrix [128, 4096], dtype torch.float32 and allow_tf32 False
Kernel: fused
Accuracy:
-> adam state - running grad mean:
Max err: 0.000003 Relative err: 0.000004
-> adam state - running grad var:
Max err: 0.000002 Relative err: 0.000002
-> params (after update):
Max err: 0.000000 Relative err: 0.000000
```

#### Discussion

##### Down Projection GEMM Shape

The motivation for the `hybrid` approach is the unconventional matrix shapes of the down projection (Step 1):

- The projection is always done such that the larger dimension of the `grad` matrix is maintained while other is projected to low rank per the `GaLore` algorithm
  - E.g., if `M >= N`, the GEMM is of shape (`M x N`) x (`N x rank`) = (`M x rank`), (`rank x M`) x (`M x N`) = (`rank x N`) otherwise
- Since `{M, N} >> rank` by definition, this results in a large reduction dimension relative to one of the output dimensions (output matrix is either fat or skinny)
- This does not fit cleanly into the `split-k / parallel reduction` `GEMM` paradigm which is more tailored for shapes where both output dims are smaller than the reduction dimension.
- Consequently, I had trouble finding an optimal kernel config using `triton` `autotuner` for the down projection step, despite tuning across many compute and io-bound configs (see `fused.triton_utils.kernels.matmul.py`).
- Benchmarking `triton`-tuned `matmul` against default `torch.matmul` for these shapes showed worse performance, for `torch.float32`

#### Effect of `TF32` tensor cores

`allow_tf32`: this has significant impact on relative performance of `triton` vs `torch` matmuls:

- Quick benchmarks of the downprojection `matmul` show that:
  - with `allow_tf32=True` for both, triton exhibits `~1.30x` performance improvement over `torch`.
  - with `allow_tf32=False`, performance of `triton` degrades significantly to `~.50x` of `torch`.

See this [`torch note`](https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere) for more details on this feature.

**Note**: This might be less of a concern given this incoming triton [PR](https://github.com/openai/triton/pull/3234), which implements a fast `TF32` trick that improves both performance and accuracy.

#### Repro

_Accuracy_

- Test accuracy of `torch` vs `hybrid` for `M=4096`, `N=4096`, `rank=128`, and `tf32` switched on:

  ```python
  pytest test/kernel/test_fused_kernels.py
  ```

_Benchmark_

- Benchmark across all kernels without `tf32`:

  ```python
  python benchmarks/bench_galore_fused_kernels.py
  ```

For additional benchmarking options:

```python
  python benchmarks/bench_galore_fused_kernels.py --help
```

#### Test Env

- GPU Device Props:
  - Name: `NVIDIA RTX A6000`
  - CC: `86`
  - Total_memory: `48676MB`
  - SM count: `84`
- Torch: `2.2.2`
- Triton: `2.2.0`
