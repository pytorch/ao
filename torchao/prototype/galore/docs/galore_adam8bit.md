## GaLore AdamW8bit Optimizer

### Overview

`GaLore` AdamW8bit optimizer utilizes `bitsandbytes` `AdamW8bit` optimizer to additionally quantize the optimizer states.

In addition to the additional ops introduced by `GaLore` to the standard `Adam` update step (see the `galore_adam.md` for details), additional dequantize / quantize steps are needed:

- one to to dequantize the quantized states for the state update
- after the states are updated, they need to quantized along and `quant_state` updated
- For `bitsandbytes` `AdamW8bit`, the `quant_state` consists of group-wise (`blocksize`) scaling factors.

The `bitsandbytes` 8bit optimizer is implemented in CUDA, with handcrafted logic for implementing each of these steps.

> The motivation for re-implementing this optimizer purely in `triton` / `torch` is to enable exploration of various fusion / optimization strategies that would be difficult with the current CUDA impl.

#### Quantization Algorithm

1. Weights are quantized in contiguous `blocksize` segments
2. Given tensor `M x N`, reshape to `-1 x blocksize`
3. Find columnwise `absmax` and normalize tensor by dividing by `absmax`
4. Reshape normalized tensor back to original shape
5. `bitsandbytes` then uses an `8-bit` [quantization code](https://github.com/TimDettmers/bitsandbytes/blob/76885a41df9e6c94b3f80b1c37374c8441b6933e/bitsandbytes/optim/optimizer.py#L146-L151), which can either be signed or unsigned -- signed for tracking `mean`, unsigned for tracking `var`.
6. The normalized tensor is then assigned to the code it is closest to:
   - E.g., given normalized value `.0412` and buckets `.0402` and `.0416`, it will be assigned the latter code.
7. **IMPORTANT**: This gives rise to a small number of edge-case errors when trying to reproduce `bitsandbytes` quantization
   - Specifically, if a normalized value falls directly between two codes there is a degree of indeterminism.
   - E.g., in the previous example, if the normalized value is `.0409`, it would be equidistant to the codes `.0402` and `.0416`.
   - See the assertions in the `test_galore_quant.py` unittest that checks that these are the only discrepancies arising from the triton implementation (run with `pytest -sv -k` flags to see the output from this test).

### bitsandbytes CUDA Source

- Adam[W]8bit [update step](https://github.com/TimDettmers/bitsandbytes/blob/fd9d072e02b74348004f197e686e168448883a9e/csrc/kernels.cu#L1770)
- Adam blockwise [quantization](https://github.com/TimDettmers/bitsandbytes/blob/fd9d072e02b74348004f197e686e168448883a9e/csrc/kernels.cu#L413) after update
- [Blockwise](https://github.com/TimDettmers/bitsandbytes/blob/fd9d072e02b74348004f197e686e168448883a9e/csrc/kernels.cu#L726) [Quantization](https://github.com/TimDettmers/bitsandbytes/blob/fd9d072e02b74348004f197e686e168448883a9e/csrc/kernels.cu#L339) kernel
