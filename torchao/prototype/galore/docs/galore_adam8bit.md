## Kernel Implementation Notes

### bitsandbytes blockwise quantization

## Algorithm

1. Weights are quantized in contiguous `blocksize` segments
2. Given tensor `M x N`, reshape to `-1 x blocksize`
3. Find columnwise `absmax` and normalize tensor by dividing by `absmax`
4. Reshape normalized tensor back to original shape
5. `bitsandbytes` then uses an `8-bit` quantization code (`F.create_dynamic_map`), which can either be signed or unsigned.
6. The normalized tensor is then assigned to the code it is closest to:
   - E.g., given normalized value `.0412` and buckets `.0402` and `.0416`, it will be assigned the latter code.
7. **IMPORTANT** This gives rise to a small number of edge-case errors when trying to reproduce `bitsandbytes` quantization
   - Specifically, if a normalized value falls directly between two codes there is a degree of indeterminism. E.g., in the previous example, if the normalized value is `.0409`, it would be equidistant to the codes `.0402` and `.0416`. See the assertions in the `test_galore_quant.py` unittest that checks that these are the only discrepancies arising from the triton implementation.

### bitsandbytes CUDA Source

- Adam[W]8bit [update step](https://github.com/TimDettmers/bitsandbytes/blob/fd9d072e02b74348004f197e686e168448883a9e/csrc/kernels.cu#L1770)
- Adam blockwise [quantization](https://github.com/TimDettmers/bitsandbytes/blob/fd9d072e02b74348004f197e686e168448883a9e/csrc/kernels.cu#L413) after update
- [Blockwise](https://github.com/TimDettmers/bitsandbytes/blob/fd9d072e02b74348004f197e686e168448883a9e/csrc/kernels.cu#L726) [Quantization](https://github.com/TimDettmers/bitsandbytes/blob/fd9d072e02b74348004f197e686e168448883a9e/csrc/kernels.cu#L339) kernel
