# Marlin QQQ

Marlin QQQ kernel is now compatible with GPUs for sm80 and above.
Marlin QQQ kernel and Marlin kernel mainly have the following differences:
1. Marlin QQQ kernel supports W4A8 mixed precision GEMM using INT8 Tensor Core, while the original Marlin kernel supports W4A16 mixed precision GEMM using FP16 Tensor Core.
2. Because the mma instruction requires that the data types of weight and activation be consistent, type conversion is required. Marlin QQQ needs to convert INT4 weight to INT8, while Marlin needs to convert INT4 weight to FP16.
3. Similar to W8A8, Marlin QQQ needs to dequant to FP16 before writing the final result because the calculation result is accumulated in INT32, while Marlin does not need this processing.

For more details about Marlin QQQ, please refer to [paper](https://arxiv.org/pdf/2406.09904).

Marlin QQQ implementation adapted from the two below sources:

* [QQQ](https://github.com/HandH1998/QQQ/tree/main)
* [vllm](https://github.com/vllm-project/vllm/tree/main)
