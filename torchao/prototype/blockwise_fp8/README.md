# Blockwise Quantization Implementation

## Overview

This directory contains the implementation of blockwise quantization introduced by DeepSeek. The method involves quantizing activations and weight matrices in blocks of 128x1 and 128x128, respectively. This approach aims to optimize the efficiency and performance of neural network computations.

## Quantization Process

### Activation Quantization
- Activations are quantized in blocks of size 128x1.
- This blockwise approach helps in reducing the memory footprint and computational load.

### Weight Matrix Quantization
- Weight matrices are quantized in blocks of size 128x128.
- The weights are quantized using the FP8 format, which balances precision and performance.

## Kernel Implementation in Triton

The kernel for blockwise quantization is implemented using Triton, a language designed for writing efficient GPU code. The Triton kernel handles the quantization process, ensuring that the operations are optimized for performance on modern GPUs.

## Illustration

![Blockwise Quantization Illustration](https://arxiv.org/html/2412.19437v1/x7.png)

*Illustration of the blockwise quantization process.*

## Original Paper

For detailed motivations and technical specifications, please refer to the original paper:
- [DeepSeek Blockwise Quantization Paper](https://arxiv.org/html/2412.19437v1)

