# Blockwise Quantization Implementation

## Overview

This directory contains the implementation of blockwise quantization introduced by DeepSeek. The method involves quantizing activations and weight matrices in blocks of 128x1 and 128x128, respectively.

## Quantization Process

### Activation Quantization
- Activations are quantized in blocks of size 128x1 using the FP8 format

### Weight Matrix Quantization
- Weights are quantized in blocks of size 128x128 using the FP8 format

## Kernel Implementation in Triton

- The kernel for blockwise quantization is implemented using Triton
- For now, the only supported types are: torch.float8_e4m3fn and torch.float8_e5m2

## Illustration

![Blockwise Quantization Illustration](https://arxiv.org/html/2412.19437v1/x7.png)

*Illustration of the blockwise quantization process.*

## Original Paper

For detailed motivations and technical specifications, please refer to the original paper:
- [DeepSeek Blockwise Quantization Paper](https://arxiv.org/html/2412.19437v1)
