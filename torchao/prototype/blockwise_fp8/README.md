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

## Benchmarks

Below are performance benchmarks measuring FP8 blockwise GEMM latency against fp16 on a single H100 GPU. 
These benchmarks can be reproduced using this [benchmarking script](https://github.com/pytorch/ao/blob/main/benchmarks/benchmark_blockwise_scaled_linear_triton.py).

|    m |     k |     n |   block_size | dtype               |   fp16_latency (ms) |   blockwise_latency (ms) |   blockwise_speedup |
|-----:|------:|------:|-------------:|:--------------------|--------------------:|-------------------------:|--------------------:|
|    1 |  8192 |  8192 |          128 | torch.float8_e4m3fn |              83.744 |                   52.224 |            1.60355  |
|    1 |  8192 | 10240 |          128 | torch.float8_e4m3fn |              99.52  |                   61.12  |            1.62827  |
|    1 |  8192 | 57344 |          128 | torch.float8_e4m3fn |             436.608 |                  234     |            1.86585  |
|    1 | 28672 |  8192 |          128 | torch.float8_e4m3fn |             233.568 |                  131.168 |            1.78068  |
|    1 |  8192 |  8192 |          128 | torch.float8_e5m2   |              84.896 |                   52.736 |            1.60983  |
|    1 |  8192 | 10240 |          128 | torch.float8_e5m2   |             100.224 |                   60.96  |            1.64409  |
|    1 |  8192 | 57344 |          128 | torch.float8_e5m2   |             441.152 |                  233.968 |            1.88552  |
|    1 | 28672 |  8192 |          128 | torch.float8_e5m2   |             233.28  |                  130.816 |            1.78327  |
|    2 |  8192 |  8192 |          128 | torch.float8_e4m3fn |              83.392 |                   53.664 |            1.55397  |
|    2 |  8192 | 10240 |          128 | torch.float8_e4m3fn |             100.192 |                   61.632 |            1.62565  |
|    2 |  8192 | 57344 |          128 | torch.float8_e4m3fn |             432.384 |                  233.664 |            1.85045  |
|    2 | 28672 |  8192 |          128 | torch.float8_e4m3fn |             233.648 |                  133.6   |            1.74886  |
|    2 |  8192 |  8192 |          128 | torch.float8_e5m2   |              83.232 |                   53.6   |            1.55284  |
|    2 |  8192 | 10240 |          128 | torch.float8_e5m2   |             100.608 |                   61.664 |            1.63155  |
|    2 |  8192 | 57344 |          128 | torch.float8_e5m2   |             432.32  |                  235.152 |            1.83847  |
|    2 | 28672 |  8192 |          128 | torch.float8_e5m2   |             233.824 |                  136.256 |            1.71606  |
|    4 |  8192 |  8192 |          128 | torch.float8_e4m3fn |              84.16  |                   52.928 |            1.59008  |
|    4 |  8192 | 10240 |          128 | torch.float8_e4m3fn |             100.544 |                   61.728 |            1.62882  |
|    4 |  8192 | 57344 |          128 | torch.float8_e4m3fn |             432.768 |                  234.944 |            1.842    |
|    4 | 28672 |  8192 |          128 | torch.float8_e4m3fn |             234.432 |                  134.432 |            1.74387  |
|    4 |  8192 |  8192 |          128 | torch.float8_e5m2   |              83.872 |                   53.408 |            1.5704   |
|    4 |  8192 | 10240 |          128 | torch.float8_e5m2   |              99.84  |                   62.24  |            1.60411  |
|    4 |  8192 | 57344 |          128 | torch.float8_e5m2   |             433.376 |                  238.272 |            1.81883  |
|    4 | 28672 |  8192 |          128 | torch.float8_e5m2   |             235.584 |                  134.08  |            1.75704  |
|    8 |  8192 |  8192 |          128 | torch.float8_e4m3fn |              83.648 |                   53.472 |            1.56433  |
|    8 |  8192 | 10240 |          128 | torch.float8_e4m3fn |             100.704 |                   62.432 |            1.61302  |
|    8 |  8192 | 57344 |          128 | torch.float8_e4m3fn |             439.104 |                  238.208 |            1.84336  |
|    8 | 28672 |  8192 |          128 | torch.float8_e4m3fn |             234.272 |                  135.072 |            1.73442  |
|    8 |  8192 |  8192 |          128 | torch.float8_e5m2   |              84.128 |                   53.728 |            1.56581  |
|    8 |  8192 | 10240 |          128 | torch.float8_e5m2   |             100.512 |                   62.976 |            1.59604  |
|    8 |  8192 | 57344 |          128 | torch.float8_e5m2   |             439.36  |                  238.496 |            1.84221  |
|    8 | 28672 |  8192 |          128 | torch.float8_e5m2   |             235.04  |                  135.424 |            1.73559  |
|   16 |  8192 |  8192 |          128 | torch.float8_e4m3fn |              83.808 |                   53.664 |            1.56172  |
|   16 |  8192 | 10240 |          128 | torch.float8_e4m3fn |              99.584 |                   63.104 |            1.57809  |
|   16 |  8192 | 57344 |          128 | torch.float8_e4m3fn |             444     |                  244.192 |            1.81824  |
|   16 | 28672 |  8192 |          128 | torch.float8_e4m3fn |             235.52  |                  133.792 |            1.76034  |
|   16 |  8192 |  8192 |          128 | torch.float8_e5m2   |              83.488 |                   53.568 |            1.55854  |
|   16 |  8192 | 10240 |          128 | torch.float8_e5m2   |             101.216 |                   63.232 |            1.60071  |
|   16 |  8192 | 57344 |          128 | torch.float8_e5m2   |             444.608 |                  245.936 |            1.80782  |
|   16 | 28672 |  8192 |          128 | torch.float8_e5m2   |             235.36  |                  133.152 |            1.7676   |
|   32 |  8192 |  8192 |          128 | torch.float8_e4m3fn |              83.872 |                   53.312 |            1.57323  |
|   32 |  8192 | 10240 |          128 | torch.float8_e4m3fn |             102.688 |                   63.264 |            1.62317  |
|   32 |  8192 | 57344 |          128 | torch.float8_e4m3fn |             441.792 |                  243.04  |            1.81777  |
|   32 | 28672 |  8192 |          128 | torch.float8_e4m3fn |             237.12  |                  133.632 |            1.77443  |
|   32 |  8192 |  8192 |          128 | torch.float8_e5m2   |              86.08  |                   53.216 |            1.61756  |
|   32 |  8192 | 10240 |          128 | torch.float8_e5m2   |             102.032 |                   63.2   |            1.61443  |
|   32 |  8192 | 57344 |          128 | torch.float8_e5m2   |             439.168 |                  245.184 |            1.79118  |
|   32 | 28672 |  8192 |          128 | torch.float8_e5m2   |             238.016 |                  134.336 |            1.7718   |
|   64 |  8192 |  8192 |          128 | torch.float8_e4m3fn |              85.888 |                   53.632 |            1.60143  |
|   64 |  8192 | 10240 |          128 | torch.float8_e4m3fn |              93.632 |                   63.936 |            1.46446  |
|   64 |  8192 | 57344 |          128 | torch.float8_e4m3fn |             471.44  |                  245.2   |            1.92268  |
|   64 | 28672 |  8192 |          128 | torch.float8_e4m3fn |             240     |                  137.424 |            1.74642  |
|   64 |  8192 |  8192 |          128 | torch.float8_e5m2   |              85.984 |                   54.016 |            1.59182  |
|   64 |  8192 | 10240 |          128 | torch.float8_e5m2   |              93.376 |                   64.032 |            1.45827  |
|   64 |  8192 | 57344 |          128 | torch.float8_e5m2   |             471.36  |                  244.576 |            1.92725  |
|   64 | 28672 |  8192 |          128 | torch.float8_e5m2   |             242.4   |                  136.096 |            1.7811   |
|  128 |  8192 |  8192 |          128 | torch.float8_e4m3fn |              91.008 |                   57.184 |            1.59149  |
|  128 |  8192 | 10240 |          128 | torch.float8_e4m3fn |              96.608 |                   67.936 |            1.42204  |
|  128 |  8192 | 57344 |          128 | torch.float8_e4m3fn |             449.6   |                  292.48  |            1.5372   |
|  128 | 28672 |  8192 |          128 | torch.float8_e4m3fn |             247.84  |                  147.232 |            1.68333  |
|  128 |  8192 |  8192 |          128 | torch.float8_e5m2   |              89.152 |                   57.248 |            1.55729  |
|  128 |  8192 | 10240 |          128 | torch.float8_e5m2   |              96.64  |                   68.784 |            1.40498  |
|  128 |  8192 | 57344 |          128 | torch.float8_e5m2   |             450.048 |                  284.16  |            1.58378  |
|  128 | 28672 |  8192 |          128 | torch.float8_e5m2   |             246.88  |                  148.064 |            1.66739  |
|  256 |  8192 |  8192 |          128 | torch.float8_e4m3fn |              85.984 |                   62.368 |            1.37866  |
|  256 |  8192 | 10240 |          128 | torch.float8_e4m3fn |             101.216 |                  104.896 |            0.964918 |
|  256 |  8192 | 57344 |          128 | torch.float8_e4m3fn |             477.984 |                  452.832 |            1.05554  |
|  256 | 28672 |  8192 |          128 | torch.float8_e4m3fn |             260.224 |                  215.392 |            1.20814  |
|  256 |  8192 |  8192 |          128 | torch.float8_e5m2   |              86.432 |                   62.048 |            1.39299  |
|  256 |  8192 | 10240 |          128 | torch.float8_e5m2   |             101.024 |                  103.904 |            0.972282 |
|  256 |  8192 | 57344 |          128 | torch.float8_e5m2   |             475.568 |                  433.792 |            1.0963   |
|  256 | 28672 |  8192 |          128 | torch.float8_e5m2   |             261.824 |                  207.968 |            1.25896  |
|  512 |  8192 |  8192 |          128 | torch.float8_e4m3fn |             117.952 |                  112.992 |            1.0439   |
|  512 |  8192 | 10240 |          128 | torch.float8_e4m3fn |             151.504 |                  166.08  |            0.912235 |
|  512 |  8192 | 57344 |          128 | torch.float8_e4m3fn |             836.848 |                  881.312 |            0.949548 |
|  512 | 28672 |  8192 |          128 | torch.float8_e4m3fn |             442.528 |                  402.464 |            1.09955  |
|  512 |  8192 |  8192 |          128 | torch.float8_e5m2   |             121.184 |                  114.592 |            1.05753  |
|  512 |  8192 | 10240 |          128 | torch.float8_e5m2   |             151.424 |                  163.296 |            0.927298 |
|  512 |  8192 | 57344 |          128 | torch.float8_e5m2   |             837.312 |                  873.664 |            0.958391 |
|  512 | 28672 |  8192 |          128 | torch.float8_e5m2   |             437.664 |                  400.928 |            1.09163  |
| 1024 |  8192 |  8192 |          128 | torch.float8_e4m3fn |             227.008 |                  224.384 |            1.01169  |
| 1024 |  8192 | 10240 |          128 | torch.float8_e4m3fn |             289.28  |                  283.872 |            1.01905  |
| 1024 |  8192 | 57344 |          128 | torch.float8_e4m3fn |            1672.13  |                 1673.34  |            0.999273 |
| 1024 | 28672 |  8192 |          128 | torch.float8_e4m3fn |             800     |                  769.152 |            1.04011  |
| 1024 |  8192 |  8192 |          128 | torch.float8_e5m2   |             224.48  |                  223.456 |            1.00458  |
| 1024 |  8192 | 10240 |          128 | torch.float8_e5m2   |             289.408 |                  283.424 |            1.02111  |
| 1024 |  8192 | 57344 |          128 | torch.float8_e5m2   |            1649.58  |                 1626.88  |            1.01396  |
| 1024 | 28672 |  8192 |          128 | torch.float8_e5m2   |             805.392 |                  768.416 |            1.04812  |
| 2048 |  8192 |  8192 |          128 | torch.float8_e4m3fn |             449.344 |                  458.272 |            0.980518 |
| 2048 |  8192 | 10240 |          128 | torch.float8_e4m3fn |             569.888 |                  586.224 |            0.972134 |
| 2048 |  8192 | 57344 |          128 | torch.float8_e4m3fn |            3275.84  |                 3251.9   |            1.00736  |
| 2048 | 28672 |  8192 |          128 | torch.float8_e4m3fn |            1614.37  |                 1555.68  |            1.03772  |
| 2048 |  8192 |  8192 |          128 | torch.float8_e5m2   |             450.624 |                  461.712 |            0.975985 |
| 2048 |  8192 | 10240 |          128 | torch.float8_e5m2   |             575.36  |                  582.016 |            0.988564 |
| 2048 |  8192 | 57344 |          128 | torch.float8_e5m2   |            3363.3   |                 3213.31  |            1.04668  |
| 2048 | 28672 |  8192 |          128 | torch.float8_e5m2   |            1574.32  |                 1525.66  |            1.03189  |
| 4096 |  8192 |  8192 |          128 | torch.float8_e4m3fn |             915.216 |                  964.592 |            0.948812 |
| 4096 |  8192 | 10240 |          128 | torch.float8_e4m3fn |            1157.18  |                 1196.42  |            0.967209 |
| 4096 |  8192 | 57344 |          128 | torch.float8_e4m3fn |            6409.98  |                 6638.3   |            0.965606 |
| 4096 | 28672 |  8192 |          128 | torch.float8_e4m3fn |            3173.76  |                 3247.23  |            0.977374 |
| 4096 |  8192 |  8192 |          128 | torch.float8_e5m2   |             898.432 |                  949.36  |            0.946355 |
| 4096 |  8192 | 10240 |          128 | torch.float8_e5m2   |            1170.62  |                 1188.45  |            0.985002 |
| 4096 |  8192 | 57344 |          128 | torch.float8_e5m2   |            6751.25  |                 6573.71  |            1.02701  |
| 4096 | 28672 |  8192 |          128 | torch.float8_e5m2   |            3155.9   |                 3179.38  |            0.992617 |
| 8192 |  8192 |  8192 |          128 | torch.float8_e4m3fn |            1868.64  |                 2022.27  |            0.92403  |
| 8192 |  8192 | 10240 |          128 | torch.float8_e4m3fn |            2336.26  |                 2621.18  |            0.891298 |
| 8192 |  8192 | 57344 |          128 | torch.float8_e4m3fn |           13004     |                13990.6   |            0.929482 |
| 8192 | 28672 |  8192 |          128 | torch.float8_e4m3fn |            6781.49  |                 6722.82  |            1.00873  |
| 8192 |  8192 |  8192 |          128 | torch.float8_e5m2   |            1865.25  |                 1983.23  |            0.940509 |
| 8192 |  8192 | 10240 |          128 | torch.float8_e5m2   |            2296.66  |                 2523.1   |            0.91025  |
| 8192 |  8192 | 57344 |          128 | torch.float8_e5m2   |           13170.9   |                14029.6   |            0.938792 |
| 8192 | 28672 |  8192 |          128 | torch.float8_e5m2   |            6688.51  |                 6699.65  |            0.998338 |

