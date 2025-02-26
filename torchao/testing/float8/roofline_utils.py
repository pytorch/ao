# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch

BYTES_PER_EL_FLOAT8 = 1
BYTES_PER_EL_BF16 = 2

gpu_name_to_specs = {
    "NVIDIA H100": {
        # https://www.nvidia.com/en-us/data-center/h100/, divide by 2 because no sparsity
        "bf16_peak_tops": 989e12,
        "fp8_peak_tops": 1979e12,
        # 2.4 TB per second, custom to Meta's H100 variant
        "peak_mem_bw_bytes_sec": 2.4e12,
        # based on quick experimental observation with sample large inputs
        "pct_achievable_gemm_tops": 0.6,
        # based on previous experience looking at pointwise triton kernels with large inputs,
        # which would hit about 2.2k GBPS on Meta's H100 variant
        "pct_achievable_mem_bw": 0.92,
    },
    "NVIDIA B200": {
        # https://resources.nvidia.com/en-us-blackwell-architecture, page 19,
        # divide by 2 because no sparsity
        "bf16_peak_tops": 2.25e15,
        "fp8_peak_tops": 4.5e15,
        "fp4_peak_tops": 9.0e15,
        # https://resources.nvidia.com/en-us-blackwell-architecture, page 20
        # 8.0 TB per second
        "peak_mem_bw_bytes_sec": 8.0e12,
        # for now, copy over from H100
        # TODO(future): measure once we have the hardware
        "pct_achievable_gemm_tops": 0.6,
        # for now, copy over from H100
        # TODO(future): measure once we have the hardware
        "pct_achievable_mem_bw": 0.92,
    },
    # TODO(future): more GPU names
}


def get_specs():
    gpu_name = torch.cuda.get_device_name(0)
    return gpu_name_to_specs[gpu_name]


# Source: run a triton kernel with a single element read/write on an H100 and
# measure GPU time from the trace
TRITON_KERNEL_1_ELEMENT_TIME_SEC = 0.002 * 0.001


def get_tensor_memory_traffic_bytes(
    dim0,
    dim1,
    fuse_with_prev=False,
    model_torch_compile_limitations=False,
):
    # assumes input bf16, output f8
    numel = dim0 * dim1

    # x_bf16 = ...
    # kernel 1:               x_bf16 -> max_abs_stage_1 -> tmp
    # kernel 2 (not modeled): tmp -> max_abs_stage_2 -> max_abs
    # kernel 3:               x_bf16, max_abs -> to_float8 -> x_fp8

    if fuse_with_prev:
        kernel_1_rw = 0
    else:
        # kernel 1: read numel, write 0 (assume size(tmp) ~ 0)
        kernel_1_rw = BYTES_PER_EL_BF16 * numel

    # kernel 3: read in bf16, write twice in float8 (row-major and col-major)
    kernel_3_rw = BYTES_PER_EL_BF16 * numel + 2 * BYTES_PER_EL_FLOAT8 * numel

    if model_torch_compile_limitations:
        # today, the kernel to do cast_to_fp8_row_major_and_col_major(input_bf16, ...)
        # has an extra memory read of the input in fp8
        # context: https://github.com/pytorch/pytorch/issues/130015
        tc_adjustment = numel * BYTES_PER_EL_FLOAT8
    else:
        tc_adjustment = 0

    return kernel_1_rw + kernel_3_rw + tc_adjustment


def get_gemm_time_sympy(M, K, N, dtype):
    specs = get_specs()
    gemm_ops = 2 * M * K * N + 2 * M * N * K + 2 * K * M * N
    if dtype is torch.bfloat16:
        peak_tops = specs["bf16_peak_tops"]
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        peak_tops = specs["fp8_peak_tops"]
    gemm_time_s = gemm_ops / peak_tops / specs["pct_achievable_gemm_tops"]
    return gemm_time_s


def get_float8_mem_sympy(
    M,
    K,
    N,
    model_torch_compile_limitations: bool = False,
):
    specs = get_specs()

    # there are three gemms in the fwd/bwd of a linear:
    #
    # input @ weight_t = output
    # MxK @ KxN => MxN
    #
    # grad_output @ weight = grad_input
    # MxN @ NxK => MxK
    #
    # input_t @ grad_output = grad_weight
    # KxM @ MxN => KxN

    #
    # forward - output
    #
    fwd_fp8_input_mem = get_tensor_memory_traffic_bytes(
        M,
        K,
        fuse_with_prev=True,
        model_torch_compile_limitations=model_torch_compile_limitations,
    )
    fwd_fp8_weight_mem = get_tensor_memory_traffic_bytes(
        K,
        N,
        fuse_with_prev=False,
        model_torch_compile_limitations=model_torch_compile_limitations,
    )
    fwd_fp8_total_mem = fwd_fp8_input_mem + fwd_fp8_weight_mem

    #
    # backward - grad_input
    #
    gi_fp8_grad_output_mem = get_tensor_memory_traffic_bytes(
        M,
        N,
        fuse_with_prev=True,
        model_torch_compile_limitations=model_torch_compile_limitations,
    )
    # already casted, assuming that we save weight from fw to bw
    # TODO: model this if FSDP float8 all-gather is on
    # TODO: model this if we don't save weight from fw to bw, and recompute instead
    gi_fp8_weight_mem = 0

    #
    # backward - grad_weight
    #
    # TODO: model this if we don't save fp8 input from fw to bw
    gw_fp8_input_t_mem = 0  # already casted
    # this should be always 0
    gw_fp8_grad_output_mem = 0  # already casted

    bwd_fp8_total_mem = (
        gi_fp8_grad_output_mem
        + gi_fp8_weight_mem
        + gw_fp8_input_t_mem
        + gw_fp8_grad_output_mem
    )
    fp8_total_mem = fwd_fp8_total_mem + bwd_fp8_total_mem
    fp8_mem_time_s = (
        fp8_total_mem / specs["peak_mem_bw_bytes_sec"] / specs["pct_achievable_mem_bw"]
    )

    # Adjust final estimate for small kernel launches
    # note that we do this adjustment here because we are assuming a minimal
    # kernel overhead in the units of seconds, and the per-gemm-input memory
    # estimations are in the units of bytes.
    num_extra_kernels = 0
    # second stage of max-abs reduction for input
    num_extra_kernels += 1
    # second stage of max-abs reduction for weight
    num_extra_kernels += 1
    # second stage of max-abs reduction for grad_output
    num_extra_kernels += 1

    extra_kernel_overhead_s = num_extra_kernels * TRITON_KERNEL_1_ELEMENT_TIME_SEC

    return fp8_mem_time_s + extra_kernel_overhead_s
