# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import sympy
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
        # based on experimental observation with sample large inputs
        "pct_achievable_gemm_tops_bf16": 0.69,
        "pct_achievable_gemm_tops_fp8": 0.78,
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
        "pct_achievable_gemm_tops_bf16": 0.69,
        "pct_achievable_gemm_tops_fp8": 0.775,
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
# TODO(future): audit this across different hardware and triton/non-triton
KERNEL_LAUNCH_OVERHEAD_SEC = 0.002 * 0.001


def get_tensor_memory_traffic_bytes(
    dim0,
    dim1,
    float8_recipe_name: Optional[str],
    mx_recipe_name: Optional[str],
    fuse_with_prev=False,
    recompute_in_bw=False,
) -> List[Union[sympy.Symbol, int]]:
    """
    Inputs: dim0 and dim1 (shape), recipes
    Outputs: list of read/write traffic byte counts, one for each kernel
    """
    # assumes input bf16, output f8
    numel = dim0 * dim1

    if float8_recipe_name == "tensorwise":
        if recompute_in_bw:
            # x_bf16 = ...
            # kernel 1:               x_bf16 -> max_abs_stage_1 -> tmp
            # kernel 2 (mem traffic not modeled): tmp -> max_abs_stage_2 -> max_abs
            # kernel 3 (fwd):         x_bf16, max_abs -> to_float8 -> x_fp8_dim0
            # kernel 4 (bwd):         x_bf16, max_abs -> to_float8 -> x_fp8_dim1
            if fuse_with_prev:
                kernel_1_rw = 0
            else:
                # kernel 1: read numel, write 0 (assume size(tmp) ~ 0)
                kernel_1_rw = BYTES_PER_EL_BF16 * numel
            # kernel 3: read in bf16, write twice in float8 (row-major and col-major)
            kernel_3_rw = BYTES_PER_EL_BF16 * numel + BYTES_PER_EL_FLOAT8 * numel
            kernel_4_rw = kernel_3_rw
            return [kernel_1_rw, 0, kernel_3_rw, kernel_4_rw]
        else:
            # x_bf16 = ...
            # kernel 1:               x_bf16 -> max_abs_stage_1 -> tmp
            # kernel 2 (mem traffic not modeled): tmp -> max_abs_stage_2 -> max_abs
            # kernel 3:               x_bf16, max_abs -> to_float8 -> x_fp8_dim0, x_fp8_dim1
            if fuse_with_prev:
                kernel_1_rw = 0
            else:
                # kernel 1: read numel, write 0 (assume size(tmp) ~ 0)
                kernel_1_rw = BYTES_PER_EL_BF16 * numel
            # kernel 3: read in bf16, write twice in float8 (row-major and col-major)
            kernel_3_rw = BYTES_PER_EL_BF16 * numel + 2 * BYTES_PER_EL_FLOAT8 * numel
            return [kernel_1_rw, 0, kernel_3_rw]

    elif float8_recipe_name == "rowwise":
        # x_bf16 = ...
        # kernel 1:               x_bf16 -> x_float8_dim0
        # kernel 2:               x_bf16 -> x_float8_dim1

        # assume that we can't fuse 1 and 2 because that would require loading
        # the entire tensor to shared memory
        # note that `recompute_in_bw` has no effect here

        if fuse_with_prev:
            # assume we can fuse one of the reads with previous op
            kernel_1_rw = 0 + BYTES_PER_EL_FLOAT8 * numel
        else:
            kernel_1_rw = BYTES_PER_EL_BF16 * numel + BYTES_PER_EL_FLOAT8 * numel

        kernel_2_rw = BYTES_PER_EL_BF16 * numel + BYTES_PER_EL_FLOAT8 * numel

        return [kernel_1_rw, kernel_2_rw]

    else:
        assert mx_recipe_name in ("mxfp8_emulated", "mxfp8_cutlass"), "unsupported"

        if recompute_in_bw:
            # x_bf16 = ...
            # kernel 1:               x_bf16 -> x_mxfp8_dim0
            # kernel 2:               x_bf16 -> x_mxfp8_dim1
            if fuse_with_prev:
                kernel_1_rw = 0 + BYTES_PER_EL_FLOAT8 * numel
            else:
                kernel_1_rw = BYTES_PER_EL_BF16 * numel + BYTES_PER_EL_FLOAT8 * numel
            kernel_2_rw = BYTES_PER_EL_BF16 * numel + BYTES_PER_EL_FLOAT8 * numel
            return [kernel_1_rw, kernel_2_rw]
        else:
            # x_bf16 = ...
            # kernel 1:               x_bf16 -> x_mxfp8_dim0, x_mxfp8_dim1
            if fuse_with_prev:
                kernel_1_rw = 0 + BYTES_PER_EL_FLOAT8 * numel * 2
            else:
                kernel_1_rw = (
                    BYTES_PER_EL_BF16 * numel + BYTES_PER_EL_FLOAT8 * numel * 2
                )
            return [kernel_1_rw]


def get_gemm_time_sympy(
    M: sympy.Symbol, K: sympy.Symbol, N: sympy.Symbol, dtype, mx_recipe_name
):
    # note: this function is currently not super accurate for small shapes:
    # when M,K,N <= 1k,1k,1k it undercounts by around 2x

    # compute bound
    specs = get_specs()
    gemm_ops = 2 * M * K * N + 2 * M * N * K + 2 * K * M * N
    if dtype is torch.bfloat16:
        peak_tops = specs["bf16_peak_tops"]
        compute_gemm_time_s = (
            gemm_ops / peak_tops / specs["pct_achievable_gemm_tops_bf16"]
        )
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        peak_tops = specs["fp8_peak_tops"]
        compute_gemm_time_s = (
            gemm_ops / peak_tops / specs["pct_achievable_gemm_tops_fp8"]
        )
    else:
        assert False, "unsupported"

    # memory bound
    num_reads = (M * K + K * N) + (K * N + N * M) + (N * M + M * K)
    num_writes = M * N + K * M + N * K

    if mx_recipe_name is not None:
        assert mx_recipe_name in ("mxfp8_emulated", "mxfp8_cutlass"), "unsupported"
        assert dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "unsupported"
        # adjust reads for MX scaling
        block_size = 32
        num_scale_reads = num_reads // block_size
        # note: e8m0 bytes per element is the same as for e4m3|e5m2
        num_reads = num_reads + num_scale_reads

    if dtype is torch.bfloat16:
        bytes_rw = num_reads * BYTES_PER_EL_BF16 + num_writes * BYTES_PER_EL_BF16
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # read in float8, output in bfloat16
        bytes_rw = num_reads * BYTES_PER_EL_FLOAT8 + num_writes * BYTES_PER_EL_BF16
    else:
        assert False, "unsupported"
    mem_gemm_time_s = (
        bytes_rw / specs["peak_mem_bw_bytes_sec"] / specs["pct_achievable_mem_bw"]
    )

    # TODO(future): this should be for each kernel and inside the max
    kernel_launch_overhead = 3 * KERNEL_LAUNCH_OVERHEAD_SEC

    return sympy.Max(compute_gemm_time_s, mem_gemm_time_s) + kernel_launch_overhead


def get_float8_mem_sympy(
    M,
    K,
    N,
    float8_recipe_name: Optional[str],
    mx_recipe_name: Optional[str],
    enable_fusion_modeling: bool,
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
        float8_recipe_name,
        mx_recipe_name,
        fuse_with_prev=enable_fusion_modeling,
        recompute_in_bw=False,
    )
    fwd_fp8_weight_mem = get_tensor_memory_traffic_bytes(
        K,
        N,
        float8_recipe_name,
        mx_recipe_name,
        fuse_with_prev=False,
        recompute_in_bw=True,
    )
    fwd_fp8_total_mem = [*fwd_fp8_input_mem, *fwd_fp8_weight_mem]

    #
    # backward - grad_input
    #
    gi_fp8_grad_output_mem = get_tensor_memory_traffic_bytes(
        M,
        N,
        float8_recipe_name,
        mx_recipe_name,
        fuse_with_prev=enable_fusion_modeling,
        recompute_in_bw=False,
    )
    # already casted, assuming that we save weight from fw to bw
    # TODO: model this if FSDP float8 all-gather is on
    # TODO: model this if we don't save weight from fw to bw, and recompute instead
    gi_fp8_weight_mem = [0]

    #
    # backward - grad_weight
    #
    # TODO: model this if we don't save fp8 input from fw to bw
    gw_fp8_input_t_mem = [0]  # already casted
    # this should be always 0
    gw_fp8_grad_output_mem = [0]  # already casted

    bwd_fp8_total_mem = [
        *gi_fp8_grad_output_mem,
        *gi_fp8_weight_mem,
        *gw_fp8_input_t_mem,
        *gw_fp8_grad_output_mem,
    ]
    # list of bytes
    fp8_total_mem = [*fwd_fp8_total_mem, *bwd_fp8_total_mem]

    # list of seconds
    fp8_mem_time_s = [
        x / specs["peak_mem_bw_bytes_sec"] / specs["pct_achievable_mem_bw"]
        for x in fp8_total_mem
    ]

    # take max of kernel_overhead, r/w time
    fp8_mem_time_s = [sympy.Max(x, KERNEL_LAUNCH_OVERHEAD_SEC) for x in fp8_mem_time_s]

    # reduce to single expression
    res = 0
    for x in fp8_mem_time_s:
        res = res + x

    return res
