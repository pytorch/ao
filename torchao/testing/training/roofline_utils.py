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
        "pct_achievable_gemm_tops": 0.78,
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
        "pct_achievable_gemm_tops": 0.78,
        # for now, copy over from H100
        # TODO(future): measure once we have the hardware
        "pct_achievable_mem_bw": 0.92,
    },
    "AMD Instinct MI300X": {
        # https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf, page 1,
        "bf16_peak_tops": 1307e12,
        "fp8_peak_tops": 2614e12,
        # 5.3 TB per second
        "peak_mem_bw_bytes_sec": 5.3e12,
        # for now, copy over from H100
        # TODO(future): run measurement on hardware
        "pct_achievable_gemm_tops": 0.78,
        # for now, copy over from H100
        # TODO(future): run measurement on hardware
        "pct_achievable_mem_bw": 0.92,
    },
    "NVIDIA GeForce RTX 5090": {
        # https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
        "bf16_peak_tops": 209.5e12,
        "fp8_peak_tops": 419e12,
        "fp4_peak_tops": 1676e12,
        "peak_mem_bw_bytes_sec": 1.792e15,
    },
    # TODO(future): more GPU names
}


def get_specs(gpu_name: Optional[str] = None):
    if gpu_name is None:
        gpu_name = torch.cuda.get_device_name(0)
    return gpu_name_to_specs[gpu_name]


# Source: run a triton kernel with a single element read/write on an H100 and
# measure GPU time from the trace
# TODO(future): audit this across different hardware and triton/non-triton
KERNEL_LAUNCH_OVERHEAD_SEC = 0.002 * 0.001


def get_tensor_memory_traffic_ovhd_s(
    specs,
    dim0,
    dim1,
    tensor_role: str,
    float8_recipe_name: Optional[str],
    mx_recipe_name: Optional[str],
    fuse_with_prev=False,
) -> List[Union[sympy.Symbol, float]]:
    """
    Calculates the roofline estimate of casting one of the gemm inputs
    (input, weight or grad_output) to float8 in fwd+bwd.

    Inputs: dim0 and dim1 (shape), tensor_role (input|weight|grad_output), recipe names
    Outputs: list of read/write traffic overhead in seconds, one for each kernel
    """
    # assumes input bf16, output f8
    numel = dim0 * dim1

    res_bytes = None
    if float8_recipe_name == "tensorwise":
        if tensor_role == "weight":
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
            res_bytes = [kernel_1_rw, 0, kernel_3_rw, kernel_4_rw]
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
            res_bytes = [kernel_1_rw, 0, kernel_3_rw]

    elif float8_recipe_name == "rowwise":
        if tensor_role == "weight":
            # x_bf16 = ...
            # kernel 1 (fwd):         x_bf16_dim0 -> x_float8_dim0
            # kernel 2 (bwd):         x_bf16_dim0 -> x_bf16_dim1
            # kernel 3 (bwd):         x_bf16_dim1 -> x_float8_dim1
            # assume that we can't fuse 2 and 3 because that would require loading
            # the entire tensor to shared memory
            if fuse_with_prev:
                # assume we can fuse one of the reads with previous op
                kernel_1_rw = 0 + BYTES_PER_EL_FLOAT8 * numel
            else:
                kernel_1_rw = BYTES_PER_EL_BF16 * numel + BYTES_PER_EL_FLOAT8 * numel
            kernel_2_rw = BYTES_PER_EL_BF16 * numel * 2
            kernel_3_rw = BYTES_PER_EL_BF16 * numel + BYTES_PER_EL_FLOAT8 * numel
            res_bytes = [kernel_1_rw, kernel_2_rw, kernel_3_rw]
        else:
            # x_bf16 = ...
            # kernel 1:               x_bf16_dim0 -> x_float8_dim0, x_bf16_dim1
            # kernel 2:               x_bf16_dim1 -> x_float8_dim1
            # assume that we can't fuse 1 and 2 because that would require loading
            # the entire tensor to shared memory
            if fuse_with_prev:
                # assume we can fuse one of the reads with previous op
                kernel_1_rw = (
                    0 + BYTES_PER_EL_FLOAT8 * numel + BYTES_PER_EL_BF16 * numel
                )
            else:
                kernel_1_rw = (
                    BYTES_PER_EL_BF16 * numel
                    + BYTES_PER_EL_FLOAT8 * numel
                    + BYTES_PER_EL_BF16 * numel
                )
            kernel_2_rw = BYTES_PER_EL_BF16 * numel + BYTES_PER_EL_FLOAT8 * numel
            res_bytes = [kernel_1_rw, kernel_2_rw]

    elif float8_recipe_name == "rowwise_with_gw_hp":
        if tensor_role in ("input", "grad_output"):
            # x_bf16 = ...
            # kernel 1 (fwd): x_bf16_dim0 -> x_float8_dim0
            # bwd: no-op
            if fuse_with_prev:
                kernel_1_rw = 0 + BYTES_PER_EL_FLOAT8 * numel
            else:
                kernel_1_rw = BYTES_PER_EL_BF16 * numel + BYTES_PER_EL_FLOAT8 * numel
            res_bytes = [kernel_1_rw]
        elif tensor_role == "weight":
            # x_bf16 = ...
            # kernel 1 (fwd): w_bf16 -> w_float8_dim0, w_scale_dim0
            # kernel 2 (bwd): w_scale_dim0 -> w_scale_tensorwise
            # kernel 3 (bwd): w_bf16, w_scale_tensorwise -> w_float8_dim1
            kernel_1_rw = BYTES_PER_EL_BF16 * numel + BYTES_PER_EL_FLOAT8 * numel
            kernel_2_rw = 0
            kernel_3_rw = BYTES_PER_EL_BF16 * numel + BYTES_PER_EL_FLOAT8 * numel
            res_bytes = [kernel_1_rw, kernel_2_rw, kernel_3_rw]
        else:
            assert False, "unsupported"

    else:
        assert mx_recipe_name in (
            "mxfp8_emulated",
            "mxfp8_cublas",
            "mxfp8_cublas_rceil",
        ), "unsupported"
        # For now, assume that we can't profitably fuse kernel 1 and kernel 2
        # x_bf16 = ...
        # kernel 1:               x_bf16 -> x_mxfp8_dim0
        # kernel 2:               x_bf16 -> x_mxfp8_dim1
        if fuse_with_prev:
            kernel_1_rw = 0 + BYTES_PER_EL_FLOAT8 * numel
        else:
            kernel_1_rw = BYTES_PER_EL_BF16 * numel + BYTES_PER_EL_FLOAT8 * numel
        kernel_2_rw = BYTES_PER_EL_BF16 * numel + BYTES_PER_EL_FLOAT8 * numel
        res_bytes = [kernel_1_rw, kernel_2_rw]

    # convert from bytes to seconds
    res_s = [
        x / specs["peak_mem_bw_bytes_sec"] / specs["pct_achievable_mem_bw"]
        for x in res_bytes
    ]

    # take max of kernel_overhead, r/w time
    res_s = [sympy.Max(x, KERNEL_LAUNCH_OVERHEAD_SEC) for x in res_s]

    return res_s


def get_individual_gemm_time_sympy(
    M: sympy.Symbol,
    K: sympy.Symbol,
    N: sympy.Symbol,
    dtype,
    mx_recipe_name,
    gpu_name: Optional[str] = None,
) -> sympy.Symbol:
    # compute bound
    specs = get_specs(gpu_name)
    gemm_ops = 2 * M * K * N
    if dtype is torch.bfloat16:
        peak_tops = specs["bf16_peak_tops"]
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        peak_tops = specs["fp8_peak_tops"]
    else:
        assert False, "unsupported"
    compute_gemm_time_s = gemm_ops / peak_tops / specs["pct_achievable_gemm_tops"]

    # memory bound
    num_reads = M * K + K * N
    num_writes = M * N

    if mx_recipe_name is not None:
        assert mx_recipe_name in (
            "mxfp8_emulated",
            "mxfp8_cublas",
            "mxfp8_cublas_rceil",
        ), "unsupported"
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

    return sympy.Max(compute_gemm_time_s, mem_gemm_time_s, KERNEL_LAUNCH_OVERHEAD_SEC)


def get_gemm_time_sympy(
    M: sympy.Symbol,
    K: sympy.Symbol,
    N: sympy.Symbol,
    dtype,
    float8_recipe_name: Optional[str],
    mx_recipe_name: Optional[str],
    gpu_name: Optional[str],
):
    # next: add rowwise_with_gw_hp here
    # note: this function is currently not super accurate for small shapes:
    # when M,K,N <= 1k,1k,1k it undercounts by around 2x

    gemm_dtype_input, gemm_dtype_grad_input, gemm_dtype_grad_weight = (
        dtype,
        dtype,
        dtype,
    )
    if float8_recipe_name == "rowwise_with_gw_hp":
        gemm_dtype_grad_weight = torch.bfloat16

    gemm_output_time_s = get_individual_gemm_time_sympy(
        M, K, N, gemm_dtype_input, mx_recipe_name, gpu_name
    )
    gemm_grad_input_time_s = get_individual_gemm_time_sympy(
        M, N, K, gemm_dtype_grad_input, mx_recipe_name, gpu_name
    )
    gemm_grad_weight_time_s = get_individual_gemm_time_sympy(
        K, M, N, gemm_dtype_grad_weight, mx_recipe_name, gpu_name
    )
    total = gemm_output_time_s + gemm_grad_input_time_s + gemm_grad_weight_time_s
    return total


def get_float8_mem_sympy(
    M,
    K,
    N,
    float8_recipe_name: Optional[str],
    mx_recipe_name: Optional[str],
    enable_fusion_modeling: bool,
    gpu_name: Optional[str] = None,
):
    specs = get_specs(gpu_name)

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

    fwd_fp8_input_mem = get_tensor_memory_traffic_ovhd_s(
        specs,
        M,
        K,
        tensor_role="input",
        float8_recipe_name=float8_recipe_name,
        mx_recipe_name=mx_recipe_name,
        fuse_with_prev=enable_fusion_modeling,
    )
    fwd_fp8_weight_mem = get_tensor_memory_traffic_ovhd_s(
        specs,
        K,
        N,
        tensor_role="weight",
        float8_recipe_name=float8_recipe_name,
        mx_recipe_name=mx_recipe_name,
        fuse_with_prev=False,
    )
    gi_fp8_grad_output_mem = get_tensor_memory_traffic_ovhd_s(
        specs,
        M,
        N,
        tensor_role="grad_output",
        float8_recipe_name=float8_recipe_name,
        mx_recipe_name=mx_recipe_name,
        fuse_with_prev=enable_fusion_modeling,
    )

    res = sum([*fwd_fp8_input_mem, *fwd_fp8_weight_mem, *gi_fp8_grad_output_mem])
    return res


def get_inference_tensor_memory_traffic_ovhd_s(
    specs,
    dim0,
    dim1,
    tensor_role: str,
    float8_recipe_name: Optional[str],
    fuse_with_prev=False,
) -> List[Union[sympy.Symbol, float]]:
    """
    Inference version of `get_tensor_memory_traffic_ovhd_s`.
    The only thing happening here is we quantize the activation.
    """
    assert float8_recipe_name == "rowwise", "unsupported"
    assert fuse_with_prev is False, "unsupported"

    # assumes input bf16, output f8
    numel = dim0 * dim1

    res_bytes = None

    assert tensor_role == "input"
    # x_bf16 = ...
    # kernel 1:               x_bf16 -> x_fp8
    kernel_1_rw = BYTES_PER_EL_BF16 * numel + BYTES_PER_EL_FLOAT8 * numel
    res_bytes = [
        kernel_1_rw,
    ]

    # convert from bytes to seconds
    res_s = [
        x / specs["peak_mem_bw_bytes_sec"] / specs["pct_achievable_mem_bw"]
        for x in res_bytes
    ]

    # take max of kernel_overhead, r/w time
    res_s = [sympy.Max(x, KERNEL_LAUNCH_OVERHEAD_SEC) for x in res_s]

    return res_s


def get_inference_float8_mem_sympy(
    M,
    K,
    N,
    float8_recipe_name: Optional[str],
    gpu_name: Optional[str] = None,
):
    specs = get_specs(gpu_name)
    # input @ weight_t = output
    # MxK @ KxN => MxN
    fwd_fp8_input_mem = get_inference_tensor_memory_traffic_ovhd_s(
        specs,
        M,
        K,
        tensor_role="input",
        float8_recipe_name=float8_recipe_name,
        fuse_with_prev=False,
    )
    res = sum([*fwd_fp8_input_mem])
    return res


def get_inference_gemm_time_sympy(
    M: sympy.Symbol,
    K: sympy.Symbol,
    N: sympy.Symbol,
    dtype,
    float8_recipe_name: Optional[str],
    gpu_name: Optional[str],
):
    assert float8_recipe_name == "rowwise" or float8_recipe_name is None, "unsupported"
    # note: this function is currently not super accurate for small shapes:
    # when M,K,N <= 1k,1k,1k it undercounts by around 2x
    gemm_output_time_s = get_individual_gemm_time_sympy(M, K, N, dtype, None, gpu_name)
    return gemm_output_time_s
