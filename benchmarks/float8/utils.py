# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import collections
import re
from typing import Optional


def profiler_output_to_time_by_kernel_name(prof):
    """
    Input: a profiler with captured events.
    Output: a deduplicated list of GPU time in nanoseconds grouped by CPU kernel name

    Note that if there are user_annotations in the captured events, `torch.profiler`
    will include their time in the total GPU time displayed at the bottom of
    `key_averages.table()`. The filter below excludes them to prevent double
    counting.
    """
    key_averages = prof.key_averages()
    thresh = 1e-10
    kernel_name_to_gpu_time_us = collections.defaultdict(float)
    for e in key_averages:
        # manually filter top-level CPU events with attributed CUDA time
        # example CPU event row:
        #                                               aten::addmm         0.83%      76.554us         0.98%      90.846us      90.846us       1.022ms        31.82%       1.022ms       1.022ms             1
        # and it maps to this CUDA event:
        #   sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize256x64...         0.00%       0.000us         0.00%       0.000us       0.000us       1.022ms        31.82%       1.022ms       1.022ms             1
        if not (e.self_cpu_time_total > thresh and e.self_device_time_total > thresh):
            continue
        kernel_name_to_gpu_time_us[e.key] = e.self_device_time_total
    return kernel_name_to_gpu_time_us


def profiler_output_to_gpu_time_for_key(prof, key):
    """
    Input: an event name
    Output: sum of GPU time of all events with that name in `prof`

    This is useful to get the total time of a user annotation
    """
    total = 0
    for e in prof.profiler.function_events:
        if e.key == key:
            total += e.device_time_total
    return total


def kernel_name_to_category(k):
    # number prefix is for easy sorting
    if k in ("aten::mm", "aten::addmm", "aten::_scaled_mm"):
        return "0_gemm"
    elif (
        # max(abs(tensor))
        ("abs" in k and "max" in k)
        or
        # casting pointwise to float8
        ("clamp" in k)
        or
        # things related to scaled_mm
        ("scaled_mm" in k)
        or
        # syncing amaxes and scales
        ("roll" in k)
    ):
        # note: the above filter is approximate and will give false
        # positives if model code contains other code to abs/max/clamp
        return "1_f8_overhead"
    return "2_other"


def parse_bw_and_kernel_name(line):
    """
    Input: a single line of stdout of TORCHINDUCTOR_PROFILE=1 output, such as
        0.257ms         0.537 GB         2092.43GB/s     triton_red_fused_native_layer_norm_0
    Output: the bandwidth value and the kernel name, or None and None
    """
    result = re.search(".* ([0-9\.]+)GB/s.*(triton_[a-z_0-9]+)", line)
    if result:
        return result.group(1), result.group(2)
    else:
        return None, None


def get_name_to_shapes_iter(
    shape_gen_name: str,
    M: Optional[int],
    K: Optional[int],
    N: Optional[int],
):
    if shape_gen_name == 'llama':
        assert M == K == N == None, \
            f'M, K, N arguments not supported for shape_gen_name {shape_gen_name}'
        bsz, seq_len = 4, 4096
        M = bsz * seq_len
        # LLaMa 2 70B single-node weight shapes
        # assumes fused attn.wqkv and ffn.w13
        # source: https://fburl.com/gsheet/g8onr7rh
        name_to_shapes_70b = {
            "attn.wqkv": (M, 8192, 1280),
            "attn.w0": (M, 1024, 8192),
            "ffn.w13": (M, 8192, 7168),
            "ffn.w2": (M, 3584, 8192),
        }
        return name_to_shapes_70b.items()

    elif shape_gen_name == 'square':
        assert M == K == N == None, \
            f'M, K, N arguments not supported for shape_gen_name {shape_gen_name}'
        name_to_shapes = {}
        min_power_of_2 = 5  # 32
        max_power_of_2 = 16  # 65,536
        for idx, power_of_2 in enumerate(range(min_power_of_2, max_power_of_2 + 1)):
            val = 2 ** power_of_2
            name_to_shapes[idx] = val, val, val
        return name_to_shapes.items()

    elif shape_gen_name == 'sweep':
        assert M == K == N == None, \
            f'M, K, N arguments not supported for shape_gen_name {shape_gen_name}'
        name_to_shapes = {}
        min_p2 = 5  # 32
        max_p2 = 16  # 65,536
        counter = 0
        for M_p2 in range(min_p2, max_p2 + 1):
            M = 2 ** M_p2
            for K_p2 in range(min_p2, max_p2 + 1):
                K = 2 ** K_p2
                for N_p2 in range(min_p2, max_p2 + 1):
                    N = 2 ** N_p2
                    name_to_shapes[counter] = M, K, N
                    counter += 1
        return name_to_shapes.items()

    elif shape_gen_name == 'custom':
        assert M is not None and K is not None and N is not None, \
            'M, K, N must be specified for custom shape_gen'
        name_to_shapes = {
            1: (M, K, N),
        }
        return name_to_shapes.items()

    raise AssertionError(f'unknown shape_gen_name {shape_gen_name}')
