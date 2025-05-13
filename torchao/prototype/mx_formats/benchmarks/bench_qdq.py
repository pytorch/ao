# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmarking mx quantize/dequantize
"""

from typing import Optional

import fire
import tabulate
import torch
from torch.profiler import ProfilerActivity, profile

from torchao.prototype.mx_formats import config
from torchao.prototype.mx_formats.constants import (  # noqa: E501
    SUPPORTED_ELEM_DTYPES,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.utils import benchmark_torch_function_in_microseconds


def run(profile_folder: Optional[str] = None):
    headers = [
        "elem_dtype",
        "use_fp4_custom_triton_dequant_kernel",
        "q_time_us",
        "q_mem_bw_tb_s",
        "dq_time_us",
        "dq_mem_bw_tb_s",
    ]
    results = []

    data_hp = torch.randn(1, 4096, 11008, dtype=torch.bfloat16, device="cuda")

    for elem_dtype in SUPPORTED_ELEM_DTYPES:
        for use_fp4_custom_triton_dequant_kernel in (False, True):
            config.use_fp4_custom_triton_dequant_kernel = (
                use_fp4_custom_triton_dequant_kernel
            )

            if (
                elem_dtype != torch.float4_e2m1fn_x2
                and use_fp4_custom_triton_dequant_kernel  # noqa: E501
            ):
                # custom_triton_kernels only works for fp4
                continue

            print(
                "elem_dtype",
                elem_dtype,
                "use_fp4_custom_triton_dequant_kernel",
                use_fp4_custom_triton_dequant_kernel,
            )

            data_lp = MXTensor.to_mx(data_hp, elem_dtype, block_size=32)

            if not use_fp4_custom_triton_dequant_kernel:
                quant = torch.compile(MXTensor.to_mx, fullgraph=True)
                dequant = torch.compile(data_lp.to_dtype, fullgraph=True)
            else:
                # As of 2024-04, torch.compile didn't work with the
                # handwritten triton kernel,
                # crashed on tl.interleave:
                # https://github.com/pytorch/pytorch/issues/123967
                # As of 2024-05-24, now there is message asking to convert to
                # an opaque custom op:
                # https://gist.github.com/vkuzo/0b0b90dca03bdb8e0446e4135644238a  # noqa: E501
                # TODO(future): make this better
                quant = MXTensor.to_mx
                dequant = data_lp.to_dtype

            # warm up
            quant(data_hp, elem_dtype, block_size=32)
            res = dequant(torch.bfloat16)

            if profile_folder is not None:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                ) as prof:
                    for _ in range(5):
                        quant(data_hp, elem_dtype, block_size=32)
                        dequant(torch.bfloat16)
                prof.export_chrome_trace(
                    profile_folder
                    + f"/mx_qdq_{elem_dtype}_{use_fp4_custom_triton_dequant_kernel}.json"  # noqa: E501
                )

            q_execution_time_us = benchmark_torch_function_in_microseconds(
                quant, data_hp, elem_dtype, block_size=32
            )
            dq_execution_time_us = benchmark_torch_function_in_microseconds(
                dequant, torch.bfloat16
            )
            print(f"q time: {q_execution_time_us} us")
            print(f"dq time: {dq_execution_time_us} us")

            # memory reads per element:
            byte_per_stored_element = 1.0  # fp8 or 2xfp4
            byte_per_stored_exp_element = 1.0  # e8m0
            byte_per_dequantized_element = 2.0  # bfloat16
            mem_reads_writes_bytes = (
                # read raw data
                (data_lp._data.numel() * byte_per_stored_element)
                +
                # read exponent
                (data_lp._scale_e8m0.numel() * byte_per_stored_exp_element)
                +
                # write dequant
                (res.numel() * byte_per_dequantized_element)
            )
            # note: the above also works for quant, with reads/writes in
            # reverse

            q_mem_bw_tb_s = (mem_reads_writes_bytes / 1e12) / (
                q_execution_time_us / 1e6
            )
            dq_mem_bw_tb_s = (mem_reads_writes_bytes / 1e12) / (
                dq_execution_time_us / 1e6
            )
            print(f"q mem bw: {q_mem_bw_tb_s} TB/s")
            print(f"dq mem bw: {dq_mem_bw_tb_s} TB/s")

            results.append(
                (
                    elem_dtype,
                    use_fp4_custom_triton_dequant_kernel,
                    q_execution_time_us,
                    q_mem_bw_tb_s,
                    dq_execution_time_us,
                    dq_mem_bw_tb_s,
                )
            )
            config.use_fp4_custom_triton_dequant_kernel = False

            torch._dynamo.reset()

    print(tabulate.tabulate(results, headers=headers, floatfmt=".2f"))


if __name__ == "__main__":
    fire.Fire(run)
