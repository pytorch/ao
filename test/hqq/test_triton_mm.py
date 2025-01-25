# Skip entire test if following module not available, otherwise CI failure
import pytest

triton = pytest.importorskip(
    "triton", minversion="3.0.0", reason="Triton > 3.0.0 required to run this test"
)
hqq = pytest.importorskip("hqq", reason="hqq required to run this test")
hqq_quantize = pytest.importorskip(
    "hqq.core.quantize", reason="hqq required to run this test"
)
HQQLinear = hqq_quantize.HQQLinear
BaseQuantizeConfig = hqq_quantize.BaseQuantizeConfig

import itertools

import torch

from torchao.prototype.hqq import pack_2xint4, triton_mixed_mm

# Test configs
SHAPES = [
    [16, 128, 128],
    [16, 4096, 4096],
]

DTYPES = [torch.bfloat16, torch.float16]
GROUP_SIZES = [64, 128]
AXES = [1]  # Only axis = 1 supported
TRANSPOSED = [False, True]
TRITON_KERNEL_TYPE = ["compute_bound"]  # ["max_autotune", "compute_bound"]

TEST_CONFIGS = list(
    itertools.product(SHAPES, GROUP_SIZES, AXES, DTYPES, TRANSPOSED, TRITON_KERNEL_TYPE)
)

BASE_QUANT_CONFIG = {
    "optimize": True,
    "view_as_float": False,
    "nbits": 4,
    "bitpack": False,
    "axis": 1,
}


def check(expected, actual, msg="", max_diff=1e-3, verbose=False):
    passed = torch.allclose(expected, actual, atol=max_diff, rtol=max_diff)
    if verbose:
        max_err = (expected - actual).abs().max()
        if not passed:
            print_msg = f"{msg}:\nFailed! Max error: {max_err}"
            try:
                from termcolor import colored
            except ImportError:
                print(print_msg)
            else:
                print(colored(print_msg, "red", attrs=["bold"]))

        else:
            print_msg = f"{msg}:\nPassed! Max error: {max_err}"
            try:
                from termcolor import colored
            except ImportError:
                print(print_msg)
            else:
                print(colored(print_msg, "green", attrs=["bold"]))

    return passed


def _arg_to_id(arg):
    if isinstance(arg, list):
        return "x".join([str(x) for x in arg])
    return str(arg)


@pytest.mark.parametrize(
    "shape, group_size, axis, dtype, transposed, kernel_type",
    TEST_CONFIGS,
    ids=_arg_to_id,
)
def test_mixed_mm(
    shape, group_size, axis, dtype, transposed, kernel_type, quant_dtype=torch.uint8
):
    qcfg = {
        **BASE_QUANT_CONFIG,
        **dict(group_size=group_size, axis=axis),
    }
    M, N, K = shape

    linear = torch.nn.Linear(K, N, bias=False, dtype=dtype, device="cuda")

    quant_config = BaseQuantizeConfig(
        quant_zero=False, quant_scale=False, offload_meta=False, view_as_float=False
    )
    quant_config.update({"weight_quant_params": qcfg})
    hqq_linear = HQQLinear(linear, quant_config, compute_dtype=dtype, del_orig=False)
    W_q, meta = hqq_linear.W_q, hqq_linear.meta
    W_q = W_q.to(dtype=quant_dtype)
    W_q = (
        W_q.reshape(meta["shape"])
        if quant_config["weight_quant_params"]["bitpack"] == False
        else W_q
    )
    W_dq = hqq_linear.dequantize()

    scales, zeros = meta["scale"], meta["zero"]
    scales = scales.reshape(N, -1)
    zeros = zeros.reshape(N, -1)

    packed_w = pack_2xint4(W_q.T)

    if transposed:
        x = torch.randn(M, N, dtype=dtype, device="cuda")
        hqq_out = x @ W_dq

        tt_out = triton_mixed_mm(
            x,
            packed_w,
            scales.T,
            zeros.T,
            transposed=True,
            group_size=group_size,
            fp8_fast_accum=False,
            kernel_type=kernel_type,
        )

    else:
        x = torch.randn(M, K, dtype=dtype, device="cuda")
        hqq_out = x @ W_dq.T

        tt_out = triton_mixed_mm(
            x,
            packed_w,
            scales.T,
            zeros.T,
            transposed=False,
            group_size=group_size,
            fp8_fast_accum=False,
            kernel_type=kernel_type,
        )
    assert check(
        hqq_out,
        tt_out,
        max_diff=1e-2 if dtype == torch.bfloat16 else 1e-3,
        verbose=True,
    )


# Only for debugging kernel without dependency on HQQ and with no autotuning
def _test_mixed_mm(
    shape,
    group_size,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    axis=1,
    dtype=torch.float16,
    transposed=True,
    kernel_type="debug",
    quant_dtype=torch.uint8,
):
    qcfg = {
        **BASE_QUANT_CONFIG,
        **dict(group_size=group_size, axis=axis),
    }
    M, N, K = shape

    quant_config = BaseQuantizeConfig(
        quant_zero=False, quant_scale=False, offload_meta=False, view_as_float=False
    )
    quant_config.update({"weight_quant_params": qcfg})
    W_q = torch.randint(0, int(2**4), size=(N, K), dtype=quant_dtype, device="cuda")

    scales = torch.arange((N * K) // group_size, dtype=dtype, device="cuda")[:, None]
    zeros = torch.zeros_like(scales)
    W_dq = ((W_q.reshape(-1, group_size) - zeros) * scales).reshape(N, K)
    scales = scales.reshape(N, -1)
    zeros = zeros.reshape(N, -1)

    packed_w = pack_2xint4(W_q.T)

    if transposed:
        x = torch.randn(M, N, dtype=dtype, device="cuda")
        hqq_out = x @ W_dq

        tt_out = triton_mixed_mm(
            x,
            packed_w,
            scales.T,
            zeros.T,
            transposed=True,
            group_size=group_size,
            fp8_fast_accum=False,
            kernel_type=kernel_type,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

    else:
        x = torch.randn(M, K, dtype=dtype, device="cuda")
        hqq_out = x @ W_dq.T

        tt_out = triton_mixed_mm(
            x,
            packed_w,
            scales.T,
            zeros.T,
            transposed=False,
            group_size=group_size,
            fp8_fast_accum=False,
            kernel_type=kernel_type,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
    msg = f"shape={shape}, group_size={group_size}, axis={axis}, dtype={dtype}, transposed={transposed}, kernel_type={kernel_type}, quant_dtype={quant_dtype}"

    check(
        hqq_out,
        tt_out,
        msg=msg,
        max_diff=1e-2 if dtype == torch.bfloat16 else 1e-3,
        verbose=True,
    )


if __name__ == "__main__":
    # _test_mixed_mm(transposed=False)
    M, N, K = shape = [32, 128, 128]
    BLOCK_M, BLOCK_N, BLOCK_K = shape
    BLOCK_K = K // 2
    BLOCK_N = N // 2
    group_size = BLOCK_K
    _test_mixed_mm(
        shape,
        group_size=group_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        transposed=False,
    )
    _test_mixed_mm(
        shape,
        group_size=group_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        transposed=True,
    )
