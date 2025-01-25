import itertools

import pytest

# Skip entire test if triton is not available, otherwise CI failure
try:
    import triton  # noqa: F401
except ImportError:
    pytest.skip("triton is not installed", allow_module_level=True)

import torch
from galore_test_utils import get_kernel, make_copy, make_data

torch.manual_seed(0)
MAX_DIFF_no_tf32 = 1e-5
MAX_DIFF_tf32 = 1e-3


def run_test(kernel, exp_avg, exp_avg2, grad, proj_matrix, params, allow_tf32):
    # Copy to use for first run -- needed because of autotuning and inplace ops
    (
        exp_avg_autotune_copy,
        exp_avg2_autotune_copy,
        grad_autotune_copy,
        proj_matrix_autotune_copy,
        params_autotune_copy,
    ) = make_copy(exp_avg, exp_avg2, grad, proj_matrix, params)

    # Copy to use for second run to check accuracy
    (
        exp_avg_test_copy,
        exp_avg2_test_copy,
        grad_test_copy,
        proj_matrix_test_copy,
        params_test_copy,
    ) = make_copy(exp_avg, exp_avg2, grad, proj_matrix, params)

    print(
        f"Running with {grad.shape[0]} x {grad.shape[1]} grad (param) shape, GaLore orthogonal matrix {list(proj_matrix.shape)}, dtype {grad.dtype} and allow_tf32 {allow_tf32}\n"
        f"Kernel: {kernel}",
        flush=True,
    )

    ref_op = get_kernel("ref")
    test_op = get_kernel(kernel)

    # Reference run
    ref_out = ref_op(
        grad,
        proj_matrix,
        exp_avg,
        exp_avg2,
        params,
    )

    # Autotune
    _ = test_op(
        grad_autotune_copy,
        proj_matrix_autotune_copy,
        exp_avg_autotune_copy,
        exp_avg2_autotune_copy,
        params_autotune_copy,
        store=False,
        allow_tf32=allow_tf32,
    )

    # Accuracy run
    test_out = test_op(
        grad_test_copy,
        proj_matrix_test_copy,
        exp_avg_test_copy,
        exp_avg2_test_copy,
        params_test_copy,
        store=True,
        allow_tf32=allow_tf32,
    )
    print("Accuracy:")

    output_names = [
        "adam state - running grad mean",
        "adam state - running grad var",
        "params (after update)",
    ]
    MAX_DIFF = MAX_DIFF_tf32 if allow_tf32 else MAX_DIFF_no_tf32
    for name, ref, tt in zip(output_names, ref_out, test_out):
        max_diff = (ref - tt).abs().max()
        print(f"-> {name}:\n  Max err: {max_diff:.6f}")
        assert max_diff < MAX_DIFF


KERNELS = ["hybrid"]  #  "fused"]
DTYPES = [torch.float32]  # torch.float16
ROW_DIMS = [4096]
COL_DIMS = [4096]  # , 11008]
RANKS = [128]
ALLOW_TF32 = [False]  # , True]

TEST_CONFIGS = list(
    itertools.product(KERNELS, DTYPES, ROW_DIMS, COL_DIMS, RANKS, ALLOW_TF32)
)

# TEST_CONFIGS = TEST_CONFIGS[0:1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("kernel, dtype, M, N, rank, allow_tf32", TEST_CONFIGS)
def test_galore_fused_kernels(kernel, dtype, M, N, rank, allow_tf32):
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    exp_avg, exp_avg2, grad, proj_matrix, params = make_data(M, N, rank, dtype)
    run_test(kernel, exp_avg, exp_avg2, grad, proj_matrix, params, allow_tf32)
