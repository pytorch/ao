import sys
import unittest

if sys.version_info < (3, 11):
    raise unittest.SkipTest("requires Python >= 3.11")

try:
    import triton
except ImportError:
    raise unittest.SkipTest("requires triton")


import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import run_tests

from torchao.prototype.dora.kernels.matmul import triton_mm
from torchao.prototype.dora.kernels.smallk import triton_mm_small_k

torch.manual_seed(0)

def _arg_to_id(arg):
    if isinstance(arg, (tuple, list)):
        return "x".join([str(x) for x in arg])
    return str(arg)


def check(expected, actual, dtype):
    if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        raise ValueError(f"Unsupported dtype: {dtype}")
    diff = (expected - actual).abs().max()
    print(f"diff: {diff}")
    # assert diff < atol
    return diff


M = 4096
N = 4096
Ks = [int(2**i) for i in range(4, 7)]
FUSED_DORA_SHAPES = [(M, N, K) for K in Ks[:1]]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
STORE_ACC = [False]
EPILOGUE_NORM = [True, False]
ADD_SOURCE = [True]
MAGNITUDE_VECTOR = [True]

class TestDoraColumnNorm(common_utils.TestCase):

    @common_utils.parametrize("shape", FUSED_DORA_SHAPES)
    @common_utils.parametrize("store_acc", STORE_ACC)
    @common_utils.parametrize("epilogue_norm", EPILOGUE_NORM)
    @common_utils.parametrize("add_source", ADD_SOURCE)
    @common_utils.parametrize("magnitude_vector", MAGNITUDE_VECTOR)
    @common_utils.parametrize("dtype", DTYPES)
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_dora_column_norm(
        self, shape, store_acc, epilogue_norm, add_source, magnitude_vector, dtype
    ):
        if not (store_acc or epilogue_norm):
            # pytest.skip("Either store_acc or epilogue_norm must be True")
            raise unittest.SkipTest("Either store_acc or epilogue_norm must be True")

        M, N, K = shape
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(K, N, device="cuda", dtype=dtype)
        source = torch.randn(M, N, device="cuda", dtype=dtype)
        magnitude = torch.randn(M, device="cuda", dtype=dtype)

        c_ref = torch.matmul(A, B)
        norm2_ref = 1 / c_ref.norm(2, dim=1)
        source_ref = source + c_ref
        source_norm2_ref = 1 / (source + c_ref).norm(2, dim=1)
        source_norm2_magnitude_ref = magnitude * source_norm2_ref

        # First test small K only kernel, no epilogue
        # source = None  # source  # None
        # magnitude = None  # magnitude  # None

        tt_out = triton_mm_small_k(
            A,
            B,
            source=source if add_source else None,
            magnitude=magnitude if magnitude_vector else None,
            epilogue_norm=epilogue_norm,
            store_acc=store_acc,
        )

        if store_acc:
            c_test = tt_out[0] if epilogue_norm else tt_out
            if add_source:
                check(source_ref, c_test, dtype)
            else:
                check(c_ref, c_test, dtype)

        if epilogue_norm:
            norm2_test = tt_out[1] if store_acc else tt_out
            if add_source:
                if magnitude_vector:
                    check(source_norm2_magnitude_ref, norm2_test, dtype)
                else:
                    check(source_norm2_ref, norm2_test, dtype)
            else:
                check(norm2_ref, norm2_test, dtype)


BATCH_SIZES = [int(2**i) for i in range(6)]
SEQ_LENS = [512]
IN_FEATURES = [4096]
OUT_FEATURES = [4096]
FUSED_MATMUL_SHAPES = [
    (bs * seqlen, in_features, out_features)
    for bs, seqlen, in_features, out_features in zip(
        BATCH_SIZES, SEQ_LENS, IN_FEATURES, OUT_FEATURES
    )
]
EPILOGUE_ELEMENTWISE_ADD = [True]
EPILOGUE_BROADCAST_SCALE = [True]

class TestDoraMatmul(common_utils.TestCase):
    @common_utils.parametrize("shape", FUSED_MATMUL_SHAPES[:1])
    @common_utils.parametrize("dtype", DTYPES)
    @common_utils.parametrize("epilogue_add", EPILOGUE_ELEMENTWISE_ADD)
    @common_utils.parametrize("epilogue_scale", EPILOGUE_BROADCAST_SCALE)
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_dora_matmul(shape, dtype, epilogue_add, epilogue_scale):
        M, K, N = shape
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(K, N, device="cuda", dtype=dtype)
        C = torch.randn(M, N, device="cuda", dtype=dtype) if epilogue_add else None
        scale = torch.randn(N, device="cuda", dtype=dtype) if epilogue_scale else None

        D_ref = torch.matmul(A, B)
        if epilogue_add:
            D_ref += C
        if epilogue_scale:
            D_ref *= scale.unsqueeze(0)

        D_test = triton_mm(A, B, epilogue_source=C, epilogue_scale=scale)
        check(D_ref, D_test, dtype)


MODES = ["default"]

class TestDoraMatmulCompile(common_utils.TestCase):
    @common_utils.parametrize("shape", FUSED_MATMUL_SHAPES[:1])
    @common_utils.parametrize("dtype", DTYPES)
    @common_utils.parametrize("epilogue_add", EPILOGUE_ELEMENTWISE_ADD)
    @common_utils.parametrize("epilogue_scale", EPILOGUE_BROADCAST_SCALE)
    @common_utils.parametrize("mode", MODES)
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skip("TODO: torch.compile does not work with custom kernel")
    def test_dora_matmul_compile(shape, dtype, epilogue_add, epilogue_scale, mode):
        M, K, N = shape
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(K, N, device="cuda", dtype=dtype)
        C = torch.randn(M, N, device="cuda", dtype=dtype) if epilogue_add else None
        scale = torch.randn(N, device="cuda", dtype=dtype) if epilogue_scale else None

        D_ref = torch.matmul(A, B)
        if epilogue_add:
            D_ref += C
        if epilogue_scale:
            D_ref *= scale.unsqueeze(0)

        D_test = triton_mm(A, B, epilogue_source=C, epilogue_scale=scale)
        check(D_ref, D_test, dtype)

        triton_compiled = torch.compile(triton_mm, mode=mode)
        D_compiled = triton_compiled(A, B, epilogue_source=C, epilogue_scale=scale)
        check(D_ref, D_compiled, dtype)


if __name__ == "__main__":
    run_tests()
