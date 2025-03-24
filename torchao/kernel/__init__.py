from torchao.utils import lazy_import

bsr_dense_addmm = lazy_import("torchao.kernel.bsr_triton_ops", "bsr_dense_addmm")
int_scaled_matmul = lazy_import("torchao.kernel.intmm", "int_scaled_matmul")
safe_int_mm = lazy_import("torchao.kernel.intmm", "safe_int_mm")

__all__ = [
    "bsr_dense_addmm",
    "safe_int_mm",
    "int_scaled_matmul",
]
