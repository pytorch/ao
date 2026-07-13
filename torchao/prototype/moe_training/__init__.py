from torchao.prototype.moe_training.fp8_grouped_mm import (
    _to_fp8_rowwise_then_scaled_grouped_mm,
)
from torchao.prototype.moe_training.mxfp8_grouped_mm import (
    _to_mxfp8_then_scaled_grouped_mm,
)

__all__ = [
    "fp8_blockwise_grouped_mm",
    "_to_mxfp8_then_scaled_grouped_mm",
    "_to_fp8_rowwise_then_scaled_grouped_mm",
]


def __getattr__(name: str):
    if name == "fp8_blockwise_grouped_mm":
        from torchao.prototype.moe_training.blockwise_fp8 import (
            fp8_blockwise_grouped_mm,
        )

        return fp8_blockwise_grouped_mm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
