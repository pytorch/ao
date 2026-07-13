from torchao.prototype.moe_training.fp8_grouped_mm import (
    _to_fp8_rowwise_then_scaled_grouped_mm,
)
from torchao.prototype.moe_training.mxfp8_grouped_mm import (
    _to_mxfp8_then_scaled_grouped_mm,
)

__all__ = [
    "_to_fp8_blockwise_then_scaled_grouped_mm",
    "_to_mxfp8_then_scaled_grouped_mm",
    "_to_fp8_rowwise_then_scaled_grouped_mm",
]


def __getattr__(name: str):
    if name == "_to_fp8_blockwise_then_scaled_grouped_mm":
        from torchao.prototype.moe_training.blockwise_fp8 import (
            _to_fp8_blockwise_then_scaled_grouped_mm,
        )

        return _to_fp8_blockwise_then_scaled_grouped_mm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
