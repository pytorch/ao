import importlib as _importlib

__all__ = [
    "_quantize_then_scaled_grouped_mm",
    "_to_mxfp8_then_scaled_grouped_mm",
    "_to_fp8_rowwise_then_scaled_grouped_mm",
]


def __getattr__(name: str):
    if name == "_to_fp8_rowwise_then_scaled_grouped_mm":
        mod = _importlib.import_module("torchao.prototype.moe_training.fp8_grouped_mm")
        return mod._to_fp8_rowwise_then_scaled_grouped_mm
    if name == "_to_mxfp8_then_scaled_grouped_mm":
        mod = _importlib.import_module(
            "torchao.prototype.mx_formats.grouped_mm.mxfp8_grouped_mm"
        )
        mod = _importlib.import_module("torchao.prototype.mx_formats.grouped_mm")
        return mod._to_mxfp8_then_scaled_grouped_mm
    if name == "_quantize_then_scaled_grouped_mm":
        mod = _importlib.import_module("torchao.prototype.mx_formats.grouped_mm.tensor")
        return mod._quantize_then_scaled_grouped_mm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
