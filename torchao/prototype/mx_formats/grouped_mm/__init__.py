import importlib as _importlib

__all__ = [
    "GroupedMMConfig",
    "MXFP8GroupedMMConfig",
    "MXFP8GroupedMMRecipe",
    "_to_mxfp8_then_scaled_grouped_mm",
    "_quantize_then_scaled_grouped_mm",
    "ScaledGroupedMMTensor",
]

_LAZY_IMPORT_MAP = {
    "GroupedMMConfig": "torchao.prototype.mx_formats.grouped_mm.config",
    "MXFP8GroupedMMConfig": "torchao.prototype.mx_formats.grouped_mm.config",
    "MXFP8GroupedMMRecipe": "torchao.prototype.mx_formats.grouped_mm.config",
    "_to_mxfp8_then_scaled_grouped_mm": "torchao.prototype.mx_formats.grouped_mm.mxfp8_grouped_mm",
    "ScaledGroupedMMTensor": "torchao.prototype.mx_formats.grouped_mm.tensor",
    "_quantize_then_scaled_grouped_mm": "torchao.prototype.mx_formats.grouped_mm.tensor",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORT_MAP:
        mod = _importlib.import_module(_LAZY_IMPORT_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
