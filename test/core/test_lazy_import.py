import importlib
import sys

import pytest


# Skip tests if PyTorch is not installed; torchao requires torch at import time
pytest.importorskip("torch", reason="Requires PyTorch for torchao import")


def _cleanup_torchao_modules():
    # Remove torchao and its submodules from sys.modules to force a fresh import
    for name in list(sys.modules.keys()):
        if name == "torchao" or name.startswith("torchao."):
            sys.modules.pop(name, None)


def test_top_level_import_is_lightweight():
    _cleanup_torchao_modules()
    importlib.invalidate_caches()

    import torchao  # noqa: F401  - just to import

    # Heavy submodules should not be imported eagerly
    assert "torchao.quantization" not in sys.modules
    assert "torchao.experimental" not in sys.modules
    assert "torchao.experimental.op_lib" not in sys.modules


def test_accessing_lazy_attrs_triggers_import():
    _cleanup_torchao_modules()
    importlib.invalidate_caches()

    import torchao

    # Accessing lazy attributes should import their source module
    _ = torchao.autoquant  # triggers torchao.quantization import
    assert "torchao.quantization" in sys.modules
    assert hasattr(torchao, "autoquant")

    _ = torchao.quantize_  # already imported module should provide symbol
    assert hasattr(torchao, "quantize_")


def test_lazy_submodule_resolution():
    _cleanup_torchao_modules()
    importlib.invalidate_caches()

    import torchao

    # Accessing a lazy submodule should import it on demand
    _ = torchao.dtypes
    assert "torchao.dtypes" in sys.modules

