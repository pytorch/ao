import logging
import warnings
from importlib.util import find_spec
from typing import Any

# torch/nested/_internal/nested_tensor.py:417: UserWarning: Failed to initialize NumPy: No module named 'numpy'
# Suppress numpy warning
warnings.filterwarnings(
    "ignore", message="Failed to initialize NumPy: No module named 'numpy'"
)

# We use this "hack" to set torchao.__version__ correctly
# the version of ao is dependent on environment variables for multiple architectures
# For local development this will default to whatever is version.txt
# For release builds this will be set the version+architecture_postfix
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torchao")
except PackageNotFoundError:
    __version__ = "unknown"  # In case this logic breaks don't break the build

# Lazy loading mechanism
class LazyLoader:
    def __init__(self, lib_name: str):
        self.lib_name = lib_name
        self._mod = None
        self._loading = False

    def __getattr__(self, name: str) -> Any:
        if self._loading:
            raise AttributeError(f"No attribute {name} - recursive load detected")
        if self._mod is None:
            self._loading = True
            try:
                import importlib
                self._mod = importlib.import_module(self.lib_name)
            finally:
                self._loading = False
        return getattr(self._mod, name)

    def __repr__(self):
        return f"LazyLoader({self.lib_name})"

# C++ extensions loader
class CppExtensionLoader:
    def __init__(self):
        self._loaded = False
        self._ops = None
        self._loading = False
        self._torch = None

    def __getattr__(self, name: str) -> Any:
        if self._loading:
            raise AttributeError(f"No attribute {name} - recursive load detected")
        if not self._loaded:
            self._loading = True
            try:
                self._load_extensions()
            finally:
                self._loading = False
                self._loaded = True
        if self._ops is None:
            raise AttributeError(f"No attribute {name} - C++ extensions not available")
        return getattr(self._ops, name)

    def _load_extensions(self):
        try:
            # Lazy import torch only when needed
            if self._torch is None:
                import torch
                self._torch = torch
            from pathlib import Path
            
            so_files = list(Path(__file__).parent.glob("_C*.so"))
            if len(so_files) > 0:
                assert len(so_files) == 1, f"Expected one _C*.so file, found {len(so_files)}"
                self._torch.ops.load_library(str(so_files[0]))
                import importlib
                self._ops = importlib.import_module("torchao.ops")
        except Exception as e:
            logging.warning(f"Failed to load C++ extensions: {e}")

    def __repr__(self):
        status = "loading" if self._loading else "not_loaded" if not self._loaded else "loaded"
        return f"CppExtensionLoader({status})"

# Experimental ops loader
class ExperimentalOpsLoader:
    def __init__(self):
        self._loaded = False
        self._ops = {}
        self._loading = False

    def __getattr__(self, name: str) -> Any:
        if self._loading:
            raise AttributeError(f"No attribute {name} - recursive load detected")
        if not self._loaded:
            self._loading = True
            try:
                self._load_ops()
            finally:
                self._loading = False
                self._loaded = True
        if name not in self._ops:
            raise AttributeError(f"No attribute {name} in experimental ops")
        return self._ops[name]

    def _load_ops(self):
        try:
            if find_spec("torchao.experimental.op_lib"):
                import importlib
                op_lib = importlib.import_module("torchao.experimental.op_lib")
                for name in getattr(op_lib, "__all__", []):
                    self._ops[name] = getattr(op_lib, name)
        except Exception as e:
            logging.warning(f"Failed to load experimental ops: {e}")

    def __repr__(self):
        status = "loading" if self._loading else "not_loaded" if not self._loaded else "loaded"
        return f"ExperimentalOpsLoader({status})"

def _lazy_import(name: str):
    """Helper to create lazy imports"""
    return LazyLoader(name)

# Initialize all loaders
dtypes = _lazy_import("torchao.dtypes")
optim = _lazy_import("torchao.optim")
swizzle = _lazy_import("torchao.swizzle")
testing = _lazy_import("torchao.testing")
ops = CppExtensionLoader()
experimental = ExperimentalOpsLoader()

# Lazy functions that avoid importing torch/numpy until absolutely needed
def quantize_():
    """Lazy load quantization module"""
    from torchao.quantization import quantize_ as _quantize
    return _quantize

def autoquant():
    """Lazy load autoquant module"""
    from torchao.quantization import autoquant as _autoquant
    return _autoquant

__all__ = [
    "dtypes",
    "autoquant",
    "optim",
    "quantize_",
    "swizzle",
    "testing",
    "ops",
    "experimental",
]
