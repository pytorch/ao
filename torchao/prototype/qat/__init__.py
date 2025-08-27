# Temporary location for prototype QAT features that will
# eventually live in torchao/quantization/qat

from .nvfp4 import (
    NVFP4FakeQuantizeConfig,
    NVFP4FakeQuantizer,
)

__all__ = [
    "NVFP4FakeQuantizeConfig",
    "NVFP4FakeQuantizer",
]
