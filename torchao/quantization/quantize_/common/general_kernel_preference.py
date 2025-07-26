from enum import Enum

import torch

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5


# can switch to StrEnum (https://docs.python.org/3/library/enum.html#enum.StrEnum)
# after python 3.10 is end of life (https://devguide.python.org/versions/)
class GeneralKernelPreference(str, Enum):
    """Enum for specifying the groups of kernels that's used for quantization, matrix multiplication
    or other compute ops for quantized tensor
    """

    """Use the most efficient quantize and mm kernels chosen for user based on hardware and library availabilities and versions etc., for example, for float8 dynamic quantization, in SM89, use scaled_mm and in SM90+ and when fbgemm library is installed, we'll use `torch.ops.fbgemm.f8f8bf16_rowwise`
    """
    DEFAULT = "default"

    """Use torch native quantized mm kernels such as `torch._scaled_mm` (short term) and
    (https://github.com/pytorch/pytorch/issues/157950) (long term)
    """
    TORCH = "torch"

    """Use fbgemm quantized mm kernels such as `torch.ops.fbgemm.f8f8bf16_rowwise`
    """
    FBGEMM = "fbgemm"


if TORCH_VERSION_AT_LEAST_2_5:
    torch.serialization.add_safe_globals([GeneralKernelPreference])
