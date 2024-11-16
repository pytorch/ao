from typing import Optional, Tuple

import torch

from torchao.dtypes.uintx.uintx import _DTYPE_TO_BIT_WIDTH, UintxTensor
from torchao.prototype.quantization.codebook.codebook_ops import (
    choose_qparams_codebook,
    dequantize_codebook,
    quantize_codebook,
)
from torchao.utils import TorchAOBaseTensor

aten = torch.ops.aten


class CodebookQuantizedTensor(TorchAOBaseTensor):
    """
    Codebook quantized tensor subclass.

    Codebook (lookup table) quantization involves partitioning the input tensor into blocks, and replacing each block
    with the index of the closest entry in a predefined codebook.

    Fields:
      codes (torch.Tensor): Tensor of indices representing blocks in the original tensor. Each index
         maps to a corresponding codebook entry.
      codebook (torch.Tensor): Tensor representing the quantization codebook, where each entry
         corresponds to a block in the original tensor. Shape is `(codebook_size, out_block_size, in_block_size)`.
      block_size (Tuple[int, ...]): Granularity of quantization, specifying the dimensions of tensor
         blocks that share the same quantization parameters.
      shape (torch.Size): Shape of the original high-precision tensor.
      dtype (torch.dtype): dtype of the original high-precision tensor.
    """

    @staticmethod
    def __new__(
        cls,
        codes: torch.Tensor,
        codebook: torch.Tensor,
        block_size: Tuple[int, ...],
        shape: torch.Size,
        dtype=None,
        strides=None,
    ):
        kwargs = {}
        kwargs["device"] = codes.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else codes.layout
        )
        kwargs["dtype"] = dtype
        if strides is not None:
            kwargs["strides"] = strides
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        codes: torch.Tensor,
        codebook: torch.Tensor,
        block_size: Tuple[int, ...],
        shape: torch.Size,
        dtype=None,
        strides=None,
    ):
        self.codes = codes
        self.codebook = codebook
        self.block_size = block_size

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(codes={self.codes}, codebook={self.codebook}, block_size={self.block_size}, "
            f"shape={self.shape}, device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    def _quantization_type(self):
        return f"shape={self.shape}, block_size={self.block_size}, codebook_size={self.codebook.size(0)}, device={self.device}, code_dtype={self.codes.dtype}"

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if output_dtype is None:
            output_dtype = torch.float32

        if isinstance(self.codes, UintxTensor):
            codes = self.codes.get_plain()
            codes = codes.to(torch.int32)  # not sure how to index with uint8
        else:
            codes = self.codes
        return dequantize_codebook(
            codes,
            self.codebook,
            output_dtype=output_dtype,
        )

    def __tensor_flatten__(self):
        return ["codes", "codebook"], [self.block_size, self.shape, self.dtype]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        codes = tensor_data_dict["codes"]
        codebook = tensor_data_dict["codebook"]
        block_size, shape, dtype = tensor_attributes
        return cls(
            codes,
            codebook,
            block_size,
            shape if outer_size is None else outer_size,
            dtype=dtype,
            strides=outer_stride,
        )

    @classmethod
    def from_float(
        cls,
        input_tensor: torch.Tensor,
        block_size: Tuple[int, ...],
        code_dtype: torch.dtype,
        chunk_size: int = 1024,
    ):
        """
        Creates a CodebookQuantizedTensor from a floating-point tensor by performing codebook quantization.

        Args:
            input_tensor (torch.Tensor): The input floating-point tensor to quantize.
            block_size (Tuple[int, ...]): The size of the blocks for which codes are assigned.
            code_dtype (torch.dtype): The dtype of the codes.
            chunk_size (int): The chunk size to use during quantization (to control memory usage).
        """

        codebook = choose_qparams_codebook(input_tensor, block_size, code_dtype)

        codes = quantize_codebook(input_tensor, codebook, chunk_size, code_dtype)
        if code_dtype in _DTYPE_TO_BIT_WIDTH:
            codes = UintxTensor.from_uint8(codes, dtype=code_dtype)

        return cls(
            codes,
            codebook,
            block_size,
            input_tensor.shape,
            dtype=input_tensor.dtype,
        )

    def to(self, *args, **kwargs):
        # I'm not sure if this is right
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        return self.__class__(
            self.codes.to(device),
            self.codebook.to(device),
            self.block_size,
            self.shape,
            **kwargs,
        )

    def _apply_fn_to_data(self, fn):
        # Apply function to only codes?
        return self.__class__(
            fn(self.codes),
            self.codebook,
            self.block_size,
            self.shape,
            dtype=self.dtype,
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func in CODEBOOK_TORCH_FUNCTIONS:
            return CODEBOOK_TORCH_FUNCTIONS[func](*args, **kwargs)

        if any(isinstance(arg, cls) for arg in args):
            # Dequantize all instances of CodebookQuantizedTensor in args
            new_args = tuple(
                arg.dequantize() if isinstance(arg, cls) else arg for arg in args
            )

            return func(*new_args, **kwargs)
        else:
            return NotImplemented

    def detach(self):
        """
        Returns a new `CodebookQuantizedTensor`.
        """
        return self.__class__(
            self.codes.detach(),
            self.codebook.detach(),
            self.block_size,
            self.shape,
            dtype=self.dtype,
        )

    def requires_grad_(self, requires_grad=True):
        """
        Modifies the tensor's `requires_grad` status in-place.
        """
        self.codes.requires_grad_(requires_grad)
        self.codebook.requires_grad_(requires_grad)
        return self


CODEBOOK_TORCH_FUNCTIONS = {}


def implements_torch_function(torch_function):
    def decorator(func):
        CODEBOOK_TORCH_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements_torch_function(torch.Tensor.detach)
def function_detach(tensor, *args, **kwargs):
    return tensor.detach()


@implements_torch_function(torch.Tensor.requires_grad_)
def function_requires_grad_(tensor, *args, **kwargs):
    return tensor.requires_grad_(*args, **kwargs)
