from typing import Optional, Tuple

import torch

from torchao.dtypes.uintx.uintx_layout import _DTYPE_TO_BIT_WIDTH, UintxTensor
from torchao.prototype.quantization.codebook.codebook_ops import (
    choose_qparams_codebook,
    dequantize_codebook,
    quantize_codebook,
)
from torchao.quantization.quant_api import _get_linear_subclass_inserter
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
      scales (torch.Tensor): Scaling factors for each scale block.
      shape (torch.Size): Shape of the original high-precision tensor.
      dtype (torch.dtype): dtype of the original high-precision tensor.
    """

    @staticmethod
    def __new__(
        cls,
        codes: torch.Tensor,
        codebook: torch.Tensor,
        block_size: Tuple[int, ...],
        scales: torch.Tensor,
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
        scales: torch.Tensor,
        shape: torch.Size,
        dtype=None,
        strides=None,
    ):
        self.codes = codes
        self.codebook = codebook
        self.block_size = block_size
        self.scales = scales
        self._dtype = dtype  # not sure this is right

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(codes={self.codes}, codebook={self.codebook}, block_size={self.block_size}, scales={self.scales}, "
            f"shape={self.shape}, device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    def _quantization_type(self):
        return f"shape={self.shape}, block_size={self.block_size}, codebook_size={self.codebook.size(0)}, device={self.device}, code_dtype={self.codes.dtype}"

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if output_dtype is None:
            output_dtype = self.dtype

        if isinstance(self.codes, UintxTensor):
            codes = self.codes.get_plain()
        else:
            codes = self.codes
        if codes.dtype != torch.int32:
            # TODO: Investigate and support not casting to torch.int32 for indexing to improve performance
            codes = codes.to(torch.int32)
        return dequantize_codebook(
            codes,
            self.codebook,
            self.scales,
            output_dtype=output_dtype,
        )

    def __tensor_flatten__(self):
        return ["codes", "codebook", "scales"], [
            self.block_size,
            self.shape,
            self.dtype,
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        codes = tensor_data_dict["codes"]
        codebook = tensor_data_dict["codebook"]
        scales = tensor_data_dict["scales"]
        block_size, shape, dtype = tensor_attributes
        return cls(
            codes,
            codebook,
            block_size,
            scales,
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
        scale_block_size: int,
        chunk_size: int = 1024,
    ):
        """
        Creates a CodebookQuantizedTensor from a floating-point tensor by performing codebook quantization.

        Args:
            input_tensor (torch.Tensor): The input floating-point tensor to quantize.
            block_size (Tuple[int, ...]): The size of the blocks for which codes are assigned.
            code_dtype (torch.dtype): The dtype of the codes.
            scale_block_size (int): The size of the blocks that share a scale.
            chunk_size (int): The chunk size to use during quantization (to control memory usage).
        """

        codebook, scales = choose_qparams_codebook(
            input_tensor.to(torch.float32), block_size, scale_block_size, code_dtype
        )  # .to(torch.float32) because I think k_means isn't numerically stable

        codes = quantize_codebook(
            input_tensor.to(torch.float32), codebook, scales, chunk_size, code_dtype
        )
        if code_dtype in _DTYPE_TO_BIT_WIDTH:
            codes = UintxTensor.from_uint8(codes, dtype=code_dtype)

        codebook = codebook.to(input_tensor.dtype)
        scales = scales.to(input_tensor.dtype)

        return cls(
            codes,
            codebook,
            block_size,
            scales,
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
            self.scales.to(device),
            self.shape,
            **kwargs,
        )

    def _apply_fn_to_data(self, fn):
        # Apply function to only codes?
        return self.__class__(
            fn(self.codes),
            self.codebook,
            self.block_size,
            self.scales,
            self.shape,
            dtype=self.dtype,
        )

    def detach(self):
        """
        Returns a new `CodebookQuantizedTensor`.
        """
        return self.__class__(
            self.codes.detach(),
            self.codebook.detach(),
            self.block_size,
            self.scales.detach(),
            self.shape,
            dtype=self.dtype,
        )

    def requires_grad_(self, requires_grad=True):
        """
        Modifies the tensor's `requires_grad` status in-place.
        """
        self.codes.requires_grad_(requires_grad)
        self.codebook.requires_grad_(requires_grad)
        self.scales.requires_grad_(requires_grad)
        return self

    @property
    def dtype(self):
        # Hacky way to avoid recursion with __torch_function__
        return self._dtype


def codebook_weight_only(
    dtype=torch.uint4,
    block_size: Tuple[int, int] = (1, 1),
    scale_block_size: int = None,
):
    """
    Applies codebook weight-only quantization to linear layers.

    Args:
        dtype: torch.uint1 to torch.uint8, torch.int32 supported.
        block_size: Tuple of (out_features, in_features) to control quantization granularity.
        scale_block_size (int): The size of the blocks that share a scale
    Returns:
        Callable for quantization transformation.
    """

    def apply_codebook_quantization(weight, scale_block_size):
        if weight.numel() > 2**27:
            return weight  # k_means is too numerically unstable
        if scale_block_size is None:
            scale_block_size = weight.shape[1]
        quantized = CodebookQuantizedTensor.from_float(
            weight,
            block_size=block_size,
            code_dtype=dtype,
            scale_block_size=scale_block_size,
        )
        return quantized

    return _get_linear_subclass_inserter(
        apply_codebook_quantization, scale_block_size=scale_block_size
    )



import logging

from torch.utils._python_dispatch import return_and_correct_aliasing

logger = logging.getLogger(__name__)

# aten = torch.ops.aten

_CODEBOOK_QLINEAR_DISPATCH_TABLE = {}

def register_codebook_quantized_linear_dispatch(dispatch_condition, impl):
    """
    Register a dispatch for codebook-based quantized linear op with a (condition, impl) pair.
    Both must accept (input, weight, bias).
    """
    _CODEBOOK_QLINEAR_DISPATCH_TABLE[dispatch_condition] = impl

def deregister_codebook_quantized_linear_dispatch(dispatch_condition):
    if dispatch_condition in _CODEBOOK_QLINEAR_DISPATCH_TABLE:
        del _CODEBOOK_QLINEAR_DISPATCH_TABLE[dispatch_condition]
    else:
        logger.warning(
            f"Attempting to deregister non-existent codebook dispatch condition: {dispatch_condition}"
        )

class CodebookLinearNotImplementedError(NotImplementedError):
    """Thin wrapper around NotImplementedError to make codebook errors more explicit."""
    pass

@staticmethod
def _codebook_linear_op(input_tensor: torch.Tensor,
                        weight_tensor: CodebookQuantizedTensor,
                        bias: torch.Tensor):
    """
    Tries each (dispatch_condition, impl) in the codebook quantized linear dispatch table.
    Raises if no specialized path is found.
    """
    for condition, impl in _CODEBOOK_QLINEAR_DISPATCH_TABLE.items():
        if condition(input_tensor, weight_tensor, bias):
            return impl(input_tensor, weight_tensor, bias)
    raise CodebookLinearNotImplementedError(
        "No specialized codebook dispatch found for quantized linear op."
    )

# Attach the _codebook_linear_op to the CodebookQuantizedTensor class
CodebookQuantizedTensor._codebook_linear_op = _codebook_linear_op

def adapt_codebook_1x16(cqt):
    """
    Given a CodebookQuantizedTensor `cqt` with block_size=(1, 16),
    reshape codebook, codes, scales into the layout needed by AQLM’s 1x16 kernel.

    Returns:
        codebooks_aqlm, codes_aqlm, scales_aqlm
    """
    # We expect codebook.shape == [codebook_size, 1, 16].
    # AQLM requires shape [num_codebooks=1, codebook_size, out_group_size=1, in_group_size=16].
    codebooks_aqlm = cqt.codebook.unsqueeze(0) #.contiguous()

    # AQLM expects codes.shape == [num_out_groups, num_in_groups, num_codebooks].
    # `cqt.codes` is [out_groups, in_groups], we just add the last dim:
    codes_aqlm = cqt.codes.unsqueeze(-1) #.contiguous()

    # AQLM expects scales.shape == [num_out_groups, 1, 1, 1].
    # `cqt.scales` is [num_out_groups, num_scale_groups=1, 1] do:
    scales_aqlm = cqt.scales.unsqueeze(-1) #.contiguous()

    return codebooks_aqlm, codes_aqlm, scales_aqlm

def _linear_aqlm_code1x16_check(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    bias: torch.Tensor
) -> bool:

    # don't need adapt_codebook_1x16 and other reshaping if refactored to follow AQLM data representation
    codebook_size, out_group_size, in_group_size = weight_tensor.codebook.shape
    num_codebooks = 1 # right now this is hardcoded, won't be if supporting AQLM

    return (
        isinstance(weight_tensor, CodebookQuantizedTensor)
        and (weight_tensor.codebook.device.type, num_codebooks, codebook_size, out_group_size) == (
            "cuda",
            1,
            65536,
            1,
        ) 
        and in_group_size in [8, 16]
    )

def _linear_aqlm_code1x16_impl(
    input_tensor: torch.Tensor,
    weight_tensor: CodebookQuantizedTensor,
    bias: torch.Tensor
) -> torch.Tensor:
    """
    Codebook linear implementation using the `code1x16_matmat_dequant` op.
    """

    codebook, codes, scales = adapt_codebook_1x16(weight_tensor)
    from torchao.ops import code1x16_matmat, code1x16_matmat_dequant

    return code1x16_matmat(
        input=input_tensor,
        codes=codes,
        codebooks=codebook,
        scales=scales,
        bias=bias,
    )

register_codebook_quantized_linear_dispatch(
    _linear_aqlm_code1x16_check,
    _linear_aqlm_code1x16_impl
)


implements = CodebookQuantizedTensor.implements

@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if not input_tensor.is_floating_point():
        raise NotImplementedError(
            f"{func} is not implemented for non floating point input"
        )

    # using try/except here so that we can have a general fallback when input_tensor/weight_tensor
    # is not picked up by any of the dispatch paths in `_codebook_linear_op`, this allows us to
    # make the branches easier to understand in `_codebook_linear_op`
    try:
        return weight_tensor._codebook_linear_op(input_tensor, weight_tensor, bias)
    except CodebookLinearNotImplementedError:
        if isinstance(input_tensor, CodebookQuantizedTensor):
            input_tensor = input_tensor.dequantize()
        if isinstance(weight_tensor, CodebookQuantizedTensor):
            weight_tensor = weight_tensor.dequantize()
        return torch.nn.functional.linear(input_tensor, weight_tensor, bias)