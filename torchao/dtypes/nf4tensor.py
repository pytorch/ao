import functools
import math
import sys
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch._prims_common import make_contiguous_strides_for
from torch.distributed.device_mesh import DeviceMesh

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

aten = torch.ops.aten

c10d_functional = torch.ops.c10d_functional


NF4_OPS_TABLE: Dict[Any, Any] = {}


_INNER_TENSOR_NAMES_FOR_SHARDING = [
    "quantized_scalers",
    "quantization_factor",
    "quantized_data",
]

# Note: Quantize in Chunks
# During quantization to NF4, one of the steps to convert from the original float number
# to the index of the nearest value in the NF4 format. This can cause a large memory spike
# Due to intermediates of the quantization process. Instead we process the original
# tensor in chunks. This is a tradeoff between memory and speed. This number seems to
# strike a good balance between memory and speed
CHUNK_SIZE = 1024**2


def same_metadata(a: "NF4Tensor", b: "NF4Tensor"):
    both_nf4 = isinstance(a, NF4Tensor) and isinstance(b, NF4Tensor)
    return (
        both_nf4
        and a.block_size == b.block_size
        and a.scaler_block_size == b.scaler_block_size
        and a.n_blocks == b.n_blocks
    )


def implements(aten_ops):
    """Use this decorator to implement a function for an aten op in __torch_dispatch__"""

    def decorator(func):
        for op in aten_ops:
            NF4_OPS_TABLE[op] = func
        return func

    return decorator


def construct_nf4_args(nf4tensor: "NF4Tensor", kwargs: Optional[Dict[str, Any]] = None):
    if kwargs is None:
        kwargs = {}
    tensor_meta = SubclassTensorArgs(
        kwargs.get("size", nf4tensor.size()),
        kwargs.get("stride", nf4tensor.stride()),
        kwargs.get("storage_offset", nf4tensor.storage_offset()),
        kwargs.get("dtype", nf4tensor.dtype),
        kwargs.get("device", nf4tensor.device),
        kwargs.get("requires_grad", nf4tensor.requires_grad),
    )
    return (
        tensor_meta,
        kwargs.get("block_size", nf4tensor.block_size),
        kwargs.get("n_blocks", nf4tensor.n_blocks),
        kwargs.get("scaler_block_size", nf4tensor.scaler_block_size),
        kwargs.get("quantized_scalers", nf4tensor.quantized_scalers),
        kwargs.get("quantization_factor", nf4tensor.quantization_factor),
        kwargs.get("scaler_mean", nf4tensor.scaler_mean),
        kwargs.get("quantized_data", nf4tensor.quantized_data),
        kwargs.get("nf4", nf4tensor.nf4),
    )


# __torch_dispatch__ utils: apply aten op to inner tensors
def apply_to_inner_tensors(nf4tensor: "NF4Tensor", aten_op, args, kwargs):
    attr_to_tensor = {}
    for attr in _INNER_TENSOR_NAMES_FOR_SHARDING:
        attr_to_tensor[attr] = aten_op(getattr(nf4tensor, attr), *args, **kwargs)
    return attr_to_tensor


# __torch_function__ utils: call tensor ops from inner tensors
def call_from_inner_tensors(nf4tensor: "NF4Tensor", method_name: str, args, kwargs):
    attr_to_tensor = {}
    for attr in _INNER_TENSOR_NAMES_FOR_SHARDING:
        inner_tensor = getattr(nf4tensor, attr)
        func = getattr(inner_tensor, method_name)
        attr_to_tensor[attr] = func(*args, **kwargs)
    return attr_to_tensor


class CompareOp(Enum):
    EQ = auto()
    LT = auto()


def expect_num_of_args(op: CompareOp, num: int, msg: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(aten_op, args, kwargs=None):
            if op == CompareOp.LT and not (len(args) < num):
                raise NotImplementedError(msg)
            return func(aten_op, args, kwargs)

        return wrapper

    return decorator


def expect_arg_value_at_k(k: int, op: CompareOp, value: Any, msg: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(aten_op, args, kwargs=None):
            if op == CompareOp.EQ and not (args[k] == value):
                raise NotImplementedError(msg + str(args[k]))
            return func(aten_op, args, kwargs)

        return wrapper

    return decorator


def expect_args_len_at_k(k: int, op: CompareOp, value: Any, msg: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(aten_op, args, kwargs=None):
            if op == CompareOp.LT and not (len(args[k]) < value):
                raise NotImplementedError(msg + str(len(args[k])))
            elif op == CompareOp.EQ and not (len(args[k]) == value):
                raise NotImplementedError(msg + str(len(args[k])))
            return func(aten_op, args, kwargs)

        return wrapper

    return decorator


@implements([torch.ops.aten.detach])
def noop_detach(func, *args, **kwargs):
    return args[0][0]


@implements([torch.ops.aten.clone.default])
def clone(func, *args, **kwargs):
    return to_nf4(args[0][0].get_original_weight())


@implements(
    [
        aten.detach.default,
    ]
)
def nf4_detach(aten_op, args, kwargs=None):
    nf4tensor = args[0]
    updated_attrs = apply_to_inner_tensors(nf4tensor, aten_op, args[1:], kwargs)
    return NF4Tensor(*construct_nf4_args(nf4tensor, updated_attrs))


@implements(
    [
        aten.empty_like.default,
    ]
)
def nf4_empty_like(aten_op, args, kwargs=None):
    nf4tensor = args[0]
    updated_attrs = apply_to_inner_tensors(nf4tensor, aten_op, args[1:], kwargs)
    if kwargs is not None and len(kwargs):
        for key, value in kwargs.items():
            updated_attrs[key] = value
    return NF4Tensor(*construct_nf4_args(nf4tensor, updated_attrs))


@implements(
    [
        aten.split.Tensor,
    ]
)
def nf4_split(aten_op, args, kwargs=None):
    if len(args) == 3 and args[2] != 0:
        raise NotImplementedError(f"aten.split(NF4Tensor, dim={args[2]})")
    nf4tensor = args[0]
    num_chunks = nf4tensor.size(0) // args[1]

    attr_to_chunks = {}
    for attr in _INNER_TENSOR_NAMES_FOR_SHARDING:
        inner_tensor = getattr(nf4tensor, attr)
        assert (
            inner_tensor.numel() % num_chunks == 0
        ), f"{attr}.numel() not divisible by {num_chunks}"
        chunks = aten_op(inner_tensor, inner_tensor.numel() // num_chunks, **kwargs)
        attr_to_chunks[attr] = chunks

    orig_dim = nf4tensor.dim()
    if orig_dim == 1:
        chunked_size = (nf4tensor.size(0) // num_chunks,)
    elif orig_dim == 2:
        chunked_size = (nf4tensor.size(0) // num_chunks, nf4tensor.size(1))
    else:
        chunked_size = ()
        raise NotImplementedError(
            f"aten.split(NF4Tensor) wherer NF4Tensor.dim() = {orig_dim}"
        )

    nf4_chunks = []
    for idx in range(num_chunks):
        updated_attrs = {"size": chunked_size}
        for attr, chunks in attr_to_chunks.items():
            updated_attrs[attr] = chunks[idx]
        nf4_chunks.append(NF4Tensor(*construct_nf4_args(nf4tensor, updated_attrs)))
    return nf4_chunks


@implements(
    [
        aten.new_zeros.default,
    ]
)
@expect_args_len_at_k(1, CompareOp.LT, 3, "aten.view(NF4Tensor) with len(size)=")
def nf4_new_zeros(aten_op, args, kwargs=None):
    nf4tensor = args[0]
    new_size = tuple(args[1])

    if nf4tensor.numel() % math.prod(new_size) != 0:
        raise NotImplementedError(f"aten.new_zeros(NF4Tensor) with new size {new_size}")
    ratio = nf4tensor.numel() // math.prod(new_size)

    updated_attrs = {}
    for attr in _INNER_TENSOR_NAMES_FOR_SHARDING:
        inner_tensor = getattr(nf4tensor, attr)
        assert (
            inner_tensor.size(0) % ratio == 0
        ), f"{attr}.numel() must be divisible by {ratio}"
        inner_tensor = aten_op(inner_tensor, [inner_tensor.size(0) // ratio], **kwargs)
        updated_attrs[attr] = inner_tensor
    updated_attrs["size"] = new_size

    return NF4Tensor(*construct_nf4_args(nf4tensor, updated_attrs))


@implements(
    [
        aten.slice.Tensor,
    ]
)
@expect_num_of_args(CompareOp.LT, 5, "aten.slice(NF4Tensor) with customized step")
@expect_arg_value_at_k(1, CompareOp.EQ, 0, "aten.slice(NF4Tensor) with dim=")
@expect_arg_value_at_k(2, CompareOp.EQ, 0, "aten.slice(NF4Tensor) with start=")
def nf4_slice(aten_op, args, kwargs=None):
    nf4tensor = args[0]
    # for tensor 512 x 512, tensor[:, :512] dispatch to
    # aten.slice(dim = 0, end=sys.maxsize)
    if args[3] not in [nf4tensor.size(0), sys.maxsize]:
        raise NotImplementedError(f"aten.slice(NF4Tensor) with end={args[3]}")
    return NF4Tensor(*construct_nf4_args(nf4tensor))


@implements(
    [
        aten.view.default,
    ]
)
@expect_args_len_at_k(1, CompareOp.EQ, 1, "aten.view(NF4Tensor) with len(size)=")
def nf4_view(aten_op, args, kwargs=None):
    nf4tensor = args[0]
    size = args[1]
    if size[0] != -1:
        raise NotImplementedError(f"aten.view(NF4Tensor) with size={size}")
    updated_attrs = apply_to_inner_tensors(nf4tensor, aten_op, args[1:], kwargs)
    updated_attrs.update(
        {
            "size": [nf4tensor.numel()],
            "stride": (1,),
        }
    )
    return NF4Tensor(*construct_nf4_args(nf4tensor, updated_attrs))


@implements(
    [
        aten.as_strided.default,
    ]
)
@expect_args_len_at_k(
    1, CompareOp.LT, 3, "aten.as_strided(NF4Tensor) only support dim <= 2 but got dim="
)
def nf4_as_strided(aten_op, args, kwargs=None):
    nf4tensor = args[0]
    size = args[1]
    stride = tuple(args[2])
    storage_offset = args[3]
    if math.prod(size) != nf4tensor.numel():
        raise NotImplementedError(
            f"aten.as_strided(NF4Tensor) different numel={nf4tensor.numel()} and size={size}"
        )
    if stride != make_contiguous_strides_for(size):
        raise NotImplementedError(
            f"aten.as_strided(NF4Tensor) only support continuous stride={make_contiguous_strides_for(size)} but got stride={stride}"
        )
    if nf4tensor.storage_offset() != storage_offset:
        raise NotImplementedError(
            f"aten.as_strided(NF4Tensor) only support original storage offset {nf4tensor.storage_offset()} but got {storage_offset}"
        )
    kwargs = {
        "size": torch.Size(size),
        "stride": stride,
        "storage_offset": storage_offset,
    }
    return NF4Tensor(*construct_nf4_args(nf4tensor, kwargs))


@implements([torch.ops.aten._to_copy.default])
def _to_copy(func, *args, **kwargs):
    if not args[0][0].is_contiguous():
        assert args[0][0].t().is_contiguous()
        return func(args[0][0].t()).t()
    out = args[0][0].get_original_weight().to(args[1]["dtype"])
    if "device" in args[1]:
        out = out.to(args[1]["device"])
    return out


@implements([torch.ops.aten.to.dtype])
def to_dtype(func, *args, **kwargs):
    if not args[0][0].is_contiguous():
        assert args[0][0].t().is_contiguous()
        return torch.ops.aten.to.dtype(args[0][0].t(), args[0][1]).t()
    return args[0][0].get_original_weight().to(args[0][1])


@implements([torch.ops.aten.t.default])
def t_default(func, *args, **kwargs):
    a = args[0][0]
    tensor_meta = SubclassTensorArgs(
        a.size(),
        (a.stride(1), a.stride(0)),
        a.storage_offset(),
        a.dtype,
        a.device,
        a.requires_grad,
    )
    b = NF4Tensor(
        tensor_meta,
        a.block_size,
        a.n_blocks,
        a.scaler_block_size,
        a.quantized_scalers,
        a.quantization_factor,
        a.scaler_mean,
        a.quantized_data,
        a.nf4,
    )
    return b


@implements([torch.ops.aten.mm.default])
def mm_default(func, *args, **kwargs):
    return linear_nf4(args[0][0], args[0][1])


@implements(
    [
        aten.copy_.default,
    ]
)
def copy_(func, *args, **kwargs):
    original: NF4Tensor = args[0][0]
    copy_in: torch.Tensor = args[0][1]

    # Base Case

    if same_metadata(original, copy_in):
        original_tensors = original.__tensor_flatten__()[0]
        for tensor_name in original_tensors:
            getattr(original, tensor_name).copy_(getattr(copy_in, tensor_name))
        return

    # Convert Non NF4Tensor into NF4 for copy in
    if not isinstance(copy_in, NF4Tensor):
        copy_in_nf4 = NF4Tensor.from_tensor(
            copy_in.to(original.device), original.block_size, original.scaler_block_size
        )
        return original.copy_(copy_in_nf4)

    # Other Tensor is not a NF4Tensor
    full_precision = copy_in.get_original_weight()
    same_meta_nf4 = NF4Tensor.from_tensor(
        full_precision, original.block_size, original.scaler_block_size
    )
    return original.copy_(same_meta_nf4)


@implements(
    [
        aten.is_pinned.default,
    ]
)
def nf4_is_pinned(aten_op, args, kwargs=None):
    nf4tensor = args[0]
    for attr in _INNER_TENSOR_NAMES_FOR_SHARDING:
        inner_tensor = getattr(nf4tensor, attr)
        if not aten_op(inner_tensor, *(args[1:]), **kwargs):
            return False
    return True


@implements(
    [
        aten._pin_memory.default,
    ]
)
def nf4_pin_memory(aten_op, args, kwargs=None):
    nf4tensor = args[0]
    updated_attrs = apply_to_inner_tensors(nf4tensor, aten_op, args[1:], kwargs)
    return NF4Tensor(*construct_nf4_args(nf4tensor, updated_attrs))


@implements(
    [
        aten.cat.default,
    ]
)
def nf4_cat(aten_op: torch._ops.OpOverload, args, kwargs=None):
    tensors_to_cat = args[0]
    assert all(isinstance(t, torch.Tensor) for t in tensors_to_cat)
    remaining_args = args[1:]

    ts = []
    for t in tensors_to_cat:
        assert isinstance(t, torch.Tensor)

        if isinstance(t, NF4Tensor):
            ts.append(t.get_original_weight())
        else:
            ts.append(t)

    dtype = ts[0].dtype
    assert all(t.dtype == dtype for t in ts)

    if kwargs is None:
        kwargs = {}

    tensors = aten_op(ts, *remaining_args, **kwargs)
    return tensors


@dataclass(frozen=True)
class SubclassTensorArgs:
    original_shape: torch.Size

    original_strides: Tuple
    storage_offset: int
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool


def get_block_absmax(input_tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    """Iterate through a flattened tensor getting the absmax scalers for each block

    Args:
        input_tensor: Input tensor to get scalers for
        block_size: Block size for the scanning window
    Returns:
        torch.Tensor: Tensor of scalers for each block
    """
    assert input_tensor.dim() == 1, "Input tensor must be flattened"
    assert (
        (input_tensor.numel() % block_size) == 0
    ), f"Input tensor must be divisible by block size, got {input_tensor.numel()} and {block_size}"

    n_blocks = input_tensor.numel() // block_size
    blocks = input_tensor.view(n_blocks, block_size)
    block_scalers = blocks.abs().max(dim=1).values
    return block_scalers


class NF4Tensor(torch.Tensor):
    """NF4Tensor class for converting a weight to the QLoRA NF4 format"""

    @torch._dynamo.disable
    def __new__(
        cls,
        # Args related for base tensor construction
        tensor_meta: SubclassTensorArgs,
        # Args stored on the instance
        block_size: int,
        n_blocks: int,
        scaler_block_size: int,
        quantized_scalers: torch.Tensor,
        quantization_factor: torch.Tensor,
        scaler_mean: torch.Tensor,
        quantized_data: torch.Tensor,
        nf4: torch.Tensor,
    ):
        """Create a new NF4Tensor object
        Args:
            tensor_meta: Metadata for the tensor
            block_size: Size of the quantization block
            n_blocks: Number of blocks to cover the full tensor
            scaler_block_size: Block size for the scalar quantization
            quantized_scalers: Quantized scalers data' represented a uint8 tensor
            quantization_factor: Quantization factor, single scalar represented as torch.Tensor
            scaler_mean: Mean of the scalers
            quantized_data: Quantized data represented as uint8 tensor
            nf4: NF4 tensor LUT for the quantization and dequantization

        """

        nf4tensor = torch.Tensor._make_wrapper_subclass(
            cls,
            tensor_meta.original_shape,
            tensor_meta.original_strides,
            tensor_meta.storage_offset,
            # Picked some floating dtype, but we need dtype extensibility
            dtype=tensor_meta.dtype,
            device=tensor_meta.device,
            requires_grad=tensor_meta.requires_grad,
        )
        return nf4tensor

    @torch._dynamo.disable
    def __init__(
        self,
        tensor_meta: SubclassTensorArgs,
        block_size: int,
        n_blocks: int,
        scaler_block_size: int,
        quantized_scalers: torch.Tensor,
        quantization_factor: torch.Tensor,
        scaler_mean: torch.Tensor,
        quantized_data: torch.Tensor,
        nf4: torch.Tensor,
    ):
        """Initialize the NF4Tensor class"""
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.scaler_block_size = scaler_block_size
        self.quantized_scalers = quantized_scalers
        self.quantization_factor = quantization_factor
        self.scaler_mean = scaler_mean
        self.quantized_data = quantized_data
        self.nf4 = nf4

    @classmethod
    @torch.no_grad()
    def from_tensor(
        cls,
        input_tensor: torch.Tensor,
        block_size: int,
        scaler_block_size: int,
    ):
        assert (
            input_tensor.dim() <= 2
        ), f"expect input tensor dim <= 2 but got dim = {input_tensor.dim()}"
        assert (
            input_tensor.numel() % block_size == 0
        ), f"Input tensor must be divisible by block size, got {input_tensor.numel()} and {block_size}"
        assert input_tensor.is_contiguous, "Input tensor must be contiguous!"
        # I think I want do this
        # assert not input_tensor.requires_grad, "Input tensor must not require grad"
        device = input_tensor.device
        # Cache the tensor on the class def
        nf4 = torch.tensor(
            [
                -1.0000,
                -0.6962,
                -0.5251,
                -0.3949,
                -0.2844,
                -0.1848,
                -0.0911,
                0.0000,
                0.0796,
                0.1609,
                0.2461,
                0.3379,
                0.4407,
                0.5626,
                0.7230,
                1.0000,
            ],
            device=device,
            dtype=input_tensor.dtype,
        )
        n_blocks = input_tensor.numel() // block_size
        # Double quantization
        (
            quantized_scalers,
            quantization_factor,
            scaler_mean,
        ) = cls.double_quantize_scalers(
            input_tensor.flatten(), block_size, scaler_block_size
        )
        quantized_data = cls.convert_to_norm_float_weight(
            input_tensor, n_blocks, block_size, nf4
        )
        tensor_meta = SubclassTensorArgs(
            input_tensor.size(),
            input_tensor.stride(),
            input_tensor.storage_offset(),
            input_tensor.dtype,
            input_tensor.device,
            input_tensor.requires_grad,
        )
        return cls(
            tensor_meta,
            block_size,
            n_blocks,
            scaler_block_size,
            quantized_scalers,
            quantization_factor,
            scaler_mean,
            quantized_data,
            nf4=nf4,
        )

    @staticmethod
    def double_quantize_scalers(
        input_tensor: torch.Tensor,
        block_size: int,
        scaler_block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used to achieve the double quantization of the scalers
        We take the input tensor first calculate the absmax quantization factors for each block.
        We then find the mean of our positive absmax scalers. We subtract this mean from the scalers
        And then we calculate the absmax quantization factors for each block again. We then quantize the scalers to int8.

        Args:
            input_tensor: Input tensor to convert to QLoRA format, typically a weight tensor

        Returns:
            torch.Tensor: Tensor of per_block quantization factors stored in int8 format
                size: (n_blocks)
            torch.Tensor: Tensor of per_scaler_block quantization factors stored in int16 format
                size: (n_scaler_blocks)
        """
        assert input_tensor.dim() == 1, "Input tensor must be flattened"
        assert (
            (input_tensor.numel() % scaler_block_size) == 0
        ), f"Input tensor must be divisible by block size, got {input_tensor.numel()} and {scaler_block_size}"

        # First round of quantization
        # Produces: A tensor of size (n_blocks) of input_tensor.dtype
        scalers_1 = get_block_absmax(input_tensor, block_size)
        scalers_1_mean = scalers_1.mean()
        scalers_1 = scalers_1 - scalers_1_mean
        # Second round of quantization
        assert (
            scalers_1.numel() % scaler_block_size == 0
        ), f"Number of scalers must be divisible by scaler block size, got {scalers_1.numel()} scaler_block_size {scaler_block_size} "
        n_scaler_blocks = scalers_1.numel() // scaler_block_size
        scaler_blocks = scalers_1.view(n_scaler_blocks, scaler_block_size)

        scaler_absmax = get_block_absmax(scalers_1, scaler_block_size)
        scaler_absmax = scaler_absmax.unsqueeze(-1).expand(
            n_scaler_blocks, scaler_block_size
        )

        quantization_factor = 256 / (2 * scaler_absmax)
        # Length equal to weight numel // block_size
        quantized_scaler_blocks = scaler_blocks * quantization_factor
        quantized_scaler_blocks = quantized_scaler_blocks.round()
        quantized_scaler_blocks = quantized_scaler_blocks.clamp(-128, 127)

        # This is needed to make sure that quantization_factor remains a repeated view of n_scaler_blocks
        # For some reason the 127/scaler_absmax realizes n_scaler entries when only n_scaler_blocks are needed
        # The following will grab the first entry for the n_scaler_blocks which is the same across the scaler_block_size

        quantization_factor = quantization_factor[:, 0]

        return (
            quantized_scaler_blocks.flatten().to(torch.int8),
            quantization_factor.view(n_scaler_blocks).contiguous(),
            scalers_1_mean,
        )

    def dequantize_scalers(
        self,
        input_tensor: torch.Tensor,
        quantization_factor: torch.Tensor,
        scaler_block_size: int,
    ) -> torch.Tensor:
        """Used to unpack the double quantized scalers

        Args:
            input_tensor: Input tensor to convert to QLoRA format this is the quantized scalers in int8 format
            quantization_factor: Tensor of per_scaler_block quantization factors stored in inpt_weight.dtype
            scaler_block_size: Scaler block size to use for double quantization.

        """
        assert input_tensor.dim() == 1, "Input tensor must be flattened"
        assert (
            (input_tensor.numel() % scaler_block_size) == 0
        ), f"Input tensor must be divisible by block size, got {input_tensor.numel()} and {scaler_block_size}"
        n_scaler_blocks = input_tensor.numel() // scaler_block_size
        input_tensor = input_tensor.view(n_scaler_blocks, scaler_block_size)
        dequantized = (input_tensor / quantization_factor.unsqueeze(-1)).flatten().to(
            self.dtype
        ) + self.scaler_mean
        return dequantized

    @staticmethod
    def convert_to_norm_float_weight(
        input_tensor: torch.Tensor, n_blocks: int, block_size: int, nf4: torch.Tensor
    ) -> torch.Tensor:
        """Convert a tensor to the normalized float weight format"""
        flattened_tensor = input_tensor.flatten()
        #  Since we are using uint8 we will encode 2 entries per byte
        numel = input_tensor.numel()
        assert (
            numel % 2 == 0
        ), "Number of elements must be even just to not have to think about the end"
        # Reshape the flattened tensor into blocks of size self.block_size
        blocks = flattened_tensor.view(n_blocks, block_size)

        # Scale the blocks
        scalers = get_block_absmax(input_tensor.flatten(), block_size)
        scales = scalers.unsqueeze(-1).expand(n_blocks, block_size)
        scaled_blocks = blocks / scales

        # Returns a flattened tensor with each element quantized to nf4 index
        # See Note: Quantize in Chunks
        quantized_blocks = torch.empty(
            numel, dtype=torch.uint8, device=input_tensor.device
        )
        flattened = scaled_blocks.flatten()
        for chunk_num in range(math.ceil(numel / CHUNK_SIZE)):
            start = chunk_num * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, numel)
            quantized_blocks[start:end] = NF4Tensor.quantize_tensor_nearest(
                flattened[start:end], nf4
            ).to(torch.uint8)

        # Combine the quantized elements into uint8 values
        # This lays out two consecutive elements in the same byte
        # [a, b, c, d] -> [ab, cd]
        # The size of combined blocks will be half the size of the original tensor
        combined_blocks = quantized_blocks[::2] << 4 | quantized_blocks[1::2]

        return combined_blocks.to(torch.uint8)

    def get_original_weight(self) -> torch.Tensor:
        """Get the original weight from the normalized float weight format"""
        # Since we are using uint8 we will decode 2 entries per byte
        # Shift elements down 4 and select out the bottom 4 bits
        first_elements = (self.quantized_data >> 4).to(torch.long)
        second_elements = (self.quantized_data & 0b1111).to(torch.long)

        # Dequantize every element
        dequantized_first = self.dequantize(first_elements, self.nf4)
        dequantized_second = self.dequantize(second_elements, self.nf4)

        # Build up matrix of scalers repeated for each element in the block
        # Since first and second elements make up a full block
        # we expand out to half the size of the full block
        scalers = self.dequantize_scalers(
            self.quantized_scalers, self.quantization_factor, self.scaler_block_size
        )
        repeated = scalers.unsqueeze(-1).expand(scalers.size(0), self.block_size // 2)

        scaled_first = dequantized_first * repeated.flatten()
        scaled_second = dequantized_second * repeated.flatten()

        # Flip them to be vertical and them stack them together horizontally
        # Upon flattening this will interleave the elements
        scaled_first = scaled_first.unsqueeze(-1).transpose(0, 1)
        scaled_second = scaled_second.unsqueeze(-1).transpose(0, 1)
        return torch.stack([scaled_first, scaled_second], dim=-1).reshape(self.shape)

    @staticmethod
    def quantize_tensor_nearest(value: torch.Tensor, nf4: torch.Tensor) -> torch.Tensor:
        """Quantize a float16 tensor to nf4 format to nearest and not rounded up"""
        value = value.unsqueeze(-1)  # (numel, 1)
        # Compare the value tensor with the nf4 tensor element-wise
        diff = (value - nf4).abs()
        closest_nf4 = diff.min(dim=-1).indices
        return closest_nf4

    @staticmethod
    def dequantize(value: torch.Tensor, nf4: torch.Tensor) -> torch.Tensor:
        """Dequantize a nf4 value to bfloat16 format"""
        # return nf4.index_select(0, value)
        return nf4[value]

    def __repr__(self) -> str:
        return f"Quantized Data: {self.quantized_data}\nScalers: {self.quantized_scalers}\n"

    def __str__(self) -> str:
        return f"NF4Tensor({self.shape}, {self.block_size})"

    def __tensor_flatten__(self):
        tensor_meta = SubclassTensorArgs(
            self.shape,
            self.stride(),
            self.storage_offset(),
            self.dtype,
            self.device,
            self.requires_grad,
        )
        ctx = {
            "block_size": self.block_size,
            "n_blocks": self.n_blocks,
            "scaler_block_size": self.scaler_block_size,
            "tensor_meta": tensor_meta,
        }
        return [
            "quantized_data",
            "scaler_mean",
            "quantization_factor",
            "quantized_scalers",
            "nf4",
        ], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, metadata, outer_size, outer_stride):
        assert len(inner_tensors) == 5, "Expected 5 inner tensors"
        return NF4Tensor(
            metadata["tensor_meta"],
            metadata["block_size"],
            metadata["n_blocks"],
            metadata["scaler_block_size"],
            inner_tensors["quantized_scalers"],
            inner_tensors["quantization_factor"],
            inner_tensors["scaler_mean"],
            inner_tensors["quantized_data"],
            inner_tensors["nf4"],
        )

    @classmethod
    @torch._dynamo.disable
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        """TODO we are not supporting torch dispatch at the moment
        instead we have created a Autograd.Function to handle the linear
        """
        # All ops in the  NF4_OPS_TABLE expect NF4 Tensors as inputs
        # And don't support mixed tensor subclasses. This will trigger the handler for
        # the next type in the dispatch list

        def allowed_subclasses(type):
            return (
                issubclass(cls, type)
                or issubclass(torch._subclasses.fake_tensor.FakeTensor, type)
                or issubclass(
                    torch._subclasses.functional_tensor.FunctionalTensor, type
                )
            )

        if not all(allowed_subclasses(t) for t in types):
            return NotImplemented("Up to the next one to handle")

        if func in NF4_OPS_TABLE:
            return NF4_OPS_TABLE[func](func, args, kwargs)
        raise NotImplementedError(
            f"NF4Tensor dispatch: attempting to run {func}, this is not supported"
        )

    # Do not force the Float8Tensor type on the returned tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        try:
            if func in NF4_TORCH_FUNCTIONS:
                return NF4_TORCH_FUNCTIONS[func](*args, **kwargs)
        except NotImplementedError:
            pass

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    def fsdp_pre_all_gather(
        self, mesh: DeviceMesh
    ) -> Tuple[Tuple[torch.Tensor, ...], Any]:
        return (
            self.quantized_scalers,
            self.quantization_factor,
            self.quantized_data,
        ), (
            SubclassTensorArgs(
                self.size(),
                self.stride(),
                self.storage_offset(),
                self.dtype,
                self.device,
                self.requires_grad,
            ),
            self.block_size,
            self.n_blocks,
            self.scaler_block_size,
            self.scaler_mean,
            self.nf4,
            mesh.get_group().size(),
        )

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, Tuple[torch.Tensor, ...]], None]:
        (quantized_scalers, quantization_factor, quantized_data) = all_gather_outputs
        (
            tensor_meta,
            block_size,
            n_blocks,
            scaler_block_size,
            scaler_mean,
            nf4,
            pg_size,
        ) = metadata
        if len(tensor_meta.original_shape) != 2:
            raise NotImplementedError(
                f"only support 2D shape but got dim={len(tensor_meta.original_shape)}"
            )

        new_shape = torch.Size(
            (tensor_meta.original_shape[0] * pg_size, tensor_meta.original_shape[1])
        )
        new_tensor_meta = replace(tensor_meta, original_shape=new_shape)
        if out is not None:
            # TODO: add param dtype for mixed precision
            assert isinstance(out, NF4Tensor), f"{type(out)}"
            assert (
                quantized_scalers.untyped_storage().data_ptr()
                == out.quantized_scalers.untyped_storage().data_ptr()
                and quantization_factor.untyped_storage().data_ptr()
                == out.quantization_factor.untyped_storage().data_ptr()
                and quantized_data.untyped_storage().data_ptr()
                == out.quantized_data.untyped_storage().data_ptr()
            ), "Expects out's data to be the all-gather output"
            return

        return nf4_constructor(
            new_tensor_meta,
            block_size,
            n_blocks,
            scaler_block_size,
            quantized_scalers,
            quantization_factor,
            scaler_mean,
            quantized_data,
            nf4,
        ), (quantized_scalers, quantization_factor, quantized_data)


class LinearNF4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: NF4Tensor):
        """Save the quantized nf4 weight for backward pass"""
        ctx.save_for_backward(weight)
        return F.linear(input, weight.to(input.dtype))

    @staticmethod
    def backward(ctx, grad_output):
        """The nf4 weight will never require grad so we can just return the grad_output @ weight.to(grad_output.dtype)"""
        weight: NF4Tensor = ctx.saved_tensors[0]
        return grad_output @ weight.to(grad_output.dtype), None


def linear_nf4(input: torch.Tensor, weight: NF4Tensor) -> torch.Tensor:
    """Apply a linear operation with the NF4Tensor weight

    Args:
        input: Input tensor
        weight: NF4Tensor weight
    """
    return LinearNF4.apply(input, weight)


def to_nf4(tensor, block_size: int = 64, scaler_block_size: int = 256):
    """Convert a given tensor to normalized float 4-bit tensor."""
    return NF4Tensor.from_tensor(tensor, block_size, scaler_block_size)


NF4_TORCH_FUNCTIONS = {}


def implements_torch_function(torch_function):
    def decorator(func):
        functools.update_wrapper(func, torch_function)
        NF4_TORCH_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements_torch_function(torch.Tensor.to)
def function_to_dtype(*args, **kwargs):
    tensor = args[0]
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
        *args[1:], **kwargs
    )

    # dtype is specified -> dequantize
    if dtype is not None:
        return tensor.get_original_weight().to(
            device, dtype, non_blocking, memory_format=convert_to_format
        )

    # dtype is not specified -> keep NF4
    updated_attrs = dict(device=device)
    tensor_attrs, _ = tensor.__tensor_flatten__()
    for attr in tensor_attrs:
        inner_tensor = getattr(tensor, attr)
        updated_attrs[attr] = inner_tensor.to(
            device, dtype, non_blocking, memory_format=convert_to_format
        )
    return NF4Tensor(*construct_nf4_args(tensor, updated_attrs))


@implements_torch_function(torch.Tensor.cpu)
def function_cpu(*args, **kwargs):
    # Tensor.cpu(self, memory_format)
    return args[0].to("cpu", *args[1:], **kwargs)


@implements_torch_function(torch.Tensor.cuda)
def function_cuda(*args, **kwargs):
    # Tensor.cuda(self, device, non_blocking, memory_format)
    tensor = args[0]
    updated_attrs = dict()
    tensor_attrs, _ = tensor.__tensor_flatten__()
    for attr in tensor_attrs:
        inner_tensor = getattr(tensor, attr)
        updated_attrs[attr] = inner_tensor.cuda(*args[1:], **kwargs)
    updated_attrs["device"] = updated_attrs[tensor_attrs[0]].device
    return NF4Tensor(*construct_nf4_args(tensor, updated_attrs))


@implements_torch_function(F.linear)
def _(*args, **kwargs):
    input = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None
    out = LinearNF4.apply(input, weight)
    if bias is not None:
        out = out + bias
    return out


@torch._dynamo.allow_in_graph
def nf4_constructor(
    tensor_meta: SubclassTensorArgs,
    block_size: int,
    n_blocks: int,
    scaler_block_size: int,
    quantized_scalers: torch.Tensor,
    quantization_factor: torch.Tensor,
    scaler_mean: torch.Tensor,
    quantized_data: torch.Tensor,
    nf4: torch.Tensor,
):
    return NF4Tensor(
        tensor_meta,
        block_size,
        n_blocks,
        scaler_block_size,
        quantized_scalers,
        quantization_factor,
        scaler_mean,
        quantized_data,
        nf4,
    )


if TORCH_VERSION_AT_LEAST_2_5:
    torch.serialization.add_safe_globals([NF4Tensor])
    torch.serialization.add_safe_globals([NF4Tensor])
