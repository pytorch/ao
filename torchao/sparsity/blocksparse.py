from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton, triton_op
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.quantization.quant_api import _get_linear_subclass_inserter
from torchao.utils import TorchAOBaseTensor

from torchao.kernel.bsr_triton_ops import bsr_dense_addmm, broadcast_batch_dims

aten = torch.ops.aten

@triton.jit
def sum_with_offsets_kernel(values, crow_indices, output, BLOCK_SIZE: tl.constexpr):
    # For each kernel invokation, we assume we are dealing with a specific row

    pid = tl.program_id(0)
    
    # Compute the start and end offset for our given row
    start = tl.load(crow_indices + pid)
    end = tl.load(crow_indices + pid + 1)

    # Number of nonzero elements in the row 
    row_nnz = end - start
    BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE

    row_block_arange = tl.arange(0, BLOCK_SIZE)
    inner_block_arange = tl.arange(0, BLOCK_SIZE)

    # Calculate correct pointer offset 
    values += BLOCK_ELEMENTS * start + BLOCK_SIZE * row_block_arange[:, None] + inner_block_arange[None, :]

    # Accumulate rowwise
    acc = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)

    # Loop over the block and accumulate the sum using offsets
    for i in range(row_nnz):
        # should I be store /loading the sumprod?
        acc += tl.sum(tl.load(values), axis=1)
        # tl.device_print("vals", tl.load(values))
        # tl.device_print("acc", acc)

        # move to next block in values
        values += BLOCK_ELEMENTS


    # Write the result to the output
    output_arange = tl.arange(0, BLOCK_SIZE)
    tl.store(output + BLOCK_SIZE * pid + output_arange, acc.to(output.dtype.element_ty))

@triton_op("blocksparse::sum", mutates_args=())
def sum_with_offsets(
    values: torch.Tensor,
    crow_indices: torch.Tensor,
    M: int,
) -> torch.Tensor:

    # Define the block size and number of blocks
    BLOCK_SIZE = values.shape[1]
    num_offsets = crow_indices.numel() - 1
    grid = lambda meta: (triton.cdiv(num_offsets, 1), )
    
    # Allocate output tensor
    y = torch.empty((M, 1), dtype=torch.bfloat16, device='cuda')

    # Launch the kernel
    wrap_triton(sum_with_offsets_kernel)[grid](values, crow_indices, y, BLOCK_SIZE)
    
    # Sum the partial results on the CPU
    return y

@torch.library.custom_op("blocksparse::int_addmm", mutates_args=())
def blocksparse_int_addmm(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    A: torch.Tensor,
    left_alpha: torch.Tensor,
    right_alpha: torch.Tensor,
) -> torch.Tensor:
    assert values.dtype == torch.int8
    M = left_alpha.shape[-1]
    K = A.shape[-2]
    N = A.shape[-1]
    weight_bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, values, size=(M, K))
    original_batch_dims_broadcasted = broadcast_batch_dims(
        blocksparse_int_addmm, weight_bsr, A
    )
    out = A.new_empty(original_batch_dims_broadcasted + (M, N), dtype=torch.bfloat16)
    return bsr_dense_addmm(
        out,
        weight_bsr,
        A,
        alpha=1,
        beta=0,
        out=out,
        left_alpha=left_alpha,
        right_alpha=right_alpha,
    ).t()


@torch.library.register_fake("blocksparse::int_addmm")
def blocksparse_int_addmm_abstract(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    A: torch.Tensor,
    left_alpha: torch.Tensor,
    right_alpha: torch.Tensor,
) -> torch.Tensor:
    N = A.shape[-1]
    M = left_alpha.shape[-1]
    # to have the same strides as the transposed result
    return torch.empty((M, N), dtype=torch.bfloat16, device=A.device).t()


# bsr wrapper custom op
@torch.library.custom_op("blocksparse::linear", mutates_args=())
def blocksparse_linear(
    A: torch.Tensor,
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    M: int,
    K: int,
    bias: torch.Tensor,
) -> torch.Tensor:
    weight_bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, values, size=(M, K))
    return torch.nn.functional.linear(A, weight_bsr, bias)


@torch.library.register_fake("blocksparse::linear")
def blocksparse_linear_abstract(
    A: torch.Tensor,
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    M: int,
    K: int,
    bias: torch.Tensor,
) -> torch.Tensor:
    new_shape = A.shape[:-1] + (M,)
    return torch.empty(new_shape, dtype=A.dtype, device=A.device)


# bsr wrapper custom op
@torch.library.custom_op("blocksparse::addmm", mutates_args=())
def blocksparse_addmm(
    x_padded: torch.Tensor,
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    M: int,
    K: int,
    bias: torch.Tensor,
) -> torch.Tensor:
    assert bias is None
    bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, values, size=(M, K))
    N_padded = x_padded.shape[1]
    out = x_padded.new_empty((M, N_padded))
    bsr_dense_addmm(
        out,
        bsr,
        x_padded,
        # (M, K),
        alpha=1,
        beta=0,
        out=out,
    )
    return out


@torch.library.register_fake("blocksparse::addmm")
def blocksparse_addmm_abstract(
    x_padded: torch.Tensor,
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    M: int,
    K: int,
    bias: torch.Tensor,
) -> torch.Tensor:
    N_padded = x_padded.shape[1]
    return x_padded.new_empty((M, N_padded))


# Subclass definition
class BlockSparseTensor(TorchAOBaseTensor):
    bsr_crow_indices: Optional[torch.Tensor]
    bsr_col_indices: Optional[torch.Tensor]
    bsr_values: Optional[torch.Tensor]
    blocksize: int 

    __slots__ = ["bsr_crow_indices", "bsr_col_indices",  "bsr_values"]

    @staticmethod
    def __new__(  # noqa: PYI034
        cls,
        shape: torch.Size,
        blocksize: int, 
        bsr_crow_indices: Optional[torch.Tensor],
        bsr_col_indices: Optional[torch.Tensor],
        bsr_values: Optional[torch.Tensor],
        requires_grad: bool = False,
    ):
        if bsr_values is None:
            raise ValueError(
                "No values passed to BlockSparseTensor: bsr_values must be provided!"
            )
        else:
            previous_tensor = bsr_values

        kwargs = {
            "device": previous_tensor.device,
            "dtype": previous_tensor.dtype,
            "layout": previous_tensor.layout,
            "requires_grad": requires_grad,
        }
        tensor = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]
        tensor.blocksize = blocksize
        tensor.bsr_crow_indices = bsr_crow_indices
        tensor.bsr_values = bsr_values
        tensor.bsr_col_indices = bsr_col_indices
        return tensor

    def __repr__(self) -> str:  # type: ignore[override]
        assert hasattr(self, "shape")
        return f"{self.__class__.__name__}(shape={self.shape})"

    def __tensor_flatten__(self) -> Tuple[List[str], Tuple[torch.Size, bool, int]]:
        inner_tensors = list(
            filter(lambda x: getattr(self, x) is not None, self.__slots__)
        )
        tensor_meta = (self.shape, self.requires_grad, self.blocksize)
        return inner_tensors, tensor_meta

    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors,
        tensor_meta: Tuple[torch.Size, bool, int],
        outer_size,
        outer_stride,
    ) -> torch.Tensor:
        shape, requires_grad, blocksize = tensor_meta
        # print("unflatten", outer_size, outer_stride)
        return cls(
            shape=shape,
            blocksize=blocksize, 
            bsr_crow_indices=inner_tensors.get("bsr_crow_indices", None),
            bsr_col_indices=inner_tensors.get("bsr_col_indices", None),
            bsr_values=inner_tensors.get("bsr_values", None),
            requires_grad=requires_grad,
        )


    @classmethod
    def from_dense(cls, dense_tensor, blocksize):
        bsr_tensor = dense_tensor.to_sparse_bsr(blocksize)
        # bsr_tensor_t = dense_tensor.t().contiguous().to_sparse_bsr(blocksize)
        return cls(
            shape=dense_tensor.shape,
            blocksize=blocksize, 
            bsr_crow_indices=bsr_tensor.crow_indices(),
            bsr_col_indices=bsr_tensor.col_indices(),
            bsr_values=bsr_tensor.values(),
            requires_grad=False,
        )

    def apply_fn_to_shard(self, func):
        return BlockSparseTensor(
            shape=self.shape,
            blocksize=self.blocksize, 
            bsr_crow_indices=func(self.bsr_crow_indices),
            bsr_col_indices=func(self.bsr_col_indices),
            bsr_values=func(self.bsr_values),
            requires_grad=self.requires_grad,
        )


# Subclass op dispatch registration
implements = BlockSparseTensor.implements


@implements(aten.detach.default)
def block_sparse_detach(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0].apply_fn_to_shard(torch.detach)
    )


@implements(aten.unsqueeze.default)
def block_sparse_unsqueeze(func, types, args, kwargs):
    assert len(args) == 2
    assert len(kwargs) == 0
    assert args[-1] == 2
    bsr = args[0]
    assert bsr.dim() == 2
    assert not bsr.requires_grad
    return BlockSparseTensor(bsr.shape + (1,),
                             bsr.blocksize, 
            bsr.crow_indices(),
            bsr.col_indices(),
            bsr.values().unsqueeze(-1), 
            requires_grad=False)


@implements(aten.mul.Tensor)
def block_sparse_mul(func, types, args, kwargs):
    assert len(args) == 2
    assert len(kwargs) == 0
    bsr, t = args

    def my_mul(bsr, t):
        assert isinstance(bsr, BlockSparseTensor)
        assert isinstance(t, torch.Tensor)
        assert bsr.dim() == 3
        assert t.dim() == 3
        assert not bsr.requires_grad
        assert t.size(0) == 1
        t_blocked = t.view(t.size(0), t.size(1) // bsr.blocksize, bsr.blocksize, 1)
        masked_t = t_blocked.transpose(0, 1).index_select(0, bsr.col_indices())
        new_values = bsr.values() * masked_t
        return BlockSparseTensor(bsr.shape,
                                 bsr.blocksize,        
                                 bsr.crow_indices(),
                                 bsr.col_indices(),
                                 new_values)

    if isinstance(bsr, torch.Tensor) and isinstance(t, BlockSparseTensor):
        return my_mul(t, bsr)
    return my_mul(bsr, t)


@implements(aten.sum.dim_IntList)
def block_sparse_sum(func, types, args, kwargs):
    bsr, dim = args
    assert type(dim) == list
    assert len(dim) == 1
    dim = dim[0]
    bsr_dim = bsr.dim()
    assert dim == 1
    return torch.ops.blocksparse.sum(bsr.values(), bsr.crow_indices(), bsr.shape[0])
    

@implements(aten.values.default)
def block_sparse_values(func, types, args, kwargs):
    return args[0].bsr_values.detach()


@implements(aten.crow_indices.default)
def block_sparse_crow_indices(func, types, args, kwargs):
    return args[0].bsr_crow_indices.detach()


@implements(aten.col_indices.default)
def block_sparse_col_indices(func, types, args, kwargs):
    return args[0].bsr_col_indices.detach()

@implements(aten._nnz.default)
def block_sparse__nnz(func, types, args, kwargs):
    return args[0].bsr_values.shape[0]


@implements(torch.nn.functional.linear)
def block_sparse_linear(func, types, args, kwargs):
    x_orig, w, bias = args
    x = x_orig.reshape(-1, x_orig.size(-1)).t()
    M = w.shape[0]
    K = w.shape[1]
    N = x.shape[1]

    out = torch.ops.blocksparse.addmm(
        x,
        w.crow_indices(),
        w.col_indices(),
        w.values(),
        M,
        K,
        None,
    )
    out_orig = out.t()
    if bias is None:
        return out_orig

    return out_orig + bias
