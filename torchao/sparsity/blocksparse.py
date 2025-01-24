from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.quantization.quant_api import _get_linear_subclass_inserter
from torchao.utils import TorchAOBaseTensor

from .bsr_triton_ops import bsr_dense_addmm, broadcast_batch_dims

aten = torch.ops.aten


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
    # TODO: Change this to call into Triton kernel directly like int_addmm
    # This way we know we must be on the hot path
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
    weight_bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, values, size=(M, K))
    N_padded = x_padded.shape[1]
    out = x_padded.new_empty((M, N_padded))
    bsr_dense_addmm(
        out,
        weight_bsr,
        x_padded,
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

    __slots__ = ["bsr_crow_indices", "bsr_col_indices", "bsr_values"]

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


    def dense(self):
        return torch.sparse_bsr_tensor(
            crow_indices=self.bsr_crow_indices,
            col_indices=self.bsr_col_indices,
            values=self.bsr_values,
            size=self.shape,
        ).to_dense()


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
            bsr.values().unsqueeze(-1))


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
        t_blocked = t.view(t.size(0), t.size(1) // 64, 64, 1)
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
    out = torch.empty((bsr.shape[0], bsr.shape[2]), dtype=bsr.dtype, device=bsr.device)
    crow_indices = bsr.crow_indices()
    blocksize = bsr.blocksize

    for i in range(crow_indices.shape[0]-1):
        start, stop = crow_indices[i], crow_indices[i+1]
        temp_sum = bsr.values()[start:stop]
        temp_sum = temp_sum.sum(dim=0).sum(dim=1)
        out[i * blocksize : (i + 1) * blocksize] = temp_sum
    
    return out


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

def next_power_of_two(n):
    assert n > 0
    return 2 ** (n.bit_length())


@implements(torch.nn.functional.linear)
def block_sparse_linear(func, types, args, kwargs):
    # linear(x, w^t)
    # linear(w, x^t)^t
    x_orig, w, bias = args
    # # TODO: Change this to do padding to make sure blocksparse.linear works
    # return torch.ops.blocksparse.linear(
    #     x, w.crow_indices(), w.col_indices(), w.values(), w.shape[0], w.shape[1], bias
    # )
    x = x_orig.reshape(-1, x_orig.size(-1)).t()
    M = w.shape[0]
    K = w.shape[1]
    N = x.shape[1]
    # TODO: Replace this with mul + sum for the mv case similar to
    # https://github.com/pytorch/pytorch/blob/a9685767773157440c162caaf125856e04e2981f/torch/_inductor/decomposition.py#L292
    # use .to_dense to get a baseline implementation that works and then use NJT for .sum and such
    # if x.size(-1) == 1:
    #     # print("USING THIS")
    #     # breakpoint()
    #     out = (torch.mul(w.unsqueeze(2), x.unsqueeze(0))).sum(dim=1)
    #     out_orig = out.t().reshape(x_orig.shape[:-1] + (M,))
    #     if bias is None:
    #         special_ret = out_orig
    #     else:
    #         special_ret = out_orig + bias
    #     return special_ret
    # else:
    # N_padded = max(16, next_power_of_two(N))
    # x_padded = torch.nn.functional.pad(x, (0, N_padded - N), 'constant', 0)
    out = torch.ops.blocksparse.addmm(
        x,
        w.crow_indices(),
        w.col_indices(),
        w.values(),
        M,
        K,
        None,
    )
    # out_orig = out[:, :x.size(-1)].t().reshape(x_orig.shape[:-1] + (M,))
    out_orig = out.t()
    if bias is None:
        return out_orig

    return out_orig + bias


def block_sparse_weight(blocksize=64):
    return _get_linear_subclass_inserter(
        partial(BlockSparseTensor.from_dense, blocksize=blocksize)
    )
