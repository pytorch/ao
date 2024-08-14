import torch
from typing import Optional, Tuple, List, Dict, Any, Callable

import torch
from typing import Optional, Tuple, List, Dict, Any, Callable
from torch.sparse._triton_ops import bsr_dense_addmm_meta, broadcast_batch_dims, prepare_inputs, bsr_dense_addmm
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.dtypes.utils import (
    _implements,
    _dispatch__torch_function__,
    _dispatch__torch_dispatch__,
)
aten = torch.ops.aten

@torch.library.custom_op("blocksparse::linear", mutates_args=())
def blocksparse_linear(A: torch.Tensor, crow_indices: torch.Tensor, col_indices: torch.Tensor, values: torch.Tensor, M: int, K: int, bias: torch.Tensor) -> torch.Tensor:
    shape = A.shape
    A_2d = A.view(-1, shape[-1])
    bias = bias.unsqueeze(1).expand(-1, A_2d.shape[0])
    weight_bsr = BlockSparseTensor(
            shape = torch.Size([M, K]),
            bsr_crow_indicies=crow_indices,
            bsr_col_indicies=col_indices,
            bsr_values=values,
        )
    res = bsr_dense_addmm(bias, weight_bsr, A_2d.t())
    res = res.view(*shape[:-1], -1)
    return res
    # return bsr_dense_addmm(bias, weight, A_2d)

# # Write the FakeTensor kernel
@torch.library.register_fake("blocksparse::linear")
def blocksparse_linear_abstract(A: torch.Tensor, crow_indices: torch.Tensor, col_indices: torch.Tensor, values: torch.Tensor, M: int, K:int , bias: torch.Tensor) -> torch.Tensor:
    new_shape = A.shape[:-1] + (bias.shape[0],)
    return torch.empty(new_shape, dtype=A.dtype, device=A.device)


class BlockSparseTensor(torch.Tensor):
    bsr_crow_indicies: Optional[torch.Tensor]
    bsr_col_indicies: Optional[torch.Tensor]
    bsr_values: Optional[torch.Tensor]

    __slots__ = ["bsr_crow_indicies", "bsr_col_indicies", "bsr_values"] 

    implements = classmethod(_implements)
    __torch_dispatch__ = classmethod(_dispatch__torch_dispatch__)
    __torch_function__ = classmethod(_dispatch__torch_function__)

    @staticmethod
    def __new__(  # noqa: PYI034
        cls,
        shape: torch.Size,
        bsr_crow_indicies: Optional[torch.Tensor],
        bsr_col_indicies: Optional[torch.Tensor],
        bsr_values: Optional[torch.Tensor],
        transposed: bool = False,
        requires_grad: bool = False,
    ):
        if bsr_values is not None:
            previous_tensor = bsr_values
        else:
            raise ValueError("bsr values must be provided!")

        kwargs = {
            "device": previous_tensor.device,
            "dtype": previous_tensor.dtype,
            "layout": previous_tensor.layout,
            "requires_grad": requires_grad,
        }
        tensor = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]
        tensor.bsr_crow_indicies = bsr_crow_indicies
        tensor.bsr_col_indicies = bsr_col_indicies
        tensor.bsr_values = bsr_values
        tensor.transposed = transposed
        return tensor

    def __repr__(self) -> str:  # type: ignore[override]
        assert hasattr(self, "shape")
        return f"{self.__class__.__name__}(shape={self.shape})"

    def __tensor_flatten__(
        self,
    ) -> Tuple[List[str], Tuple[torch.Size, int, bool]]:
        inner_tensors = list(
            filter(lambda x: getattr(self, x) is not None, self.__slots__)
        )
        tensor_meta = (
            self.shape,
            self.requires_grad,
        )
        return inner_tensors, tensor_meta

    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors,
        tensor_meta: Tuple[torch.Size, int, bool],
        outer_size,
        outer_stride,
    ) -> torch.Tensor:
        shape,  requires_grad = tensor_meta
        return cls(
            shape=shape,
            bsr_crow_indicies=inner_tensors.get("bsr_crow_indicies", None),
            bsr_col_indicies=inner_tensors.get("bsr_col_indicies", None),
            bsr_values=inner_tensors.get("bsr_values", None),
            requires_grad=requires_grad,
        )

    @classmethod
    def from_dense(cls, dense_tensor, blocksize):
        bsr_tensor = dense_tensor.to_sparse_bsr(blocksize)
        crow_indicies = bsr_tensor.crow_indices()
        col_indicies = bsr_tensor.col_indices()
        values = bsr_tensor.values()
        return cls(
            shape=dense_tensor.shape,
            bsr_crow_indicies=crow_indicies,
            bsr_col_indicies=col_indicies,
            bsr_values=values,
            transposed=False,
            requires_grad=False,
        )

    def apply_fn_to_shard(self, func):
        return BlockSparseTensor(
            shape = self.shape,
            bsr_crow_indicies=func(self.bsr_crow_indicies),
            bsr_col_indicies=func(self.bsr_col_indicies),
            bsr_values=func(self.bsr_values),
            transposed=self.transposed,
            requires_grad=self.requires_grad,
        )

    def transpose(self):
        return BlockSparseTensor(
            shape = torch.Size([self.shape[-1], self.shape[0]]),
            bsr_crow_indicies=self.bsr_crow_indicies,
            bsr_col_indicies=self.bsr_col_indicies,
            bsr_values=self.bsr_values,
            transposed=not self.transposed,
            requires_grad=self.requires_grad,
        )


implements = BlockSparseTensor.implements

@implements(aten.detach.default)
def block_sparse_detach(func, types, args, kwargs):
    return return_and_correct_aliasing(func, args, kwargs, args[0].apply_fn_to_shard(torch.detach))

@implements(aten.t.default)
def block_sparse_transpose(func, types, args, kwargs):
    return return_and_correct_aliasing(func, args, kwargs, args[0].transpose())

@implements(aten.values.default)
def block_sparse_values(func, types, args, kwargs):
    return args[0].bsr_values.detach()

@implements(aten.crow_indices.default)
def block_sparse_crow_indicies(func, types, args, kwargs):
    return args[0].bsr_crow_indicies.detach()

@implements(aten.col_indices.default)
def block_sparse_col_indices(func, types, args, kwargs):
    return args[0].bsr_col_indicies.detach()

@implements(aten._nnz.default)
def block_sparse__nnz(func, types, args, kwargs):
    return args[0].bsr_values.shape[0]
