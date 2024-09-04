from functools import partial

import torch
from typing import Optional, Tuple, List, Dict, Any, Callable
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.utils import TorchAOBaseTensor
from torchao.quantization.quant_api import _get_linear_subclass_inserter

aten = torch.ops.aten

# bsr wrapper custom op
@torch.library.custom_op("blocksparse::linear", mutates_args=())
def blocksparse_linear(A: torch.Tensor, crow_indices: torch.Tensor, col_indices: torch.Tensor, values: torch.Tensor, M: int, K: int, bias: torch.Tensor) -> torch.Tensor:
    weight_bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, values, size=(M, K))
    return torch.nn.functional.linear(A, weight_bsr, bias)

@torch.library.register_fake("blocksparse::linear")
def blocksparse_linear_abstract(A: torch.Tensor, crow_indices: torch.Tensor, col_indices: torch.Tensor, values: torch.Tensor, M: int, K:int , bias: torch.Tensor) -> torch.Tensor:
    new_shape = A.shape[:-1] + (bias.shape[0],)
    return torch.empty(new_shape, dtype=A.dtype, device=A.device)

# Subclass definition
class BlockSparseTensor(TorchAOBaseTensor):
    bsr_crow_indices: Optional[torch.Tensor]
    bsr_col_indices: Optional[torch.Tensor]
    bsr_values: Optional[torch.Tensor]

    __slots__ = ["bsr_crow_indices", "bsr_col_indices", "bsr_values"]

    @staticmethod
    def __new__(  # noqa: PYI034
        cls,
        shape: torch.Size,
        bsr_crow_indices: Optional[torch.Tensor],
        bsr_col_indices: Optional[torch.Tensor],
        bsr_values: Optional[torch.Tensor],
        requires_grad: bool = False,
    ):
        if bsr_values is None:
            raise ValueError("bsr values must be provided!")
        else:
            previous_tensor = bsr_values

        kwargs = {
            "device": previous_tensor.device,
            "dtype": previous_tensor.dtype,
            "layout": previous_tensor.layout,
            "requires_grad": requires_grad,
        }
        tensor = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]
        tensor.bsr_crow_indices = bsr_crow_indices
        tensor.bsr_col_indices = bsr_col_indices
        tensor.bsr_values = bsr_values
        return tensor

    def __repr__(self) -> str:  # type: ignore[override]
        assert hasattr(self, "shape")
        return f"{self.__class__.__name__}(shape={self.shape})"

    def __tensor_flatten__(self) -> Tuple[List[str], Tuple[torch.Size, bool]]:
        inner_tensors = list(
            filter(lambda x: getattr(self, x) is not None, self.__slots__)
        )
        tensor_meta = (self.shape, self.requires_grad)
        return inner_tensors, tensor_meta

    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors,
        tensor_meta: Tuple[torch.Size, bool],
        outer_size,
        outer_stride,
    ) -> torch.Tensor:
        shape,  requires_grad = tensor_meta
        return cls(
            shape=shape,
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
            bsr_crow_indices=bsr_tensor.crow_indices(),
            bsr_col_indices=bsr_tensor.col_indices(),
            bsr_values=bsr_tensor.values(),
            requires_grad=False,
        )

    def apply_fn_to_shard(self, func):
        return BlockSparseTensor(
            shape = self.shape,
            bsr_crow_indices=func(self.bsr_crow_indices),
            bsr_col_indices=func(self.bsr_col_indices),
            bsr_values=func(self.bsr_values),
            requires_grad=self.requires_grad,
        )

# Subclass op dispatch registration
implements = BlockSparseTensor.implements

@implements(aten.detach.default)
def block_sparse_detach(func, types, args, kwargs):
    return return_and_correct_aliasing(func, args, kwargs, args[0].apply_fn_to_shard(torch.detach))

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
    x, w, bias = args
    return torch.ops.blocksparse.linear(x,
                                        w.crow_indices(),
                                        w.col_indices(),
                                        w.values(),
                                        w.shape[0], w.shape[1], bias)

def block_sparse_weight(blocksize=64):
    return _get_linear_subclass_inserter(partial(BlockSparseTensor.from_dense, blocksize=blocksize))
