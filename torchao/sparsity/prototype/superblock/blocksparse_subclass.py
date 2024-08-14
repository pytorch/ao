import torch
from typing import Optional, Tuple, List, Dict, Any, Callable

import torch
from typing import Optional, Tuple, List, Dict, Any, Callable
from torch.sparse._triton_ops import bsr_dense_addmm_meta, broadcast_batch_dims, bsr_dense_addmm


@torch.library.custom_op("blocksparse::dense_addmm", mutates_args=())
def custom_bsr_op(bias : torch.Tensor, weight : torch.Tensor, A_2d : torch.Tensor) -> torch.Tensor:
    return bsr_dense_addmm(bias, weight, A_2d)

# Write the FakeTensor kernel
@torch.library.register_fake("blocksparse::dense_addmm")
def custom_bsr_op_abstract(bias : torch.Tensor, weight : torch.Tensor, A_2d : torch.Tensor) -> torch.Tensor:
    return bsr_dense_addmm_meta(bias, weight, A_2d)

def block_sparse_detach(func, types, args, kwargs):
    self = args[0]
    return BlockSparseTensor(
        shape=args[0].shape,
        bsr_crow_indicies=args[0].bsr_crow_indicies.detach(),
        bsr_col_indicies=args[0].bsr_col_indicies.detach(),
        bsr_values=args[0].bsr_values.detach(),
        blocksize=args[0].blocksize,
        requires_grad=args[0].requires_grad,
        transposed = args[0].transposed,
    )

def block_sparse_t(func, types, args, kwargs):
    return BlockSparseTensor(
        shape=args[0].shape,
        bsr_crow_indicies=args[0].bsr_crow_indicies.detach(),
        bsr_col_indicies=args[0].bsr_col_indicies.detach(),
        bsr_values=args[0].bsr_values.detach(),
        blocksize=args[0].blocksize,
        requires_grad=args[0].requires_grad,
        transposed = not args[0].transposed,
    )

def block_sparse_values(func, types, args, kwargs):
    return args[0].bsr_values.detach()

def block_sparse_crow_indicies(func, types, args, kwargs):
    return args[0].bsr_crow_indicies.detach()

def block_sparse_col_indicies(func, types, args, kwargs):
    return args[0].bsr_col_indicies.detach()

def block_sparse__nnz(func, types, args, kwargs):
    return args[0].bsr_values.numel()
    
def block_sparse_dense_addmm(func, types, args, kwargs):
    bias, weight, A_2d = args
    return bsr_dense_addmm(bias, weight, A_2d)

class BlockSparseTensor(torch.Tensor):
    bsr_crow_indicies: Optional[torch.Tensor]
    bsr_col_indicies: Optional[torch.Tensor]
    bsr_values: Optional[torch.Tensor]
    blocksize: int

    __slots__ = ["bsr_crow_indicies", "bsr_col_indicies", "bsr_values"] 

    SPARSE_DISPATCH = {
        torch.ops.aten.detach: block_sparse_detach,
        torch.ops.aten.t: block_sparse_t,
        torch.ops.aten.values: block_sparse_values,
        torch.ops.aten.crow_indices: block_sparse_crow_indicies,
        torch.ops.aten.col_indices: block_sparse_col_indicies,
        torch.ops.aten._nnz: block_sparse__nnz,
    }

    @staticmethod
    def __new__(  # noqa: PYI034
        cls,
        shape: torch.Size,
        bsr_crow_indicies: Optional[torch.Tensor],
        bsr_col_indicies: Optional[torch.Tensor],
        bsr_values: Optional[torch.Tensor],
        blocksize: int,
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
        tensor.blocksize = blocksize
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
            self.blocksize,
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
        shape, blocksize, requires_grad = tensor_meta
        return cls(
            shape=shape,
            bsr_crow_indicies=inner_tensors.get("bsr_crow_indicies", None),
            bsr_col_indicies=inner_tensors.get("bsr_col_indicies", None),
            bsr_values=inner_tensors.get("bsr_values", None),
            blocksize=blocksize,
            requires_grad=requires_grad,
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func is torch.nn.functional.linear:
            A, weight, bias = (
                args[0],
                args[1],
                args[2] if len(args)>2 else None
            )

            shape = A.shape
            A_2d = A.view(-1, shape[-1])
            bias = bias.unsqueeze(1).expand(-1, A_2d.shape[0])
            # weight_bsr = torch.sparse_bsr_tensor(weight.crow_indicies,
            #                                      weight.col_indicies,
            #                                      weight.values,
            #                                      size=weight.shape)
            # return func(A, weight, bias)
            res = torch.ops.blocksparse.dense_addmm(bias, weight, A_2d.t())
            # res = bsr_dense_addmm(bias, weight, A_2d.t())
            return res.view(*shape[:-1], -1)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    # __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs) -> Any:
        if func._overloadpacket not in cls.SPARSE_DISPATCH:
            raise NotImplementedError(
                f"{cls.__name__} only supports a specific set of operations, "
                f"can't perform requested op ({func.__name__})"
            )
        return cls.SPARSE_DISPATCH[func._overloadpacket](func, types, args, kwargs)

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
            blocksize=blocksize,
            transposed=False,
            requires_grad=bsr_tensor.requires_grad,
        )
