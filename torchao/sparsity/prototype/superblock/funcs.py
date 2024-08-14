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
