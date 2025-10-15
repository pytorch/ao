# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torchao.utils import TorchAOBaseTensor

__all__ = ["CutlassSemiSparseFp8Tensor"]
aten = torch.ops.aten

class CutlassSemiSparseFp8Tensor(TorchAOBaseTensor):
    tensor_data_names = ["sparse", "scale", "meta"]

    def __new__(
        cls,
        sparse: torch.Tensor,
        meta: torch.Tensor,
        scale: torch.Tensor,
    ):
        kwargs = {}
        kwargs["device"] = sparse.device
        kwargs["dtype"] = scale.dtype
        kwargs["requires_grad"] = False
        shape = (sparse.shape[0], 2 * sparse.shape[-1])
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]


    def __init__(
        self,
        sparse: torch.Tensor,
        meta: torch.Tensor,
        scale: torch.Tensor,
    ):
        super().__init__()
        self.sparse = sparse
        self.meta = meta
        self.scale = scale

    def _quantization_type(self):
        return f"shape={self.shape}, device={self.device}, dtype={self.dtype}"

    
    @classmethod
    def from_hp(
        ):
        raise NotImplementedError("CutlassSemiSparseFp8Tensor.from_hp is not implemented yet")

    
implements = CutlassSemiSparseFp8Tensor.implements
implements_torch_function = CutlassSemiSparseFp8Tensor.implements_torch_function

CutlassSemiSparseFp8Tensor.__module__ = "torchao.quantization"

# Allow a model with CutlassSemiSparseFp8Tensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([CutlassSemiSparseFp8Tensor])
