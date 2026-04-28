# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from torchao.utils import TorchAOBaseTensor


class GPTQObserverTensor(TorchAOBaseTensor):
    tensor_data_names = ["hp_data", "total_batches"]
    optional_tensor_data_names = ["hessian"]
    tensor_attribute_names = []

    def __new__(cls, hp_data: torch.Tensor, total_batches, hessian=None):
        shape = hp_data.shape
        kwargs = {}
        kwargs["device"] = hp_data.device
        kwargs["dtype"] = hp_data.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, hp_data: torch.Tensor, total_batches, hessian=None):
        super().__init__()
        self.hp_data = hp_data
        self.hessian = hessian
        if isinstance(total_batches, torch.Tensor):
            self.total_batches = total_batches
        elif len(self.hp_data.shape) == 3:
            # TODO(future PR): audit whether we need to change this
            # from `total_batches` (current) to something like `total_tokens`,
            # to ensure that each token is weighted equally in the 3d case.
            self.total_batches = torch.zeros(
                self.hp_data.shape[0], dtype=torch.int64, device=self.hp_data.device
            )
        else:
            self.total_batches = torch.zeros(
                1, dtype=torch.int64, device=self.hp_data.device
            )

        # initialize hessian
        if self.hessian is None:
            assert self.hp_data.is_contiguous()
            feature_dim = self.hp_data.shape[-1]
            if len(self.hp_data.shape) == 2:
                self.hessian = torch.zeros(
                    feature_dim,
                    feature_dim,
                    dtype=torch.float32,
                    device=self.hp_data.device,
                )
            else:
                assert len(self.hp_data.shape) == 3, "unsupported"
                expert_dim = self.hp_data.shape[0]
                self.hessian = torch.zeros(
                    expert_dim,
                    feature_dim,
                    feature_dim,
                    dtype=torch.float32,
                    device=self.hp_data.device,
                )

    @staticmethod
    def _update_single_hessian(
        x: torch.Tensor, hessian: torch.Tensor, total_batches: torch.Tensor
    ):
        """Update a single 2D Hessian and total_batches in-place."""
        shape = x.shape
        n = 1 if len(shape) == 2 else shape[0]
        x = x.reshape(-1, shape[-1])

        # cast to Python int64 for optimal type promotion semantics
        # Note: there is definitely a better way to get ^, saving for
        # a follow-up PR. For now, this preserves numerics.
        tb = total_batches.item()
        if tb > 0:
            hessian *= tb / (tb + n)

        total_batches += n
        # cast to Python int64 for optimal type promotion semantics
        # Note: there is definitely a better way to get ^, saving for
        # a follow-up PR. For now, this preserves numerics.
        tb = total_batches.item()

        x = ((2 / tb) ** (1 / 2)) * x.t()
        hessian += x.matmul(x.t())

    def update_2d(self, input: torch.Tensor):
        x = input.float().to(self.hp_data.device)
        self._update_single_hessian(x, self.hessian, self.total_batches[0:1])

    def update_3d(self, input: torch.Tensor):
        x = input.float().to(self.hp_data.device)
        # TODO(future PR): optimize if this is too slow
        for e_idx in range(self.hessian.shape[0]):
            x_cur = x[e_idx]
            h_cur = self.hessian[e_idx]
            total_batches = self.total_batches[e_idx : e_idx + 1]
            self._update_single_hessian(x_cur, h_cur, total_batches)

    def update_3d_with_offs(self, input: torch.Tensor, offs: torch.Tensor):
        x = input.float().to(self.hp_data.device)
        # offs is cumulative end indices; expert e gets rows [prev_end : offs[e]]
        # Pull offs to CPU once to avoid a GPU->CPU sync per expert.
        # TODO(future PR): optimize if this is too slow
        offs_cpu = offs.tolist()
        prev_end = 0
        for e_idx in range(self.hessian.shape[0]):
            end = offs_cpu[e_idx]
            if end == prev_end:
                continue
            x_cur = x[prev_end:end]
            h_cur = self.hessian[e_idx]
            total_batches = self.total_batches[e_idx : e_idx + 1]
            self._update_single_hessian(x_cur, h_cur, total_batches)
            prev_end = end

    @classmethod
    def from_hp(cls, hp_tensor):
        return GPTQObserverTensor(hp_tensor, 0, None)


implements = GPTQObserverTensor.implements
implements_torch_function = GPTQObserverTensor.implements_torch_function
aten = torch.ops.aten


@implements(aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if isinstance(weight_tensor, GPTQObserverTensor):
        weight_tensor.update_2d(input_tensor.detach())
        return F.linear(input_tensor, weight_tensor.hp_data, bias)
    else:
        raise ValueError(
            f"Expected weight_tensor to be GPTQObserverTensor, got: {type(weight_tensor)}"
        )


@implements(aten.transpose.int)
def _(func, types, args, kwargs):
    self, dim0, dim1 = args[0], args[1], args[2]
    assert {dim0, dim1} == {-2, -1} or {dim0, dim1} == {
        self.hp_data.ndim - 2,
        self.hp_data.ndim - 1,
    }, f"only transpose of last two dims is supported, got dims {dim0}, {dim1}"
    new_data = func(self.hp_data, dim0, dim1)
    new_hessian = func(self.hessian, dim0, dim1)
    return GPTQObserverTensor(new_data, self.total_batches, new_hessian)


@implements(aten.bmm.default)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor = (
        args[0],
        args[1],
    )
    weight_tensor.update_3d(input_tensor.detach())
    return func(input_tensor, weight_tensor.hp_data)


@implements([aten._grouped_mm.default])
def _(func, types, args, kwargs):
    mat_a, mat_b = args[0], args[1]
    offs = args[2] if len(args) > 2 else kwargs.get("offs", None)
    assert offs is not None, "offs is required for grouped_mm"
    assert isinstance(mat_b, GPTQObserverTensor)
    mat_b.update_3d_with_offs(mat_a.detach(), offs)
    return func(mat_a, mat_b.hp_data, offs)
