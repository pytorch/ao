# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# mobicham's tinygemm hqq eval script
import torch

device = "cuda"


import copy

import torch.nn.functional as F
from hqq.core.quantize import Quantizer
from hqq.core.utils import *  # noqa: F401, F403
from torch import Tensor, nn

from torchao.dtypes.utils import is_device
from torchao.utils import check_cpu_version


class HQQLinearTorchWeightOnlyInt4(torch.nn.Module):
    def __init__(
        self,
        linear_layer: nn.Module | None,
        quant_config: dict,
        del_orig: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        initialize: bool = True,
        inner_k_tiles=8,
        padding=True,
    ):
        super().__init__()

        self.ready = False
        self.in_gpu = False
        self.bias = None
        self.device = device
        self.compute_dtype = compute_dtype
        self.quant_config = copy.deepcopy(quant_config)
        self.del_orig = del_orig

        weight_quant_params = self.quant_config["weight_quant_params"]
        self.groupsize = weight_quant_params["group_size"]
        self.nbits = weight_quant_params["nbits"]
        self.inner_k_tiles = inner_k_tiles
        self.padding = padding

        assert self.nbits in [1, 2, 4], "Unsupported nbits"
        assert self.groupsize in [None, 32, 64, 128, 256], "Unsupported groupsize"
        assert self.inner_k_tiles in [2, 4, 8], "Unsupported tile"

        self.linear_layer = linear_layer
        self.compute_dtype = compute_dtype

        if initialize:
            self.initialize()

    ###################### Initializers ######################
    def initialize_with_hqq_quants(self, W_q, meta, bias=None):
        self.padding = (
            False  # Force padding off, a bit tricky to post-pad with grouping
        )

        self.set_shape(meta["shape"])
        self.process_hqq_quants(W_q, meta)
        self.bias = bias
        self.ready = True
        self.in_gpu = True
        torch.cuda.empty_cache()

        return self

    def initialize(self):
        if self.linear_layer is not None:
            W = self.linear_layer.weight.data
            self.set_shape(W.shape)

            if self.in_features_diff > 0:
                W = F.pad(W, pad=(0, self.in_features_diff), value=0)

            W_q, meta = self.quantize(W, **self.quant_config)
            self.process_hqq_quants(W_q, meta)
            del W_q, meta

            self.bias = (
                None
                if (self.linear_layer.bias is None)
                else self.linear_layer.bias.to(
                    dtype=self.compute_dtype, device=self.device
                )
            )

        if self.del_orig:
            del self.linear_layer

        self.ready = True
        self.in_gpu = True
        torch.cuda.empty_cache()

        return self

    ###################### Quantize/packing ######################

    def quantize(
        self,
        W: Tensor,
        weight_quant_params: dict,
        scale_quant_params=dict | None,
        zero_quant_params=dict | None,
        offload_meta=False,
    ):
        W_q, meta = Quantizer.quantize(
            W,
            **weight_quant_params,
            device=self.device,
            compute_dtype=self.compute_dtype,
            bitpack=False,
        )

        # ToDO: meta quantization

        return W_q, meta

    # TODO: move these to utils
    @torch.no_grad()
    def reshape_meta_axis1(self, meta_tensor, new_group_size, shape):
        meta_tensor = meta_tensor.repeat([1, shape[1]]).reshape(shape)
        meta_tensor = torch.mean(
            meta_tensor.reshape([-1, new_group_size]), axis=1, keepdim=True
        )
        return meta_tensor

    def find_multiple(self, n: int, k: int) -> int:
        if n % k == 0:
            return n
        return n + k - (n % k)

    def set_shape(self, shape):
        self.shape = shape
        self.in_features = shape[1]
        self.out_features = shape[0]

        self.origin_in_features = self.in_features
        if self.padding:
            self.in_features = self.find_multiple(self.in_features, 1024)

        self.in_features_diff = self.in_features - self.origin_in_features

    @torch.no_grad()
    def process_hqq_quants(self, W_q, meta):
        scales = meta["scale"]
        zeros = meta["zero"]
        shape = meta["shape"]

        if meta["packing"] is not None:
            W_q = Quantizer.unpack[meta["packing"]](W_q)

        if self.groupsize is None:
            self.groupsize = 128
            W_q = W_q.reshape([-1, self.groupsize])
            scales = self.reshape_meta_axis1(scales, self.groupsize, shape)
            zeros = self.reshape_meta_axis1(zeros, self.groupsize, shape)

        W_q_torch, scales_torch, zeros_torch = self.hqq_quants_to_torch_quants(
            W_q=W_q, scales=scales, zeros=zeros, shape=shape, nbits=self.nbits
        )
        if check_cpu_version(W_q.device):
            self.weight_int4pack = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
                W_q_torch, self.inner_k_tiles
            )
        else:
            self.weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(
                W_q_torch, self.inner_k_tiles
            )
        self.scales_and_zeros = self.pack_scales_and_zeros(scales_torch, zeros_torch)

        del W_q_torch, scales_torch, zeros_torch
        torch.cuda.empty_cache()

    @torch.no_grad()
    def hqq_quants_to_torch_quants(
        self, W_q: Tensor, scales: Tensor, zeros: Tensor, shape, nbits=4
    ):
        W_q = W_q.to(dtype=self.compute_dtype, device=self.device)
        scales = scales.to(dtype=self.compute_dtype, device=self.device)
        zeros = zeros.to(dtype=self.compute_dtype, device=self.device)

        max_int = 2**nbits - 1
        min_int = 0
        dump = 2 ** (nbits - 1)

        # HQQ -> torch logic
        new_zeros = (scales * dump) - zeros * scales

        min_val = new_zeros - scales * dump

        # group_quantize_tensor_from_qparams
        W_r = (W_q - zeros) * scales

        W_q = (
            W_r.sub(min_val)
            .div(scales)
            .round()
            .clamp_(min_int, max_int)
            .to(torch.int32)
            .reshape(shape)
            .contiguous()
        )
        if not is_device(W_q.device.type, "cpu"):
            W_q = (W_q[::, ::2] << 4 | W_q[::, 1::2]).to(torch.uint8)

        # group_dequantize_tensor_from_qparams
        # W_r = W_q*scales + min_val

        scales = scales.contiguous().reshape(shape[0], -1)
        new_zeros = new_zeros.contiguous().reshape(shape[0], -1)

        return W_q, scales, new_zeros

    def pack_scales_and_zeros(self, scales, zeros):
        assert scales.shape == zeros.shape
        assert scales.dtype == torch.bfloat16
        assert zeros.dtype == torch.bfloat16
        return (
            torch.cat(
                [
                    scales.reshape(scales.size(0), scales.size(1), 1),
                    zeros.reshape(zeros.size(0), zeros.size(1), 1),
                ],
                2,
            )
            .transpose(0, 1)
            .contiguous()
        )

    ###################### Forward/matmul ######################

    @torch.jit.ignore()
    def matmul(self, x):
        origin_x_size = x.size()
        x = x.reshape(-1, origin_x_size[-1])
        if check_cpu_version(x.device):
            c = torch.ops.aten._weight_int4pack_mm_for_cpu(
                x, self.weight_int4pack, self.groupsize, self.scales_and_zeros
            )
        else:
            c = torch.ops.aten._weight_int4pack_mm(
                x, self.weight_int4pack, self.groupsize, self.scales_and_zeros
            )
        new_shape = origin_x_size[:-1] + (self.out_features,)
        c = c.reshape(new_shape)
        return c

    # TODO without matmul
    def dequantize(self):
        return self.matmul(
            torch.eye(self.in_features, dtype=self.compute_dtype, device=self.device)
        )[: self.origin_in_features].t()

    # TODO: backward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.compute_dtype)

        if self.in_features_diff > 0:
            x = F.pad(x, pad=(0, self.in_features_diff))

        out = self.matmul(x)

        if self.bias is not None:
            out += self.bias
        return out
