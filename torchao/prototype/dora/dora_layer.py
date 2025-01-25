import logging

import bitsandbytes as bnb
import torch
import torch.nn as nn
from prototypes.dora.kernels.matmul import triton_mm
from prototypes.dora.kernels.smallk import triton_mm_small_k

logger = logging.getLogger(__name__)


# Adapted from https://github.com/AnswerDotAI/fsdp_qlora/blob/dora/scripts/dora.py
class DoRALayer(nn.Module):
    """DoRA Update"""

    def __init__(
        self, in_features, out_features, lora_rank, device, dtype, *args, **kwargs
    ):
        super().__init__()

        # LoRA layers
        std_dev = 1 / torch.sqrt(torch.tensor(lora_rank).float())
        lora_A_param = nn.Parameter(
            torch.randn(lora_rank, in_features).to(device=device, dtype=dtype) * std_dev
        )
        self.lora_A = nn.Linear(
            in_features, lora_rank, bias=False, device=device, dtype=dtype
        )
        setattr(self.lora_A, "weight", lora_A_param)

        self.lora_B = nn.Linear(
            lora_rank, out_features, bias=False, device=device, dtype=dtype
        )
        self.lora_B.weight.data.zero_()

    def forward(self, x, base_weight):
        # LoRA update, shape `bs x seq_len x in-features @ in-features x lora-rank @ lora-rank x out-features = bs x seq_len x out-features`
        output = self.lora_B(self.lora_A(x))

        # DoRA Section 4.3. Column norm no gradient update.
        column_norm = (
            (base_weight + self.lora_B.weight @ self.lora_A.weight)
            .norm(p=2, dim=1)
            .detach()
        )

        return output, column_norm


class DoRALinear(nn.Module):
    """Reference DoRA Update Layer

    out = (x @ base_weight + lora_out) * magnitude_scale
    where:
    `lora_out = lora_B(lora_A(x)`
    `magnitude_scale = (base_weight + lora_B @ lora_A).norm(p=2, dim=1) * magnitude_vector`

    base_weight is the weight of the frozen `linear` layer of shape `out_features x in_features`.

    In QDoRA, the base weight is quantized and needs an additional dequantization step.
    In this base DoRA layer, a placeholder (no-op) `dequantize` method stub is provided, which simply
    returns the base weight.

    For `bnb` and `hqq`, the respective `dequantize` method can be substituted.
    """

    def __init__(self, base_layer, lora_rank, *args, **kwargs):
        super().__init__()

        # Get original (dequantized) weight dtype
        dtype = getattr(
            base_layer, "compute_dtype", next(base_layer.parameters()).dtype
        )
        device = next(base_layer.parameters()).device
        self.base_layer = base_layer

        # Initialize magnitude vec - TODO: this is clunky, better way to init?
        base_weight = self.dequantize().clone().cuda()
        self.magnitude_vec = nn.Parameter(base_weight.norm(p=2, dim=1))

        del base_weight
        torch.cuda.empty_cache()

        #  DoRA layer
        self.dora_layer = DoRALayer(
            base_layer.in_features,
            base_layer.out_features,
            lora_rank,
            device,
            dtype,
            *args,
            **kwargs,
        )

    def dequantize(self):
        return self.base_layer.weight

    def forward(self, x, *args, **kwargs):
        # Out shape is either bs, seqlen, out_features or bs * seqlen, out_features
        assert x.ndim == 2 or x.ndim == 3, "Expected 2D or 3D input"
        dq_base_weight = self.dequantize()
        out_shape = [*x.shape[:-1], dq_base_weight.shape[0]]
        # Reshape to (bs * seqlen, out_features)
        x = x.reshape(-1, x.shape[-1])

        # LoRA update
        lora_A_weight = self.dora_layer.lora_A.weight
        lora_B_weight = self.dora_layer.lora_B.weight
        lora_out = (x @ lora_A_weight.T) @ lora_B_weight.T

        # DoRA magnitude scale
        column_norm = (dq_base_weight + lora_B_weight @ lora_A_weight).norm(p=2, dim=1)
        magnitude_scale = self.magnitude_vec / column_norm

        # DoRA update
        dora_out = (x @ dq_base_weight.T + lora_out) * magnitude_scale[None, :]
        dora_out = dora_out.reshape(*out_shape)

        return dora_out

    def forward_fused(self, x, *args, **kwargs):
        """Reorders computation as well employs two fused kernels to speed up computation.

        See README.md for description of fused kernels.
        """
        assert x.ndim == 2 or x.ndim == 3, "Expected 2D or 3D input"

        dq_base_weight = self.dequantize()
        # Out shape is either bs, seqlen, out_features or bs * seqlen, out_features
        out_shape = [*x.shape[:-1], dq_base_weight.shape[0]]
        # Reshape to (bs * seqlen, out_features)
        x = x.reshape(-1, x.shape[-1])

        # LoRA update
        lora_A_weight = self.dora_layer.lora_A.weight
        lora_B_weight = self.dora_layer.lora_B.weight
        lora_out = (x @ lora_A_weight.T) @ lora_B_weight.T

        # DoRA magnitude
        # Fused kernel #1: `magnitude_scale = (base_weight + lora_B @ lora_A).norm(p=2, dim=1) * magnitude_vector`
        magnitude_scale = triton_mm_small_k(
            lora_B_weight,
            lora_A_weight,
            epilogue_norm=True,
            source=dq_base_weight,
            magnitude=self.magnitude_vec,
            store_acc=False,
        )
        # DoRA update
        # Fused kernel #2:  `out = (x @ base_weight + lora_out) * magnitude_scale`
        dora_out = triton_mm(
            x,
            dq_base_weight.T,
            epilogue_source=lora_out,
            epilogue_scale=magnitude_scale,
        )
        dora_out = dora_out.reshape(out_shape)

        return dora_out

    # For profiling
    def forward_instrumented(self, x, *args, **kwargs):
        annotation_ctx = kwargs.pop("annotation_ctx")
        with annotation_ctx("##dora_forward"):
            with annotation_ctx("##base_layer"):
                result = self.base_layer(x, *args, **kwargs)

            with annotation_ctx("##dora_layer"):
                dq_weight = self.dequantize()
                output, column_norm = self.dora_layer(x, dq_weight)

            with annotation_ctx("##dora_rescale"):
                result += output
                result = result / column_norm.view(1, 1, -1)
                result = result * self.magnitude_vec.view(1, 1, -1)

        return result


class BNBDoRALinear(DoRALinear):
    def dequantize(self):
        return bnb.functional.dequantize_4bit(
            self.base_layer.weight.data, self.base_layer.weight.quant_state
        )


class HQQDoRALinear(DoRALinear):
    def dequantize(self):
        return self.base_layer.dequantize()
