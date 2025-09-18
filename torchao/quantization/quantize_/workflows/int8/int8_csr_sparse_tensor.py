import warnings
from typing import List, Optional, Tuple

import torch

from torchao.quantization.quant_primitives import (
    MappingType,
    choose_qparams_affine,
    quantize_affine,
)
from torchao.utils import TorchAOBaseTensor

__all__ = ["Int8CsrSparseTensor"]
aten = torch.ops.aten


class Int8CsrSparseTensor(TorchAOBaseTensor):
    # include a8_dynamic in attributes so it serializes cleanly
    tensor_data_names = ["qdata", "scale", "zero_point"]
    tensor_attribute_names = ["shape", "block_size", "mapping", "a8_mode"]

    def __new__(
        cls,
        qdata: torch.Tensor,  # sparse CSR int8 (values are int8)
        scale: torch.Tensor,  # per-row (out,) in fp
        zero_point: torch.Tensor,  # usually 0 for symmetric
        shape: Tuple[int, int],
        block_size: Tuple[int, int],
        mapping: MappingType,
        a8_mode: str = "int8_sym_per_token",  # "noop" | "int8_sym_per_token" | "int8_asym_per_token"
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            device=qdata.device,
            dtype=scale.dtype,
            requires_grad=False,
        )

    def __init__(
        self,
        qdata,
        scale,
        zero_point,
        shape,
        block_size,
        mapping,
        a8_mode="int8_sym_per_token",
    ):
        super().__init__()
        if a8_mode not in ("noop", "int8_sym_per_token", "int8_asym_per_token"):
            raise ValueError(f"Invalid a8_mode: {a8_mode}")
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        # self.shape = shape
        self.block_size = block_size
        self.mapping = mapping
        self.a8_mode = a8_mode  # <—— passed from config

    @staticmethod
    def _magnitude_prune(x: torch.Tensor, target: Optional[float]) -> torch.Tensor:
        if target is None or not (0.0 < target < 1.0):
            return x
        tmp = x.detach().clone()
        flat = tmp.abs().view(-1)
        k = int(flat.numel() * (1 - target))
        if k <= 0:
            return tmp.zero_()
        thresh = flat.kthvalue(k).values
        tmp.mul_(tmp.abs() >= thresh)
        return tmp

    @classmethod
    def from_hp(
        cls,
        w: torch.Tensor,
        *,
        target_sparsity: Optional[float] = 0.9,
        mapping: MappingType = MappingType.SYMMETRIC,
        block_size: Optional[List[int]] = None,
        eps: float = 1e-6,
        act_mode: str = "int8_sym_per_token",  # <—— NEW knob for W8A8
    ) -> "Int8CsrSparseTensor":
        assert w.dim() == 2  # weight must be 2D ( Out, In )
        out_features, in_features = w.shape
        if block_size is None:
            block_size = [1, in_features]
        block_size = tuple(block_size)

        # per-row int8
        scale, zp = choose_qparams_affine(
            input=w,
            mapping_type=mapping,
            block_size=block_size,
            target_dtype=torch.int8,
            quant_min=-127,
            quant_max=127,
            eps=eps,
        )
        wq = quantize_affine(
            input=w,
            block_size=block_size,
            scale=scale,
            zero_point=zp,
            output_dtype=torch.int8,
            quant_min=-127,
            quant_max=127,
        )

        pruned = cls._magnitude_prune(wq.to(torch.int8), target_sparsity)
        q_csr = pruned.to_sparse_csr()

        scale = scale.to(w.dtype)
        zp = zp.to(torch.int32)

        return cls(
            qdata=q_csr,
            scale=scale,
            zero_point=zp,
            shape=(out_features, in_features),
            block_size=block_size,
            mapping=mapping,
            a8_mode=act_mode,  # <—— propagate
        )


implements = Int8CsrSparseTensor.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    from torchao.quantization.quant_primitives import (
        MappingType,
        choose_qparams_affine,
        quantize_affine,
    )

    x, W, bias = args[0], args[1], (args[2] if len(args) > 2 else None)
    assert isinstance(W, Int8CsrSparseTensor)  # weight must be Int8CsrSparseTensor

    # flatten to 2D
    x2d = x.view(-1, x.shape[-1])  # (B*, K)

    # ── Weight-only path (no activation quant) ────────────────────────────────
    if W.a8_mode in (None, "noop"):
        y = torch.mm(W.qdata.to(torch.float32), x2d.to(torch.float32).t()).t()
        # apply weight scales only
        y = y * W.scale.view(1, -1)
        if bias is not None:
            y = y + bias.to(y.dtype)
        return y.view(*x.shape[:-1], W.shape[0])

    # ── Dynamic A8 (per-token/row) ───────────────────────────────────────────
    if W.a8_mode == "int8_sym_per_token":
        act_mapping = MappingType.SYMMETRIC
        qmin, qmax = -127, 127
        a_dtype = torch.int8
    elif W.a8_mode == "int8_asym_per_token":
        act_mapping = MappingType.ASYMMETRIC
        qmin, qmax = 0, 255
        a_dtype = torch.uint8  # <<< IMPORTANT: uint8 for 0..255
    else:
        raise ValueError(f"Invalid a8_mode: {W.a8_mode}")

    a_scale, a_zp = choose_qparams_affine(
        input=x2d,
        mapping_type=act_mapping,
        block_size=(1, x2d.shape[1]),  # per-row
        target_dtype=a_dtype,
        quant_min=qmin,
        quant_max=qmax,
        eps=1e-6,
    )
    xq = quantize_affine(
        input=x2d,
        block_size=(1, x2d.shape[1]),
        scale=a_scale,
        zero_point=a_zp,
        output_dtype=a_dtype,
        quant_min=qmin,
        quant_max=qmax,
    )

    # Center activations for asymmetric: use (xq - zp)
    if act_mapping == MappingType.ASYMMETRIC:
        if a_zp.dim() == 2:
            a_zp_row = a_zp.squeeze(-1).to(torch.float32)
        else:
            a_zp_row = a_zp.to(torch.float32)
        xq_f32 = xq.to(torch.float32) - a_zp_row.view(-1, 1)
    else:
        xq_f32 = xq.to(torch.float32)

    # float fallback for SpMM (replace with int8 CSR kernel when available)
    warnings.warn(
        "Falling back to SPMM FP32 computation for linear() (as int8 SpMM kernel is not available).",
        UserWarning,
        stacklevel=2,
    )
    y = torch.mm(W.qdata.to(torch.float32), xq_f32.t()).t()  # (B*, O)

    # apply outer-product of scales: (B*,1)*(1,O)
    if a_scale.dim() == 2:  # (B*,1)
        a_scale_row = a_scale.squeeze(-1)
    else:
        a_scale_row = a_scale
    y = y * (a_scale_row.view(-1, 1) * W.scale.view(1, -1))

    if bias is not None:
        y = y + bias.to(y.dtype)

    # restore original batch shape
    out_features = W.shape[0]
    y = y.view(*x.shape[:-1], out_features)
    return y


Int8CsrSparseTensor.__module__ = "torchao.quantization"
torch.serialization.add_safe_globals([Int8CsrSparseTensor])
