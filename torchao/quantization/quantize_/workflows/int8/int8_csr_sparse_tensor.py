import warnings
from typing import List, Optional, Tuple

import torch
from torch import _dynamo

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
        # self.shape = torch.Size(shape) // AttributeError: attribute 'shape' of 'torch._C.TensorBase' objects is not writable
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


# ---- alias: return self unchanged (avoid touching strides) ----
for _name in ("default", "default_1", "default_2", "default_3"):
    _op = getattr(aten.alias, _name, None)
    if _op is not None:

        @implements([_op])
        def _alias_noop(func, types, args, kwargs):
            self = args[0]
            return self.detach()  # avoid touching strides


# ---- access_subclass_inner_tensor: return a dense, strided placeholder ----
_access_ns = getattr(torch.ops, "aten", None)
_access_all = (
    getattr(_access_ns, "access_subclass_inner_tensor", None) if _access_ns else None
)
if _access_all is not None:
    for _name in ("default", "default_1", "default_2", "default_3"):
        _op = getattr(_access_all, _name, None)
        if _op is not None:

            @implements([_op])
            def _access_inner_tensor(func, types, args, kwargs):
                """Provide a strided stand-in for tracing/export metadata checks."""
                self = args[0]  # Int8CsrSparseTensor
                return torch.empty(0, dtype=torch.int8, device=self.qdata.device)


# ---- alias_copy: same behavior as alias (return distinct wrapper) ----
for _name in ("default", "default_1", "default_2", "default_3"):
    _op = getattr(getattr(aten, "alias_copy", None), _name, None)
    if _op is not None:

        @implements([_op])
        def _alias_copy_view(func, types, args, kwargs):
            self = args[0]
            return self.detach()


# ---- copy_: enforce metadata equality and match test error string ----
_op = getattr(aten.copy_, "default", None)
if _op is not None:

    @implements([_op])
    def _copy_(func, types, args, kwargs):
        self = args[0]
        src = args[1]
        # If not our wrapper(s), fall back to original op
        if not isinstance(self, Int8CsrSparseTensor) or not isinstance(
            src, Int8CsrSparseTensor
        ):
            return func(*args, **(kwargs or {}))

        # Compare metadata we care about
        same_meta = (
            tuple(self.shape) == tuple(src.shape)
            and tuple(self.block_size) == tuple(src.block_size)
            and self.mapping == src.mapping
            and self.a8_mode == src.a8_mode
            and self.scale.dtype == src.scale.dtype
            and self.zero_point.dtype == src.zero_point.dtype
        )

        if not same_meta:
            # NOTE: test expects the misspelling "mistach"
            raise ValueError(
                f"Not supported args for copy_ due to metadata mistach: ({src}, {self})"
            )

        # Perform the copy (deep copy of payloads; keep attributes in sync)
        self.qdata = src.qdata.clone()
        self.scale = src.scale.clone()
        self.zero_point = src.zero_point.clone()
        self.block_size = src.block_size
        self.mapping = src.mapping
        self.a8_mode = src.a8_mode
        return self


# ... where we register the override for linear() ...
@_dynamo.disable
@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    from torchao.quantization.quant_primitives import (
        MappingType,
        choose_qparams_affine,
        quantize_affine,
    )

    # ---- robust arg parsing: support positional AND kwargs ----
    def _get(name, pos, default=None):
        if len(args) > pos:
            return args[pos]
        if kwargs:
            return kwargs.get(name, default)
        return default

    x = _get("input", 0)
    W = _get("weight", 1)
    bias = _get("bias", 2, None)

    # If signature is odd or not our wrapper (e.g., FakeTensor), defer
    if x is None or W is None or not isinstance(W, Int8CsrSparseTensor):
        return func(*args, **(kwargs or {}))

    # flatten to 2D
    x2d = x.view(-1, x.shape[-1])  # (B*, K)

    # IMPORTANT: never let CSR leak into the graph for export/tracing.
    # Convert weight to a dense, strided tensor before matmul.
    W_dense_f32 = W.qdata.to_dense().to(torch.float32)

    # ── Weight-only path (no activation quant) ────────────────────────────────
    if W.a8_mode in (None, "noop"):
        y = torch.mm(W_dense_f32, x2d.to(torch.float32).t()).t()
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
        a_dtype = torch.uint8
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
        a_zp_row = (a_zp.squeeze(-1) if a_zp.dim() == 2 else a_zp).to(torch.float32)
        xq_f32 = xq.to(torch.float32) - a_zp_row.view(-1, 1)
    else:
        xq_f32 = xq.to(torch.float32)

    # float fallback for SpMM using *dense* weight
    warnings.warn(
        "Falling back to SPMM FP32 computation for linear() (as int8 SpMM kernel is not available).",
        UserWarning,
        stacklevel=2,
    )
    y = torch.mm(W_dense_f32, xq_f32.t()).t()  # (B*, O)

    # apply outer-product of scales: (B*,1)*(1,O)
    a_scale_row = a_scale.squeeze(-1) if a_scale.dim() == 2 else a_scale
    y = y * (a_scale_row.view(-1, 1) * W.scale.view(1, -1))

    if bias is not None:
        y = y + bias.to(y.dtype)

    out_features = W.shape[0]
    return y.view(*x.shape[:-1], out_features)


Int8CsrSparseTensor.__module__ = "torchao.quantization"
torch.serialization.add_safe_globals([Int8CsrSparseTensor])
