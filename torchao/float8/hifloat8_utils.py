from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional, Tuple

import torch


_HIFLOAT4_E2M1_NAME = "float4_e2m1fn_x2"
_HIFLOAT4_E1M2_NAME = "float4_e1m2fn_x2"
_HIFLOAT4_NAME_HINTS = (
    "hifloat4",
    _HIFLOAT4_E2M1_NAME,
    _HIFLOAT4_E1M2_NAME,
)


@lru_cache(None)
def _load_torch_npu() -> Tuple[Optional[Any], Optional[type]]:
    """
    Best-effort import of torch_npu and its HiFloat8 tensor type.

    We intentionally catch broad exceptions because importing torch_npu can
    raise RuntimeError in environments without Ascend/NPU support or when
    another accelerator is active.
    """
    try:
        import torch_npu  # type: ignore
    except Exception:
        return None, None

    hif8_cls = getattr(torch_npu, "HiFloat8Tensor", None)
    if hif8_cls is None:
        try:
            from torch_npu.utils.hif8_tensor import _HiFloat8Tensor  # type: ignore

            hif8_cls = _HiFloat8Tensor
        except Exception:
            hif8_cls = None

    return torch_npu, hif8_cls


@lru_cache(None)
def _load_torch_npu_hifloat4_dtypes() -> Tuple[Optional[Any], Tuple[Any, ...]]:
    try:
        import torch_npu  # type: ignore
    except Exception:
        return None, tuple()

    dtypes = []
    for name in (_HIFLOAT4_E2M1_NAME, _HIFLOAT4_E1M2_NAME):
        dtype_obj = getattr(torch_npu, name, None)
        if dtype_obj is not None:
            dtypes.append(dtype_obj)
        try:
            cd_dtype = getattr(torch_npu._C._cd.DType, name, None)
            if cd_dtype is not None:
                dtypes.append(cd_dtype)
        except Exception:
            pass
    return torch_npu, tuple(dtypes)


def _torch_float4_e2m1_dtype() -> Optional[torch.dtype]:
    return getattr(torch, _HIFLOAT4_E2M1_NAME, None)


def _looks_like_hifloat8_dtype(dtype: Any) -> bool:
    if dtype is None:
        return False
    name = getattr(dtype, "name", None)
    if isinstance(name, str) and name.lower() == "hifloat8":
        return True
    name = getattr(dtype, "__name__", None)
    if isinstance(name, str) and name.lower() == "hifloat8":
        return True
    try:
        dtype_str = str(dtype).lower()
    except Exception:
        return False
    if "hifloat8" in dtype_str and "tensor" not in dtype_str:
        return True
    try:
        repr_str = repr(dtype).lower()
    except Exception:
        return False
    return "hifloat8" in repr_str and "tensor" not in repr_str


def _looks_like_hifloat4_dtype(dtype: Any) -> bool:
    if dtype is None:
        return False

    for name_attr in ("name", "__name__"):
        name = getattr(dtype, name_attr, None)
        if isinstance(name, str) and name.lower() in _HIFLOAT4_NAME_HINTS:
            return True

    try:
        dtype_str = str(dtype).lower()
    except Exception:
        return False
    if "tensor" not in dtype_str and any(tok in dtype_str for tok in _HIFLOAT4_NAME_HINTS):
        return True

    try:
        repr_str = repr(dtype).lower()
    except Exception:
        return False
    return "tensor" not in repr_str and any(
        tok in repr_str for tok in _HIFLOAT4_NAME_HINTS
    )


def is_hifloat8_dtype(dtype: Any) -> bool:
    torch_npu, _ = _load_torch_npu()
    if torch_npu is not None:
        try:
            if dtype == torch_npu.hifloat8:
                return True
        except Exception:
            pass
        try:
            hif8 = getattr(torch_npu._C._cd.DType, "hifloat8", None)
            if hif8 is not None and dtype == hif8:
                return True
        except Exception:
            pass
    return _looks_like_hifloat8_dtype(dtype)


def is_hifloat4_dtype(dtype: Any) -> bool:
    torch_float4 = _torch_float4_e2m1_dtype()
    if torch_float4 is not None:
        try:
            if dtype == torch_float4:
                return True
        except Exception:
            pass

    _, torch_npu_hif4_dtypes = _load_torch_npu_hifloat4_dtypes()
    for hif4_dtype in torch_npu_hif4_dtypes:
        try:
            if dtype == hif4_dtype:
                return True
        except Exception:
            pass
    return _looks_like_hifloat4_dtype(dtype)


def is_hifloatx_dtype(dtype: Any) -> bool:
    return is_hifloat8_dtype(dtype) or is_hifloat4_dtype(dtype)


def is_hifloat8_tensor(tensor: Any) -> bool:
    _, hif8_cls = _load_torch_npu()
    return hif8_cls is not None and isinstance(tensor, hif8_cls)


def is_hifloat4_tensor(tensor: Any) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    try:
        return is_hifloat4_dtype(tensor.dtype)
    except Exception:
        return False


def is_hifloatx_tensor(tensor: Any) -> bool:
    return is_hifloat8_tensor(tensor) or is_hifloat4_tensor(tensor)


def to_hifloat8(tensor: torch.Tensor) -> torch.Tensor:
    torch_npu, hif8_cls = _load_torch_npu()
    if torch_npu is None or hif8_cls is None:
        raise RuntimeError(
            "HiFloat8 support requires torch_npu with Ascend/NPU runtime available."
        )
    return hif8_cls.to_hifloat8(tensor)


def _resolve_hifloat4_target_dtype(preferred_dtype: Optional[Any] = None) -> Optional[Any]:
    if preferred_dtype is not None and is_hifloat4_dtype(preferred_dtype):
        return preferred_dtype

    torch_float4 = _torch_float4_e2m1_dtype()
    if torch_float4 is not None:
        return torch_float4

    torch_npu, _ = _load_torch_npu_hifloat4_dtypes()
    if torch_npu is not None:
        hif4 = getattr(torch_npu, _HIFLOAT4_E2M1_NAME, None)
        if hif4 is not None:
            return hif4
    return None


def get_hifloat4_npu_dtype() -> Any:
    torch_npu, _ = _load_torch_npu_hifloat4_dtypes()
    if torch_npu is None:
        raise RuntimeError("HiFloat4 support requires torch_npu runtime.")
    hif4 = getattr(torch_npu, _HIFLOAT4_E2M1_NAME, None)
    if hif4 is None:
        raise RuntimeError(
            "torch_npu.float4_e2m1fn_x2 is not available in this torch_npu build."
        )
    return hif4


def to_hifloat4(tensor: torch.Tensor, dtype: Optional[Any] = None) -> torch.Tensor:
    target_dtype = _resolve_hifloat4_target_dtype(dtype)
    if target_dtype is not None:
        try:
            return tensor.to(target_dtype)
        except Exception:
            pass

    torch_npu, _ = _load_torch_npu_hifloat4_dtypes()
    if torch_npu is None:
        raise RuntimeError(
            "HiFloat4 support requires torch_npu with float4_e2m1fn_x2 support."
        )

    if tensor.device.type != "npu":
        tensor = tensor.npu()
    tensor = tensor.contiguous().detach()
    if tensor.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        tensor = tensor.float()

    if hasattr(torch_npu._C, "_cd_init"):
        torch_npu._C._cd_init()
    tex = torch_npu._C._cd
    dtype_e2m1 = getattr(tex.DType, _HIFLOAT4_E2M1_NAME, None)
    if dtype_e2m1 is None:
        raise RuntimeError(
            "torch_npu._C._cd.DType.float4_e2m1fn_x2 is not available in this torch_npu build."
        )

    data = tex.cast_to_fp8(tensor.view(1, -1), dtype_e2m1).view(tensor.size())
    torch_float4 = _torch_float4_e2m1_dtype()
    if torch_float4 is not None and data.dtype == torch.uint8:
        try:
            data = data.view(torch_float4)
        except Exception:
            pass
    return data


def to_hifloatx(tensor: torch.Tensor, dtype: Any) -> torch.Tensor:
    if is_hifloat8_dtype(dtype) or _looks_like_hifloat8_dtype(dtype):
        return to_hifloat8(tensor)
    if is_hifloat4_dtype(dtype) or _looks_like_hifloat4_dtype(dtype):
        return to_hifloat4(tensor, dtype=dtype)
    raise ValueError(f"Unsupported HiFloat dtype: {dtype}")


def _decode_hifloat8_byte_reference(v: int) -> float:
    # Reference fallback used when torch_npu cast kernels are unavailable.
    if v == 0 or v == 128:
        return 0.0
    if v == 239:
        return -32768.0
    if v == 111:
        return 32768.0
    if v >= 128:
        sign = -1.0
    else:
        sign = 1.0
    dot_4_bits = v & 120
    dot_4_value = dot_4_bits >> 3
    if dot_4_value >= 12:
        exponent = v & 30
        exponent_int = exponent >> 1
        if exponent_int >= 8:
            exponent_value = -exponent_int
        else:
            exponent_value = exponent_int + 8
        fra_int = v & 1
        m_value = 1.0 + fra_int * 0.5
    elif dot_4_value >= 8:
        exponent = v & 28
        exponent_int = exponent >> 2
        if exponent_int >= 4:
            exponent_value = -exponent_int
        else:
            exponent_value = exponent_int + 4
        fra_int = v & 3
        m_value = 1.0 + fra_int * 0.25
    elif dot_4_value >= 4:
        exponent = v & 24
        exponent_int = exponent >> 3
        if exponent_int >= 2:
            exponent_value = -exponent_int
        else:
            exponent_value = exponent_int + 2
        fra_int = v & 7
        m_value = 1.0 + fra_int * 0.125
    elif dot_4_value >= 2:
        exponent = v & 8
        exponent_sign = exponent >> 3
        if exponent_sign >= 1:
            exponent_value = -1
        else:
            exponent_value = 1
        fra_int = v & 7
        m_value = 1.0 + fra_int * 0.125
    elif dot_4_value == 1:
        exponent_value = 0
        fra_int = v & 7
        m_value = 1.0 + fra_int * 0.125
    elif dot_4_value == 0:
        m_value = 1.0
        exponent_value = (v & 7) - 23
    else:
        return 0.0
    return sign + pow(2.0, exponent_value) * m_value


@lru_cache(None)
def _decode_hifloat8_lut() -> Tuple[float, ...]:
    """
    Decode all 256 HiFloat8 byte patterns.

    Prefer torch_npu cast kernels (same path as HiFloat8Tensor cast), and
    fall back to the Python reference decode when NPU runtime is unavailable.
    """
    torch_npu, _ = _load_torch_npu()
    if torch_npu is not None:
        try:
            if torch_npu.npu.is_available():
                if hasattr(torch_npu._C, "_cd_init"):
                    torch_npu._C._cd_init()
                tex = torch_npu._C._cd
                # Mirror torch_npu HiFloat8 cast path:
                # torch_npu/utils/hif8_tensor.py -> tex.cast_from_fp8(..., DType.hifloat8, DType.float32)
                raw = torch.arange(256, dtype=torch.uint8, device="npu").view(1, -1)
                decoded = tex.cast_from_fp8(raw, tex.DType.hifloat8, tex.DType.float32)
                return tuple(float(x) for x in decoded.view(-1).cpu().tolist())
        except Exception:
            pass
    return tuple(_decode_hifloat8_byte_reference(v) for v in range(256))


def _decode_hifloat8_byte(v: int) -> float:
    if not (0 <= v <= 255):
        raise ValueError(f"HiFloat8 byte must be in [0, 255], got {v}")
    return _decode_hifloat8_lut()[v]


@lru_cache(None)
def hifloat8_max_abs() -> float:
    return max(abs(x) for x in _decode_hifloat8_lut())


@lru_cache(None)
def hifloat8_min_max() -> Tuple[float, float]:
    vals = _decode_hifloat8_lut()
    return min(vals), max(vals)


@lru_cache(None)
def hifloat4_min_max() -> Tuple[float, float]:
    torch_float4 = _torch_float4_e2m1_dtype()
    if torch_float4 is not None:
        try:
            finfo = torch.finfo(torch_float4)
            return float(finfo.min), float(finfo.max)
        except Exception:
            pass

    torch_npu, _ = _load_torch_npu_hifloat4_dtypes()
    if torch_npu is not None:
        try:
            if torch_npu.npu.is_available():
                if hasattr(torch_npu._C, "_cd_init"):
                    torch_npu._C._cd_init()
                tex = torch_npu._C._cd
                dtype_e2m1 = getattr(tex.DType, _HIFLOAT4_E2M1_NAME, None)
                if dtype_e2m1 is not None:
                    raw = torch.arange(256, dtype=torch.uint8, device="npu").view(1, -1)
                    decoded = tex.cast_from_fp8(raw, dtype_e2m1, tex.DType.float32)
                    vals = tuple(float(x) for x in decoded.view(-1).cpu().tolist())
                    if vals:
                        return min(vals), max(vals)
        except Exception:
            pass

    # Fallback for fp4 e2m1 finite range.
    return -6.0, 6.0


@lru_cache(None)
def hifloat4_max_abs() -> float:
    hif4_min, hif4_max = hifloat4_min_max()
    return max(abs(hif4_min), abs(hif4_max))


def hifloatx_min_max(dtype: Any) -> Tuple[float, float]:
    if is_hifloat8_dtype(dtype) or _looks_like_hifloat8_dtype(dtype):
        return hifloat8_min_max()
    if is_hifloat4_dtype(dtype) or _looks_like_hifloat4_dtype(dtype):
        return hifloat4_min_max()
    raise TypeError(f"Unsupported HiFloat dtype: {dtype}")


def hifloatx_max_abs(dtype: Any) -> float:
    if is_hifloat8_dtype(dtype) or _looks_like_hifloat8_dtype(dtype):
        return hifloat8_max_abs()
    if is_hifloat4_dtype(dtype) or _looks_like_hifloat4_dtype(dtype):
        return hifloat4_max_abs()
    raise TypeError(f"Unsupported HiFloat dtype: {dtype}")


def hifloat8_raw_int8(tensor: torch.Tensor) -> torch.Tensor:
    """
    Return raw HiFloat8 payload as int8 tensor suitable for npu_quant_matmul.
    """
    if is_hifloat8_tensor(tensor):
        raw = tensor._data  # type: ignore[attr-defined]
    else:
        raw = tensor
    return raw.contiguous().to(torch.int8)


def hifloat4_raw_int8(tensor: torch.Tensor) -> torch.Tensor:
    """
    Return raw HiFloat4 payload as int8 tensor suitable for npu_quant_matmul.
    """
    raw = tensor
    try:
        if is_hifloat4_tensor(tensor):
            raw = tensor.view(torch.uint8)
    except Exception:
        raw = tensor
    return raw.contiguous().to(torch.int8)


def hifloatx_raw_int8(tensor: torch.Tensor) -> torch.Tensor:
    if is_hifloat8_tensor(tensor):
        return hifloat8_raw_int8(tensor)
    if is_hifloat4_tensor(tensor):
        return hifloat4_raw_int8(tensor)
    return tensor.contiguous().to(torch.int8)


def hifloat8_scales_to_npu(
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_features: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Map Float8TrainingTensor scales to npu_quant_matmul scale/pertoken_scale.
    """
    pertoken_scale = None
    if a_scale is not None:
        pertoken_scale = a_scale.reciprocal()
        if pertoken_scale.numel() == 1:
            pertoken_scale = pertoken_scale.reshape(1)
        elif pertoken_scale.dim() == 2 and pertoken_scale.shape[-1] == 1:
            pertoken_scale = pertoken_scale.reshape(-1)

    if b_scale is not None:
        scale = b_scale.reciprocal()
        if scale.numel() == 1:
            scale = scale.reshape(1)
        elif scale.dim() == 2 and scale.shape[0] == 1:
            scale = scale.reshape(-1)
    else:
        scale = torch.ones((out_features,), device=a_scale.device, dtype=torch.float32)

    return scale, pertoken_scale


def hifloat4_scales_to_npu(
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_features: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    return hifloat8_scales_to_npu(a_scale, b_scale, out_features)


def hifloatx_scales_to_npu(
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_features: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    return hifloat8_scales_to_npu(a_scale, b_scale, out_features)
