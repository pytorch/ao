import re

import pytest
import torch
import torch.nn.functional as F

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_qat import MoEQATConfig
from torchao.prototype.moe_qat.tensor import FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizedWeightWrapperTensor
from torchao.quantization.granularity import PerRow, PerTensor
from torchao.quantization.qat.fake_quantize_config import FakeQuantizeConfigBase, Float8FakeQuantizeConfig
from torchao.quantization.quant_api import quantize_
from torchao.utils import TorchAOBaseTensor

from .reference_moe import MoE
from .testing_utils import _expert_weight_filter, _set_seed, device, moe_model, target_devices, use_grouped_mm


# =========================================================================
# Test __init__
# =========================================================================


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, expected_match", [
    (FakeQuantizedWeightWrapperBaseTensor, r"^Must specify `weight_config` in FakeQuantizedWeightWrapperBaseTensor\.$"),
    (Float8FakeQuantizedWeightWrapperTensor, r"^Only `Float8FakeQuantizeConfig` is supported for `weight_config` in Float8FakeQuantizedWeightWrapperTensor\.$"),
])
def test_wrapper_init_requires_weight_config(wrapper_cls, expected_match, device):
    """__init__ raises ValueError when weight_config is None."""
    w = torch.randn(64, 128, device=device)
    with pytest.raises(ValueError, match=expected_match):
        wrapper_cls(w, weight_config=None)


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", [
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), None),
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
def test_wrapper_init_stores_attrs(wrapper_cls, weight_config, act_config, device):
    """__init__ stores _data, weight_config, and activation_config."""
    w = torch.randn(64, 128, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    assert wrapper._data is w
    assert wrapper.weight_config is weight_config
    assert wrapper.activation_config is act_config


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, expected_match", [
    (Float8FakeQuantizedWeightWrapperTensor, r"^Only `Float8FakeQuantizeConfig` is supported for `weight_config` in Float8FakeQuantizedWeightWrapperTensor\.$"),
])
def test_wrapper_init_rejects_invalid_weight_config(wrapper_cls, expected_match, device):
    """Wrapper subclass rejects non-matching weight_config type."""
    class DummyConfig(FakeQuantizeConfigBase):
        pass

    w = torch.randn(64, 128, device=device)
    with pytest.raises(ValueError, match=expected_match):
        wrapper_cls(w, weight_config=DummyConfig())


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, expected_match", [
    (Float8FakeQuantizedWeightWrapperTensor, r"^Only `Float8FakeQuantizeConfig` is supported for `activation_config` in Float8FakeQuantizedWeightWrapperTensor\.$"),
])
def test_wrapper_init_rejects_invalid_activation_config(wrapper_cls, expected_match, device):
    """Wrapper subclass rejects non-matching activation_config type."""
    class DummyConfig(FakeQuantizeConfigBase):
        pass

    w = torch.randn(64, 128, device=device)
    with pytest.raises(ValueError, match=expected_match):
        wrapper_cls(w, activation_config=DummyConfig(), weight_config=Float8FakeQuantizeConfig())


# =========================================================================
# to_tensor
# =========================================================================


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config", [
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig()),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig()),
])
def test_wrapper_to_tensor(wrapper_cls, weight_config, device):
    """to_tensor returns the underlying raw tensor."""
    w = torch.randn(64, 128, device=device)
    wrapper = wrapper_cls(w, weight_config=weight_config)
    result = wrapper.to_tensor()
    assert type(result) is torch.Tensor
    assert result is w


# =========================================================================
# __repr__
# =========================================================================


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config", [
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig()),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig()),
])
def test_wrapper_repr(wrapper_cls, weight_config, device):
    w = torch.randn(64, 128, device=device)
    wrapper = wrapper_cls(w, weight_config=weight_config)
    expected = f"{wrapper_cls.__name__}(data={w}, activation_config=None, weight_config={weight_config})"
    assert repr(wrapper) == expected


# =========================================================================
# __tensor_flatten__ / __tensor_unflatten__
# =========================================================================


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", [
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), None),
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
def test_wrapper_tensor_flatten(wrapper_cls, weight_config, act_config, device):
    """__tensor_flatten__ returns _data tensor and metadata dict."""
    w = torch.randn(64, 128, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    tensor_names, metadata = wrapper.__tensor_flatten__()
    assert tensor_names == ["_data"]
    assert metadata["weight_config"] is weight_config
    assert metadata["activation_config"] is act_config


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", [
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), None),
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
def test_wrapper_tensor_unflatten(wrapper_cls, weight_config, act_config, device):
    """__tensor_unflatten__ reconstructs the wrapper from flattened parts."""
    w = torch.randn(64, 128, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    # simulate PyTorch: record size/stride during flatten, pass back during unflatten
    tensor_names, metadata = wrapper.__tensor_flatten__()
    outer_size = wrapper._data.size()
    outer_stride = wrapper._data.stride()
    tensor_data_dict = {name: getattr(wrapper, name) for name in tensor_names}

    reconstructed = wrapper_cls.__tensor_unflatten__(
        tensor_data_dict, metadata, outer_size, outer_stride
    )
    assert type(reconstructed) is wrapper_cls
    assert torch.equal(reconstructed._data, w)
    assert reconstructed.weight_config is weight_config
    assert reconstructed.activation_config is act_config


@pytest.mark.parametrize("wrapper_cls, weight_config", [
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig()),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig()),
])
def test_meta_weights(wrapper_cls, weight_config):
    """Wrapper can be constructed on meta device."""
    with torch.device("meta"):
        wrapper = wrapper_cls(torch.randn(64, 128), weight_config=weight_config)
    assert wrapper._data.is_meta


# =========================================================================
# __torch_dispatch__
# =========================================================================
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", [
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), None),
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
@pytest.mark.parametrize("weight_shape, op_func", [
    # select / slice
    ((4, 64, 128), lambda x: x[0]),
    ((4, 64, 128), lambda x: x[0:2]),
    ((4, 64, 128), lambda x: x[1:]),
    ((4, 64, 128), lambda x: x[:3]),
    ((4, 64, 128), lambda x: x[::2]),
    ((4, 64, 128), lambda x: x[:, 0]),
    # index
    ((4, 64, 128), lambda x: x[torch.tensor([0, 2])]),
    ((4, 64, 128), lambda x: x[torch.tensor([True, False, True, False])]),
    # unsqueeze
    ((4, 64, 128), lambda x: x.unsqueeze(0)),
    # new_zeros
    ((4, 64, 128), lambda x: x.new_zeros(2, 64, 128)),
    # as_strided
    ((4, 64, 128), lambda x: x.as_strided((2, 64, 128), (16384, 128, 1))),
    # transpose
    ((4, 64, 128), lambda x: x.transpose(0, 1)),
    # detach
    ((4, 64, 128), lambda x: x.detach()),
    # clone
    ((4, 64, 128), lambda x: x.clone()),
    # view
    ((4, 64, 128), lambda x: x.view(4, 2, 32, 128)),
    # permute
    ((4, 64, 128), lambda x: x.permute(1, 0, 2)),
    # _to_copy
    ((4, 64, 128), lambda x: x.to(dtype=torch.float16)),
    # squeeze.dim — needs singleton dim
    ((1, 64, 128), lambda x: x.squeeze(0)),
    # squeeze (no dim) — needs singleton dim
    ((4, 1, 128), lambda x: x.squeeze()),
    # t — needs 2D shape
    ((64, 128), lambda x: x.t()),
    # split — returns tuple
    ((8, 64, 128), lambda x: torch.split(x, 2)),
    # _pin_memory — requires CUDA, skipped on CPU
    ((4, 64, 128), lambda x: x.pin_memory()),
])
@pytest.mark.parametrize("device", target_devices)
def test_wrapper_preserves_subclass(wrapper_cls, weight_config, act_config, weight_shape, op_func, device):
    """All ops in _ops_to_preserve_subclass return the wrapper subclass.

    _unsafe_index.Tensor has no public API to trigger it directly.
    c10d.scatter_.default requires distributed runtime — tested in test_fsdp2.py.
    copy_ is tested separately in test_wrapper_dispatch_copy_ because it is an in-place op.
    """
    def apply_assertions(result, ref_result):
        assert isinstance(result, wrapper_cls)
        assert result.weight_config is weight_config
        assert result.activation_config is act_config
        assert torch.equal(result._data, ref_result)

    weight = torch.randn(*weight_shape, device=device)
    wrapper = wrapper_cls(weight, activation_config=act_config, weight_config=weight_config)

    with torch._C.DisableTorchFunctionSubclass():
        try:
            result = op_func(wrapper)
            ref_result = op_func(wrapper._data)
        except RuntimeError as e:
            if "Cannot access accelerator device" in str(e) and device != "cuda":
                pytest.skip("pin_memory requires CUDA")
            raise

        if isinstance(result, tuple):
            for r, ref in zip(result, ref_result):
                apply_assertions(r, ref)
        else:
            apply_assertions(result, ref_result)


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", [
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), None),
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
def test_wrapper_dispatch_copy_(wrapper_cls, weight_config, act_config, device):
    """copy_ via __torch_dispatch__ returns self and updates _data in-place."""
    w = torch.randn(4, 64, 128, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    with torch._C.DisableTorchFunctionSubclass():
        w2 = torch.randn(4, 64, 128, device=device)
        wrapper2 = wrapper_cls(w2, activation_config=act_config, weight_config=weight_config)
        result = wrapper.copy_(wrapper2)
        assert result is wrapper
        assert torch.equal(wrapper._data, w2)


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, bad_weight_config", [
    (
        FakeQuantizedWeightWrapperBaseTensor,
        Float8FakeQuantizeConfig(),
        Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerTensor())
    ),
    (
        Float8FakeQuantizedWeightWrapperTensor,
        Float8FakeQuantizeConfig(),
        Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerTensor())
    ),
])
def test_wrapper_dispatch_weight_config_mismatch(wrapper_cls, weight_config, bad_weight_config, device):
    """Dispatch with mismatched weight_config raises AssertionError."""
    w = torch.randn(4, 64, 128, device=device)
    w1 = wrapper_cls(w, weight_config=weight_config)
    w2 = wrapper_cls(w, weight_config=bad_weight_config)

    with pytest.raises(AssertionError, match=r"^All FakeQuantizedWeightWrapperBaseTensor instances must have the same weight_config$"):
        with torch._C.DisableTorchFunctionSubclass():
            w1.copy_(w2)


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config, bad_act_config", [
    (
        FakeQuantizedWeightWrapperBaseTensor,
        Float8FakeQuantizeConfig(),
        Float8FakeQuantizeConfig(),
        Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerTensor())
    ),
    (
        Float8FakeQuantizedWeightWrapperTensor,
        Float8FakeQuantizeConfig(),
        Float8FakeQuantizeConfig(),
        Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerTensor())
    ),
])
def test_wrapper_dispatch_activation_config_mismatch(wrapper_cls, weight_config, act_config, bad_act_config, device):
    """Dispatch with mismatched activation_config raises AssertionError."""
    w = torch.randn(4, 64, 128, device=device)
    w1 = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    w2 = wrapper_cls(w, activation_config=bad_act_config, weight_config=weight_config)

    with pytest.raises(AssertionError, match=r"^All FakeQuantizedWeightWrapperBaseTensor instances must have the same activation_config$"):
        with torch._C.DisableTorchFunctionSubclass():
            w1.copy_(w2)


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls", [
    FakeQuantizedWeightWrapperBaseTensor,
    Float8FakeQuantizedWeightWrapperTensor,
])
@pytest.mark.parametrize("func", [
    torch.ops.aten.view.default,
    torch.ops.aten.add.Tensor,
])
def test_wrapper_dispatch_no_wrapped_args(wrapper_cls, func, device):
    """Dispatch without any wrapped tensor raises AssertionError."""
    w = torch.randn(4, 64, 128, device=device)
    expected = f"^__torch_dispatch__ called on {func.__name__} without any FakeQuantizedWeightWrapperBaseTensor arguments$"
    with pytest.raises(AssertionError, match=expected):
        wrapper_cls.__torch_dispatch__(func, (wrapper_cls,), (w,))


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config", [
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig()),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig()),
])
@pytest.mark.parametrize("func", [
    torch.ops.aten.add.Tensor,
    torch.ops.aten.mul.Tensor,
])
def test_wrapper_dispatch_non_preserved_op(wrapper_cls, weight_config, func, device):
    """Dispatch of an op not in _ops_to_preserve_subclass returns a plain tensor."""
    w = torch.randn(4, 64, 128, device=device)
    wrapper = wrapper_cls(w, weight_config=weight_config)

    with torch._C.DisableTorchFunctionSubclass():
        result = func(wrapper, wrapper)
    assert type(result) is torch.Tensor
    assert not isinstance(result, FakeQuantizedWeightWrapperBaseTensor)


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", [
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), None),
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
def test_wrapper_dispatch_detach(wrapper_cls, weight_config, act_config, device):
    """detach via __torch_dispatch__ creates a new wrapper with shared configs and detached _data."""
    w = torch.randn(4, 64, 128, requires_grad=True, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    with torch._C.DisableTorchFunctionSubclass():
        result = wrapper.detach()
        assert type(result) is wrapper_cls
        assert result._data.requires_grad is False
        assert result.weight_config is weight_config
        assert result.activation_config is act_config


# =========================================================================
# Standalone torch functions tests
# =========================================================================

@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("func, args", [
    (torch.mm, (torch.randn(16, 64), None)),
    (torch.addmm, (torch.randn(128), torch.randn(16, 64), None)),
])
def test_wrapper_torch_function_not_implemented(func, args, device):
    """FakeQuantizedWeightWrapperBaseTensor.__torch_function__ raises NotImplementedError."""
    w = torch.randn(64, 128, device=device)
    wrapper = FakeQuantizedWeightWrapperBaseTensor(w, weight_config=Float8FakeQuantizeConfig())
    args = tuple(wrapper if a is None else a for a in args)
    with pytest.raises(
        NotImplementedError,
        match=r"^FakeQuantizedWeightWrapperBaseTensor is not intended to be used directly, please override `__torch_function__` in a tensor subclass for your intended derived dtype\.$"
    ):
        func(*args)


@pytest.mark.parametrize("device", target_devices)
def test_wrapper_fake_quantize_not_implemented(device):
    """FakeQuantizedWeightWrapperBaseTensor._fake_quantize raises NotImplementedError."""
    w = torch.randn(64, 128, device=device)
    with pytest.raises(NotImplementedError, match=r"^FakeQuantizedWeightWrapperBaseTensor is not intended to be used directly, please override `_fake_quantize` in a tensor subclass for your intended derived dtype\.$"):
        FakeQuantizedWeightWrapperBaseTensor._fake_quantize(w, Float8FakeQuantizeConfig())


@pytest.mark.parametrize("wrapper_cls, weight_config", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig()),
])
@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("func", [
    torch.mm,
    torch.bmm,
])
def test_wrapper_torch_function_A_is_wrapper(wrapper_cls, weight_config, func, device):
    """__torch_function__ asserts A is not a wrapper."""
    w = torch.randn(4, 64, 128, device=device) if func is torch.bmm else torch.randn(64, 128, device=device)
    w1 = wrapper_cls(w, weight_config=weight_config)
    w2 = wrapper_cls(w, weight_config=weight_config)
    with pytest.raises(AssertionError, match=rf"^A should not be a {wrapper_cls.__name__}$"):
        func(w1, w2)


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig()),
])
@pytest.mark.parametrize("func", [torch.addmm, F.linear])
def test_wrapper_torch_function_B_not_wrapper(wrapper_cls, weight_config, func, device):
    """__torch_function__ asserts B is a wrapped weight."""
    bias = wrapper_cls(torch.randn(128, device=device), weight_config=weight_config)
    A = torch.randn(16, 64, device=device)
    B = torch.randn(64, 128, device=device)
    with pytest.raises(AssertionError, match=rf"^B should be a {wrapper_cls.__name__}$"):
        func(bias, A, B) if func is torch.addmm else func(A, B, bias)


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
@pytest.mark.parametrize("A_shape, w_shape, call_fn, kwargs", [
    ((16, 64),     (64, 128),     lambda a, w: torch.mm(a, w),      {}),
    ((16, 64),     (64, 128),     lambda a, w: torch.matmul(a, w),  {}),
    ((16, 64),     (64, 128),     lambda a, w, *, bias: torch.addmm(bias, a, w),  {"bias_shape": (128,)}),
    ((1, 16, 64),  (1, 64, 128),  lambda a, w: torch.bmm(a, w),     {}),
    ((16, 64),     (128, 64),     lambda a, w: F.linear(a, w),      {}),
    ((8, 1024),    (4, 1024, 2048), lambda a, w: torch._grouped_mm(a, w),  {}),
])
def test_wrapper_torch_function_activation_quantized_tensor(wrapper_cls, weight_config, act_config, A_shape, w_shape, call_fn, kwargs, device):
    """__torch_function__ asserts activation is not a TorchAOBaseTensor when act_config is set."""
    class DummyTensor(TorchAOBaseTensor):
        @classmethod
        def __torch_function__(cls, func, types, args, kwargs=None):
            if func in (torch.mm, torch.bmm, torch.addmm, torch.matmul, torch._grouped_mm, F.linear):
                return NotImplemented
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **(kwargs or {}))

    A = torch.randn(*A_shape, device=device).as_subclass(DummyTensor)
    B = wrapper_cls(torch.randn(*w_shape, device=device), activation_config=act_config, weight_config=weight_config)
    expected_match = r"^When an activation config is specified, the activation must not be a quantized tensor, got " + re.escape(str(type(A))) + "$"

    resolved = {}
    for k, v in kwargs.items():
        if k.endswith("_shape"):
            resolved[k[:-len("_shape")]] = torch.randn(*v, device=device)
        else:
            resolved[k] = v

    with pytest.raises(AssertionError, match=expected_match):
        call_fn(A, B, **resolved)


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig()),
])
def test_wrapper_torch_function_non_fake_quant_op(wrapper_cls, weight_config, device):
    """__torch_function__ on a non-fake-quant op passes through without fake quantization."""
    w1 = torch.randn(64, 128, device=device)
    w2 = torch.randn(64, 128, device=device)
    wrapper = wrapper_cls(w2.clone(), weight_config=weight_config)
    result = torch.add(w1, wrapper)
    expected = torch.add(w1, w2)
    assert type(result) is torch.Tensor
    assert not isinstance(result, TorchAOBaseTensor)
    assert torch.equal(result, expected)


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
def test_activation_qat_empty_input(wrapper_cls, weight_config, act_config, device):
    """Activation fake quant is skipped for empty tensors (expert with 0 tokens)."""
    w = torch.randn(64, 128, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    param = torch.nn.Parameter(wrapper)
    A = torch.randn(0, 64, device=device)
    out = torch.mm(A, param)
    assert out.shape == (0, 128), "Output should be empty when input is empty"
    assert out.numel() == 0


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig()),
])
@pytest.mark.parametrize("call_fn, A_shape, w_shape, bias_shape", [
    (lambda a, w, b: torch.addmm(b, a, w), (16, 64), (64, 128), (128,)),
    (lambda a, w, b: F.linear(a, w, b),    (16, 64), (128, 64), (128,)),
])
def test_bias_bypass(wrapper_cls, weight_config, call_fn, A_shape, w_shape, bias_shape, device):
    """Wrapped bias is unconditionally bypassed in __torch_function__."""
    A = torch.randn(*A_shape, device=device)
    w_wrapped = wrapper_cls(torch.randn(*w_shape, device=device), weight_config=weight_config)

    bias = torch.randn(*bias_shape, device=device)
    bias_wrapped = wrapper_cls(bias, weight_config=weight_config)
    out_wrapped = call_fn(A, w_wrapped, bias_wrapped)
    out_ref = call_fn(A, w_wrapped, bias)
    assert torch.equal(out_wrapped, out_ref), "bias should not be fake-quantized"



# =========================================================================
# Fake-quantization tests
# =========================================================================


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("granularity", [PerRow(dim=-1), PerRow(dim=-2)])
@pytest.mark.parametrize("wrapper_cls, weight_config", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig()),
])
def test_wrapper_fake_quantize(wrapper_cls, weight_config, granularity, device):
    """_fake_quantize applies FP8 fake quantization and returns a plain tensor."""
    w = torch.randn(1024, 2048, device=device)
    result = wrapper_cls._fake_quantize(w, weight_config, granularity)
    assert type(result) is torch.Tensor
    assert result.shape == w.shape
    assert result.dtype == w.dtype
    sqnr = compute_error(result, w)
    assert sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert sqnr > 30, f"SQNR too low ({sqnr:.1f} dB)"


@pytest.mark.parametrize("wrapper_cls, weight_config, act_config, sqnr_threshold", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None, 30),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig(), 27),
])
@pytest.mark.parametrize("call_fn, A_shape, w_shape, out_shape, kwargs", [
    # A_shape: (tokens, in_features) or (batch, tokens, in_features)
    # w_shape: (in_features, out_features), linear weight is (out_features, in_features)
    # kwargs: extra keyword arguments passed to call_fn
    # out_shape: (tokens, out_features) or (batch, tokens, out_features)
    (lambda a, w: torch.mm(a, w),                         (16, 1024),     (1024, 2048),     (16, 2048),                             {}),
    (lambda a, w: torch.bmm(a, w),                        (4, 16, 1024),  (4, 1024, 2048),  (4, 16, 2048),                          {}),
    (lambda a, w: F.linear(a, w),                      (16, 1024),     (2048, 1024),     (16, 2048),                             {}),
    (lambda a, w, *, bias=None: F.linear(a, w, bias),  (16, 1024),     (2048, 1024),     (16, 2048),     {"bias_shape": (2048,)}),
    (lambda a, w: torch.matmul(a, w),                  (16, 1024),     (1024, 2048),     (16, 2048),                             {}),
    (lambda a, w, *, bias: torch.addmm(bias, a, w),    (16, 1024),     (1024, 2048),     (16, 2048),     {"bias_shape": (2048,)}),
])
@pytest.mark.parametrize("device", target_devices)
def test_op_fake_quantize(wrapper_cls, weight_config, act_config, sqnr_threshold, call_fn, A_shape, w_shape, out_shape, kwargs, device):
    """__torch_function__ fake-quantizes weight/activation and produces good SQNR."""
    A = torch.randn(*A_shape, requires_grad=True, device=device)
    w = torch.randn(*w_shape, device=device)
    
    resolved = {}
    for k, v in kwargs.items():
        if k.endswith("_shape"):
            resolved[k[:-len("_shape")]] = torch.randn(*v, device=device)
        else:
            resolved[k] = v
    kwargs = resolved
    
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    param = torch.nn.Parameter(wrapper)

    A_ref = A.clone().detach().requires_grad_(True)
    w_ref = w.clone().detach().requires_grad_(True)

    ref_out = call_fn(A_ref, w_ref, **kwargs)
    out = call_fn(A, param, **kwargs)

    assert out.shape == out_shape

    sqnr = compute_error(out, ref_out)
    assert sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert sqnr > sqnr_threshold, f"Forward SQNR too low ({sqnr:.1f} dB)"

    ref_out.sum().backward()
    out.sum().backward()
    assert A.grad is not None
    assert compute_error(A.grad, A_ref.grad) > sqnr_threshold, f"Input grad SQNR too low ({compute_error(A.grad, A_ref.grad):.1f} dB)"
    assert param.grad is not None
    assert compute_error(param.grad, w_ref.grad) > sqnr_threshold, f"Weight grad SQNR too low ({compute_error(param.grad, w_ref.grad):.1f} dB)"


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config, sqnr_threshold", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None, 30),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig(), 27),
])
def test_op_mm(wrapper_cls, weight_config, act_config, sqnr_threshold, device):
    M, K, N = 16, 1024, 2048  # tokens, in_features, out_features

    A = torch.randn(M, K, requires_grad=True, device=device)
    w = torch.randn(K, N, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    param = torch.nn.Parameter(wrapper)

    A_ref = A.clone().detach().requires_grad_(True)
    w_ref = w.clone().detach().requires_grad_(True)
    ref_out = torch.mm(A_ref, w_ref)
    out = torch.mm(A, param)
    assert out.shape == (M, N)

    sqnr = compute_error(out, ref_out)
    assert sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert sqnr > sqnr_threshold, f"Forward SQNR too low ({sqnr:.1f} dB)"
    
    ref_out.sum().backward()
    out.sum().backward()
    assert A.grad is not None
    assert compute_error(A.grad, A_ref.grad) > sqnr_threshold, f"Input grad SQNR too low ({compute_error(A.grad, A_ref.grad):.1f} dB)"
    assert param.grad is not None
    assert compute_error(param.grad, w_ref.grad) > sqnr_threshold, f"Weight grad SQNR too low ({compute_error(param.grad, w_ref.grad):.1f} dB)"



@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config, sqnr_threshold", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None, 30),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow()), 27),
])
def test_op_bmm(wrapper_cls, weight_config, act_config, sqnr_threshold, device):
    B, M, K, N = 4, 16, 1024, 2048  # batch, tokens, in_features, out_features

    A = torch.randn(B, M, K, requires_grad=True, device=device)
    w = torch.randn(B, K, N, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    param = torch.nn.Parameter(wrapper)

    A_ref = A.clone().detach().requires_grad_(True)
    w_ref = w.clone().detach().requires_grad_(True)
    ref_out = torch.bmm(A_ref, w_ref)
    out = torch.bmm(A, param)
    assert out.shape == (B, M, N)

    sqnr = compute_error(out, ref_out)
    assert sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert sqnr > sqnr_threshold, f"Forward SQNR too low ({sqnr:.1f} dB)"

    ref_out.sum().backward()
    out.sum().backward()
    assert A.grad is not None
    assert compute_error(A.grad, A_ref.grad) > sqnr_threshold, f"Input grad SQNR too low ({compute_error(A.grad, A_ref.grad):.1f} dB)"
    assert param.grad is not None
    assert compute_error(param.grad, w_ref.grad) > sqnr_threshold, f"Weight grad SQNR too low ({compute_error(param.grad, w_ref.grad):.1f} dB)"



@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config, sqnr_threshold", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None, 30),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow()), 27),
])
def test_op_linear(wrapper_cls, weight_config, act_config, sqnr_threshold, device):
    M, K, N = 16, 1024, 2048  # tokens, in_features, out_features

    A = torch.randn(M, K, requires_grad=True, device=device)
    w = torch.randn(N, K, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    param = torch.nn.Parameter(wrapper)

    A_ref = A.clone().detach().requires_grad_(True)
    w_ref = w.clone().detach().requires_grad_(True)
    ref_out = F.linear(A_ref, w_ref)
    out = F.linear(A, param)
    assert out.shape == (M, N)

    sqnr = compute_error(out, ref_out)
    assert sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert sqnr > sqnr_threshold, f"Forward SQNR too low ({sqnr:.1f} dB)"

    ref_out.sum().backward()
    out.sum().backward()
    assert A.grad is not None
    assert compute_error(A.grad, A_ref.grad) > sqnr_threshold, f"Input grad SQNR too low ({compute_error(A.grad, A_ref.grad):.1f} dB)"
    assert param.grad is not None
    assert compute_error(param.grad, w_ref.grad) > sqnr_threshold, f"Weight grad SQNR too low ({compute_error(param.grad, w_ref.grad):.1f} dB)"



@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config, sqnr_threshold", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None, 30),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow()), 27),
])
def test_op_matmul(wrapper_cls, weight_config, act_config, sqnr_threshold, device):
    M, K, N = 16, 1024, 2048  # tokens, in_features, out_features

    A = torch.randn(M, K, requires_grad=True, device=device)
    w = torch.randn(K, N, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    param = torch.nn.Parameter(wrapper)

    A_ref = A.clone().detach().requires_grad_(True)
    w_ref = w.clone().detach().requires_grad_(True)
    ref_out = torch.matmul(A_ref, w_ref)
    out = torch.matmul(A, param)
    assert out.shape == (M, N)

    sqnr = compute_error(out, ref_out)
    assert sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert sqnr > sqnr_threshold, f"Forward SQNR too low ({sqnr:.1f} dB)"

    ref_out.sum().backward()
    out.sum().backward()
    assert A.grad is not None
    assert compute_error(A.grad, A_ref.grad) > sqnr_threshold, f"Input grad SQNR too low ({compute_error(A.grad, A_ref.grad):.1f} dB)"
    assert param.grad is not None
    assert compute_error(param.grad, w_ref.grad) > sqnr_threshold, f"Weight grad SQNR too low ({compute_error(param.grad, w_ref.grad):.1f} dB)"



@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config, sqnr_threshold", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None, 30),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow()), 27),
])
def test_op_addmm(wrapper_cls, weight_config, act_config, sqnr_threshold, device):
    M, K, N = 16, 1024, 2048  # tokens, in_features, out_features

    A = torch.randn(M, K, requires_grad=True, device=device)
    w = torch.randn(K, N, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    param = torch.nn.Parameter(wrapper)

    bias = torch.randn(N, device=device)
    A_ref = A.clone().detach().requires_grad_(True)
    w_ref = w.clone().detach().requires_grad_(True)
    ref_out = torch.addmm(bias, A_ref, w_ref)
    out = torch.addmm(bias, A, param)
    assert out.shape == (M, N)

    sqnr = compute_error(out, ref_out)
    assert sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert sqnr > sqnr_threshold, f"Forward SQNR too low ({sqnr:.1f} dB)"
    
    ref_out.sum().backward()
    out.sum().backward()
    assert A.grad is not None
    assert compute_error(A.grad, A_ref.grad) > sqnr_threshold, f"Input grad SQNR too low ({compute_error(A.grad, A_ref.grad):.1f} dB)"
    assert param.grad is not None
    assert compute_error(param.grad, w_ref.grad) > sqnr_threshold, f"Weight grad SQNR too low ({compute_error(param.grad, w_ref.grad):.1f} dB)"

@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config, sqnr_threshold", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None, 20),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig(), 20),
])
def test_op_grouped_mm(wrapper_cls, weight_config, act_config, sqnr_threshold, device):
    if device != "cuda":
        pytest.skip("grouped_mm CPU backward corrupts subsequent forward calls")

    S, E, K, N = 16, 4, 1024, 2048  # total_tokens, experts, in_features, out_features

    A = torch.randn(S, K, requires_grad=True, device=device)
    w = torch.randn(E, K, N, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    param = torch.nn.Parameter(wrapper)

    offs = torch.tensor([4, 4, 4, 4], dtype=torch.int32)
    A_ref = A.clone().detach().requires_grad_(True)
    w_ref = w.clone().detach().requires_grad_(True)
    ref_out = torch._grouped_mm(A_ref, w_ref, offs=offs)
    out = torch._grouped_mm(A, param, offs=offs)
    assert out.shape == (S, N)

    sqnr = compute_error(out, ref_out)
    assert sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert sqnr > sqnr_threshold, f"Forward SQNR too low ({sqnr:.1f} dB)"

    ref_out.backward(torch.ones_like(ref_out))
    out.backward(torch.ones_like(out))
    assert A.grad is not None
    assert compute_error(A.grad, A_ref.grad) > sqnr_threshold, f"Input grad SQNR too low ({compute_error(A.grad, A_ref.grad):.1f} dB)"
    assert param.grad is not None
    assert compute_error(param.grad, w_ref.grad) > sqnr_threshold, f"Weight grad SQNR too low ({compute_error(param.grad, w_ref.grad):.1f} dB)"


@pytest.mark.parametrize("device", target_devices)
def test_op_grouped_mm_no_activation_fake_quantization(device):
    """grouped_mm with weight-only fake quantization. Run individually on CPU —
    the grouped_mm CPU backward corrupts state for subsequent fake-quantized calls."""
    S, E, K, N = 16, 4, 1024, 2048  # total_tokens, experts, in_features, out_features

    A = torch.randn(S, K, requires_grad=True, device=device)
    w = torch.randn(E, K, N, device=device)
    wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=Float8FakeQuantizeConfig())
    param = torch.nn.Parameter(wrapper)

    offs = torch.tensor([4, 4, 4, 4], dtype=torch.int32)
    A_ref = A.clone().detach().requires_grad_(True)
    w_ref = w.clone().detach().requires_grad_(True)
    ref_out = torch._grouped_mm(A_ref, w_ref, offs=offs)
    out = torch._grouped_mm(A, param, offs=offs)
    assert out.shape == (S, N)

    sqnr = compute_error(out, ref_out)
    assert sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert sqnr > 20, f"Forward SQNR too low ({sqnr:.1f} dB)"

    ref_out.backward(torch.ones_like(ref_out))
    out.backward(torch.ones_like(out))
    assert A.grad is not None
    assert compute_error(A.grad, A_ref.grad) > 20, f"Input grad SQNR too low ({compute_error(A.grad, A_ref.grad):.1f} dB)"
    assert param.grad is not None
    assert compute_error(param.grad, w_ref.grad) > 20, f"Weight grad SQNR too low ({compute_error(param.grad, w_ref.grad):.1f} dB)"


@pytest.mark.parametrize("device", target_devices)
def test_op_grouped_mm_activation_fake_quantization(device):
    """grouped_mm with weight+activation fake quantization. Run individually on CPU —
    the grouped_mm CPU backward corrupts state for subsequent fake-quantized calls."""
    S, E, K, N = 16, 4, 1024, 2048  # total_tokens, experts, in_features, out_features

    A = torch.randn(S, K, requires_grad=True, device=device)
    w = torch.randn(E, K, N, device=device)
    wrapper = Float8FakeQuantizedWeightWrapperTensor(
        w,
        activation_config=Float8FakeQuantizeConfig(),
        weight_config=Float8FakeQuantizeConfig(),
    )
    param = torch.nn.Parameter(wrapper)

    offs = torch.tensor([4, 4, 4, 4], dtype=torch.int32)
    A_ref = A.clone().detach().requires_grad_(True)
    w_ref = w.clone().detach().requires_grad_(True)
    ref_out = torch._grouped_mm(A_ref, w_ref, offs=offs)
    out = torch._grouped_mm(A, param, offs=offs)
    assert out.shape == (S, N)

    sqnr = compute_error(out, ref_out)
    assert sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert sqnr > 20, f"Forward SQNR too low ({sqnr:.1f} dB)"

    ref_out.backward(torch.ones_like(ref_out))
    out.backward(torch.ones_like(out))
    assert A.grad is not None
    assert compute_error(A.grad, A_ref.grad) > 20, f"Input grad SQNR too low ({compute_error(A.grad, A_ref.grad):.1f} dB)"
    assert param.grad is not None
    assert compute_error(param.grad, w_ref.grad) > 20, f"Weight grad SQNR too low ({compute_error(param.grad, w_ref.grad):.1f} dB)"



@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("dtype", [torch.float32, ])
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
@pytest.mark.parametrize("call_fn, A_shape, w_shape, kwargs", [
    # torch.mm: weight [K, N] at args[1], contracted dim=-2
    (lambda a, w: torch.mm(a, w),             (16, 64),    (64, 128),     {}),
    # torch.matmul: same as mm
    (lambda a, w: torch.matmul(a, w),         (16, 64),    (64, 128),     {}),
    # torch.bmm: weight [B, K, N] at args[1], contracted dim=-2
    (lambda a, w: torch.bmm(a, w),            (4, 16, 64), (4, 64, 128),  {}),
    # F.linear: weight [N, K] at args[1], contracted dim=-1
    (lambda a, w: F.linear(a, w),             (16, 64),    (128, 64),     {}),
    # torch.addmm: weight [K, N] at args[2], contracted dim=-2
    (lambda a, w, *, bias: torch.addmm(bias, a, w), (16, 64), (64, 128), {"bias_shape": (128,)}),
    # torch._grouped_mm: weight [E, K, N] at args[1], S=16 tokens, E=4 experts
    (lambda a, w, *, offs: torch._grouped_mm(a, w, offs=offs), (16, 1024), (4, 1024, 2048), {"offs": torch.tensor([4, 4, 4, 4], dtype=torch.int32)}),
])
def test_compatibility_with_torch_compile(wrapper_cls, weight_config, act_config, call_fn, A_shape, w_shape, kwargs, dtype, device):
    """torch.compile through a Python wrapper should match eager.

    Wraps the op in a plain Python function before compiling. This is the
    correct usage pattern: ``torch.compile(forward_pass)`` forces Dynamo to
    trace through the Python call, which triggers ``__torch_function__``
    dispatch on tensor subclasses.

    Compiling a built-in C-extension directly may bypass
    ``__torch_function__``, producing a false sense of correctness.
    """
    def prepare_arguments():
        A = torch.randn(*A_shape, dtype=dtype, device=device)
        w = torch.randn(*w_shape, dtype=dtype, device=device)
        wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
        
        resolved = {}
        for k, v in kwargs.items():
            if k.endswith("_shape"):
                resolved[k[:-len("_shape")]] = torch.randn(*v, dtype=dtype, device=device)
            elif isinstance(v, torch.Tensor):
                resolved[k] = v.to(device=device)
            else:
                resolved[k] = v
        
        return A, wrapper, resolved

    def forward_pass(activation, weight, kwargs):
        return call_fn(activation, weight, **kwargs)

    torch._dynamo.reset()
    compiled = torch.compile(forward_pass, fullgraph=True)

    def run_test():
        activation, weight, resolved_kwargs = prepare_arguments()
        eager_out = forward_pass(activation, weight, resolved_kwargs)
        out = compiled(activation, weight, resolved_kwargs)
        assert out.shape == eager_out.shape
        assert torch.allclose(out, eager_out), "Compiled output should match eager output"

    # 1. Warm up the compiler with the initial shape (creates the first graph)
    run_test()

    # 2. Enter the stance to block subsequent recompilations
    with torch.compiler.set_stance("fail_on_recompile"):
        for _ in range(5):
            run_test()