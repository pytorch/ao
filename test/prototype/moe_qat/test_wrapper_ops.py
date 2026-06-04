import os
import re
import copy
import pytest
import torch
import torch.nn.functional as F
from torch.distributed._tensor import DTensor, Shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_qat import MoEQATConfig
from torchao.prototype.moe_qat.tensor import FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizedWeightWrapperTensor
from torchao.quantization.granularity import PerRow, PerTensor
from torchao.quantization.qat.fake_quantize_config import FakeQuantizeConfigBase, Float8FakeQuantizeConfig
from torchao.quantization.quant_api import quantize_
from torchao.utils import TorchAOBaseTensor

from .reference_moe import MoE
from .testing_utils import _expert_weight_filter, target_devices


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
        Float8FakeQuantizeConfig(dtype=torch.float8_e5m2, granularity=PerRow(dim=-1))
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
        match=(
            r"^FakeQuantizedWeightWrapperBaseTensor is not intended to be used directly, "
            r"please override `__torch_function__` in a tensor subclass for "
            r"your intended derived dtype\.$"
        ),
    ):
        func(*args)


@pytest.mark.parametrize("device", target_devices)
def test_wrapper_fake_quantize_not_implemented(device):
    """FakeQuantizedWeightWrapperBaseTensor._fake_quantize raises NotImplementedError."""
    w = torch.randn(64, 128, device=device)
    with pytest.raises(
        NotImplementedError,
        match=(
            r"^FakeQuantizedWeightWrapperBaseTensor is not intended to be used directly, "
            r"please override `_fake_quantize` in a tensor subclass for "
            r"your intended derived dtype\.$"
        ),
    ):
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
    ((16, 64),     (128, 64),     lambda a, w: torch.mm(a, w.T),      {}),
    ((16, 64),     (128, 64),     lambda a, w: torch.matmul(a, w.T),  {}),
    ((16, 64),     (128, 64),     lambda a, w, *, bias: torch.addmm(bias, a, w.T),  {"bias_shape": (128,)}),
    ((1, 16, 64),  (1, 128, 64),  lambda a, w: torch.bmm(a, w.transpose(-2, -1)),     {}),
    ((16, 64),     (128, 64),     lambda a, w: F.linear(a, w),      {}),
    ((8, 1024),    (4, 2048, 1024), lambda a, w: torch._grouped_mm(a, w.transpose(-2, -1)),  {}),
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
    w = torch.randn(128, 64, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    param = torch.nn.Parameter(wrapper)
    A = torch.randn(0, 64, device=device)
    out = torch.mm(A, param.T)
    assert out.shape == (0, 128), "Output should be empty when input is empty"
    assert out.numel() == 0


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig()),
])
@pytest.mark.parametrize("call_fn, A_shape, w_shape, bias_shape", [
    (lambda a, w, b: torch.addmm(b, a, w.T), (16, 64), (128, 64), (128,)),
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
@pytest.mark.parametrize("wrapper_cls, weight_config, sqnr_threshold", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), 30),
])
def test_wrapper_fake_quantize(wrapper_cls, weight_config, granularity, sqnr_threshold, device):
    """_fake_quantize applies FP8 fake quantization and returns a plain tensor."""
    if granularity.dim in (-2, 0):
        w = torch.randn(2048, 1024, device=device).T  # F-contiguous for PerRow(dim=-2)
    else:
        w = torch.randn(1024, 2048, device=device)  # C-contiguous for PerRow(dim=-1)
    result = wrapper_cls._fake_quantize(w, weight_config, granularity)
    assert type(result) is torch.Tensor
    assert result.shape == w.shape
    assert result.dtype == w.dtype
    sqnr = compute_error(result, w)
    assert sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert sqnr > sqnr_threshold, f"SQNR too low ({sqnr:.1f} dB)"


@pytest.mark.parametrize("wrapper_cls, weight_config, act_config, sqnr_threshold", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None, 30),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig(), 27),
])
@pytest.mark.parametrize("call_fn, A_shape, w_shape, out_shape, kwargs", [
    # A_shape: (tokens, in_features) or (batch, tokens, in_features)
    # w_shape: (in_features, out_features), linear weight is (out_features, in_features)
    # kwargs: extra keyword arguments passed to call_fn
    # out_shape: (tokens, out_features) or (batch, tokens, out_features)
    (lambda a, w: torch.mm(a, w.T),                         (16, 1024),     (2048, 1024),     (16, 2048),                             {}),
    (lambda a, w: torch.bmm(a, w.transpose(-2, -1)),        (4, 16, 1024),  (4, 2048, 1024),  (4, 16, 2048),                          {}),
    (lambda a, w: F.linear(a, w),                         (16, 1024),     (2048, 1024),     (16, 2048),                             {}),
    (lambda a, w, *, bias=None: F.linear(a, w, bias),     (16, 1024),     (2048, 1024),     (16, 2048),     {"bias_shape": (2048,)}),
    (lambda a, w: torch.matmul(a, w.T),                   (16, 1024),     (2048, 1024),     (16, 2048),                             {}),
    (lambda a, w, *, bias: torch.addmm(bias, a, w.T),     (16, 1024),     (2048, 1024),     (16, 2048),     {"bias_shape": (2048,)}),
])
@pytest.mark.parametrize("device", target_devices)
def test_op_fake_quantize(wrapper_cls, weight_config, act_config, sqnr_threshold, call_fn, A_shape, w_shape, out_shape, kwargs, device):
    """__torch_function__ fake-quantizes weight/activation and produces good SQNR."""
    
    activation_tensor = torch.randn(*A_shape, device=device)
    weight_tensor = torch.randn(*w_shape, device=device)

    # Prepare the wrapper tensor 
    activation = torch.nn.Parameter(activation_tensor.clone())
    weight = torch.nn.Parameter(wrapper_cls(
        weight_tensor.clone(),
        activation_config=act_config,
        weight_config=weight_config,
    ))

    resolved = {}
    for k, v in kwargs.items():
        if k.endswith("_shape"):
            resolved[k[:-len("_shape")]] = torch.nn.Parameter(torch.randn(*v, device=device))
        else:
            resolved[k] = v

    # Prepare the reference
    ref_activation = torch.nn.Parameter(activation_tensor.clone())
    ref_weight = torch.nn.Parameter(weight_tensor.clone())
    
    # No graph exists yet. So we do not need .detach().requires_grad_() after .clone()
    ref_resolved = {k : v.clone() if isinstance(v, torch.Tensor) else v for k, v in resolved.items()}


    # Run the function call
    learning_rate = 1 # set learning rate to 1 to ensure noises in new weights are not suppressed or amplified

    optimizer = torch.optim.SGD([weight], lr=learning_rate)
    out = call_fn(activation, weight, **resolved)
    
    ref_optimizer = torch.optim.SGD([ref_weight], lr=learning_rate)
    ref_out = call_fn(ref_activation, ref_weight, **ref_resolved)


    assert out.shape == out_shape
    assert out.shape == ref_out.shape

    sqnr = compute_error(out, ref_out)
    assert sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert sqnr > sqnr_threshold, f"Forward SQNR too low ({sqnr:.1f} dB)"

    out.sum().backward()
    ref_out.sum().backward()
    
    assert activation.grad is not None
    activation_grad_sqnr = compute_error(activation.grad, ref_activation.grad)
    assert activation_grad_sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert activation_grad_sqnr > sqnr_threshold, f"Input grad SQNR too low ({activation_grad_sqnr:.1f} dB)"
    
    assert weight.grad is not None
    weight_grad_sqnr = compute_error(weight.grad, ref_weight.grad)
    assert weight_grad_sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert weight_grad_sqnr > sqnr_threshold, f"Weight grad SQNR too low ({weight_grad_sqnr:.1f} dB)"

    # Update weights
    optimizer.step()
    ref_optimizer.step()

    assert not torch.equal(weight_tensor, weight), "weight should be updated"
    assert not torch.equal(weight_tensor, ref_weight), "ref_weight is not updated. Bug in test_op_fake_quantize"
    new_weight_sqnr = compute_error(weight, ref_weight)
    assert new_weight_sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
    assert new_weight_sqnr > sqnr_threshold, f"New weight SQNR too low ({new_weight_sqnr:.1f} dB)"



@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config, sqnr_threshold", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None, 20),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig(), 20),
])
def test_op_grouped_mm(wrapper_cls, weight_config, act_config, sqnr_threshold, device):
    """grouped_mm with fake-quantized weight produces good forward+backward SQNR. GPU-only."""
    if device != "cuda":
        pytest.skip("grouped_mm CPU backward corrupts subsequent forward calls")

    S, E, K, N = 16, 4, 1024, 2048  # total_tokens, experts, in_features, out_features

    A = torch.randn(S, K, requires_grad=True, device=device)
    w = torch.randn(E, N, K, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    param = torch.nn.Parameter(wrapper)

    offs = torch.tensor([4, 4, 4, 4], dtype=torch.int32)
    A_ref = A.clone().detach().requires_grad_(True)
    w_ref = w.clone().detach().requires_grad_(True)
    ref_out = torch._grouped_mm(A_ref, w_ref.transpose(-2, -1), offs=offs)
    out = torch._grouped_mm(A, param.transpose(-2, -1), offs=offs)
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


@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
@pytest.mark.parametrize("call_fn, A_shape, w_shape, kwargs", [
    # torch.mm: weight stored as [N, K], transposed at call site
    (lambda a, w: torch.mm(a, w.T),             (16, 64),    (128, 64),     {}),
    # torch.matmul: same as mm
    (lambda a, w: torch.matmul(a, w.T),         (16, 64),    (128, 64),     {}),
    # torch.bmm: weight stored as [B, N, K], transposed at call site
    (lambda a, w: torch.bmm(a, w.transpose(-2, -1)), (4, 16, 64), (4, 128, 64),  {}),
    # F.linear: weight [N, K] at args[1], contracted dim=-1 (no transpose needed)
    (lambda a, w: F.linear(a, w),               (16, 64),    (128, 64),     {}),
    # torch.addmm: weight stored as [N, K], transposed at call site
    (lambda a, w, *, bias: torch.addmm(bias, a, w.T), (16, 64), (128, 64), {"bias_shape": (128,)}),
    # torch._grouped_mm: weight stored as [E, N, K], transposed at call site
    (lambda a, w, *, offs: torch._grouped_mm(a, w.transpose(-2, -1), offs=offs), (16, 1024), (4, 2048, 1024), {
        "offs": torch.tensor([4, 4, 4, 4], dtype=torch.int32), "_skip_cpu": True
    }),
])
def test_compatibility_with_torch_compile(wrapper_cls, weight_config, act_config, call_fn, A_shape, w_shape, kwargs, device, fullgraph):
    """torch.compile through a Python wrapper should match eager."""
    if device == "cpu" and kwargs.get("_skip_cpu", False):
        pytest.skip("grouped_mm is not fully supported on CPU yet.")

    def prepare_arguments():
        A = torch.randn(*A_shape, device=device).requires_grad_(True)
        w = torch.randn(*w_shape, device=device).requires_grad_(True)
        wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
        
        resolved = {}
        for k, v in kwargs.items():
            if k.endswith("_shape"):
                resolved[k[:-len("_shape")]] = torch.randn(*v, device=device).requires_grad_(True)
            elif isinstance(v, torch.Tensor):
                resolved[k] = v.to(device=device)
            else:
                resolved[k] = v
        
        return A, wrapper, resolved

    def eager_forward(activation, weight, kwargs):
        return call_fn(activation, weight, **kwargs)

    compiled_forward = torch.compile(eager_forward, fullgraph=fullgraph)

    def run_test():
        eager_activation, eager_weight, eager_kwargs = prepare_arguments()
        compiled_activation = eager_activation.clone().detach().requires_grad_(True)
        compiled_weight = eager_weight.clone().detach().requires_grad_(True)
        compiled_kwargs = {
            k: copy.deepcopy(v) if not isinstance(v, torch.Tensor) else (
                v.clone().detach().requires_grad_(True) if v.requires_grad else v.clone().detach()
            )
            for k, v in eager_kwargs.items()
        }

        eager_out = eager_forward(eager_activation, eager_weight, eager_kwargs)
        compiled_out = compiled_forward(compiled_activation, compiled_weight, compiled_kwargs)
        assert eager_out.shape == compiled_out.shape
        assert torch.allclose(compiled_out, eager_out), "Compiled output should match eager output"

        eager_out.sum().backward()
        compiled_out.sum().backward()

        assert eager_activation.grad is not None
        assert torch.allclose(eager_activation.grad, compiled_activation.grad), \
            "Compiled activation grad should match eager activation grad"

        assert eager_weight.grad is not None
        assert torch.allclose(eager_weight.grad, compiled_weight.grad), \
            "Compiled weight grad should match eager weight grad"

    # 1. Warm up the compiler with the initial shape (creates the first graph)
    run_test()

    # 2. Enter the stance to block subsequent recompilations
    with torch.compiler.set_stance("fail_on_recompile"):
        for _ in range(5):
            run_test()


