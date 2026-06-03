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
from .testing_utils import _expert_weight_filter, _set_seed, target_devices


# =========================================================================
# Test __init__
# =========================================================================


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, expected_match", [
    (
        FakeQuantizedWeightWrapperBaseTensor,
        r"^Must specify `weight_config` in FakeQuantizedWeightWrapperBaseTensor\.$"
    ),
    (
        Float8FakeQuantizedWeightWrapperTensor, 
        r"^Only `Float8FakeQuantizeConfig` is supported for `weight_config` in Float8FakeQuantizedWeightWrapperTensor\.$"
    ),
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
@pytest.mark.parametrize("wrapper_cls, weight_config, expected_match", [
    (
        Float8FakeQuantizedWeightWrapperTensor,
        None, 
        r"^Only `Float8FakeQuantizeConfig` is supported for `weight_config` in Float8FakeQuantizedWeightWrapperTensor\.$"
    ),
    (
        Float8FakeQuantizedWeightWrapperTensor, 
        Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow(dim=-2)), 
        r"^Only the row-wise granularity is supported\.$"
    ),
    (
        Float8FakeQuantizedWeightWrapperTensor, 
        Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerTensor()), 
        r"^Only the row-wise granularity is supported\.$"
    ),
])
def test_wrapper_init_rejects_invalid_weight_config(wrapper_cls, weight_config, expected_match, device):
    """Wrapper subclass rejects non-matching weight_config type or granularity."""
    if weight_config is None:
        class DummyConfig(FakeQuantizeConfigBase):
            pass
        weight_config = DummyConfig()

    w = torch.randn(64, 128, device=device)
    with pytest.raises(ValueError, match=expected_match):
        wrapper_cls(w, weight_config=weight_config)


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, expected_match", [
    (
        Float8FakeQuantizedWeightWrapperTensor, 
        r"^Only `Float8FakeQuantizeConfig` is supported for `activation_config` in Float8FakeQuantizedWeightWrapperTensor\.$"
    ),
])
def test_wrapper_init_rejects_invalid_activation_config(wrapper_cls, expected_match, device):
    """Wrapper subclass rejects non-matching activation_config type."""
    class DummyConfig(FakeQuantizeConfigBase):
        pass

    w = torch.randn(64, 128, device=device)
    with pytest.raises(ValueError, match=expected_match):
        wrapper_cls(w, activation_config=DummyConfig(), weight_config=Float8FakeQuantizeConfig())


# =========================================================================
# __deepcopy__
# =========================================================================


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", [
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), None),
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
def test_wrapper_deepcopy(wrapper_cls, weight_config, act_config, device):
    """deepcopy creates an independent wrapper with independent _data."""
    w = torch.randn(128, 64, device=device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    wrapper_copy = copy.deepcopy(wrapper)

    assert wrapper_copy is not wrapper
    assert isinstance(wrapper, wrapper_cls)
    assert isinstance(wrapper_copy, wrapper_cls)

    assert wrapper_copy._data is not wrapper._data
    assert torch.equal(wrapper_copy._data, wrapper._data)

    assert wrapper_copy.weight_config == wrapper.weight_config
    assert wrapper_copy.weight_config is not wrapper.weight_config

    assert wrapper_copy.activation_config == wrapper.activation_config
    if act_config is not None:
        assert wrapper_copy.activation_config is not wrapper.activation_config
    
    activation = torch.nn.Parameter(torch.randn(16, 64, device=device))
    if wrapper_cls is FakeQuantizedWeightWrapperBaseTensor:
        with pytest.raises(
            NotImplementedError,
            match=(
                r"FakeQuantizedWeightWrapperBaseTensor is not intended to be used directly, "
                r"please override `__torch_function__` in a tensor subclass for your intended derived dtype\."
            )
        ):
            torch.mm(activation, wrapper_copy.T)
    else:
        out = torch.mm(activation, wrapper.T)
        out_copy = torch.mm(activation, wrapper_copy.T)
        assert torch.equal(out, out_copy), "The cloned tensor should yield identical results."


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
# FSDP hook unit tests
# =========================================================================


@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", [
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), None),
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
@pytest.mark.parametrize("tensor_dtype", [
    torch.float32, torch.bfloat16,
])
@pytest.mark.parametrize("param_dtype", [
    torch.float32, torch.bfloat16,
])
def test_fsdp_pre_all_gather(wrapper_cls, weight_config, act_config, tensor_dtype, param_dtype):
    """fsdp_pre_all_gather casts _data to mp_policy.param_dtype and returns it."""
    w = torch.randn(64, 128, dtype=tensor_dtype)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype)

    all_gather_inputs, metadata = wrapper.fsdp_pre_all_gather(
        None, None, None, None, mp_policy
    )
    (data,) = all_gather_inputs
    assert metadata == ()
    assert data.dtype == param_dtype
    assert data.shape == w.shape
    assert torch.equal(data, w.to(param_dtype))


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("on_meta", [False, True])
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", [
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), None),
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
def test_fsdp_post_all_gather_first_step(wrapper_cls, weight_config, act_config, device, on_meta):
    """fsdp_post_all_gather with out=None creates a new wrapper with preserved configs."""
    w_device = "meta" if on_meta else device
    w = torch.empty(64, 128, device=w_device)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)
    gathered = torch.empty(4, 64, 128, device=device)

    result, inner_tensors = wrapper.fsdp_post_all_gather(
        (gathered,), None, torch.float32, out=None
    )
    assert type(result) is wrapper_cls
    assert result._data is gathered
    assert isinstance(inner_tensors, tuple)
    assert len(inner_tensors) == 1
    assert inner_tensors[0] is gathered
    assert result.weight_config is weight_config
    assert result.activation_config is act_config


@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", [
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), None),
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fsdp_post_all_gather_existing_out_same_dtype(wrapper_cls, weight_config, act_config, dtype):
    """out is bare wrapper, same dtype: configs restored, storage pointer verified."""
    w = torch.empty(2, 32, 64, device="meta")  # different shape/device — must not be used
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    out = wrapper_cls(torch.randn(4, 64, 128, dtype=dtype), weight_config=weight_config)
    out.activation_config = None
    out.weight_config = None

    data = out._data
    result = wrapper.fsdp_post_all_gather(
        (data,), None, dtype, out=out
    )
    assert result is None
    assert out._data is data  # storage-sharing: same pointer reused
    assert out.weight_config is weight_config
    assert out.activation_config is act_config

    # If data has the same dtype but different storage, the assertion should fire
    storage_mismatch = data.clone()
    with pytest.raises(AssertionError):
        wrapper.fsdp_post_all_gather(
            (storage_mismatch,), None, dtype, out=out
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
def test_fsdp_post_all_gather_existing_out_same_dtype_dtensor(wrapper_cls, weight_config, act_config, dtype):
    """out is DTensor with wrapped local_tensor — configs restored on local_tensor."""
    os.environ.update({"MASTER_ADDR": "localhost", "MASTER_PORT": "12355", "RANK": "0", "WORLD_SIZE": "1"})


    w = torch.empty(2, 32, 64, device="meta")
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    mesh = DeviceMesh("cpu", torch.arange(1))
    out_data = torch.randn(4, 64, 128, dtype=dtype)
    out_local = wrapper_cls(out_data, weight_config=weight_config)
    out = DTensor.from_local(out_local, mesh, [Shard(0)])
    out._local_tensor.activation_config = None
    out._local_tensor.weight_config = None

    data = out._local_tensor._data
    result = wrapper.fsdp_post_all_gather(
        (data,), None, dtype, out=out
    )

    assert result is None
    assert out._local_tensor._data is data
    assert out._local_tensor.weight_config is weight_config
    assert out._local_tensor.activation_config is act_config

    # If data has the same dtype but different storage, the assertion should fire
    storage_mismatch = data.clone()
    with pytest.raises(AssertionError):
        wrapper.fsdp_post_all_gather(
            (storage_mismatch,), None, dtype, out=out
        )


@pytest.mark.parametrize("in_dtype, out_dtype", [
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16),
])
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", [
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), None),
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
def test_fsdp_post_all_gather_existing_out_cross_dtype(wrapper_cls, weight_config, act_config, in_dtype, out_dtype):
    """out is bare wrapper, different dtype: configs restored, out_data.copy_(data)."""
    w = torch.empty(2, 32, 64, device="meta")  # different shape/device — must not be used
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    out = wrapper_cls(torch.randn(4, 64, 128, dtype=out_dtype), weight_config=weight_config)
    out.activation_config = None
    out.weight_config = None

    data = torch.randn(4, 64, 128, dtype=in_dtype)
    out_data_before = out._data
    result = wrapper.fsdp_post_all_gather(
        (data,), None, out_dtype, out=out
    )

    assert result is None
    assert out._data is out_data_before  # in-place copy_: same object
    assert out.weight_config is weight_config
    assert out.activation_config is act_config
    assert torch.equal(out._data, data.to(out_dtype))

    # If param_dtype doesn't match out_data.dtype, the assertion should fire
    bad_param_dtype = torch.float64
    expected_msg = (
        f"^`out`\\(dtype={out_dtype}\\) dose not match "
        f"the mixed precision policy param_dtype {bad_param_dtype}$"
    )
    with pytest.raises(AssertionError, match=expected_msg):
        wrapper.fsdp_post_all_gather(
            (data,), None, bad_param_dtype, out=out
        )


@pytest.mark.parametrize("in_dtype, out_dtype", [
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16),
])
@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
def test_fsdp_post_all_gather_existing_out_cross_dtype_dtensor(wrapper_cls, weight_config, act_config, in_dtype, out_dtype):
    """out is DTensor with wrapped local_tensor, cross-dtype: configs restored, copy_ applied."""
    os.environ.update({"MASTER_ADDR": "localhost", "MASTER_PORT": "12355", "RANK": "0", "WORLD_SIZE": "1"})


    w = torch.empty(2, 32, 64, device="meta")
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    mesh = DeviceMesh("cpu", torch.arange(1))
    out_data = torch.randn(4, 64, 128, dtype=out_dtype)
    out_local = wrapper_cls(out_data, weight_config=weight_config)
    out = DTensor.from_local(out_local, mesh, [Shard(0)])
    out._local_tensor.activation_config = None
    out._local_tensor.weight_config = None

    data = torch.randn(4, 64, 128, dtype=in_dtype)
    out_local_tensor_data_before = out._local_tensor._data
    result = wrapper.fsdp_post_all_gather(
        (data,), None, out_dtype, out=out
    )

    assert result is None
    assert out._local_tensor._data is out_local_tensor_data_before  # in-place copy_: same object
    assert out._local_tensor.weight_config is weight_config
    assert out._local_tensor.activation_config is act_config
    assert torch.equal(out._local_tensor._data, data.to(out_dtype))

    # If param_dtype doesn't match out_data.dtype, the assertion should fire
    bad_param_dtype = torch.float64
    expected_msg = (
        f"^`out`\\(dtype={out_dtype}\\) dose not match "
        f"the mixed precision policy param_dtype {bad_param_dtype}$"
    )
    with pytest.raises(AssertionError, match=expected_msg):
        wrapper.fsdp_post_all_gather(
            (data,), None, bad_param_dtype, out=out
        )


@pytest.mark.parametrize("wrapper_cls, weight_config, act_config", [
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), None),
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
def test_fsdp_post_all_gather_existing_out_wrong_type(wrapper_cls, weight_config, act_config):
    """out with wrong type raises RuntimeError."""
    w = torch.randn(4, 64, 128)
    wrapper = wrapper_cls(w, activation_config=act_config, weight_config=weight_config)

    out_type = re.escape(str(type(torch.randn(1))))
    expected_msg = (
        f"^expected out to be {re.escape(wrapper_cls.__name__)} or "
        f"DTensor with local_tensor={re.escape(wrapper_cls.__name__)}, "
        f"but got {out_type}$"
    )
    with pytest.raises(RuntimeError, match=expected_msg):
        wrapper.fsdp_post_all_gather(
            (w,), None, torch.float32, out=torch.randn(4, 64, 128)
        )