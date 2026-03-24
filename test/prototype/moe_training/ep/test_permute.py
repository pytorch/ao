import pytest
import torch

from torchao.utils import is_cuda_version_at_least, is_sm_at_least_100

if torch.cuda.is_available() and not (
    is_sm_at_least_100() and is_cuda_version_at_least(12, 8)
):
    pytest.skip("Test requires CUDA 12.8+ with SM >= 100", allow_module_level=True)

from torchao.prototype.moe_training.ep import permute_mxfp8_fwd_hp_bwd
from torchao.prototype.moe_training.ep.permute import permute_and_pad
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.quantization.utils import compute_error
from torchao.utils import get_available_devices

_DEVICES = get_available_devices()[1:]  # Exclude CPU since this test is for GPU kernels


@pytest.fixture(scope="module", params=_DEVICES)
def device(request):
    return request.param


def test_mxfp8_permute_forward(device: str):
    tokens = 64
    dim = 128
    num_experts = 8
    ep_degree = 1
    block_size = 32

    input_tensor = torch.randn(tokens, dim, device=device, dtype=torch.bfloat16)

    mx_input = MXTensor.to_mx(
        input_tensor, elem_dtype=torch.float8_e4m3fn, block_size=block_size
    )

    # Create num_tokens_per_expert tensor
    tokens_per_expert = tokens // num_experts
    num_tokens_per_expert = torch.full(
        (num_experts,), tokens_per_expert, dtype=torch.int32, device=device
    )

    (
        padded_shape,
        mx_output,
        permuted_indices,
        num_tokens_per_expert_padded,
        offsets,
    ) = permute_mxfp8_fwd_hp_bwd(
        mx_input,
        num_tokens_per_expert,
        ep_degree,
        num_experts,
        block_size,
    )

    # BF16 reference
    (
        _,
        ref_output,
        _,
        _,
        _,
    ) = permute_and_pad(
        input_tensor,
        num_tokens_per_expert,
        ep_degree,
        num_experts,
        block_size,
    )

    # Compare outputs
    output = mx_output.dequantize()
    sqnr = compute_error(output, ref_output)
    assert sqnr >= 30.0, f"SQNR too low: {sqnr} dB"

    # Note: backward is tested in an e2e integration test with other mxfp8 EP pipeline components
