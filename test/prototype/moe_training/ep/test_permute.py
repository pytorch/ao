import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torchao.prototype.moe_training.ep import permute
from torchao.prototype.mx_formats.mx_tensor import MXTensor


class TestPermute(TestCase):
    def setUp(self):
        super().setUp()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    def test_forward_output_is_mxtensor(self):
        tokens = 64
        dim = 128
        num_experts = 8
        ep_degree = 1
        block_size = 32

        input_tensor = torch.randn(
            tokens, dim, device=self.device, dtype=torch.bfloat16
        )

        mx_input = MXTensor.to_mx(
            input_tensor, elem_dtype=torch.float8_e4m3fn, block_size=block_size
        )

        # Create num_tokens_per_expert tensor
        tokens_per_expert = tokens // num_experts
        num_tokens_per_expert = torch.full(
            (num_experts,), tokens_per_expert, dtype=torch.int32, device=self.device
        )

        (
            padded_shape,
            output,
            permuted_indices,
            num_tokens_per_expert_padded,
            offsets,
        ) = permute(
            mx_input,
            num_tokens_per_expert,
            ep_degree,
            num_experts,
            block_size,
            use_mxfp8=True,
        )

        assert isinstance(output, MXTensor)
        assert output.qdata.shape[0] >= tokens  # May have padding
        assert output.scale.shape[0] >= tokens

    def test_backward_pass(self):
        tokens = 64
        dim = 128
        num_experts = 8
        ep_degree = 1
        block_size = 32

        input_tensor = torch.randn(
            tokens, dim, device=self.device, dtype=torch.bfloat16
        )

        mx_input = MXTensor.to_mx(
            input_tensor, elem_dtype=torch.float8_e4m3fn, block_size=block_size
        )

        # Create num_tokens_per_expert tensor
        tokens_per_expert = tokens // num_experts
        num_tokens_per_expert = torch.full(
            (num_experts,), tokens_per_expert, dtype=torch.int32, device=self.device
        )

        (
            padded_shape,
            output,
            permuted_indices,
            num_tokens_per_expert_padded,
            offsets,
        ) = permute(
            mx_input,
            num_tokens_per_expert,
            ep_degree,
            num_experts,
            block_size,
            use_mxfp8=True,
        )
        assert isinstance(output, MXTensor)


if __name__ == "__main__":
    run_tests()
