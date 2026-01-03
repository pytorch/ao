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
        input_tensor = torch.randn(
            tokens, dim, device=self.device, dtype=torch.bfloat16
        )

        mx_input = MXTensor.to_mx(
            input_tensor, elem_dtype=torch.float8_e4m3fn, block_size=32
        )

        permuted_indices = torch.randperm(tokens, device=self.device)
        padded_shape = torch.Size([tokens + 1, dim])

        output = permute(mx_input, permuted_indices, padded_shape)

        assert isinstance(output, MXTensor)
        assert output.qdata.shape[0] == tokens
        assert output.scale.shape[0] == tokens

    def test_backward_pass(self):
        tokens = 64
        dim = 128
        input_tensor = torch.randn(
            tokens, dim, device=self.device, dtype=torch.bfloat16
        )

        mx_input = MXTensor.to_mx(
            input_tensor, elem_dtype=torch.float8_e4m3fn, block_size=32
        )

        permuted_indices = torch.randperm(tokens, device=self.device)
        padded_shape = torch.Size([tokens + 1, dim])

        output = permute(mx_input, permuted_indices, padded_shape)
        assert isinstance(output, MXTensor)


if __name__ == "__main__":
    run_tests()
