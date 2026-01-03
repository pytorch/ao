import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torchao.prototype.moe_training.ep import unpermute


class TestUnpermute(TestCase):
    def setUp(self):
        super().setUp()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    def test_forward_output_is_bf16(self):
        tokens = 64
        dim = 128
        input_tensor = torch.randn(
            tokens, dim, device=self.device, dtype=torch.bfloat16
        )

        permuted_indices = torch.randperm(tokens, device=self.device)
        padded_shape = torch.Size([tokens + 1, dim])

        output = unpermute(input_tensor, permuted_indices, padded_shape)

        assert output.dtype == torch.bfloat16
        assert output.shape[0] == tokens

    def test_backward_receives_mxtensor(self):
        tokens = 64
        dim = 128
        input_tensor = torch.randn(
            tokens, dim, device=self.device, dtype=torch.bfloat16
        )

        permuted_indices = torch.randperm(tokens, device=self.device)
        padded_shape = torch.Size([tokens + 1, dim])

        output = unpermute(input_tensor, permuted_indices, padded_shape)

        assert output.dtype == torch.bfloat16
        assert output.shape[0] == tokens


if __name__ == "__main__":
    run_tests()
