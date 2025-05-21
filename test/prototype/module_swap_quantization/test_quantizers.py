import unittest

import torch

from torchao.prototype.quantization.module_swap import IntQuantizer


class TestIntQuantizer(unittest.TestCase):
    def test_get_scale_param_size(self) -> None:
        x = torch.FloatTensor([0, 5, 10, 15])
        group_size = 4
        scale_param_size = IntQuantizer.get_scale_param_size(x, group_size)
        assert scale_param_size == torch.Size([1])

        x = torch.FloatTensor([0, 5, 10, 15])
        group_size = 2
        scale_param_size = IntQuantizer.get_scale_param_size(x, group_size)
        assert scale_param_size == torch.Size([2])

        x = torch.FloatTensor([0, 5, 10, 15, 20, 25, 30, 35]).reshape(2, 4)
        group_size = 4
        scale_param_size = IntQuantizer.get_scale_param_size(x, group_size)
        assert scale_param_size == torch.Size([2, 1])

        x = torch.FloatTensor([0, 5, 10, 15, 20, 25, 30, 35]).reshape(2, 4)
        group_size = 2
        scale_param_size = IntQuantizer.get_scale_param_size(x, group_size)
        assert scale_param_size == torch.Size([2, 2])

    def test_get_qmin_qmax(self) -> None:
        qmin, qmax = IntQuantizer.get_qmin_qmax(4, signed=False)
        assert qmin == 0
        assert qmax == 15

        qmin, qmax = IntQuantizer.get_qmin_qmax(4, signed=True)
        assert qmin == -8
        assert qmax == 7

    def test_get_scale_offset_asymmetric(self) -> None:
        x = torch.FloatTensor([0, 5, 10, 15])
        group_size = 4
        quantization_mode = "asymmetric"
        q_min = 0
        q_max = 15
        scale, offset = IntQuantizer.get_scale_offset(
            x, group_size, quantization_mode, q_min, q_max
        )
        assert scale == 1
        assert offset == 0

    def test_get_scale_offset_symmetric(self) -> None:
        x = torch.FloatTensor([-8, -5, -3, -1, 0, 1, 3, 5, 7])
        group_size = 9
        quantization_mode = "symmetric"
        q_min = -8
        q_max = 7
        scale, offset = IntQuantizer.get_scale_offset(
            x, group_size, quantization_mode, q_min, q_max
        )
        assert scale == 1
        assert offset is None

    def test_quantize_forward(self) -> None:
        x = torch.FloatTensor([0, 5, 10, 15])
        scale = torch.FloatTensor([1])
        offset = torch.FloatTensor([0])
        group_size = 4
        q_min = 0
        q_max = 15
        output = IntQuantizer.quantize_forward(
            x, scale, offset, q_min, q_max, group_size
        )

        torch.testing.assert_close(output, torch.FloatTensor([0, 5, 10, 15]))

    def test_quantize_forward_asymmetric_clipping(self) -> None:
        x = torch.FloatTensor([0, 5, 10, 100])
        scale = torch.FloatTensor([1])
        offset = torch.FloatTensor([0])
        group_size = 4
        q_min = 0
        q_max = 15
        output = IntQuantizer.quantize_forward(
            x, scale, offset, q_min, q_max, group_size
        )

        torch.testing.assert_close(output, torch.FloatTensor([0, 5, 10, 15]))

    def test_quantize_forward_symmetric(self) -> None:
        x = torch.FloatTensor([0, 1, 2, 3])
        scale = torch.FloatTensor([1.0])
        group_size = 4
        q_min = -8
        q_max = 7
        output = IntQuantizer.quantize_forward(
            x, scale, offset=None, group_size=group_size, q_min=q_min, q_max=q_max
        )

        torch.testing.assert_close(output, torch.FloatTensor([0, 1, 2, 3]))

    def test_quantize_forward_symmetric_clipping(self) -> None:
        x = torch.FloatTensor([0, 1, 2, 10])
        scale = torch.FloatTensor([1.0])
        group_size = 4
        q_min = -8
        q_max = 7
        output = IntQuantizer.quantize_forward(
            x, scale, offset=None, group_size=group_size, q_min=q_min, q_max=q_max
        )

        torch.testing.assert_close(output, torch.FloatTensor([0, 1, 2, 7]))

    def test_get_scale_offset_from_min_max(self) -> None:
        x_min = torch.FloatTensor([-8])
        x_max = torch.FloatTensor([7])
        q_min = 0
        q_max = 15
        scale, offset = IntQuantizer.get_scale_offset_from_min_max(
            x_min, x_max, q_min, q_max
        )
        assert scale == 1
        assert offset == 8

    def test_get_scale_offset_from_min_max_tensorized(self) -> None:
        x_min = torch.FloatTensor([-8, 0])
        x_max = torch.FloatTensor([7, 15])
        q_min = 0
        q_max = 15
        scale, offset = IntQuantizer.get_scale_offset_from_min_max(
            x_min, x_max, q_min, q_max
        )
        assert torch.allclose(scale, torch.FloatTensor([1, 1]))
        assert torch.allclose(offset, torch.FloatTensor([8, 0]))

    def test_get_scale_from_min_max(self) -> None:
        x_min = torch.FloatTensor([-8])
        x_max = torch.FloatTensor([7])
        q_min = -8
        q_max = 7
        scale = IntQuantizer.get_scale_from_min_max(x_min, x_max, q_min, q_max)

        assert scale == 1

    def test_get_scale_from_min_max_vectorized(self) -> None:
        x_min = torch.FloatTensor([-8, -16])
        x_max = torch.FloatTensor([7, 14])
        q_min = -8
        q_max = 7
        scale = IntQuantizer.get_scale_from_min_max(x_min, x_max, q_min, q_max)

        assert torch.allclose(scale, torch.FloatTensor([1, 2]))


class TestCodebookQuantizer(unittest.TestCase):
    def test_codebook_quantizer(self) -> None:
        pass

    def test_vector_quantizer(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
