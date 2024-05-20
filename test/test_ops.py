import torch
from torch.testing._internal.common_utils import TestCase, IS_FBCODE
from torch.testing._internal.optests import opcheck
import torchao
from torchao.quantization.utils import TORCH_VERSION_AFTER_2_4
import unittest
from parameterized import parameterized


# torch.testing._internal.optests.generate_tests.OpCheckError: opcheck(op, ...):
# test_faketensor failed with module 'torch' has no attribute '_custom_ops' (scroll up for stack trace)
@unittest.skipIf(IS_FBCODE, "Skipping the test in fbcode since we don't have TARGET file for kernels")
class TestOps(TestCase):
    def _create_tensors_with_iou(self, N, iou_thresh):
        # force last box to have a pre-defined iou with the first box
        # let b0 be [x0, y0, x1, y1], and b1 be [x0, y0, x1 + d, y1],
        # then, in order to satisfy ops.iou(b0, b1) == iou_thresh,
        # we need to have d = (x1 - x0) * (1 - iou_thresh) / iou_thresh
        # Adjust the threshold upward a bit with the intent of creating
        # at least one box that exceeds (barely) the threshold and so
        # should be suppressed.
        boxes = torch.rand(N, 4) * 100
        boxes[:, 2:] += boxes[:, :2]
        boxes[-1, :] = boxes[0, :]
        x0, y0, x1, y1 = boxes[-1].tolist()
        iou_thresh += 1e-5
        boxes[-1, 2] += (x1 - x0) * (1 - iou_thresh) / iou_thresh
        scores = torch.rand(N)
        return boxes, scores

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(not TORCH_VERSION_AFTER_2_4, "skipping when torch verion is 2.3 or lower")
    def test_nms(self):
        iou = 0.2
        boxes, scores = self._create_tensors_with_iou(1000, iou)
        boxes = boxes.cuda()
        scores = scores.cuda()

        # smoke test
        _ = torchao.ops.nms(boxes, scores, iou)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.nms, (boxes, scores, iou), test_utils=test_utils)

    def _create_fp6_inputs(self, BS: int, OC: int, IC: int):
        # Randomly initialize each bytes. The highest value for randint() is set the the max value of uint32_t.
        fp6_weight = torch.randint(4294967295, (OC, IC // 16 * 3)).to(torch.int)
        fp16_scale = torch.rand(OC).half() + 0.5
        fp16_activation = torch.rand(BS, IC).half() + 0.5
        return fp6_weight, fp16_scale, fp16_activation

    def test_prepack_fp6_weight(self):
        OC = 256
        IC = 256
        fp6_weight, _, _ = self._create_fp6_inputs(0, OC, IC)

        # smoke test
        torchao.ops.prepack_fp6_weight(fp6_weight)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.prepack_fp6_weight, (fp6_weight,), test_utils=test_utils)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fp16_to_fp6(self):
        OC = 256
        IC = 256
        fp16_weight = torch.randn((OC, IC), dtype=torch.float16)

        # smoke test
        torchao.ops.fp16_to_fp6_original(fp16_weight)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.fp16_to_fp6_original, (fp16_weight,), test_utils=test_utils)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fp16act_fp6weight_linear(self):
        BS = 2
        OC = 256
        IC = 256
        splitK = 1
        fp6_weight, fp16_scale, fp16_activation = self._create_fp6_inputs(BS, OC, IC)

        fp6_weight_packed = torchao.ops.prepack_fp6_weight(fp6_weight)
        act_cuda = fp16_activation.cuda()
        weight_cuda = fp6_weight_packed.cuda()
        scale_cuda = fp16_scale.cuda()

        # smoke test
        torchao.ops.fp16act_fp6weight_linear(act_cuda, weight_cuda, scale_cuda, splitK)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.fp16act_fp6weight_linear, (act_cuda, weight_cuda, scale_cuda, splitK), test_utils=test_utils)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fp6_weight_dequant(self):
        OC = 256
        IC = 256
        fp6_weight, fp16_scale, _ = self._create_fp6_inputs(0, OC, IC)

        # smoke test
        torchao.ops.fp6_weight_dequant(fp6_weight, fp16_scale)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.fp6_weight_dequant, (fp6_weight, fp16_scale), test_utils=test_utils)

    # adapted from https://github.com/usyd-fsalab/fp6_llm/blob/main/tests/python/kernel_test.py
    @parameterized.expand([(1, 2048, 4096, 5), (2, 8192, 8192, 6)])
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fp6_matmul_correctness(self, BS, OC, IC, splitK):
        fp6_weight, fp16_scale, fp16_activation = self._create_fp6_inputs(BS, OC, IC)

        fp6_weight_packed = torchao.ops.prepack_fp6_weight(fp6_weight)
        act_cuda = fp16_activation.cuda()
        weight_cuda = fp6_weight_packed.cuda()
        scale_cuda = fp16_scale.cuda()

        results_fp6 = torchao.ops.fp16act_fp6weight_linear(act_cuda, weight_cuda, scale_cuda, splitK)

        fp16_weight = torchao.ops.fp6_weight_dequant(fp6_weight, fp16_scale).cuda()
        results_fp16 = act_cuda @ fp16_weight.T

        error = (results_fp6 - results_fp16).abs()
        relative_error = error / results_fp16.abs()
        assert relative_error.mean() < 1e-2


class TestFp6(TestCase):
    def _skip_cpu(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available. We don't compile for CPU-only build")

    @parameterized.expand([(device, dtype) for device in ["cpu", "cuda"] for dtype in [torch.float32, torch.float16, torch.bfloat16]])
    def test_to_fp6_unpacked(self, device, dtype):
        self._skip_cpu()
        inputs = torch.randn(128, 128, device=device, dtype=dtype)

        # smoke test
        torchao.ops.to_fp6_unpacked(inputs)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.to_fp6_unpacked, (inputs,), test_utils=test_utils)

    @parameterized.expand([(device, dtype) for device in ["cpu", "cuda"] for dtype in [torch.float32, torch.float16, torch.bfloat16]])
    def test_to_fp6_packed(self, device, dtype):
        self._skip_cpu()
        inputs = torch.randn(128, 128, device=device, dtype=dtype)

        # smoke test
        torchao.ops.to_fp6_packed(inputs)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.to_fp6_packed, (inputs,), test_utils=test_utils)

    @parameterized.expand([(device, dtype) for device in ["cpu", "cuda"] for dtype in [torch.float32, torch.float16, torch.bfloat16]])
    def test_from_fp6_unpacked(self, device, dtype):
        self._skip_cpu()
        inputs = torch.randint(256, size=(128, 128 // 4 * 3), device=device, dtype=torch.uint8)

        # smoke test
        torchao.ops.from_fp6_unpacked(inputs, dtype)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.from_fp6_unpacked, (inputs, dtype), test_utils=test_utils)

    @parameterized.expand([(device, dtype) for device in ["cpu", "cuda"] for dtype in [torch.float32, torch.float16, torch.bfloat16]])
    def test_from_fp6_packed(self, device, dtype):
        self._skip_cpu()
        inputs = torch.randint(256, size=(128, 128 // 4 * 3), device=device, dtype=torch.uint8)

        # smoke test
        torchao.ops.from_fp6_packed(inputs, dtype)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.from_fp6_packed, (inputs, dtype), test_utils=test_utils)

    def test_to_fp6_unpacked_shape(self):
        for shape in [(), (0,), (10,), (20, 20)]:
            x = torch.randn(shape)
            result = torchao.ops.to_fp6_unpacked(x)
            assert result.shape == shape

    def test_to_fp6_packed_shape(self):
        for shape in [(4,), (20, 20)]:
            x = torch.randn(shape)
            result = torchao.ops.to_fp6_packed(x)
            assert result.shape == shape[:-1] + (shape[-1] // 4 * 3,)

    @parameterized.expand(
        [
            (0.0,    0b000000),  # exact values
            (1.0,    0b001100),  # normal numbers
            (1.25,   0b001101),
            (28.0,   0b011111),  # max
            (0.1875, 0b000011),  # subnormal number
            (0.0625, 0b000001),  # min
            (29.0,   0b011111),  # normal round down
            (26.0,   0b011110),  # normal round to nearest even
            (0.1251, 0b000010),  # subnormal round down
            (0.0314, 0b000001),  # subnormal round up
            (0.03,   0b000000),  # underflow
        ]
    )
    def test_to_fp6_unpacked_correctness(self, input, output):
        self._skip_cpu()
        for device in ("cpu", "cuda"):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                x = torch.tensor(input, device=device, dtype=dtype)
                assert torchao.ops.to_fp6_unpacked(x).item() == output
                assert torchao.ops.to_fp6_unpacked(-x).item() == (output | 0b100000)

    @parameterized.expand([(device, dtype) for device in ["cpu", "cuda"] for dtype in [torch.float32, torch.float16, torch.bfloat16]])
    def test_to_fp6_packed_correctness(self, device, dtype):
        x = torch.randn(128, 128, device=device, dtype=dtype)
        results_unpacked = torchao.ops.to_fp6_unpacked(x)
        results_packed = torchao.ops.to_fp6_packed(x)

        val0, val1, val2, val3 = results_unpacked.unflatten(-1, (-1, 4)).unbind(-1)
        bits0 = (val0 << 2) | (val1 >> 4)  # 0000 0011
        bits1 = (val1 << 4) | (val2 >> 2)  # 1111 2222
        bits2 = (val2 << 6) | (val3);      # 2233 3333

        expected_packed = torch.stack([bits0, bits1, bits2], dim=-1).flatten(-2)
        assert (results_packed == expected_packed).all()

    @parameterized.expand([30.0, -100.0, float("inf"), float("nan")])
    def test_to_fp6_exception(self, input):
        self._skip_cpu()
        x = torch.tensor(input)
        with self.assertRaises(Exception):
            torchao.ops.to_fp6_unpacked(x)
        with self.assertRaises(Exception):
            torchao.ops.to_fp6_packed(x)

    @parameterized.expand(
        [
            (0b000000, 0.0),
            (0b001100, 1.0),
            (0b011111, 28.0),
            (0b000001, 0.0625),
            (0b001110, 1.5),
            (0b000011, 0.1875),
        ]
    )
    def test_from_fp6_unpacked_correctness(self, input, output):
        self._skip_cpu()
        for device in ("cpu", "cuda"):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                x = torch.tensor(input, device=device, dtype=torch.uint8)
                result = torchao.ops.from_fp6_unpacked(x, dtype)
                assert result.dtype == dtype
                assert result.item() == output

                x = torch.tensor(input | 0b100000, device=device, dtype=torch.uint8)
                result = torchao.ops.from_fp6_unpacked(x, dtype)
                assert result.dtype == dtype
                assert result.item() == -output

    @parameterized.expand([(device, dtype) for device in ["cpu", "cuda"] for dtype in [torch.float32, torch.float16, torch.bfloat16]])
    def test_from_fp6_packed_correctness(self, device, dtype):
        x = torch.randint(256, (128, 128 // 4 * 3), device=device, dtype=torch.uint8)
        results = torchao.ops.from_fp6_packed(x, dtype=dtype)

        bits0, bits1, bits2 = x.unflatten(-1, (-1, 3)).unbind(-1)
        x_unpacked0 = bits0 >> 2
        x_unpacked1 = ((bits0 & 0x3) << 4) | (bits1 >> 4)
        x_unpacked2 = ((bits1 & 0xF) << 2) | (bits2 >> 6)
        x_unpacked3 = bits2 & 0x3F

        x_unpacked = torch.stack([x_unpacked0, x_unpacked1, x_unpacked2, x_unpacked3], dim=-1).flatten(-2)
        expected = torchao.ops.from_fp6_unpacked(x_unpacked, dtype)
        assert (results == expected).all()


if __name__ == "__main__":
    unittest.main()
