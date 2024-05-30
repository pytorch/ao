import torch
from torch.testing._internal.common_utils import TestCase, IS_FBCODE
from torch.testing._internal.optests import opcheck
import torchao
from torchao.quantization.utils import TORCH_VERSION_AFTER_2_4
import unittest
from parameterized import parameterized
import pytest

try:
    import torchao.ops
except RuntimeError:
    pytest.skip("torchao.ops not available")


# torch.testing._internal.optests.generate_tests.OpCheckError: opcheck(op, ...):
# test_faketensor failed with module 'torch' has no attribute '_custom_ops' (scroll up for stack trace)
@pytest.mark.filterwarnings("ignore:create_unbacked_symint is deprecated, please use new_dynamic_size instead:UserWarning")
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
    def test_fp16_to_fp6_original(self):
        OC = 256
        IC = 256
        fp16_weight = torch.randn((OC, IC), dtype=torch.float16)

        # the original FP16->FP6 kernel checks for overflow/underflow
        fp16_weight.clip_(-28.0, 28.0)
        fp16_weight[fp16_weight.abs() < 0.0625] = 0.0

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

        fp16_weight = torchao.dtypes.from_float6_e3m2(fp6_weight.view(torch.uint8), dtype=torch.float16) * fp16_scale[:, None]
        results_fp16 = act_cuda @ fp16_weight.cuda().T

        error = (results_fp6 - results_fp16).abs()
        relative_error = error / results_fp16.abs()
        assert relative_error.mean() < 1e-2


if __name__ == "__main__":
    unittest.main()
