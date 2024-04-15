import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import torchao
from torchao.quantization.utils import TORCH_VERSION_AFTER_2_4
import unittest

def is_nightly_with_dev_cud(version):
    return "dev" in version and "cu" in version

# torch.testing._internal.optests.generate_tests.OpCheckError: opcheck(op, ...):
# test_faketensor failed with module 'torch' has no attribute '_custom_ops' (scroll up for stack trace)
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

    @unittest.skipIf(is_nightly_with_dev(torch.__version__), " NotImplementedError: Could not run 'torchao::nms' with arguments from the 'CUDA' backend")
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


if __name__ == "__main__":
    unittest.main()
