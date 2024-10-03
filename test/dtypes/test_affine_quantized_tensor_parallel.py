from torchao.testing.utils import copy_tests, TorchAOTensorParallelTestCase
from torch.testing._internal.common_utils import run_tests
from torchao.quantization import int8_weight_only
import torch

class TestAffineQuantizedTensorParallel(TorchAOTensorParallelTestCase):
    pass

is_H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)

copy_tests(TorchAOTensorParallelTestCase, TestAffineQuantizedTensorParallel, "aqt_tp")

if __name__ == "__main__":
    if not is_H100:
        run_tests()
    else:
        print("Skipping TestAffineQuantizedTensorParallel: not supported on H100")
