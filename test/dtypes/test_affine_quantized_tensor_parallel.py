from torchao.testing.utils import copy_tests, TorchAOTensorParallelTestCase
from torch.testing._internal.common_utils import run_tests
from torchao.quantization import int8_weight_only

class TestAffineQuantizedTensorParallel(TorchAOTensorParallelTestCase):
    pass

is_H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)

if not is_H100:
    copy_tests(TorchAOTensorParallelTestCase, TestAffineQuantizedTensorParallel, "aqt_tp")
else:
    print("Skipping TestAffineQuantizedTensorParallel because it doesn't run on H100")

if __name__ == "__main__":
    run_tests()
