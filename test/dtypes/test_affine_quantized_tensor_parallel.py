from torchao.testing.utils import copy_tests, TorchAOTensorParallelTestCase
from torch.testing._internal.common_utils import run_tests
from torchao.quantization import int8_weight_only

class TestAffineQuantizedTensorParallel(TorchAOTensorParallelTestCase):
    pass


copy_tests(TorchAOTensorParallelTestCase, TestAffineQuantizedTensorParallel, "aqt_tp")

if __name__ == "__main__":
    run_tests()
