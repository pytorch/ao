import torch
from torchao.testing.utils import copy_tests, TorchAOTensorParallelTestCase
from torch.testing._internal.common_utils import run_tests
from torchao.quantization import int8_weight_only, float8_weight_only, float8_dynamic_activation_float8_weight
from torchao.quantization.observer import PerRow, PerTensor

class TestInt8woAffineQuantizedTensorParallel(TorchAOTensorParallelTestCase):
    QUANT_METHOD_FN = staticmethod(int8_weight_only)
copy_tests(TorchAOTensorParallelTestCase, TestInt8woAffineQuantizedTensorParallel, "int8wo_tp")

# Run only on H100
if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0):
    class TestFloat8woAffineQuantizedTensorParallel(TorchAOTensorParallelTestCase):
        QUANT_METHOD_FN = staticmethod(float8_weight_only)
    copy_tests(TorchAOTensorParallelTestCase, TestFloat8woAffineQuantizedTensorParallel, "fp8wo_tp")

# Run only on H100
if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0):
    class TestFloat8dqRowAffineQuantizedTensorParallel(TorchAOTensorParallelTestCase):
        QUANT_METHOD_FN = staticmethod(float8_dynamic_activation_float8_weight)
        QUANT_METHOD_KWARGS = {"granularity": PerRow()}
    copy_tests(TorchAOTensorParallelTestCase, TestFloat8dqRowAffineQuantizedTensorParallel, "fp8dqr_tp")

# Run only on H100
if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0):
    class TestFloat8dqTensorAffineQuantizedTensorParallel(TorchAOTensorParallelTestCase):
        QUANT_METHOD_FN = staticmethod(float8_dynamic_activation_float8_weight)
        QUANT_METHOD_KWARGS = {"granularity": PerTensor()}
    copy_tests(TorchAOTensorParallelTestCase, TestFloat8dqTensorAffineQuantizedTensorParallel, "fp8dqt_tp")

if __name__ == "__main__":
    run_tests()
