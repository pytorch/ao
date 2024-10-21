import unittest
import torch
from torchao.dtypes.affine_quantized_tensor import (
    to_affine_quantized_intx,
    ZeroPointDomain,
    PlainAQTTensorImpl,
    PlainLayout,
    TensorCoreTiledAQTTensorImpl,
    TensorCoreTiledLayout,
    MappingType,
)

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_3,
)
from torchao.quantization import (
    uintx_weight_only,
    int4_weight_only,
)

cuda_available = torch.cuda.is_available()

#Parameters
device            = 'cuda:0'
compute_dtype     = torch.bfloat16
group_size        = 64
mapping_type      = MappingType.ASYMMETRIC
block_size        = (1, group_size) #axis=1
preserve_zero     = False
zero_point_domain = ZeroPointDomain.FLOAT
zero_point_dtype  = compute_dtype
inner_k_tiles     = 8
in_features       = 4096
out_features      = 11800
torch_seed        = 100


def _init_data(in_features, out_features, compute_dtype, device, torch_seed):
    torch.random.manual_seed(torch_seed)
    linear_layer = torch.nn.Linear(in_features, out_features, bias=False).to(device)
    x = torch.randn((1, linear_layer.in_features), dtype=torch.float, device=device)/20.
    y_ref = linear_layer(x)
    W = linear_layer.weight.data.clone().to(device=device, dtype=compute_dtype)
    return W, x, y_ref

def _eval_hqq(dtype):
    W, x, y_ref = _init_data(in_features, out_features, compute_dtype, device, torch_seed)

    dummy_linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False)
    dummy_linear.weight.data = W
    if dtype == torch.uint4:
        q_tensor_hqq = int4_weight_only(group_size=max(block_size), use_hqq=True)(dummy_linear).weight
    else:
        q_tensor_hqq = uintx_weight_only(dtype, group_size=max(block_size), use_hqq=True)(dummy_linear).weight

    quant_linear_layer = torch.nn.Linear(W.shape[1], W.shape[0], bias=False, device=W.device)
    del quant_linear_layer.weight
    quant_linear_layer.weight = q_tensor_hqq
    dequantize_error = (W - q_tensor_hqq.dequantize()).abs().mean().item()
    dot_product_error = (y_ref - quant_linear_layer(x.to(compute_dtype))).abs().mean().item()

    return dequantize_error, dot_product_error


@unittest.skipIf(not cuda_available, "Need CUDA available")
@unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_3, "Need torch 2.3+")
class TestHQQ(unittest.TestCase):
    def _test_hqq(self, dtype=None, ref_dequantize_error=None, ref_dot_product_error=None):
        if(dtype is None): return
        dequantize_error, dot_product_error = _eval_hqq(dtype)
        self.assertTrue(dequantize_error < ref_dequantize_error)
        self.assertTrue(dot_product_error < ref_dot_product_error)

    def test_hqq_plain_8bit(self):
        self._test_hqq(dtype=torch.uint8, ref_dequantize_error=5e-5, ref_dot_product_error=0.00013)

    def test_hqq_plain_7bit(self):
        self._test_hqq(dtype=torch.uint7, ref_dequantize_error=6e-05, ref_dot_product_error=0.000193)

    def test_hqq_plain_6bit(self):
        self._test_hqq(dtype=torch.uint6, ref_dequantize_error=0.0001131, ref_dot_product_error=0.000353)

    def test_hqq_plain_5bit(self):
        self._test_hqq(dtype=torch.uint5, ref_dequantize_error=0.00023, ref_dot_product_error=0.000704)

    def test_hqq_plain_4bit(self):
        self._test_hqq(dtype=torch.uint4, ref_dequantize_error=0.000487, ref_dot_product_error=0.001472)

    def test_hqq_plain_3bit(self):
        self._test_hqq(dtype=torch.uint3, ref_dequantize_error=0.00101, ref_dot_product_error=0.003047)

    def test_hqq_plain_2bit(self):
        self._test_hqq(dtype=torch.uint2, ref_dequantize_error=0.002366, ref_dot_product_error=0.007255)

if __name__ == "__main__":
    unittest.main()
