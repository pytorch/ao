import unittest
import torch
from torchao.prototype.hqq.core import HQQQuantizer
from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    ZeroPointDomain,
    PlainAQTLayout,
    PlainLayoutType,
    TensorCoreTiledAQTLayout,
    TensorCoreTiledLayoutType,
    MappingType,
)

torch.random.manual_seed(100)

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


in_features, out_features = 4096, 11800
linear_layer = torch.nn.Linear(in_features, out_features, bias=False, device=device)
x = torch.randn((1, linear_layer.in_features), dtype=torch.float, device=device)/20.
y_ref = linear_layer(x)
W = linear_layer.weight.data.clone().to(device=device, dtype=compute_dtype)

def _eval_hqq(nbits, W, y_ref, layout_type):
    q_tensor_hqq = AffineQuantizedTensor.from_float(
            input_float=W,
            mapping_type=mapping_type,
            block_size=block_size,
            target_dtype=torch.int32 if isinstance(layout_type, TensorCoreTiledLayoutType) else torch.uint8,
            quant_min=0,
            quant_max=2**nbits - 1,
            zero_point_domain=zero_point_domain,
            preserve_zero=preserve_zero,
            layout_type=layout_type,
            use_hqq=True,
            )

    quant_linear_layer = torch.nn.Linear(W.shape[1], W.shape[0], bias=False, device=W.device)
    del quant_linear_layer.weight 
    quant_linear_layer.weight = q_tensor_hqq
    dequantize_error = (W - q_tensor_hqq.dequantize()).abs().mean().item()
    dot_product_error = (y_ref - quant_linear_layer(x.to(compute_dtype))).abs().mean().item()

    return dequantize_error, dot_product_error

class TestHQQ(unittest.TestCase):
    def test_hqq_plain_8bit(self):
        dequantize_error, dot_product_error = _eval_hqq(8, W, y_ref, PlainLayoutType())
        self.assertTrue(dequantize_error < 5e-5)
        self.assertTrue(dot_product_error < 0.00013)

    def test_hqq_plain_7bit(self):
        dequantize_error, dot_product_error = _eval_hqq(7, W, y_ref, PlainLayoutType())
        self.assertTrue(dequantize_error < 6e-05)
        self.assertTrue(dot_product_error < 0.000193)

    def test_hqq_plain_6bit(self):
        dequantize_error, dot_product_error = _eval_hqq(6, W, y_ref, PlainLayoutType())
        self.assertTrue(dequantize_error < 0.0001131)
        self.assertTrue(dot_product_error < 0.000353)

    def test_hqq_plain_5bit(self):
        dequantize_error, dot_product_error = _eval_hqq(5, W, y_ref, PlainLayoutType())
        self.assertTrue(dequantize_error < 0.00023)
        self.assertTrue(dot_product_error < 0.000704)

    def test_hqq_plain_4bit(self):
        dequantize_error, dot_product_error = _eval_hqq(4, W, y_ref, PlainLayoutType())
        self.assertTrue(dequantize_error < 0.000487)
        self.assertTrue(dot_product_error < 0.001472)

    def test_hqq_tensorcore_4bit(self):
        dequantize_error, dot_product_error = _eval_hqq(4, W, y_ref, TensorCoreTiledLayoutType(inner_k_tiles=inner_k_tiles))
        self.assertTrue(dequantize_error < 0.000487)
        self.assertTrue(dot_product_error < 0.00147)

    def test_hqq_plain_3bit(self):
        dequantize_error, dot_product_error = _eval_hqq(3, W, y_ref, PlainLayoutType())
        self.assertTrue(dequantize_error < 0.00101)
        self.assertTrue(dot_product_error < 0.003047)

    def test_hqq_plain_2bit(self):
        dequantize_error, dot_product_error = _eval_hqq(2, W, y_ref, PlainLayoutType())
        self.assertTrue(dequantize_error < 0.002366)
        self.assertTrue(dot_product_error < 0.007255)

if __name__ == "__main__":
    unittest.main()