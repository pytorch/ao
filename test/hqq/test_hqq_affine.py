import unittest
import torch
from torchao.dtypes.affine_quantized_tensor import (
    to_affine_quantized_intx,
    ZeroPointDomain,
    PlainAQTLayout,
    PlainLayoutType,
    TensorCoreTiledAQTLayout,
    TensorCoreTiledLayoutType,
    MappingType,
)

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_4,
    TORCH_VERSION_AT_LEAST_2_5,
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
    linear_layer = torch.nn.Linear(in_features, out_features, bias=False, device=device)
    x = torch.randn((1, linear_layer.in_features), dtype=torch.float, device=device)/20.
    y_ref = linear_layer(x)
    W = linear_layer.weight.data.clone().to(device=device, dtype=compute_dtype)
    return W, x, y_ref

def _eval_hqq(nbits, layout_type):
    W, x, y_ref  = _init_data(in_features, out_features, compute_dtype, device, torch_seed)
    
    #Plain layout
    target_dtype = torch.uint8
    #Tensorcore layout
    if isinstance(layout_type, TensorCoreTiledLayoutType):
    	target_dtype = torch.uint8 if TORCH_VERSION_AT_LEAST_2_5 else torch.int32
    	    	
    q_tensor_hqq = to_affine_quantized_intx(
            input_float=W,
            mapping_type=mapping_type,
            block_size=block_size,
            target_dtype=target_dtype,
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


class TestHQQBase(unittest.TestCase):
    @unittest.skipIf(not cuda_available, "Need CUDA available")
    def test_hqq(self, nbits=None, layout_type=None, ref_dequantize_error=None, ref_dot_product_error=None):
        if(nbits is None): return
        dequantize_error, dot_product_error = _eval_hqq(nbits=nbits, layout_type=layout_type)
        self.assertTrue(dequantize_error < ref_dequantize_error)
        self.assertTrue(dot_product_error < ref_dot_product_error)

class TestHQQ8Bit(TestHQQBase):
    def test_hqq_plain_8bit(self):
        self.test_hqq(nbits=8, layout_type=PlainLayoutType(), ref_dequantize_error=5e-5, ref_dot_product_error=0.00013)

class TestHQQ7Bit(TestHQQBase):
    def test_hqq_plain_7bit(self):
        self.test_hqq(nbits=7, layout_type=PlainLayoutType(), ref_dequantize_error=6e-05, ref_dot_product_error=0.000193)

class TestHQQ6Bit(TestHQQBase):
    def test_hqq_plain_6bit(self):
        self.test_hqq(nbits=6, layout_type=PlainLayoutType(), ref_dequantize_error=0.0001131, ref_dot_product_error=0.000353)

class TestHQQ5Bit(TestHQQBase):
    def test_hqq_plain_5bit(self):
        self.test_hqq(nbits=5, layout_type=PlainLayoutType(), ref_dequantize_error=0.00023, ref_dot_product_error=0.000704)

class TestHQQ4bit(TestHQQBase):
    def test_hqq_plain_4bit(self):
        self.test_hqq(nbits=4, layout_type=PlainLayoutType(), ref_dequantize_error=0.000487, ref_dot_product_error=0.001472)
    
    def test_hqq_tensorcore_4bit(self):
        self.test_hqq(nbits=4, layout_type=TensorCoreTiledLayoutType(inner_k_tiles=inner_k_tiles), ref_dequantize_error=0.000487, ref_dot_product_error=0.00147)

class TestHQQ3Bit(TestHQQBase):
    def test_hqq_plain_3bit(self):
        self.test_hqq(nbits=3, layout_type=PlainLayoutType(), ref_dequantize_error=0.00101, ref_dot_product_error=0.003047)

class TestHQQ2Bit(TestHQQBase):
    def test_hqq_plain_2bit(self):
        self.test_hqq(nbits=2, layout_type=PlainLayoutType(), ref_dequantize_error=0.002366, ref_dot_product_error=0.007255)

if __name__ == "__main__":
    unittest.main()
