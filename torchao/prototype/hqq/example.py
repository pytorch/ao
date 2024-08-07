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

#Parameters
device, compute_dtype = "cuda:0", torch.bfloat16
group_size, axis = 64, 1

linear_layer = torch.nn.Linear(4096, 11800, bias=False, device=device)
x = torch.randn((1, linear_layer.in_features), dtype=torch.float, device=device)/20.
y_ref = linear_layer(x)
W = linear_layer.weight.data.clone().to(device=device, dtype=compute_dtype)
del linear_layer.weight 
################################################################################################

for nbits in list(range(2, 9))[::-1]:
    print('------------------------------------------------------------------------------')
    q_tensor_default = AffineQuantizedTensor.from_float(
            input_float=W,
            mapping_type=MappingType.ASYMMETRIC,
            block_size=[1, group_size],
            target_dtype=torch.uint8,
            quant_min=0,
            quant_max=2**nbits - 1,
            preserve_zero=False,#Important
            zero_point_domain= ZeroPointDomain.FLOAT,
            layout_type=PlainLayoutType(),
            )

    linear_layer.weight = q_tensor_default
    print("nbits", nbits, "| Default dequantization error", (W - q_tensor_default.dequantize()).abs().mean().item())
    print("nbits", nbits, '| Default Dot product error', (y_ref - linear_layer(x.to(compute_dtype))).abs().mean().item())
    # 4-bit Default dequantization error 0.001953125
    # 4-bit Default Dot product error 0.0057801781222224236


    q_tensor_hqq = AffineQuantizedTensor.from_float(
            input_float=W,
            mapping_type=MappingType.ASYMMETRIC,
            block_size=[1, group_size],
            target_dtype=torch.uint8,
            quant_min=0,
            quant_max=2**nbits - 1,
            preserve_zero=False,#Important
            zero_point_domain= ZeroPointDomain.FLOAT,
            layout_type=PlainLayoutType(),
            use_hqq=True,
            )

    linear_layer.weight = q_tensor_hqq
    print("nbits", nbits, "| HQQ dequantization error", (W - q_tensor_hqq.dequantize()).abs().mean().item())
    print("nbits", nbits, '| HQQ Dot product error', (y_ref - linear_layer(x.to(compute_dtype))).abs().mean().item())
    # 4-bit HQQ dequantization error 0.0004863739013671875
    # 4-bit HQQ Dot product error 0.0014263123739510775
