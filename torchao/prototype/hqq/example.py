import torch

from torchao.dtypes import PlainLayout, TensorCoreTiledLayout
from torchao.dtypes.affine_quantized_tensor import (
    to_affine_quantized_intx,
)
from torchao.quantization import (
    MappingType,
    ZeroPointDomain,
)

# Parameters
device, compute_dtype = "cuda:0", torch.bfloat16
group_size, axis = 64, 1
in_features, out_features = 4096, 11800

torch.random.manual_seed(100)
linear_layer = torch.nn.Linear(in_features, out_features, bias=False, device=device)
x = torch.randn((1, linear_layer.in_features), dtype=torch.float, device=device) / 20.0
y_ref = linear_layer(x)
W = linear_layer.weight.data.clone().to(device=device, dtype=compute_dtype)
del linear_layer.weight

################################################################################################
# AffineQuantizedTensor example
################################################################################################
print("-------------------------------------------------------------------")
print("AffineQuantizedTensor example")
print("-------------------------------------------------------------------")
mapping_type = MappingType.ASYMMETRIC
block_size = (1, group_size)
target_dtype = torch.uint8  # until sub-byte dtypes are supported
preserve_zero = False
zero_point_domain = ZeroPointDomain.FLOAT
zero_point_dtype = compute_dtype
_layout = PlainLayout()

for nbits in list(range(2, 9))[::-1]:
    print(
        "------------------------------------------------------------------------------"
    )
    q_tensor_default = to_affine_quantized_intx(
        input_float=W,
        mapping_type=mapping_type,
        block_size=block_size,
        target_dtype=target_dtype,
        quant_min=0,
        quant_max=2**nbits - 1,
        zero_point_domain=zero_point_domain,
        preserve_zero=preserve_zero,
        _layout=_layout,
    )

    linear_layer.weight = q_tensor_default
    print(
        "nbits",
        nbits,
        "| Default dequantization error",
        (W - q_tensor_default.dequantize()).abs().mean().item(),
    )
    print(
        "nbits",
        nbits,
        "| Default Dot product error",
        (y_ref - linear_layer(x.to(compute_dtype))).abs().mean().item(),
    )
    # nbits 4 | Default dequantization error 0.001953125
    # nbits 4 | Default Dot product error 0.005926903802901506

    q_tensor_hqq = to_affine_quantized_intx(
        input_float=W,
        mapping_type=mapping_type,
        block_size=block_size,
        target_dtype=target_dtype,
        quant_min=0,
        quant_max=2**nbits - 1,
        zero_point_domain=zero_point_domain,
        preserve_zero=preserve_zero,
        _layout=_layout,
        use_hqq=True,
    )

    linear_layer.weight = q_tensor_hqq
    print(
        "nbits",
        nbits,
        "| HQQ dequantization error",
        (W - q_tensor_hqq.dequantize()).abs().mean().item(),
    )
    print(
        "nbits",
        nbits,
        "| HQQ Dot product error",
        (y_ref - linear_layer(x.to(compute_dtype))).abs().mean().item(),
    )
    # nbits 4 | HQQ dequantization error 0.0004863739013671875
    # nbits 4 | HQQ Dot product error 0.0014713306445628405

################################################################################################
# quant_api example
################################################################################################
print("-------------------------------------------------------------------")
print("Quant API example")
print("-------------------------------------------------------------------")

from torchao.quantization.quant_api import int4_weight_only

nbits = 4
target_dtype = torch.int32
inner_k_tiles = 8
_layout = TensorCoreTiledLayout(inner_k_tiles=inner_k_tiles)

int4_weight_only_patch_fct = int4_weight_only(
    group_size=group_size, inner_k_tiles=inner_k_tiles
)
linear_layer_default = torch.nn.Linear(
    in_features, out_features, bias=False, device=device
)
linear_layer_default.weight.data = W.clone()
linear_layer_default = int4_weight_only_patch_fct(linear_layer_default)
print(
    "nbits",
    nbits,
    "| Default dequantization error",
    (W - linear_layer_default(torch.eye(W.shape[1], dtype=W.dtype, device=W.device)).T)
    .abs()
    .mean()
    .item(),
)
print(
    "nbits",
    nbits,
    "| Default Dot product error",
    (y_ref - linear_layer_default(x.to(compute_dtype))).abs().mean().item(),
)
# nbits 4 | Default dequantization error 0.000492095947265625
# nbits 4 | Default Dot product error 0.0015244047390297055


q_tensor_hqq = to_affine_quantized_intx(
    input_float=W,
    mapping_type=mapping_type,
    block_size=block_size,
    target_dtype=target_dtype,
    quant_min=0,
    quant_max=2**nbits - 1,
    zero_point_domain=zero_point_domain,
    preserve_zero=preserve_zero,
    _layout=_layout,
    use_hqq=True,
)
linear_layer.weight = q_tensor_hqq
print(
    "nbits",
    nbits,
    "| HQQ dequantization error",
    (W - linear_layer(torch.eye(W.shape[1], dtype=W.dtype, device=W.device)).T)
    .abs()
    .mean()
    .item(),
)
print(
    "nbits",
    nbits,
    "| HQQ Dot product error",
    (y_ref - linear_layer(x.to(compute_dtype))).abs().mean().item(),
)
# nbits 4 | HQQ dequantization error 0.0004863739013671875
# nbits 4 | HQQ Dot product error 0.0014699687017127872
