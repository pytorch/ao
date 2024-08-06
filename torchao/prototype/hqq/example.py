import torch
from torchao.prototype.hqq.core import HQQQuantizer
from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    ZeroPointDomain,
    PlainAQTLayout,
    PlainLayoutType,
)

#Parameters
device, compute_dtype = "cuda:0", torch.bfloat16
nbits, group_size, axis = 4, 64, 1

linear_layer = torch.nn.Linear(4096, 11800, bias=False)
W = linear_layer.weight.data.clone()

verbose = True  # For debugging the optimizer

################################################################################################
# # Uses raw_output=True to produce the same output as hqq lib
# W_q, scale, zero, shape = HQQQuantizer.quantize(
#     W,
#     nbits=nbits,
#     group_size=group_size,
#     axis=axis,
#     compute_dtype=compute_dtype,
#     device=device,
#     verbose=verbose,
#     raw_output=True,
# )
# W_r = ((W_q.to(zero.dtype) - zero) * scale).view(shape)
# print("Check error manually / raw_output=False", (linear_layer.weight.data.cuda() - W_r.float()).abs().mean().item())
# # compute_dtype bfloat16: 0.0004856811137869954
# # compute_dtype  float16: 0.00048531172797083855
################################################################################################

# Uses raw_output=False to produce AffineQuantizedTensor compatible output
W_q, scale, zero, shape = HQQQuantizer.quantize(
    W,
    nbits=nbits,
    group_size=group_size,
    axis=axis,
    compute_dtype=compute_dtype,
    device=device,
    verbose=verbose,
    raw_output=False,
)

W_r = ((W_q.to(zero.dtype).view([-1, group_size]) - (2**nbits) / 2) * scale + zero).view(shape)
print("Check error manually / raw_output=True", (linear_layer.weight.data.cuda() - W_r.float()).abs().mean().item())
# compute_dtype bfloat16: 0.0004856870509684086
# compute_dtype float16 : 0.00048532348591834307


layout_tensor = PlainAQTLayout.from_plain(
    int_data=W_q, scale=scale, zero_point=zero, layout_type=PlainLayoutType()
)

q_tensor = AffineQuantizedTensor(
    layout_tensor=layout_tensor,
    block_size=[1, group_size], # axis=1
    shape=shape,
    quant_min=0,
    quant_max=2**nbits - 1,
    zero_point_domain=ZeroPointDomain.FLOAT,
    dtype=torch.bfloat16,
)

print("Check error via AffineQuantizedTensor", (W.cuda() - q_tensor.dequantize().float()).abs().mean().item())

