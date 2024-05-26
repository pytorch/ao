import os
os.environ['TORCH_LOGS'] = "output_code"

import torch
from bitnet_example.bitnet_lib import BitLinearTrain, quantize_weights, pack_int2, unpack_uint8_to_trinary2

test_layer = torch.rand(1024, 16, 8) * 500.0 - 250.0
original_fake_layer = quantize_weights(test_layer).to(torch.int8).to("cuda")
shifted_fake_layer = (original_fake_layer + 1.0).to(torch.uint8).to("cuda")

packed = pack_int2(shifted_fake_layer)
unpacked = unpack_uint8_to_trinary2(packed).to("cuda")

print(original_fake_layer.dtype)
print(unpacked.dtype)
print(unpacked.allclose(original_fake_layer))
assert(unpacked.allclose(original_fake_layer))