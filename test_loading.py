# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from torchao.quantization import (
    Float8Tensor,
    Int4TilePackedTo4dTensor,
    IntxUnpackedToInt8Tensor,
)

model_name = "torchao-testing/opt-125m-ModuleFqnToConfig-v1-regex-0.14.0.dev"
device = "cuda"
input_text = "What are we having for dinner?"
max_new_tokens = 10

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    dtype=torch.bfloat16,
)
for i in range(12):
    if i == 3:
        assert isinstance(
            quantized_model.model.decoder.layers[i].self_attn.q_proj.weight,
            Int4TilePackedTo4dTensor,
        )
        assert isinstance(
            quantized_model.model.decoder.layers[i].self_attn.k_proj.weight,
            Int4TilePackedTo4dTensor,
        )
        assert isinstance(
            quantized_model.model.decoder.layers[i].self_attn.v_proj.weight,
            Int4TilePackedTo4dTensor,
        )
    else:
        assert isinstance(
            quantized_model.model.decoder.layers[i].self_attn.q_proj.weight,
            Float8Tensor,
        )
        assert isinstance(
            quantized_model.model.decoder.layers[i].self_attn.k_proj.weight,
            Float8Tensor,
        )
        assert isinstance(
            quantized_model.model.decoder.layers[i].self_attn.v_proj.weight,
            Float8Tensor,
        )
    assert isinstance(
        quantized_model.model.decoder.layers[i].self_attn.out_proj.weight,
        IntxUnpackedToInt8Tensor,
    )

tokenizer = AutoTokenizer.from_pretrained(model_name)

input_ids = tokenizer(input_text, return_tensors="pt").to(device)

output = quantized_model.generate(**input_ids, max_new_tokens=max_new_tokens)
print(tokenizer.decode(output[0], skip_special_tokens=True))
