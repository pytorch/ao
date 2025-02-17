# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
import unittest

import torch
from torch.export import exported_program

from torchao.dtypes import PlainLayout
from torchao.experimental.packed_linear_int8_dynamic_activation_intx_weight_layout import (
    PackedLinearInt8DynamicActivationIntxWeightLayout,
)
from torchao.experimental.q_dq_layout import QDQLayout
from torchao.experimental.quant_api import int8_dynamic_activation_intx_weight, intx_weight
from torchao.quantization.granularity import PerGroup, PerRow
from torchao.quantization.quant_api import quantize_
from torchao.utils import unwrap_tensor_subclass

granularity = PerRow()
m = 3
k0 = 512
k1 = 256
k2 = 128
k3 = 1024
weight_dtype = torch.int8
has_weight_zeros = True
layers = [
    torch.nn.Linear(k0, k1, bias=False),
]
model = torch.nn.Sequential(*layers)

# model = Repro()
# quantize_(
#     model,
#     intx_weight(
#         weight_dtype=torch.int4,
#         granularity=PerRow(),
#         has_weight_zeros=False,
#         layout=QDQLayout(),
#     ),
# )

activations = torch.randn(2, 1, m, k0, dtype=torch.float32)

exported = torch.export.export(model, (activations,), strict=True)


from coremltools.optimize.torch.quantization.quantization_config import (
            LinearQuantizerConfig,
            QuantizationScheme,
)
from coremltools.optimize.torch.quantization._coreml_quantizer import CoreMLQuantizer

config = LinearQuantizerConfig.from_dict(
    {
        "global_config": {
            "quantization_scheme": QuantizationScheme.affine,
            "activation_dtype": torch.quint8,
            "weight_dtype": "qint4",
            "weight_per_channel": True,
        }
    }
)
quantizer = CoreMLQuantizer(config)
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
em = exported.module()
em = prepare_pt2e(em, quantizer)
print("PREPARED", em)
em = convert_pt2e(em)
print("CONVERTED", em)

exported = torch.export.export(em, (activations,))

print("EXPORTED")
print(exported)

# for node in exported.graph.nodes:
#     print(node.name, node.meta["val"].shape, node.meta["val"].dtype)

print(exported_program.graph_signature())


# print("EXPORTED AFTER")
# print(exported)
# # eager_results = model(activations)

# # unwrapped_model = copy.deepcopy(model)
# # unwrap_tensor_subclass(model)
# # print("Unwrapped model")
# # print(model[0])

# # print("Exporting quantized model")
# # exported = torch.export.export(model, (activations,), strict=True)
# # exported_results = exported.module()(activations)

# # print(exported)


# import coremltools as ct
# # import torch

# # class Model(torch.nn.Module):
# #     def forward(self, x):
# #         noise = torch.randn(x.shape)
# #         return x + noise

# # model = Model()
# # inputs = (torch.randn(1,3,16,16),)
# # ep = torch.export.export(model, inputs)
# # print(ep)

# exported._verifiers[0].dialect = "ATEN"
# print(exported)
# converted_model = ct.convert(
#     exported,
#     convert_to="mlprogram",
# )
# print(converted_model._mil_program)
