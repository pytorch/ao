# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

from torchao.prototype.parq.quant import (
    StretchedIntxWeightConfig,
    StretchedUnifTorchaoQuantizer,
)
from torchao.prototype.quantization.int8_lut_tensor.int8_lut_tensor import Int8LutTensor
from torchao.prototype.tensor_conversion.api import _convert_model_for_aarch64
from torchao.quantization import MappingType
from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    quantize_,
)
from torchao.quantization.quantize_.workflows.intx.intx_opaque_tensor import (
    IntxOpaqueTensor,
    _is_kernel_library_loaded,
)
from torchao.quantization.utils import compute_error


class ToyLinearModelWithTiedEmbedding(torch.nn.Module):
    def __init__(self, d0=512, d1=512, d2=256, d3=128, d4=32):
        super().__init__()
        self.embedding1 = torch.nn.Embedding(d0, d1)
        self.embedding2 = torch.nn.Embedding(d0, d1)
        self.embedding3 = torch.nn.Embedding(d0, d1)

        self.linear1 = torch.nn.Linear(d1, d2, bias=False)
        self.linear2 = torch.nn.Linear(d2, d3, bias=True)
        self.linear3 = torch.nn.Linear(d3, d4, bias=False)
        self.linear4 = torch.nn.Linear(d4, d1, bias=False)

        self.lm_head1 = torch.nn.Linear(d1, d0, bias=False)
        self.lm_head2 = torch.nn.Linear(d1, d0, bias=False)
        self.lm_head3 = torch.nn.Linear(d1, d0, bias=False)

        # Tie weights
        # lm_head1 / lm_head2 form one tied weight group
        self.embedding2.weight = self.embedding1.weight
        self.lm_head1.weight = self.embedding1.weight
        self.lm_head2.weight = self.embedding1.weight

        # lm_head3 forms a separate tied weight group
        self.lm_head3.weight = self.embedding3.weight

    def example_inputs(
        self,
        lead_dim=(1,),
        dtype=torch.bfloat16,
    ):
        return (
            torch.randint(
                0,
                self.embedding1.num_embeddings,
                size=lead_dim,
                dtype=torch.int64,
                device="cpu",
            ),
        )

    def forward(self, x):
        x = self.embedding1(x) + self.embedding2(x) + self.embedding3(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.lm_head1(x) + self.lm_head2(x) + self.lm_head3(x)
        return x


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    yield
    torch._dynamo.reset()  # reset cache between tests


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("granularity", [PerGroup(32), PerAxis(0)])
@pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "lead_dim",
    [
        (1,),
        (5,),
        (7, 2),
    ],
)
@pytest.mark.skipif(
    not _is_kernel_library_loaded(), reason="Kernel library is not loaded"
)
def test_aarch64_conversion(dtype, granularity, bit_width, lead_dim):
    torch.manual_seed(0)

    model = ToyLinearModelWithTiedEmbedding()
    model = model.to(dtype)
    example_inputs = model.example_inputs(lead_dim, dtype)

    # Quantize linear 2 and 3 with PARQ
    quantizer = StretchedUnifTorchaoQuantizer(bit_width)
    config = StretchedIntxWeightConfig(
        b=bit_width,
        quant_min=quantizer.quant_min,
        quant_max=quantizer.quant_max,
        granularity=granularity,
        activation_quantization="int8_asym_per_token",
    )
    quantize_(model, config, filter_fn=lambda m, fqn: fqn in ["linear2", "linear3"])

    # Quantize linear 1 and 4 with int8 dynamic activation
    config = Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int4,
        weight_granularity=granularity,
        weight_mapping_type=MappingType.SYMMETRIC,
    )
    quantize_(
        model,
        config,
        filter_fn=lambda m, fqn: fqn
        in ["linear1", "linear4", "lm_head1", "lm_head2", "lm_head3"],
    )

    # Quantize embedding 1, 2, and 3 with weight only
    config = IntxWeightOnlyConfig(
        weight_dtype=torch.int4,
        granularity=granularity,
        mapping_type=MappingType.SYMMETRIC,
    )
    quantize_(
        model,
        config,
        filter_fn=lambda m, fqn: fqn in ["embedding1", "embedding2", "embedding3"],
    )
    model_out = model(*example_inputs)

    # Convert to optimized model
    _convert_model_for_aarch64(model)

    # Check expected tensor subclass
    assert isinstance(model.linear2.weight, Int8LutTensor)
    assert isinstance(model.linear3.weight, Int8LutTensor)
    assert isinstance(model.linear1.weight, IntxOpaqueTensor)
    assert isinstance(model.linear4.weight, IntxOpaqueTensor)

    # Assert tied params
    tied_group1_id = id(model.embedding1.weight)
    assert id(model.embedding2.weight) == tied_group1_id
    assert id(model.lm_head1.weight) == tied_group1_id
    assert id(model.lm_head2.weight) == tied_group1_id

    assert id(model.lm_head3.weight) == id(model.embedding3.weight)
    assert id(model.lm_head3.weight) != tied_group1_id

    # Compare converted out with original out
    converted_out = model(*example_inputs)
    sqnr = compute_error(model_out, converted_out)
    sqnr_threshold = 30
    assert sqnr > sqnr_threshold, f"sqnr: {sqnr}"

    # Check exported graph for correct ops
    ep = torch.export.export(model, example_inputs)
    expected_counts = {
        "torch.ops.torchao._shared_embedding_": 3,
        "torch.ops.torchao._linear_8bit_act_": 7,
        "torch.ops.aten.linear.default": 0,
        "torch.ops.aten.embedding.default": 0,
    }
    for line, cnt in expected_counts.items():
        assert ep.graph_module.code.count(line) == cnt, (
            f"expected {cnt} {line} in {ep.graph_module.code}"
        )
