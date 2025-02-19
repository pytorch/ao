# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch

from torchao.dtypes import PlainLayout
from torchao.experimental.packed_linear_int8_dynamic_activation_intx_weight_layout import (
    PackedLinearInt8DynamicActivationIntxWeightLayout,
)
from torchao.experimental.quant_api import (
    int8_dynamic_activation_intx_weight,
)
from torchao.quantization.granularity import (
    PerRow,
)
from torchao.quantization.quant_api import quantize_
from torchao.quantization.quant_primitives import MappingType


class TestPackedLinearInt8DynamicActivationIntxWeightLayoutAten(unittest.TestCase):
    def test_accuracy(self):
        """
        Checks the accuracy of PackedLinearInt8DynamicActivationIntxWeightLayout() by comparing
        its results to the results of a reference model that uses PlainLayout()
        """
        granularities = [PerRow()]
        m = 32
        n = 128
        k = 256
        activations = torch.randn(m, k)
        weight_mapping_type = MappingType.SYMMETRIC_NO_CLIPPING_ERR
        model = torch.nn.Sequential(*[torch.nn.Linear(k, n, bias=False)])

        for weight_dtype in [
            torch.int4,
        ]:
            for has_weight_zeros in [True]:
                for granularity in granularities:
                    print(
                        f"Testing weight_dtype={weight_dtype}, has_weight_zeros={has_weight_zeros}, granularity={granularity}"
                    )
                    quantized_model = copy.deepcopy(model)
                    quantize_(
                        quantized_model,
                        int8_dynamic_activation_intx_weight(
                            weight_dtype=weight_dtype,
                            granularity=granularity,
                            has_weight_zeros=has_weight_zeros,
                            weight_mapping_type=weight_mapping_type,
                            layout=PackedLinearInt8DynamicActivationIntxWeightLayout(
                                target="aten"
                            ),  # default
                        ),
                    )

                    quantized_model_reference = copy.deepcopy(model)
                    quantize_(
                        quantized_model_reference,
                        int8_dynamic_activation_intx_weight(
                            weight_dtype=weight_dtype,
                            granularity=granularity,
                            has_weight_zeros=has_weight_zeros,
                            layout=PlainLayout(),
                        ),
                    )

                    with torch.no_grad():
                        res = quantized_model(activations)
                        ref = quantized_model_reference(activations)

                    mean_err = ((res - ref).abs() / ref).mean()
                    self.assertTrue(mean_err < 0.04)


if __name__ == "__main__":
    unittest.main()
