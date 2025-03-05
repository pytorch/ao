# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
General configuration objects for post-training quantization.

TODO: write this
"""

from dataclasses import dataclass

from torchao.dtypes import PlainLayout
from torchao.dtypes.utils import Layout


@dataclass
class WeightQuantizationConfig:
    """
    TODO: write this.
    """
    dtype: Union[torch.dtype, TorchAODType]
    granularity: Granularity
    mapping_type: MappingType
    scale_precision: torch.dtype
    zero_point_precision: torch.dtype
    zero_point_domain: ZeroPointDomain
    is_dynamic: bool = True
    layout: Layout = PlainLayout()


@dataclass
class ActivationQuantizationConfig:
    """
    TODO: write this.
    """
    dtype: Union[torch.dtype, TorchAODType]
    granularity: Granularity
    mapping_type: MappingType
    scale_precision: torch.dtype
    zero_point_precision: torch.dtype
    zero_point_domain: ZeroPointDomain
    is_dynamic: bool = True
    layout: Layout = PlainLayout()


@dataclass
class QuantizationConfig(AOBaseConfig):
    """
    TODO: write this.
    """

    weight_config: WeightQuantizationConfig
    activation_config: ActivationQuantizationConfig
