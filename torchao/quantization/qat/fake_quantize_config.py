# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch

from torchao.core.config import AOBaseConfig
from torchao.float8.config import e4m3_dtype
from torchao.float8.inference import (
    FP8Granularity,
    _normalize_granularity,
)
from torchao.quantization.granularity import (
    Granularity,
    PerAxis,
    PerGroup,
    PerRow,
    PerTensor,
    PerToken,
)
from torchao.quantization.quant_primitives import (
    _SUB_BYTE_INT_BOUNDS,
    _SUB_BYTE_UINT_BOUNDS,
    MappingType,
    TorchAODType,
    ZeroPointDomain,
)
from torchao.utils import _is_float8_type

from .utils import _log_deprecation_warning

from .utils import _log_deprecation_warning


class FakeQuantizeConfigBase(abc.ABC):
    """
    Base class for representing fake quantization config.
    """

    pass


@dataclass
class Float8FakeQuantizeConfig(FakeQuantizeConfigBase):
    """
    Config for float8 fake quantization, targeting :class:`~torchao.quantization.Float8Tensor`.

    Args:
       dtype (torch.dtype): the dtype for float8 Tensor
       granularity (FP8Granularity): the granularity for the Tensor, currently either PerRow() or PerTensor()
       hp_value_lb (Optional[float]): the lower bound for high precision floating point value for calculating scale
       hp_value_ub (Optional[float]): the upper bound for high precision floating point value for calculating scale
    """

    dtype: torch.dtype = e4m3_dtype
    granularity: FP8Granularity = PerRow()
    hp_value_lb: Optional[float] = None
    hp_value_ub: Optional[float] = None

    def __post_init__(self):
        """
        Verify dtype and granularity are the ones we support.
        """
        if not _is_float8_type(self.dtype):
            raise ValueError(f"{self.dtype} is not a float8 dtype")
        if isinstance(self.granularity, type):
            raise ValueError(
                "Please specify the granularity object instead of the class, e.g. PerRow() instead of PerRow"
            )
        if type(self.granularity) not in [PerRow, PerTensor]:
            raise ValueError(
                f"Expected PerRow or PerTensor granularity, got {self.granularity}"
            )


@dataclass
class IntxFakeQuantizeConfig(FakeQuantizeConfigBase):
    """
    Config for how to fake quantize weights or activations,
    targeting integer dtypes up to torch.int8.

    Args:
        dtype: dtype to simulate during fake quantization, e.g. torch.int8.
            For PyTorch versions older than 2.6, you may use `TorchAODType` to represent
            torch.int1 to torch.int7 instead, e.g. TorchAODType.INT4.
        granularity: granularity of scales and zero points, e.g. PerGroup(32).
            We also support the following strings:
               1) 'per_token': equivalent to PerToken()
               2) 'per_channel': equivalent to PerAxis(0)
               3) 'per_group': equivalent to PerGroup(group_size), must be combined
                   with separate `group_size` kwarg, Alternatively, just set the
                   `group_size` kwarg and leave this field empty.
        mapping_type: whether to use symmetric (default) or asymmetric quantization
            Alternatively, set `is_symmetric` (bool) and leave this field empty.
        scale_precision: scale dtype (default torch.fp32)
        zero_point_precision: zero point dtype (default torch.int32)
        zero_point_domain: whether zero point is in integer (default) or float domain
        is_dynamic: whether to use dynamic (default) or static scale and zero points
        range_learning (prototype): whether to learn scale and zero points during training
            (default false), not compatible with `is_dynamic`.

    Keyword args:
        group_size: size of each group in per group fake quantization,
            can be set instead of `granularity`
        is_symmetric: whether to use symmetric or asymmetric quantization,
            can be set instead of `mapping_type`

    Example usage::

        # Per token asymmetric quantization
        IntxFakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
        IntxFakeQuantizeConfig(torch.int8, PerToken(), MappingType.ASYMMETRIC)

        # Per channel symmetric quantization
        IntxFakeQuantizeConfig(torch.int4, "per_channel")
        IntxFakeQuantizeConfig(torch.int4, "per_channel", is_symmetric=True)
        IntxFakeQuantizeConfig(torch.int4, PerAxis(0), MappingType.SYMMETRIC)

        # Per group symmetric quantization
        IntxFakeQuantizeConfig(torch.int4, group_size=32)
        IntxFakeQuantizeConfig(torch.int4, group_size=32, is_symmetric=True)
        IntxFakeQuantizeConfig(torch.int4, "per_group", group_size=32, is_symmetric=True)
        IntxFakeQuantizeConfig(torch.int4, PerGroup(32), MappingType.SYMMETRIC)
    """

    dtype: Union[torch.dtype, TorchAODType]
    granularity: Granularity
    mapping_type: MappingType
    scale_precision: torch.dtype
    zero_point_precision: torch.dtype
    zero_point_domain: ZeroPointDomain
    is_dynamic: bool = True
    range_learning: bool = False
    eps: Optional[float] = None

    def __init__(
        self,
        dtype: Union[torch.dtype, TorchAODType],
        granularity: Union[Granularity, str, None] = None,
        mapping_type: Optional[MappingType] = None,
        scale_precision: torch.dtype = torch.float32,
        zero_point_precision: torch.dtype = torch.int32,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        is_dynamic: bool = True,
        range_learning: bool = False,
        eps: Optional[float] = None,
        *,
        group_size: Optional[int] = None,
        is_symmetric: Optional[bool] = None,
    ):
        if zero_point_domain is None:
            raise ValueError("Please use ZeroPointDomain.NONE instead of None")
        self.dtype = dtype
        self.granularity = self._get_granularity(granularity, group_size)
        self.mapping_type = self._get_mapping_type(mapping_type, is_symmetric)
        self.scale_precision = scale_precision
        self.zero_point_precision = zero_point_precision
        self.zero_point_domain = zero_point_domain
        self.is_dynamic = is_dynamic
        self.range_learning = range_learning
        self.eps = eps

        # Validate dtype
        all_dtypes = [torch.int8, torch.uint8]
        all_dtypes.extend(list(_SUB_BYTE_INT_BOUNDS.keys()))
        all_dtypes.extend(list(_SUB_BYTE_UINT_BOUNDS.keys()))
        if dtype not in all_dtypes:
            raise ValueError(
                "Unsupported dtype '%s', choose from %s" % (dtype, all_dtypes)
            )

        # Dynamic is not compatible with range learning
        if is_dynamic and range_learning:
            raise ValueError("`is_dynamic` is not compatible with `range_learning`")

        self.__post_init__()

    def __post_init__(self):
        """
        For deprecation only, can remove after https://github.com/pytorch/ao/issues/2630.
        """
        pass

    def _get_granularity(
        self,
        granularity: Union[Granularity, str, None],
        group_size: Optional[int],
    ) -> Granularity:
        """
        Parse the `Granularity` represented in the args.

        Granularity can be specified in one of three ways:
            1) `Granularity` object: one of PerToken(), PerAxis(), and PerGroup(group_size)
            2) str: one of 'per_token', 'per_channel', and 'per_group'
            3) None: `group_size` must be set instead, represents per group granularity
        """
        # If group_size is set, then granularity must be either "per_group" or None
        if (
            group_size is not None
            and granularity != "per_group"
            and granularity is not None
        ):
            raise ValueError(
                "`group_size` conflicts with granularity '%s'" % granularity
            )

        # Case 1: Granularity object
        if isinstance(granularity, Granularity):
            if not isinstance(granularity, (PerToken, PerAxis, PerGroup)):
                raise ValueError("Granularity '%s' is not supported" % granularity)
            if isinstance(granularity, PerAxis) and granularity.axis != 0:
                raise ValueError("Only axis=0 is supported for PerAxis granularity")
            return granularity

        # Case 2: str granularity
        if granularity == "per_token":
            return PerToken()
        elif granularity == "per_channel":
            return PerAxis(axis=0)
        elif granularity == "per_group":
            if group_size is None:
                raise ValueError(
                    "Granularity was 'per_group' but no `group_size` was set"
                )
            return PerGroup(group_size)
        elif isinstance(granularity, str):
            raise ValueError(
                "Unexpected granularity: '%s', must be one of %s"
                % (granularity, ["per_token", "per_channel", "per_group"])
            )

        # Case 3: None granularity + group_size was specified
        if granularity is not None:
            raise ValueError(
                "Granularity '%s' has unexpected type %s"
                % (granularity, type(granularity))
            )
        if group_size is None:
            raise ValueError(
                "At least one of `granularity` or `group_size` must be set"
            )
        return PerGroup(group_size)

    def _get_mapping_type(
        self,
        mapping_type: Optional[MappingType],
        is_symmetric: Optional[bool],
    ) -> MappingType:
        """
        Parse the `MappingType` represented in the args.

        Mapping type can be specified in one of two ways:
            1): `MappingType` object: one of SYMMETRIC or ASYMMETRIC
            2): is_symmetric bool
        """
        if mapping_type is not None and is_symmetric is not None:
            raise ValueError("Cannot set both `mapping_type` and `is_symmetric`")

        # Case 0: Default to symmetric
        if mapping_type is None and is_symmetric is None:
            return MappingType.SYMMETRIC

        # Case 1: MappingType object
        if mapping_type is not None:
            if mapping_type not in [MappingType.SYMMETRIC, MappingType.ASYMMETRIC]:
                raise ValueError("MappingType '%s' is not supported" % mapping_type)
            return mapping_type

        # Case 2: is_symmetric flag
        assert is_symmetric is not None
        if is_symmetric:
            return MappingType.SYMMETRIC
        else:
            return MappingType.ASYMMETRIC

    @property
    def group_size(self) -> int:
        """
        If this is per group granularity, return the group size.
        Otherwise, throw an error.
        """
        if isinstance(self.granularity, PerGroup):
            return self.granularity.group_size
        else:
            raise ValueError(
                "`group_size` is undefined for %s granularity" % self.granularity
            )

    @property
    def is_symmetric(self) -> bool:
        """
        Return True if mapping type is symmetric, else False (asymmetric).
        """
        return self.mapping_type == MappingType.SYMMETRIC

    def __setattr__(self, name: str, value: Any):
        """
        Support setting `group_size` and `is_symmetric`.
        """
        if name == "group_size":
            super().__setattr__("granularity", PerGroup(value))
        elif name == "is_symmetric":
            mapping_type = MappingType.SYMMETRIC if value else MappingType.ASYMMETRIC
            super().__setattr__("mapping_type", mapping_type)
        else:
            super().__setattr__(name, value)


# For BC
class FakeQuantizeConfig(IntxFakeQuantizeConfig):
    """
    (Deprecated) Please use :class:`~torchao.quantization.qat.IntxFakeQuantizeConfig` instead.
    """

    def __post_init__(self):
        _log_deprecation_warning(self)


# TODO: rewrite using registration API?
def _infer_fake_quantize_configs(
    base_config: AOBaseConfig,
) -> Tuple[Optional[FakeQuantizeConfigBase], Optional[FakeQuantizeConfigBase]]:
    """
    Given a base post-training quantization (PTQ) config, infer the corresponding
    `FakeQuantizeConfigBase`s for both the activations and the weights.
    This is called during the prepare phase of QAT.

    Return a 2-tuple of (activation_config, weight_config) for fake quantization.
    """
    # avoid circular imports
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Float8DynamicActivationInt4WeightConfig,
        Int4WeightOnlyConfig,
        Int8DynamicActivationInt4WeightConfig,
    )

    if isinstance(base_config, Int8DynamicActivationInt4WeightConfig):
        act_config = IntxFakeQuantizeConfig(
            dtype=torch.int8,
            granularity="per_token",
            is_symmetric=base_config.act_mapping_type == MappingType.SYMMETRIC,
        )
        weight_config = IntxFakeQuantizeConfig(
            dtype=torch.int4,
            group_size=base_config.group_size,
            is_symmetric=base_config.mapping_type == MappingType.SYMMETRIC,
        )
    elif isinstance(base_config, Int4WeightOnlyConfig):
        if base_config.version != 2:
            raise ValueError(f"Only version 2 of {type(base_config)} is supported")
        act_config = None
        weight_config = IntxFakeQuantizeConfig(
            dtype=torch.int4,
            group_size=base_config.group_size,
            is_symmetric=True,
        )
    elif isinstance(base_config, Float8DynamicActivationFloat8WeightConfig):
        if base_config.version != 2:
            raise ValueError(f"Only version 2 of {type(base_config)} is supported")
        (act_granularity, weight_granularity) = _normalize_granularity(
            base_config.granularity
        )
        act_config = Float8FakeQuantizeConfig(
            dtype=base_config.activation_dtype,
            granularity=act_granularity,
            hp_value_lb=base_config.activation_value_lb,
            hp_value_ub=base_config.activation_value_ub,
        )
        weight_config = Float8FakeQuantizeConfig(
            dtype=base_config.weight_dtype,
            granularity=weight_granularity,
        )
    elif isinstance(base_config, Float8DynamicActivationInt4WeightConfig):
        act_config = Float8FakeQuantizeConfig(
            dtype=torch.float8_e4m3fn,
            granularity=PerRow(),
        )
        weight_config = IntxFakeQuantizeConfig(
            dtype=torch.int4,
            group_size=base_config.group_size,
            is_symmetric=True,
        )
    else:
        raise ValueError("Unexpected base config: %s" % base_config)
    return (act_config, weight_config)
