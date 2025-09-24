# mypy: allow-untyped-defs

from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torchao.quantization.pt2e._numeric_debugger import (  # noqa: F401
    CUSTOM_KEY,
    FROM_NODE_KEY,
    NUMERIC_DEBUG_HANDLE_KEY,
    compare_results,
    extract_results_from_loggers,
    generate_numeric_debug_handle,
    prepare_for_propagation_comparison,
)
from torchao.quantization.pt2e.export_utils import (
    WrapperModule,
)
from torchao.quantization.pt2e.export_utils import (
    _allow_exported_model_train_eval as allow_exported_model_train_eval,
)
from torchao.quantization.pt2e.export_utils import (
    _move_exported_model_to_eval as move_exported_model_to_eval,
)
from torchao.quantization.pt2e.export_utils import (
    _move_exported_model_to_train as move_exported_model_to_train,
)
from torchao.quantization.pt2e.graph_utils import (
    bfs_trace_with_node_process,
    find_sequential_partitions,
    get_equivalent_types,
    update_equivalent_types_dict,
)

from .fake_quantize import (
    FakeQuantize,
    FakeQuantizeBase,
    FixedQParamsFakeQuantize,
    FusedMovingAvgObsFakeQuantize,
    default_dynamic_fake_quant,
    default_fake_quant,
    disable_fake_quant,
    disable_observer,
    enable_fake_quant,
    enable_observer,
)
from .observer import (
    AffineQuantizedObserverBase,
    FixedQParamsObserver,
    Granularity,
    HistogramObserver,
    MappingType,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    NoopObserver,
    ObserverBase,
    PartialWrapper,
    PerAxis,
    PerBlock,
    PerChannelMinMaxObserver,
    PerGroup,
    PerRow,
    PerTensor,
    PerToken,
    PlaceholderObserver,
    RecordingObserver,
    ReuseInputObserver,
    TorchAODType,
    UniformQuantizationObserverBase,
    ZeroPointDomain,
    get_block_size,
)

for _f in [
    compare_results,
    extract_results_from_loggers,
    generate_numeric_debug_handle,
    prepare_for_propagation_comparison,
]:
    _f.__module__ = "torchao.quantization.pt2e"


# ensure __module__ is set correctly for public APIs
ObserverOrFakeQuantize = Union[ObserverBase, FakeQuantizeBase]
ObserverOrFakeQuantize.__module__ = "torchao.quantization.pt2e"

ObserverOrFakeQuantizeConstructor = Union[
    PartialWrapper, type[ObserverBase], type[FakeQuantizeBase]
]
ObserverOrFakeQuantizeConstructor.__module__ = "torchao.quantization.pt2e"


__all__ = [
    # old fake quantizers
    "FakeQuantize",
    "FakeQuantizeBase",
    "FixedQParamsFakeQuantize",
    "FixedQParamsObserver",
    "FusedMovingAvgObsFakeQuantize",
    # old observers
    "HistogramObserver",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "MovingAveragePerChannelMinMaxObserver",
    "NoopObserver",
    "ObserverBase",
    "ObserverOrFakeQuantize",
    "PerChannelMinMaxObserver",
    "PlaceholderObserver",
    "RecordingObserver",
    "ReuseInputObserver",
    "UniformQuantizationObserverBase",
    "ObserverOrFakeQuantizeConstructor",
    "DerivedObserverOrFakeQuantize",
    # utils
    "enable_fake_quant",
    "enable_observer",
    "disable_fake_quant",
    "disable_observer",
    # export_utils
    "move_exported_model_to_eval",
    "move_exported_model_to_train",
    "allow_exported_model_train_eval",
    "WrapperModule",
    # graph_utils
    "find_sequential_partitions",
    "get_equivalent_types",
    "update_equivalent_types_dict",
    "bfs_trace_with_node_process",
    # pt2e numeric debugger
    "generate_numeric_debug_handle",
    "CUSTOM_KEY",
    "NUMERIC_DEBUG_HANDLE_KEY",
    "FROM_NODE_KEY",
    "prepare_for_propagation_comparison",
    "extract_results_from_loggers",
    "compare_results",
    # should be merged with torchao/quantization/observer.py in the future
    "AffineQuantizedObserverBase",
    "Granularity",
    "MappingType",
    "PerAxis",
    "PerBlock",
    "PerGroup",
    "PerRow",
    "PerTensor",
    "PerToken",
    "TorchAODType",
    "ZeroPointDomain",
    "get_block_size",
    "default_fake_quant",
    "default_dynamic_fake_quant",
]


class DerivedObserverOrFakeQuantize(ObserverBase):
    r"""This observer is used to describe an observer whose quantization parameters
    are derived from other observers
    """

    def __init__(
        self,
        dtype: torch.dtype,
        obs_or_fqs: list[ObserverOrFakeQuantize],
        derive_qparams_fn: Callable[
            [list[ObserverOrFakeQuantize]], tuple[Tensor, Tensor]
        ],
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        qscheme: Optional[torch.qscheme] = None,
        ch_axis: Optional[int] = None,
    ):
        super().__init__(dtype)
        self.obs_or_fqs = obs_or_fqs
        self.derive_qparams_fn = derive_qparams_fn
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.qscheme = qscheme
        self.ch_axis = ch_axis

        from .utils import is_per_channel

        if is_per_channel(self.qscheme):
            assert self.ch_axis is not None, (
                "Must provide a valid ch_axis if qscheme is per channel"
            )

    def forward(self, x: Tensor) -> Tensor:
        return x

    def calculate_qparams(self):  # type:ignore[override]
        return self.derive_qparams_fn(self.obs_or_fqs)
