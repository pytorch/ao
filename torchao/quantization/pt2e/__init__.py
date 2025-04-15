# mypy: allow-untyped-defs

from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torchao.quantization.pt2e.pt2e._numeric_debugger import (  # noqa: F401
    CUSTOM_KEY,
    NUMERIC_DEBUG_HANDLE_KEY,
    compare_results,
    extract_results_from_loggers,
    generate_numeric_debug_handle,
    prepare_for_propagation_comparison,
)
from torchao.quantization.pt2e.pt2e.export_utils import (
    _allow_exported_model_train_eval as allow_exported_model_train_eval,
)
from torchao.quantization.pt2e.pt2e.export_utils import (
    _move_exported_model_to_eval as move_exported_model_to_eval,
)
from torchao.quantization.pt2e.pt2e.export_utils import (
    _move_exported_model_to_train as move_exported_model_to_train,
)

from .fake_quantize import (
    FakeQuantize,
    FakeQuantizeBase,
    FixedQParamsFakeQuantize,
    FusedMovingAvgObsFakeQuantize,
    default_fake_quant,
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
    _PartialWrapper,
    get_block_size,
)

# ensure __module__ is set correctly for public APIs
ObserverOrFakeQuantize = Union[ObserverBase, FakeQuantizeBase]
ObserverOrFakeQuantize.__module__ = "torchao.quantization.pt2e"

for _f in [
    compare_results,
    extract_results_from_loggers,
    generate_numeric_debug_handle,
    prepare_for_propagation_comparison,
]:
    _f.__module__ = "torchao.quantization.pt2e"


_ObserverOrFakeQuantizeConstructor = Union[
    _PartialWrapper, type[ObserverBase], type[FakeQuantizeBase]
]

__all__ = [
    "FakeQuantize",
    "FakeQuantizeBase",
    "FixedQParamsFakeQuantize",
    "FixedQParamsObserver",
    "FusedMovingAvgObsFakeQuantize",
    "HistogramObserver",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "MovingAveragePerChannelMinMaxObserver",
    "NoopObserver",
    "ObserverBase",
    "ObserverOrFakeQuantize",
    "_ObserverOrFakeQuantizeConstructor",
    "PerChannelMinMaxObserver",
    "PlaceholderObserver",
    "RecordingObserver",
    "ReuseInputObserver",
    "UniformQuantizationObserverBase",
    "enable_fake_quant",
    "enable_observer",
    "move_exported_model_to_eval",
    "move_exported_model_to_train",
    "allow_exported_model_train_eval",
    # pt2e numeric debugger
    "generate_numeric_debug_handle",
    "CUSTOM_KEY",
    "NUMERIC_DEBUG_HANDLE_KEY",
    "prepare_for_propagation_comparison",
    "extract_results_from_loggers",
    "compare_results",
    # from torchao, should be merged with torchao
    # in the future
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
]


def default_eval_fn(model, calib_data):
    r"""Define the default evaluation function.

    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """
    for data, _target in calib_data:
        model(data)


class _DerivedObserverOrFakeQuantize(ObserverBase):
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
            assert (
                self.ch_axis is not None
            ), "Must provide a valid ch_axis if qscheme is per channel"

    def forward(self, x: Tensor) -> Tensor:
        return x

    def calculate_qparams(self):  # type:ignore[override]
        return self.derive_qparams_fn(self.obs_or_fqs)
