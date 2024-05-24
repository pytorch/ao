import torch
from typing import Any
from abc import ABC, abstractmethod


# API 1, single quantize call to create a quantized model with quantized state_dict
class Quantizer(ABC):
    @abstractmethod
    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:

        pass


# API 2, flow that needs calibration or training
class TwoStepQuantizer:
    @abstractmethod
    def prepare(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:

        pass

    def convert(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:

        pass
