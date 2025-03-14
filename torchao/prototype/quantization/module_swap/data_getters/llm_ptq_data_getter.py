from typing import Dict

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from torchao.prototype.quantization.module_swap.data_getters.ptq_data_getter import (
    DataGetter,
    get_module_input_data,
)
from torchao.prototype.quantization.module_swap.utils import (
    get_layer_by_name,
)


class LLMPTQDataGetter(DataGetter):
    """
    This datagetter can be used to efficiently retrieve layer-wise input data from a LlamaForCausalLM model
    The two benefits are
    1) It caches the data in between layer's residuals, so previous layers dont have to be computed
    2) Layers with the same input have their input cached and returned

    Usage is simple, give it a model, and data
    then data_getter.pop(model, layer_name) for each batch of data you require

    the actual model passed is used to get the data, so if you want to e.g. quantize
    the entire network with weight quantizers, it uses that data
    The datagetter has to be called in-order of the layer's occurence in the network, otherwise it will fail

    """

    def __init__(
        self, model: LlamaForCausalLM, data: torch.Tensor, batch_size: int
    ) -> None:
        super().__init__()
        self.initialize(model, data, batch_size)

    def initialize(self, model: nn.Module, data: torch.Tensor, batch_size: int) -> None:
        assert isinstance(model, LlamaForCausalLM)
        assert isinstance(data, torch.Tensor)

        # set attention_mask and/or position_ids
        self.layer_kwargs: Dict[str, torch.Tensor] = self.get_layer_kwargs(model, data)

        self.input_data_cache: torch.Tensor = get_module_input_data(
            model, data, model.model.layers[0], batch_size
        )
        self.current_layer_idx = 0
        self.previously_called_name: str = ""
        self.batch_size = batch_size
        self.output_data_cache: torch.Tensor = torch.zeros_like(data)
        self.matched_input_layers = [
            ["q_proj", "k_proj", "v_proj"],
            ["up_proj", "gate_proj"],
        ]

    def pop(self, model: nn.Module, name: str) -> torch.Tensor:
        assert isinstance(model, LlamaForCausalLM)
        with torch.no_grad():
            # special case for the last layer
            if name != "lm_head":
                query_layer_idx = int(name.split(".")[2])
            else:
                query_layer_idx = len(model.model.layers)

            assert (
                query_layer_idx >= self.current_layer_idx
            ), "pop() called out of order, layers have to be called in order"

            # TODO: batch the next two parts

            # progress the progress over layers
            while query_layer_idx > self.current_layer_idx:
                self.input_data_cache = model.model.layers[self.current_layer_idx](
                    self.input_data_cache, **self.layer_kwargs
                )[0]
                self.current_layer_idx += 1

            # special case for the final layer
            if name == "lm_head":
                return self.input_data_cache

            # use cached output data if the outputs are matching
            base_name = self.get_base_name(name)
            for matching_list in self.matched_input_layers:
                if base_name in matching_list:
                    previous_base_name = self.get_base_name(self.previously_called_name)
                    if previous_base_name in matching_list:
                        self.previously_called_name = name
                        return self.output_data_cache

            # get data from current requested layer
            query_layer = get_layer_by_name(model, name)
            layer_output_data = get_module_input_data(
                model.model.layers[self.current_layer_idx],
                self.input_data_cache,
                query_layer,
                self.batch_size,
                self.layer_kwargs,
            )

            # cache the data for next call
            self.output_data_cache = layer_output_data
            self.previously_called_name = name
            return layer_output_data

    def get_layer_kwargs(
        self, model: LlamaForCausalLM, data: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # get used attention_mask and position_ids
        layer_kwargs: Dict[str, torch.Tensor] = {}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                if kwargs["attention_mask"] is not None:
                    layer_kwargs["attention_mask"] = kwargs["attention_mask"]
                if kwargs["position_ids"] is not None:
                    layer_kwargs["position_ids"] = kwargs["position_ids"]
                raise ValueError

        device = model.parameters().__next__().device
        model.model.layers[0] = Catcher(model.model.layers[0])
        try:
            model(data[0:2].to(device))
        except ValueError:
            pass
        model.model.layers[0] = model.model.layers[0].module

        return layer_kwargs
