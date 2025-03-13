from typing import Dict, List

import torch
import torch.nn as nn


class ExpectedError(Exception):
    pass


class DataGetter:
    def __init__(self) -> None:
        return

    def pop(self, model: nn.Module, name: str) -> torch.Tensor:
        raise NotImplementedError()

    def get_base_name(self, name: str) -> str:
        base_name = name.split(".")[-1]
        if base_name.isnumeric():
            base_name = name.split(".")[-2]
        return base_name

    def initialize(self, model: nn.Module, data: torch.Tensor, batch_size: int) -> None:
        raise NotImplementedError()


def get_module_input_data(
    model: nn.Module,
    data: torch.Tensor,
    module: nn.Module,
    batch_size: int,
    layer_kwargs: Dict[str, torch.Tensor] = {},  # noqa
) -> torch.Tensor:
    with torch.no_grad():
        if isinstance(data, list):
            num_data = len(data)
        else:
            num_data = data.shape[0]
        num_batches = num_data // batch_size
        assert num_data % batch_size == 0

        input_data: List[torch.Tensor] = []

        def _input_data_hook(
            module: nn.Module, input: List[torch.Tensor], output: List[torch.Tensor]
        ) -> None:
            input_data.append(input[0].detach())
            assert len(input) == 1
            raise ExpectedError

        hook = module.register_forward_hook(_input_data_hook)

        for i in range(num_batches):
            try:
                this_batch = data[i * batch_size : (i + 1) * batch_size]
                this_batch = this_batch.to(next(model.parameters()).device)
                if layer_kwargs:
                    model(this_batch, **layer_kwargs)
                else:
                    model(this_batch)
            except ExpectedError:
                pass

        hook.remove()
        return_data = torch.cat(input_data, dim=0)
        return return_data
