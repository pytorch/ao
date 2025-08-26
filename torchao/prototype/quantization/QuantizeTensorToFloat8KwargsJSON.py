import json
from typing import Any, Dict

import torch

from torchao.float8.inference import Float8MMConfig
from torchao.quantization.granularity import PerRow
from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
    QuantizeTensorToFloat8Kwargs,
)

ALLOWED_QUANT_DTYPES = {
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    # add to me
}
ALLOWED_GRANUALARITY = {"PerRow": PerRow()}


class QuantizeTensorToFloat8KwargsJSONEncoder(json.JSONEncoder):
    def default(self, o):
        json_dict = {}

        if hasattr(o, "mm_config"):
            json_dict["mm_config"] = json.dumps(o.mm_config)
        if hasattr(o, "hp_value_lb"):
            json_dict["hp_value_lb"] = o.hp_value_lb
        if hasattr(o, "hp_value_ub"):
            json_dict["hp_value_ub"] = o.hp_value_ub

        json_dict["kernel_preference"] = o.kernel_preference
        json_dict["granularity"] = str(o.granularity)
        json_dict["float8_dtype"] = str(o.float8_dtype)

        return json_dict


def config_from_dict(data: Dict[str, Any]) -> QuantizeTensorToFloat8Kwargs:
    """
    Create QuantizeTensorToFloat8Kwargs instance from a dictionary.
    """
    saved_mm_config = json.loads(data.get("mm_config"))
    if saved_mm_config:
        saved_mm_config = Float8MMConfig(*saved_mm_config)

    saved_granularity = ALLOWED_GRANUALARITY.get(data.get("granularity"))
    if not saved_granularity:
        saved_granularity = PerRow()

    return QuantizeTensorToFloat8Kwargs(
        float8_dtype=ALLOWED_QUANT_DTYPES.get(data.get("float8_dtype")),
        granularity=saved_granularity,
        mm_config=saved_mm_config,
        hp_value_lb=data.get("hp_value_lb"),
        hp_value_ub=data.get("hp_value_ub"),
        kernel_preference=data.get("kernel_preference"),
    )
