# Safetensors Support in TorchAO

`torchao.prototype.safetensors` provides secure, zero-copy serialization and loading for `torchao` quantized models using the [safetensors](https://github.com/huggingface/safetensors) format.

Because safetensors only stores plain `torch.Tensor` objects, `torchao` decomposes custom tensor subclasses (like `Int8Tensor`, `Float8Tensor`, `NVFP4Tensor`) into plain data tensors and JSON metadata, and reconstructs the subclass structure on load.

## Quickstart

### Saving a Quantized Model to Safetensors

```python
import tempfile
import torch
from safetensors.torch import save_file
from torchao.prototype.safetensors.safetensors_support import flatten_tensor_state_dict

# 1. Quantize your model (e.g. SmoothQuant static INT8)
from torchao.quantization.quant_api import quantize_, Int8StaticActivationInt8WeightConfig, PerTensor
from torchao.prototype.smoothquant import SmoothQuantConfig
from torchao.quantization.quantize_.common.quantization_step import QuantizationStep

model = ... # your PyTorch model
config = SmoothQuantConfig(
    base_config=Int8StaticActivationInt8WeightConfig(granularity=PerTensor()),
    step=QuantizationStep.PREPARE,
)
quantize_(model, config)

# Calibrate with representative data (single pass)
for batch in calibration_data:
    model(batch)

# Convert to quantized model
config.step = QuantizationStep.CONVERT
quantize_(model, config)

# 2. Flatten and save to .safetensors
tensors_data_dict, metadata = flatten_tensor_state_dict(model.state_dict())
save_file(tensors_data_dict, "model_quantized.safetensors", metadata=metadata)
```

### Loading a Quantized Model from Safetensors

```python
import json
import struct
import torch
from safetensors.torch import load_file
from torchao.prototype.safetensors.safetensors_support import unflatten_tensor_state_dict

def load_data(file_path: str, device: str = "cpu"):
    loaded_tensors = load_file(file_path, device)
    with open(file_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes)
        metadata = header.get("__metadata__", {})
    return loaded_tensors, metadata

# 1. Load data and metadata from safetensors file
loaded_tensors, metadata = load_data("model_quantized.safetensors", device="cpu")

# 2. Reconstruct the tensor subclasses
reconstructed_dict, leftover = unflatten_tensor_state_dict(loaded_tensors, metadata)

# 3. Instantiate base model architecture and assign state dict
model = ... # fresh instance of model architecture
model.load_state_dict(reconstructed_dict, assign=True)

# Ready for inference!
```

## Supported Subclasses

- `Int8Tensor` (dynamic and static INT8, including SmoothQuant with `act_pre_scale`, `act_quant_scale`, `act_quant_zero_point`)
- `Float8Tensor`
- `Int4Tensor` & `Int4TilePackedTo4dTensor`
- `IntxUnpackedToInt8Tensor`
- `MXTensor`
- `NVFP4Tensor`
