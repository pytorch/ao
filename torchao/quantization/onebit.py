import torch 
from torch import Tensor
import torch.nn as nn
import torch.functional as F
from torchao.nn import RMSNorm
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter # This shouldn't be an internal function, it's useful

# TODO: How would people create other Linear nn.Modules?
class BitLinear(nn.Linear):
    
    # TODO: What is the API we'll produce to help people apply BitLienar to their models? 
    # For training

    @classmethod
    def from_float(cls, mod: nn.Linear) -> "BitLinear":
        """
        Converts a `mod` of class `torch.nn.Linear` to the `BitLinear` class
        """
        return cls(mod.in_features, mod.out_features, mod.bias is not None)

    def activation_quant(x : Tensor) -> Tensor:
        """
        Uniform Symmetric int8 quantization (not a new feature)
        """
        scale = 127.0 / x.abs(dim=-1, keepdim=True).values.clamp_(min=1e-5) # Scale = 127 / Max magnitude
        y = (x * scale).round().clamp_(-128, 127) / scale  # Quantize to -128, ..., 127
        return y # Take that tuple and call it uniform8 type, this exists in ao already

    def weight_quant(w : Tensor) -> Tensor:
        """
        Ternary quantization
        This does not store the dtype of the weights but converts at runtime
        """
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5) # Scale = 1 / Average magnitude
        u = (w * scale).round().clamp_(-1, 1) / scale # Quantize to -1, 0, 1
        return u # Take that tuple and call it 1.58 type

    def one_bit_weight_quant(w):
        scale = w.abs().mean()
        e = w.mean()
        u = (w - e).sign() * scale
        return u 

    # For inference
    def activation_norm_quant(x):
        x = RMSNorm(x)
        scale = 127.0 / x.abs.max(dim=-1, keepdim=True).values.clamp_(min=1e-5) # Scale = 127 / Max magnitude
        y = (x * scale).round().clamp_(-128, 127) # no division by scale because we are not training
        return y, scale

    def inference_forward(self, x : Tensor) -> Tensor:
        w = self.weight
        w_scale = self.weight_scale
        x_quant, x_scale = self.activation_norm_quant(x)
        y = gemm_lowbit_kernel(x_quant, w) / w_scale / x_scale # where is this magic kernel? Can we generate it?
        return y

    
    def forward(self, x : Tensor) -> Tensor:
        w = self.weight
        x_norm = RMSNorm(x) # Can we avoid the max because we have the norm?
        x_quant = x_norm + (self.activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (self.weight_quant(w) - w).detach() # quantization aware training because of detach, don't need to do this for inference anymore
        y = F.linear(x_quant, w_quant)
        return y
    


class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10, 10)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

a = MyNetwork()
print(a)
_replace_with_custom_fn_if_matches_filter(a, lambda mod: BitLinear.from_float(mod), lambda x: isinstance(x, nn.Linear)) # Need to debug this more
print(a)