import torch
import torch.nn as nn
from torchao.quantization import quantize_, int8_weight_only, int4_weight_only
from torchao.quantization.utils import compute_error
from torchao.quantization.prototype.mixed_precision.naive_intNwo import intN_weight_only
