import warnings

from torchao.optim.adam import Adam4bit, Adam8bit, AdamFp8, AdamW4bit, AdamW8bit, AdamWFp8, _AdamW
from torchao.optim.cpu_offload import CPUOffloadOptimizer

warnings.warn(
    "We have moved to torchao.optim! Please migrate your code to this new path. "
    "torchao.prototype.low_bit_optim will be removed in future versions."
)
