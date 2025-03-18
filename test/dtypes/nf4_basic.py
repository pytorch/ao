import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

import torchao
from packaging import version
from torchao.dtypes.nf4tensor import (
    _INNER_TENSOR_NAMES_FOR_SHARDING,
    NF4Tensor,
    linear_nf4,
    nf4_weight_only,
    to_nf4,
)


def _build_input_weight(embed_dim: int, device: torch.device, dtype: torch.dtype):
    torch.manual_seed(0)
    input_weight = torch.empty(embed_dim, embed_dim, device=device, dtype=dtype)
    input_weight.normal_(0, 1)
    return input_weight

torch.manual_seed(0)
dim = 512
device = "xpu"
dtype = torch.bfloat16
input_weight = _build_input_weight(dim, device, dtype)
nf4_weight = to_nf4(input_weight)

inp = torch.randn(2, 512, dtype=dtype, device=device)

out_nf4 = linear_nf4(inp, nf4_weight).sum()
