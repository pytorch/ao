import pytest

import torch
from torch import nn
from torch.nn import functional as F

from torchao.quantization.quant_api import quantize_
from torchao.prototype.scaled_grouped_mm.conversion_utils import MoETrainingConfig
from torchao.float8.float8_utils import compute_error

# model definition from torchtitan:
# https://github.com/pytorch/torchtitan/blob/768cde131105bde624160029d808e94649faf0f4/torchtitan/experiments/llama4/model/moe.py#L14
class GroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.use_grouped_mm = use_grouped_mm
        self.init_weights()

    def forward(
        self,
        x: torch.Tensor,
        num_local_tokens_per_expert: torch.Tensor | list[int] | None = None,
    ) -> torch.Tensor:
        # TODO: keeping this for loop implementation for comparison
        #       and readability, will remove later
        if not self.use_grouped_mm:
            if num_local_tokens_per_expert is not None:
                # a tuple of tensors indexed by experts
                # each with shape (tokens_per_expert(varying), dim)
                x = torch.split(
                    x,
                    split_size_or_sections=num_local_tokens_per_expert,
                    dim=0,
                )
                out_experts_splits = []
                for expert_idx, x_expert in enumerate(x):
                    w1, w2, w3 = (
                        self.w1[expert_idx],
                        self.w2[expert_idx],
                        self.w3[expert_idx],
                    )
                    h = F.silu(torch.matmul(x_expert, w1))
                    h = h * torch.matmul(x_expert, w3)
                    h = torch.matmul(h, w2)
                    # h shape (tokens_per_expert(varying), dim)
                    out_experts_splits.append(h)
                out = torch.cat(out_experts_splits, dim=0)
            else:
                # x shape (num_experts, tokens_per_expert, dim)
                h = F.silu(torch.bmm(x, self.w1))
                h = h * torch.bmm(x, self.w3)
                # out shape (num_experts, tokens_per_expert, dim)
                out = torch.bmm(h, self.w2)
            return out
        # grouped mm implementation
        if num_local_tokens_per_expert is not None:
            # https://github.com/pytorch/pytorch/pull/150374
            # NOTE: torch._gouped_mm requires bf16 dtypes
            #       and shapes to be multiple of 8
            offsets = torch.cumsum(
                num_local_tokens_per_expert, dim=0, dtype=torch.int32
            )
            # grouped mm between a 2D tensor and a 3D tensor
            assert x.dim() == 2
        else:
            offsets = None
            # fall back to regular bmm between 3D tensors
            assert x.dim() == 3
        assert (
            x.dtype == self.w1.dtype == self.w2.dtype == self.w3.dtype == torch.bfloat16
        ), "torch._grouped_mm only supports bf16 dtypes"
        h = F.silu(torch._grouped_mm(x, self.w1, offs=offsets))
        h = h * torch._grouped_mm(x, self.w3, offs=offsets)
        out = torch._grouped_mm(h, self.w2, offs=offsets)
        return out

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)

class MoE(nn.Module):
    """Toy MoE for testing. Not a complete implementation."""
    def __init__(self, 
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.experts = GroupedExperts(
            dim,
            hidden_dim,
            num_experts,
            use_grouped_mm,
        )
        #self.init_weights()

    def forward(self, x: torch.Tensor, num_local_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        return self.experts(x, num_local_tokens_per_expert=num_local_tokens_per_expert)

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)

@pytest.mark.parametrize(
    "model_class,target_fqns", [
        (MoE, ["experts"]),     # calling quantize_ on higher level module 
        (GroupedExperts, [""]), # calling quantize_ on experts directly
    ])
def test_moe_float8_training(model_class: nn.Module, target_fqns: list[str]):
    batch, seq, dim = 1, 8192, 4096
    num_experts, top_k = 2, 1

    def moe_module_filter_fn(mod: nn.Module, cur_fqn: str) -> bool:
        for target_fqn in target_fqns:
            if target_fqn in cur_fqn:
                return True
        return False

    # define MoE layer
    torch.manual_seed(42)
    model = model_class(dim=dim, hidden_dim=4*dim, num_experts=num_experts, use_grouped_mm=True).to(torch.bfloat16).cuda()
    torch.manual_seed(42)
    ref_model = model_class(dim=dim, hidden_dim=4*dim, num_experts=num_experts, use_grouped_mm=True).to(torch.bfloat16).cuda()
    for param1, param2 in zip(model.parameters(), ref_model.parameters()):
        assert torch.equal(param1, param2)

    # convert MoE to float8 training
    config = MoETrainingConfig()
    quantize_(model, config=config, filter_fn=moe_module_filter_fn)

    # inputs
    torch.manual_seed(42)
    x = torch.randn(batch*seq*top_k, dim, dtype=torch.bfloat16, requires_grad=True).cuda()
    torch.manual_seed(42)
    ref_x = torch.randn(batch*seq*top_k, dim, dtype=torch.bfloat16, requires_grad=True).cuda()

    # offsets
    tokens_per_expert = torch.tensor([batch*seq*top_k // num_experts], dtype=torch.int32).repeat(num_experts).cuda()
    ref_tokens_per_expert = tokens_per_expert.clone()

    # forward pass
    out = model(x, num_local_tokens_per_expert=tokens_per_expert)
    ref_out = ref_model(ref_x, num_local_tokens_per_expert=ref_tokens_per_expert)
    
    
    # validate SQNR is acceptable.
    # a single fp8 gemm uses SQNR >= 25.0 for testing, so for a full MoE layer
    # we'll use a slightly lower threshold.
    out_sqnr = compute_error(out, ref_out)
    assert out_sqnr.item() >= 23.0, f"SQNR must be >= 23.0, got {out_sqnr.item()}."

    # backward pass
    out.sum().backward()
    ref_out.sum().backward()

    # validate input gradients
    assert torch.allclose(x.grad, ref_x.grad)

    # validate param gradients
    for param1, param2 in zip(model.parameters(), ref_model.parameters()):
        assert torch.allclose(param1.grad, param2.grad)
