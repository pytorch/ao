import pytest
import torch
from torch import nn

from torchao.float8.float8_utils import compute_error
from torchao.prototype.scaled_grouped_mm.conversion_utils import MoETrainingConfig
from torchao.quantization.quant_api import quantize_

try:
    from torchtitan.experiments.llama4.model.args import TransformerModelArgs
    from torchtitan.experiments.llama4.model.moe import MoE
except ImportError:
    import warnings

    warnings.warn("torchtitan not installed, skipping MoE tests.")
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize(
    "target_fqns",
    ["experts"],
)
def test_moe_float8_training(target_fqns: list[str]):
    model_args = TransformerModelArgs(moe_enabled=True, num_experts=2)
    init_std = 0.2
    device = torch.device("cuda")

    # reference bf16 MoE
    ref_model = MoE(model_args).to(torch.bfloat16).cuda()
    torch.manual_seed(42)
    ref_model.init_weights(init_std, device)

    # target MoE for testing conversion
    model = MoE(model_args).to(torch.bfloat16).cuda()
    torch.manual_seed(42)
    model.init_weights(init_std, device)

    # assert starting params are identical for both models
    for param1, param2 in zip(model.parameters(), ref_model.parameters()):
        assert torch.equal(param1, param2)

    # convert MoE to float8 training
    def moe_module_filter_fn(mod: nn.Module, cur_fqn: str) -> bool:
        for target_fqn in target_fqns:
            if target_fqn in cur_fqn:
                return True
        return False

    config = MoETrainingConfig()
    quantize_(model, config=config, filter_fn=moe_module_filter_fn)

    # inputs
    batch, seq, dim = 1, 8192, 4096
    x = torch.randn(batch, seq, dim, dtype=torch.bfloat16, requires_grad=True).cuda()
    ref_x = x.clone()

    # forward pass
    out = model(x)
    ref_out = ref_model(ref_x)

    # validate SQNR between outputs is acceptable.
    # a single fp8 gemm uses SQNR >= 25.0 for testing, so for a full MoE layer
    # we'll use a slightly lower threshold.
    out_sqnr = compute_error(out, ref_out)
    assert out_sqnr.item() >= 23.0, f"SQNR must be >= 23.0, got {out_sqnr.item()}."

    # backward pass
    ref_out.sum().backward()
    out.sum().backward()

    # validate input gradients
    input_grad_sqnr = compute_error(x.grad, ref_x.grad)
    assert input_grad_sqnr.item() >= 23.0, (
        f"SQNR must be >= 23.0, got {input_grad_sqnr.item()}."
    )

    # validate param gradients
    for param1, param2 in zip(model.parameters(), ref_model.parameters()):
        param_grad_sqnr = compute_error(param1.grad, param2.grad)
        assert param_grad_sqnr.item() >= 23.0, (
            f"SQNR must be >= 23.0, got {param_grad_sqnr.item()}."
        )
