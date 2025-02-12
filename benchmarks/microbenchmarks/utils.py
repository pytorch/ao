import torch


class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=64, n=32, k=64, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.linear1 = torch.nn.Linear(k, n, bias=False).to(dtype)

    def example_inputs(self, m=1, device="cuda"):
        return (torch.randn(m, self.linear1.in_features, dtype=self.dtype, device=device),)

    def forward(self, x):
        x = self.linear1(x)
        return x


def get_default_device() -> str:
    return (
        "cuda" if torch.cuda.is_available() else
        "xpu" if torch.xpu.is_available() else
        "cpu"
    )
