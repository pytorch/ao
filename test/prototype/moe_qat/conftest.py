import pytest
import torch


@pytest.fixture(autouse=True)
def _init_test():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch._dynamo.reset()
