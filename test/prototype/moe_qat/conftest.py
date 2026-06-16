import gc

import pytest
import torch


@pytest.fixture(autouse=True)
def _init_test():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch._dynamo.reset()
    torch.compiler.reset()

    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def mock_distributed_env(monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "12355")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
