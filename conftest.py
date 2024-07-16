import os
import pytest
import subprocess

SEQUENTIAL_FILES = [
    "test/integration/test_integration.py",
    "test/test_ops.py",
    "test/prototype/test_quant_llm.py",
]

def get_free_gpus():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                                capture_output=True, text=True)
        gpu_memory = [int(x) for x in result.stdout.strip().split('\n')]
        free_gpus = [i for i, mem in enumerate(gpu_memory) if mem == 0]
        return free_gpus
    except Exception as e:
        print(f"Error getting GPU status: {e}")
        return []

@pytest.fixture(scope='session', autouse=True)
def set_cuda_visible_devices(worker_id):
    free_gpus = get_free_gpus()
    if free_gpus:
        gpu_id = free_gpus[worker_id % len(free_gpus)]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    config.pluginmanager.register(set_cuda_visible_devices)

def pytest_collection_modifyitems(config, items):
    parallel_items = []
    sequential_items = []
    for item in items:
        if any(sequential_file in item.nodeid for sequential_file in SEQUENTIAL_FILES):
            item.add_marker(pytest.mark.sequential)
            sequential_items.append(item)
        else:
            parallel_items.append(item)
    config.parallel_items = parallel_items
    config.sequential_items = sequential_items

@pytest.hookimpl(trylast=True)
def pytest_sessionstart(session):
    config = session.config
    items = config.parallel_items + config.sequential_items
    session.items = items
