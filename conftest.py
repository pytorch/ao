import os
import pytest
import subprocess

def get_free_gpus():
    try:
        # Use nvidia-smi to get the GPU utilization
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
        raise RuntimeError("No free GPUs available")

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    config.pluginmanager.register(set_cuda_visible_devices)
