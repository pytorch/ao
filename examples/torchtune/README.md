# torchtune Examples
Examples to tune language models using [torchtune](https://github.com/pytorch/torchtune).

## Setup
1. Follow the [torchao Installation](../../README.md#installation) steps.

2. Install `torchtune`:
```
pip install torchtune
```

## Run
1. Download a model (see more details [here](https://github.com/pytorch/torchtune#downloading-a-model)):
```
tune download meta-llama/Llama-3.2-1B-Instruct --output-dir /tmp/Llama-3.2-1B-Instruct --ignore-patterns "original/consolidated.00.pth"
```

2. Finetune:
- To finetune on a single device:
```
tune run --nproc_per_node 1 full_finetune_single_device.py --config ./configs/full_finetune.yaml
```

- To finetune on multiple GPUs:
```
tune run --nproc_per_node 8 full_finetune_distributed.py --config ./configs/full_finetune.yaml
```