## gpt-fused

A more handwritten version of [gpt-fast](https://github.com/pytorch-labs/gpt-fast)'s model.py for us to experiment with.

Requires the use of gpt-fast to use, but is a drop-in replacement.

To use it just set to PYTHONPATH environment variable to `PYTHONPATH=<path to ao repository>/ao/torchao/prototype/models/gpt_fused` and delete gpt-fast's model.py.

For example

```
PYTHONPATH=/home/cpuhrsch/local/ao/torchao/prototype/models/gpt_fused CUDA_VISIBLE_DEVICES=0 numactl --membind 0 --cpubind 0 python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is"
```
