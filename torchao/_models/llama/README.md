# Llama Benchmarks

The llama folder contains code/scripts for stable benchmarking llama models.

To get model weights, go to https://huggingface.co/meta-llama/Llama-2-7b and/or https://huggingface.co/meta-llama/Meta-Llama-3-8B
and follow the steps to gain access.

Then from the torchao root directory use `huggingface-cli login` and follow the steps to login, then `sh ./scripts/prepare.sh` to
download and convert the model weights

once done you can execute benchmarks from the torchao/_models/llama dir with `sh benchmarks.sh`. You can perform and benchmarking
directly using `generate.py`.
