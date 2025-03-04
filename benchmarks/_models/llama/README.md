# Llama Benchmarks

The llama folder contains code/scripts for stable benchmarking llama models.

To get model weights, go to https://huggingface.co/meta-llama/Llama-2-7b, https://huggingface.co/meta-llama/Meta-Llama-3-8B, https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
and follow the steps to gain access.

Then from the torchao root directory use `huggingface-cli login` and follow the steps to login, then `sh ./scripts/prepare.sh` to
download and convert the model weights

once done you can execute benchmarks from the benchmarks/_models/llama dir with `sh benchmarks.sh`. You can perform and benchmarking or evaluation
directly using `generate.py` or `eval.py`.

## KV Cache Quantization - Memory Efficient Inference
We've added some features to `model.py` compared to the original gpt-fast implementation in order to enable long context length (and necessarily memory efficient) inference. Specifically we've added kv_cache quantization and a linear_causal_mask implementation which are **able to reduce memory usage by 50-60%** at long context lengths.

In practice these features alongside int4 weight only quantization allow us to do Llama3.1-8B inference with a **130k context length with only 18.9 GB of peak memory.**

You can check it out yourself with `generate.py`, these features exist as a proof of concept and technical demonstration of the techniques though we're working to figure out a way to release them in a general way. Until then feel free to copy these features into your own models. The details and a full explanation can be found in this [PR](https://github.com/pytorch/ao/pull/738)

To see how these techniques scale generally we've run `generate.py` with subsets of these features for different context lengths on an A100 GPU. You can find commands to reproduce these numbers in `benchmarks.sh`

| context length (tokens) | normal peak (GB) | kv_quant peak (GB) | kv quant+linear_causal_mask peak (GB) |
|-------------------------|------------------|--------------------|---------------------------------------|
|                    8192 |            17.86 |              17.52 |                                 17.47 |
|                   16384 |            19.81 |              18.75 |                                 18.48 |
|                   32768 |            23.83 |              21.72 |                                 20.64 |
|                   65536 |             33.5 |              29.54 |                                 25.24 |
|                  131072 |            59.27 |              52.62 |                                 34.18 |

## Adding Benchmarks For New Techniques

If you want to add benchmarks that you think should be kept up to date, please try to keep the format consistent. For performance focused techniques (e.g. if they require fine-tuning or something else) add an option to run them in generate.py and an execution command in benchmarks.sh in the relevant section. If its a technique that's still in development, add it in the section for `OTHER BENCHMARKS` if there's a finalized api and you want those numbers in the main quantization README, add them in the `README BENCHMARKS` section. For accuracy focused techniques, add them in eval.py and evaluations.sh in a similar vein. Ideally techniques in the main readme will have both benchmarks and evaluations set up here so they can be monitored and reproduced easily.
