
## Performance Profiling Example

An minimal reproduction of `gpt-fast` that demonstrates usage of `torchao.profiler.TransformerPerformanceCounter`.

## Usage
```python
python generate.py --prompt "Hello my name is" --checkpoint_path path/to/model.pth --num_samples 1 --max_new_tokens 2 --save_path performance_stats.json
```
where `checkpoint_path` is the checkpoint path of the converted model weights per `gpt-fast` and `save_path` specifies where to save performance stats.


Running the above command for `llama2-7b` should print the following, with accumulated stats saved to `performance_stats.json`

```
Loading model ...
Time to load model: 20.14 seconds

==============================

Using DeviceSpec(device_type=cuda, name=NVIDIA GeForce RTX 3090, dtype=torch.bfloat16, bandwidth=936.1GB/s, flops=35.6TFLOPs, vram=25.4GB)
Model Config: ModelArgs(block_size=2048, vocab_size=32000, n_layer=32, n_head=32, dim=4096, intermediate_size=11008, n_local_heads=32, head_dim=128, rope_base=10000, norm_eps=1e-05)
Active params, Total Params: 6607343616, 6738415616

==============================

TransformerPerfCounter Metrics
PREFILL_SEQLEN-6:
  Latency = 1.26 s
  Tokens
    Total: 6 tokens
    Throughput: 5 tokens/s
  IO
    Total: 13.25 GB
    Throughput: 10.54 GB/s
    Theoretical Latency: 14.15 ms
  FLOPs 
    Total: 79.31 GFLOPs
    Throughput: 63.06 GFLOPs/s
    Theoretical Latency: 2.23 ms
  Utilization
    Bandwidth: 0.0113 %
    FLOPs: 0.0018 %

==============================

TransformerPerfCounter Metrics
DECODE_CTX-6_NUM_TOKS-1:
  Latency = 0.16 s
  Tokens
    Total: 1 tokens
    Throughput: 6 tokens/s
  IO
    Total: 13.22 GB
    Throughput: 83.27 GB/s
    Theoretical Latency: 14.13 ms
  FLOPs 
    Total: 13.22 GFLOPs
    Throughput: 83.24 GFLOPs/s
    Theoretical Latency: 0.37 ms
  Utilization
    Bandwidth: 0.0890 %
    FLOPs: 0.0023 %

==============================

Generated text for sample 0: Hello, my name is [Name

GPTFast Sample Metrics
  Time for inference 1: 6 prompt tokens 2 tokens generated, 1.57 sec total, 1.28 tokens/sec
  Bandwidth achieved: 17.22 GB/s

==============================

GPTFast Aggregate Stats
  Average tokens/sec: 1.28
  Memory used: 13.51 GB

==============================

TransformerPerfCounter
Performance Summary:
  Latency = 1.42 s
  Tokens
    Total: 7 tokens
    Throughput: 5 tokens/s
  IO
    Total: 26.47 GB
    Throughput: 18.69 GB/s
    Theoretical Latency: 28.28 ms
  FLOPs 
    Total: 92.53 GFLOPs
    Throughput: 65.33 GFLOPs/s
    Theoretical Latency: 2.60 ms
  Utilization
    Bandwidth: 0.0200 %
    FLOPs: 0.0018 %

Saving performance results to performance_stats.json
```

**Notes**
- `generate.py` script is a stripped down version of the original `gpt-fast` script and currently does not support quantization, tensor parallelism, and speculative decoding, as the primary purpose is to demonstrate basic usage of the performance tracker.
- The discrepancy between `gpt-fast` token throughput and that of `TransformerPerformanceCounter` is due to the fact that `gpt-fast` does not account for all prefill tokens 
  - `gpt-fast` only counts generated tokens -- so even though `prefill` technically generated `len(prompt) + 1` tokens, it counts the number of tokens generated during this phase as `1`, whereas `TransformerPerformanceCounter` includes all `prefill` tokens in the total token count.