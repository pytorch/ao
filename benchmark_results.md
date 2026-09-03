# GrokAdamW Benchmark Results

**Setup:**
- Model: Simple Transformer (1,040,104 parameters)
- Device: cpu
- Training Steps: 100
- Batch Size: 4
- Sequence Length: 32
- Vocabulary Size: 1000

| Optimizer | Final Loss | Total Time (s) | Time/Step (ms) | Speedup vs Adam |
|-----------|-----------|----------------|----------------|-----------------|
| Adam | 7.0207 | 0.45 | 33.12 | - |
| GrokAdamW8bit | 6.9850 | 32.59 | 2506.57 | 0.01x |