# FP6-LLM kernel

This kernel is adapted from https://github.com/usyd-fsalab/fp6_llm. It performs linear op (A @ W.T), where A is in FP16 and W is in FP6 (E3M2 without infinities and NaN).

On most hardware, this kernel is faster than FP16 linear for batch size from 1 to 128, and slower for batch size larger than or equal to 256. See https://github.com/usyd-fsalab/fp6_llm/issues/8 for a detailed discussion.

See https://github.com/pytorch/ao/pull/223 for some benchmark results.
