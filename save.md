# Grouped Triton Torch Oracles

## Current State

- Added an exact pure-Torch `abs().amax()` / BF16 RHT oracle to the small grouped amax correctness test while retaining the per-expert Triton twin comparison.
- Added an `nvfp4_quantize` oracle for grouped 2D weight rowwise codes and blocked scales. Scales and signs match exactly; FP4 magnitudes may differ by at most one index because the two implementations order reciprocal/multiply operations differently.
- No production code or public interfaces changed.

## Validation

- Full focused modules: 27 passed, 1 initial new-test failure in 267.86s. The failure was isolated to 16 of 32,768 FP4 nibbles; scales and signs matched exactly and every magnitude difference was one step.
- After applying the supported tolerance: both new oracle tests passed in 16.08s.
- `git diff --check` passed.
- Detailed failure evidence is recorded in `debug-session.md`.

## Next Action

Review and commit the test-only changes. Expected outcome: the grouped kernels retain twin parity coverage while each now has an independent Torch correctness anchor.

Confidence: HIGH  
Risk: LOW — the only tolerance permits the established one-step FP4 rounding difference while exact scale and sign assertions still detect layout, packing, or scale regressions.

Surgical Simplicity: The new quantization test protects the previously missing independent-oracle invariant; `debug-session.md` records the required failed-validation experiment, and this `save.md` is required because the run touched three files and crossed a debug transition. No helper or production abstraction was added.
