# Debug Session

## 2026-07-24 — Grouped 2D weight Torch oracle

- Hypothesis: The grouped kernel and `nvfp4_quantize` use equivalent scales but slightly different floating-point operation ordering, so values at FP4 rounding boundaries can differ by one magnitude step.
- Exact command: `git diff --check && python -m pytest -q test/prototype/moe_training/nvfp4_training/test_group_hadamard_amax_triton.py test/prototype/moe_training/nvfp4_training/test_group_quantize_2d_triton.py`
- Result: 27 passed, 1 failed. `test_group_quantize_2d_matches_torch_oracle` had 16 mismatched packed bytes out of 16,384 (0.1%); greatest byte difference was 16, consistent with a one-step difference in one nibble.
- Interpretation: The fixture aligns the scale domains, but exact packed-code equality is too strict until scale equality and unpacked FP4 sign/magnitude deltas are checked independently.
- Canonical-command status: Focused canonical pytest invocation for the two changed modules.
- Failure classification: Numeric reference-tolerance mismatch in the new test; no existing test or production failure.
- Ranked competing hypothesis: A blocked-scale or packed-code layout mismatch would produce widespread, structured differences rather than 0.1% isolated bytes.
- Next experiment: Run only the new test logic and report scale equality plus unpacked sign and magnitude-index differences.

### Experiment result

- Exact action: Reproduced the new two-expert fixture and compared scales plus unpacked FP4 nibbles independently.
- Result: Both experts' blocked scales matched bitwise. Expert 0 codes matched bitwise. Expert 1 had 16 differing nibbles, all with identical signs and magnitude-index distance exactly one; no difference exceeded one step.
- Interpretation: Confirms floating-point rounding-order sensitivity, not a scale, packing, or layout error.
- Supported fix: Keep exact scale equality and assert code signs match with magnitude indices at most one step apart, matching the existing RHT quantization test convention.
