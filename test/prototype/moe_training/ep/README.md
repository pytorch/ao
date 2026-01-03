# EP Autograd Functions Tests

Unit tests for the Expert Parallelism (EP) autograd functions with MXFP8 quantization.

## Test Files

- **test_a2a_dispatch.py**: Tests for the pink a2a_dispatch function
- **test_permute.py**: Tests for the green permute function
- **test_unpermute.py**: Tests for the purple unpermute function
- **test_a2a_combine.py**: Tests for the blue a2a_combine function
- **test_integration.py**: Integration test for the full pipeline

## Running Tests

Run individual test files:
```bash
pytest test/prototype/moe_training/ep/test_permute.py
pytest test/prototype/moe_training/ep/test_a2a_dispatch.py
```

Run all EP tests:
```bash
pytest test/prototype/moe_training/ep/
```

## Requirements

- Tests with distributed collectives (a2a_dispatch, a2a_combine, integration) require multi-GPU setup
- Single-node tests (permute, unpermute) can run on CPU or single GPU
