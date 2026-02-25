#!/bin/bash

python benchmarks/prototype/attention/eval_flux_model.py --baseline fa2 --test fa3 --compile
python benchmarks/prototype/attention/eval_flux_model.py --baseline fa3 --test fa3_fp8 --compile
python benchmarks/prototype/attention/eval_flux_model.py --baseline fa3 --test fa3_fp8 --no_fuse_rope --compile
