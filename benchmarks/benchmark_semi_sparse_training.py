# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import itertools

import pandas as pd
import torch
from segment_anything_fast import sam_model_registry
from torch.sparse import to_sparse_semi_structured
from torch.utils import benchmark

from torchao.sparsity.training import (
    SemiSparseLinear,
    swap_linear_with_semi_sparse_linear,
)
from torchao.sparsity.training.autograd import semi_structured_sparsify


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def benchmark_helper(
    functions,
    cases,
    fw: bool = False,
    bw: bool = False,
    cuda_graph: bool = False,
    compile: bool = False,
    blocked_autorange=False,
):
    assert fw or bw
    assert not (cuda_graph and compile)
    print(
        f"Running benchmarks with: fw={fw}, bw={bw}, cuda_graph={cuda_graph}, compile={compile}: "
    )

    results = []

    def handle_case(**case):
        for sparsity_config, benchmark_cls in functions.items():
            result = {
                "sparsity_config": sparsity_config,
            }
            result.update(**case)
            try:
                benchmark_object = benchmark_cls(**case)

                def run_one():
                    if fw:
                        benchmark_object.fw()
                    if bw:
                        benchmark_object.bw()

                if cuda_graph:
                    run_one()
                    benchmark_object = benchmark_cls(**case)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        run_one()

                    def run_one():
                        g.replay()

                if compile:
                    benchmark_object.model = torch.compile(
                        benchmark_object.model, mode="max-autotune"
                    )

                # benchmark
                torch.cuda.reset_peak_memory_stats()
                t0 = benchmark.Timer(
                    stmt="fn()",
                    globals={
                        "fn": run_one,
                    },
                    label="benchmark",
                )
                if blocked_autorange:
                    res = t0.blocked_autorange()
                else:
                    res = t0.adaptive_autorange(0.03, min_run_time=0.2, max_run_time=20)
                result.update(
                    {
                        "time": res.median * 1e3,
                        "memory": torch.cuda.max_memory_allocated() / 1e9,
                    }
                )
            except Exception as e:
                if "CUDA out of memory" not in str(e):
                    raise
                else:
                    result.update({"time": "OOM", "memory": "OOM"})
            finally:
                # clean up
                if "benchmark_object" in locals():
                    del benchmark_object
                if "g" in locals():
                    del g
                gc.collect()
                torch.cuda.empty_cache()
                results.append(result)

    for case in cases:
        handle_case(**case)
    return pd.DataFrame(results)


# test classes for Linear
class LinearTest(torch.nn.Module):
    def __init__(self, mkn):
        super().__init__()
        m, k, n = mkn
        self.model = torch.nn.Linear(k, n).cuda().half()
        self.input = torch.randn(
            [m, k], device="cuda", dtype=torch.half, requires_grad=True
        )
        self.grad = torch.randn([m, n], device="cuda", dtype=torch.half)

    def fw(self):
        self.out = self.model(self.input)

    def bw(self):
        self.out.backward(self.grad, retain_graph=True)


class SemiSparseLinearOfflineCompressionTest(torch.nn.Module):
    def __init__(self, mkn):
        super().__init__()
        m, k, n = mkn
        self.model = torch.nn.Linear(k, n).cuda().half()
        self.model.weight = torch.nn.Parameter(
            to_sparse_semi_structured(self.model.weight)
        )
        self.input = torch.randn(
            [m, k], device="cuda", dtype=torch.half, requires_grad=True
        )
        self.grad = torch.randn([m, n], device="cuda", dtype=torch.half)

    def fw(self):
        self.out = self.model(self.input)


class SemiSparseLinearTest(LinearTest):
    def __init__(self, mkn):
        super().__init__(mkn)
        self.model = SemiSparseLinear.from_dense(self.model)


class SemiSparseKernelTest(LinearTest):
    def __init__(self, mkn):
        super().__init__(mkn)

    def fw(self):
        self.out = semi_structured_sparsify(self.input)

    def bw(self):
        pass


# test class for ViT (SAM image encoder)
class SAMTest(torch.nn.Module):
    def __init__(self, model_type, batch_size):
        super().__init__()
        self.model = (
            sam_model_registry[model_type]().image_encoder.cuda().half().train()
        )
        self.input = torch.randn(
            batch_size,
            3,
            1024,
            1024,
            device="cuda",
            dtype=torch.half,
            requires_grad=True,
        )
        self.grad = torch.randn(
            [batch_size, 256, 64, 64], device="cuda", dtype=torch.half
        )

    def fw(self):
        self.out = self.model(self.input)

    def bw(self):
        self.out.backward(self.grad, retain_graph=True)


class SAM_W24_MLP_ONLY(SAMTest):
    def __init__(self, model_type, batch_size):
        super().__init__(model_type, batch_size)
        # Apply to just MLP linear layers of SAM image encoder (ViT)
        sparse_config = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Linear) and "mlp" in name:
                sparse_config[name] = SemiSparseLinear
        swap_linear_with_semi_sparse_linear(self.model, sparse_config)


class SAM_W24_ALL(SAMTest):
    def __init__(self, model_type, batch_size):
        super().__init__(model_type, batch_size)
        # Apply to all linear layers of SAM image encoder (ViT)
        sparse_config = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                sparse_config[name] = SemiSparseLinear
        swap_linear_with_semi_sparse_linear(self.model, sparse_config)


if __name__ == "__main__":
    print("BENCHMARKING")
    parser = argparse.ArgumentParser(
        description="run semi-structured sparse training benchmarks"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["linear", "llama3-8b", "vit"],
        help="nn.Linear/ViT-e2e benchmarking",
        default="vit",
    )
    parser.add_argument("--save", action="store_true", help="save benchmarking results")
    args = parser.parse_args()
    if args.mode == "linear":
        functions = {
            "dense_linear": LinearTest,
            "semi_sparse_linear": SemiSparseLinearTest,
            "semi_sparse_prune+compress_time_only": SemiSparseKernelTest,
        }
        cases = list(
            product_dict(
                mkn=[
                    # DINO ViT-L mlp.lin1
                    (13008, 1024, 4096),
                    # DINO ViT-L mlp.lin2
                    (13008, 4096, 1024),
                ],
            )
        )

        df = benchmark_helper(
            functions, cases, fw=True, bw=True, cuda_graph=True, blocked_autorange=True
        )
    elif args.mode == "llama3-8b":
        functions = {
            "dense_linear": LinearTest,
            "semi_sparse_linear": SemiSparseLinearOfflineCompressionTest,
        }
        batch_size = 16
        cases = list(
            product_dict(
                mkn=[
                    # attn q and o
                    (batch_size, 4096, 4096),
                    # attn k and v
                    (batch_size, 4096, 1024),
                    # mlp up and gate
                    (batch_size, 4096, 14336),
                    # mlp down
                    (batch_size, 14336, 4096),
                ],
            )
        )

        df = benchmark_helper(
            functions, cases, fw=True, bw=False, cuda_graph=True, blocked_autorange=True
        )

    elif args.mode == "vit":
        functions = {
            "ViT dense (baseline)": SAMTest,
            "ViT MLP weight 2:4 sparse": SAM_W24_MLP_ONLY,
            # "ViT all(MLP+ATN) Linear weight 2:4 sparse": SAM_W24_ALL
        }
        cases = list(product_dict(model_type=["vit_l"], batch_size=[8]))

        df = benchmark_helper(functions, cases, fw=True, bw=True, compile=True)

    print(df)
    if args.save:
        df.to_csv(f"{args.mode}_semi_structured_training_benchmarks.csv")
