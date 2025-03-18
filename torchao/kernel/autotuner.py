# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import pathlib

import torch
import triton

AUTOTUNER_DATA_PATH = os.getenv("TORCHAO_AUTOTUNER_DATA_PATH", None)


def do_bench_triton(
    fn,
    warmup=25,
    rep=100,
    grad_to_none=None,
    quantiles=None,
    fast_flush=True,
    return_mode="mean",
):
    assert return_mode in ["min", "max", "mean", "median"]
    import torch

    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float]
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    """

    fn()
    torch.cuda.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float
    )
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()


BEST_CONFIGS = None


def _save_best_configs(best_configs):
    device_name = torch.cuda.get_device_name()
    if AUTOTUNER_DATA_PATH is None:
        saved_configs = pathlib.Path.cwd() / "data.pkl"
    else:
        saved_configs = pathlib.Path(AUTOTUNER_DATA_PATH)
    logging.info(
        f"Trying to store configs for {device_name} locally under {saved_configs}"
    )
    with open(saved_configs, "wb") as f:
        import pickle

        logging.info(f"Saving best configs to file {saved_configs}")
        pickle.dump(best_configs, f)


def _load_best_configs():
    device_name = torch.cuda.get_device_name()
    import importlib

    if AUTOTUNER_DATA_PATH is None:
        saved_configs = importlib.resources.files("torchao")
        saved_configs = saved_configs / "kernel" / "configs" / "data_a100.pkl"
        if not device_name.startswith("NVIDIA A100"):
            logging.info("Warning! Loaded configurations are optimized for A100!")
    else:
        saved_configs = pathlib.Path(AUTOTUNER_DATA_PATH)
    logging.info(f"Trying to load configs for {device_name} from {saved_configs}")
    if saved_configs.is_file():
        import pickle

        with open(saved_configs, "rb") as f:
            logging.info(f"Loading best configs from file {saved_configs}")
            return pickle.load(f)


def get_arg_key(a):
    if torch.is_tensor(a):
        return (a.dtype, a.size(), a.stride())
    return (a,)


def get_args_key(args):
    return sum(tuple(get_arg_key(a) for a in args), ())


def do_bench_basic(fn, rep):
    # Modified version of Triton's basic bench
    fn()
    torch.cuda.synchronize()
    # Fast flush
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    cache.zero_()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(rep):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / rep
    return estimate_ms


def do_bench(fn, args, config, best_time=None):
    # TODO: CUDA graph compatible version
    def wrapped_fn():
        return fn(*(args + [config]))

    # Get fast estimate to abort stupid configs

    # Run it once and skip if it crashes or is 100x slower
    try:
        time = do_bench_basic(wrapped_fn, 1)
    except RuntimeError:
        time = None
    except triton.runtime.OutOfResources:
        time = None
    if time is None or (best_time is not None and time > best_time * 100):
        return float("inf")

    # Run it five times and skip if it is 10x slower
    time = do_bench_basic(wrapped_fn, 5)
    if best_time is not None and time > best_time * 10:
        return float("inf")

    # Do a regular bench
    return do_bench_triton(wrapped_fn)


def get_best_config_by_key(key):
    if key in BEST_CONFIGS:
        return BEST_CONFIGS[key][0]


def get_best_config_fn(fn, args, configs):
    global BEST_CONFIGS
    if BEST_CONFIGS is None:
        BEST_CONFIGS = _load_best_configs()

    # This means no config file was loaded
    if BEST_CONFIGS is None:
        BEST_CONFIGS = {}

    if len(configs) == 0:
        return None

    key = get_args_key(args)
    best_config = get_best_config_by_key(key)
    if best_config is not None:
        return best_config

    logging.info(f"Starting autotune search. No config found for key {key}.")

    # Search for the best config
    best_config = configs[0]
    best_time = do_bench(fn, args, configs[0])
    logging.info(" ".join(map(str, [key, best_time, best_config])))
    i = 1
    # TODO: Instead of walking this in order, a random selection
    # is maybe better to end up with a reasonable config that can be
    # used to filter bad configs sooner.
    for config in configs[1:]:
        time = do_bench(fn, args, config, best_time)
        logging.info(
            " ".join([f"{i:4d}/{len(configs):4d}", f"{time:6.3f}", str(config)])
        )
        if time < best_time:
            best_time = time
            best_config = config
        i += 1
    # Also store time, so it can be proven that the config works
    BEST_CONFIGS[key] = (best_config, best_time)
    logging.info("-- perfetto --")
    logging.info(" ".join(map(str, [best_time, best_config])))
    _save_best_configs(BEST_CONFIGS)
    return best_config
