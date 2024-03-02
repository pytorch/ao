import torch
import triton
import pickle
import logging

BEST_CONFIGS = None

logging.basicConfig(level=logging.INFO)


def _save_best_configs(best_configs):
    with open('data.pkl', 'wb') as f:
        pickle.dump(best_configs, f)
    logging.info('Saved configs in data.pkl')


def _load_best_configs():
    from pathlib import Path
    filename = Path('data.pkl')
    if filename.is_file():
        with open('data.pkl', 'rb') as f:
            logging.info('Loading configs in data.pkl')
            return pickle.load(f)


def get_arg_key(a):
    if torch.is_tensor(a):
        return (a.dtype, a.size(), a.stride())
    return (a,)


def get_args_key(args):
    return sum(tuple(get_arg_key(a) for a in args), ())


def do_bench(fn, args, config):
    return triton.testing.do_bench(lambda: fn(*(args + [config])))


def get_best_config_fn(fn, args, configs):
    global BEST_CONFIGS
    if BEST_CONFIGS is None:
        BEST_CONFIGS = _load_best_configs()
    # This means no config file was found
    if BEST_CONFIGS is None:
        BEST_CONFIGS = {}

    if len(configs) == 0:
        return None
    best_config = configs[0]
    best_time = do_bench(fn, args, configs[0])
    key = get_args_key(args)
    if key in BEST_CONFIGS:
        return BEST_CONFIGS[key][0]
    print(key, best_time, best_config)
    for config in configs[1:]:
        time = do_bench(fn, args, config)
        print(time, config)
        if time < best_time:
            best_time = time
            best_config = config
    # Also store time, so it can be proven that the config works
    BEST_CONFIGS[key] = (best_config, best_time)
    print("-- perfetto --")
    print(best_time, best_config)
    _save_best_configs(BEST_CONFIGS)
    return best_config
