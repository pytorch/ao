import torch


def benchmark_model(model, num_runs, input_tensor):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
    # benchmark
    for _ in range(num_runs):
        with torch.autograd.profiler.record_function("timed region"):
            model(input_tensor)
    
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs

def profiler_runner(path, fn, *args, **kwargs):
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True) as prof:
        result = fn(*args, **kwargs)
    prof.export_chrome_trace(path)
    return result
