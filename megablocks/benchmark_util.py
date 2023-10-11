import numpy as np
import torch


def log_benchmark(name, arguments, time, std):
    print("="*60)
    print(f"{name} Benchmark")
    print("Benchmark Parameters:")
    for (key, value) in arguments.items():
        print(f"{key} = {value}")
    print("Results:")
    print("mean time = {:.3f}ms, std time = {:.3f}ms".format(time, std))
    print("="*60)


def benchmark_function(fn, iterations=100, warmup=10):
    # Warmup iterations.
    for _ in range(warmup):
        fn()

    times = []
    for i in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            # for _ in range(5):
                # fn()
        fn()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        # if torch.distributed.get_rank() == 0:
            # prof.export_chrome_trace("trace_ps.json")
        # break
    return np.mean(times), np.std(times)
