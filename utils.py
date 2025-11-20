import time
import torch

def time_function(fn, repeat=5):
    times = []
    for _ in range(repeat):
        start = time.time()
        fn()
        end = time.time()
        times.append(end - start)
    return sum(times) / repeat

def check_gpu():
    return torch.cuda.is_available()
