import argparse
import numpy as np
import torch
from utils import time_function, check_gpu
from visualize import generate_plot
from report import export_report

def numpy_matmul(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    return lambda: A @ B

def torch_matmul(n, device):
    A = torch.rand(n, n, device=device)
    B = torch.rand(n, n, device=device)
    return lambda: A @ B

def run_benchmarks(size, device):
    results = {}

    # NumPy baseline
    numpy_time = time_function(numpy_matmul(size))
    results["numpy_matmul"] = numpy_time

    # PyTorch
    torch_time = time_function(torch_matmul(size, device))
    results[f"torch_matmul_{device}"] = torch_time

    return results

def main():
    parser = argparse.ArgumentParser(description="PyTorch MicroBenchmark Suite")
    parser.add_argument("--size", type=int, default=500, help="Matrix size NxN")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--export", action="store_true", help="Export markdown report")
    args = parser.parse_args()

    device = "cuda" if args.device == "cuda" and check_gpu() else "cpu"

    print(f"\nRunning benchmarks with size={args.size} on device={device}...\n")
    
    results = run_benchmarks(args.size, device)

    # Generate plot
    generate_plot(results)

    if args.export:
        export_report(results)

if __name__ == "__main__":
    main()
