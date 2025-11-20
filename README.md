# PyTorch MicroBenchmark Suite

A lightweight tool comparing NumPy and PyTorch execution speeds across CPU/GPU.

## Features
- NumPy vs PyTorch performance comparison
- CPU vs GPU benchmarking
- Bar chart visualization
- Markdown report generation

## Usage

```bash
python benchmark.py --size 500 --device cpu
python benchmark.py --size 800 --device cuda
python benchmark.py --size 600 --export
