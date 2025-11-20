import matplotlib.pyplot as plt

def generate_plot(results):
    names = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.ylabel("Execution Time (seconds)")
    plt.title("NumPy vs PyTorch Benchmark Performance")
    plt.xticks(rotation=25)
    plt.tight_layout()

    plt.savefig("benchmark_results.png")
    print("Saved: benchmark_results.png")
