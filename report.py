def export_report(results):
    with open("benchmark_report.md", "w") as f:
        f.write("# PyTorch MicroBenchmark Suite Report\n\n")
        f.write("| Operation | Time (sec) |\n")
        f.write("|-----------|------------|\n")

        for name, t in results.items():
            f.write(f"| {name} | {t:.6f} |\n")

    print("Saved: benchmark_report.md")
