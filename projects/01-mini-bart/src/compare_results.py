"""
Compare E0/E1/E2 evaluation summaries.

Run from the Mini-BART project root:
    python src/compare_results.py --results-dir outputs/evaluation

Reads e0_summary.json, e1_summary.json, and e2_summary.json when present.
Writes comparison.csv and metric plots into the results directory.
"""

import argparse
import csv
import json
from pathlib import Path


def load_summaries(results_dir):
    summaries = {}
    for model in ("E0", "E1", "E2"):
        path = results_dir / f"{model.lower()}_summary.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                summaries[model] = json.load(handle)
    if not summaries:
        raise FileNotFoundError(
            f"No *_summary.json files found in {results_dir}"
        )
    return summaries


def flatten_summary(summary):
    row = {
        "model": summary["model"],
        "num_examples": summary["num_examples"],
        "runtime_seconds": summary.get("runtime_seconds"),
        "examples_per_second": summary.get("examples_per_second"),
        "peak_gpu_memory_gb": summary.get("peak_gpu_memory_gb"),
        "mean_compression_ratio": summary.get("mean_compression_ratio"),
    }
    for metric, value in summary.get("rouge", {}).items():
        row[f"rouge_{metric}"] = value
    for ngram, values in summary.get("source_overlap", {}).items():
        row[f"{ngram}_copy_rate"] = values.get("copy_rate")
        row[f"{ngram}_novel_rate"] = values.get("novel_rate")
    for metric, value in summary.get("extractive_fragments", {}).items():
        row[f"fragment_{metric}"] = value
    return row


def write_csv(summaries, output_path):
    rows = [flatten_summary(summaries[key]) for key in ("E0", "E1", "E2") if key in summaries]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_plots(summaries, results_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped plots (CSV was still created).")
        return

    models = [model for model in ("E0", "E1", "E2") if model in summaries]

    def plot_metric(title, values, ylabel, filename):
        plt.figure(figsize=(7, 4.5))
        plt.bar(models, values)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(results_dir / filename, dpi=180)
        plt.close()

    rouge_keys = ("rouge1", "rouge2", "rougeL", "rougeLsum")
    for key in rouge_keys:
        values = [summaries[m].get("rouge", {}).get(key) for m in models]
        if all(value is not None for value in values):
            plot_metric(f"{key.upper()} comparison", values, "ROUGE score", f"{key}_comparison.png")

    for ngram in ("1-gram", "2-gram", "4-gram"):
        values = [summaries[m].get("source_overlap", {}).get(ngram, {}).get("copy_rate") for m in models]
        if all(value is not None for value in values):
            safe = ngram.replace("-", "")
            plot_metric(f"{ngram} source copy rate", values, "Copy rate", f"{safe}_copy_rate.png")

    for key, title, ylabel, filename in [
        ("coverage", "Extractive fragment coverage", "Coverage", "fragment_coverage.png"),
        ("density", "Extractive fragment density", "Density", "fragment_density.png"),
        ("mean_compression_ratio", "Mean compression ratio", "Summary tokens / source tokens", "compression_ratio.png"),
        ("examples_per_second", "Evaluation throughput", "Examples / second", "throughput.png"),
    ]:
        values = [
            summaries[m].get("extractive_fragments", {}).get(key)
            if key in ("coverage", "density")
            else summaries[m].get(key)
            for m in models
        ]
        if all(value is not None for value in values):
            plot_metric(title, values, ylabel, filename)


def main():
    parser = argparse.ArgumentParser(description="Compare E0/E1/E2 evaluation summaries.")
    parser.add_argument("--results-dir", default="outputs/evaluation")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries(results_dir)
    csv_path = results_dir / "comparison.csv"
    write_csv(summaries, csv_path)
    write_plots(summaries, results_dir)

    print(f"Loaded: {', '.join(summaries)}")
    print(f"Comparison table: {csv_path}")
    print("Plots saved in:", results_dir)


if __name__ == "__main__":
    main()
