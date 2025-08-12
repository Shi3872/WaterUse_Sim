import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

colors = {
    "Stylized Game": "red",
    "Parametric Game": "blue",
    "Heuristics": "green"
}

metrics = {
    "yield": "Total Yield",
    "fish": "Total Fish Stock",
    "budget": "Average Farmer Budget"
}

def plot_metrics(datasets): #median + IQR ribbon plots
    for metric_key, ylabel in metrics.items():
        plt.figure(figsize=(10, 6))
        for data in datasets:
            values = data[metric_key]
            median = np.median(values, axis=0)
            lower = np.percentile(values, 25, axis=0)
            upper = np.percentile(values, 75, axis=0)
            x = np.arange(len(median))

            label = data["label"]
            color = colors.get(label, "gray")
            plt.plot(x, median, label=label, color=color)
            plt.fill_between(x, lower, upper, alpha=0.2, color=color)

        plt.title(f"{ylabel} Over Time Across Modes")
        plt.xlabel("Year")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"Ribbon_{metric_key}.pdf", format="pdf")
        plt.show()

def plot(results):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for i, (label, data) in enumerate(results.items()):
        axs[i].boxplot(data, positions=np.arange(1, 10), widths=0.6, showfliers=True)
        axs[i].set_xticks(np.arange(1, 10))
        axs[i].set_xticklabels([f"Farmer {i}" for i in range(1, 10)])
        axs[i].set_ylabel("Annual Yield")
        axs[i].set_title(f"Decentralized\n{label.replace('_', ' ').capitalize()}")

    plt.tight_layout()
    plt.savefig(f"idk.pdf", format="pdf")
    plt.show()