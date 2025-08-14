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

def water_plot(sim_delta0, sim_delta1):
    years = range(1, len(sim_delta0.july_inflows) + 1)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(years, sim_delta0.july_inflows, label="RWV (Real July Water)", color="black", linewidth=2)
    plt.plot(years, sim_delta0.predicted_water_history, label="EWA (δ=0)", linestyle="--", color="red")
    plt.plot(years, sim_delta1.predicted_water_history, label="EWA (δ=1)", linestyle="--", color="blue")
    plt.xlabel("Year")
    plt.ylabel("Water Availability")
    plt.title("Real vs Predicted July Water Availability")
    plt.legend()
    plt.grid(True)
    plt.show()

def box_plot(results_dict):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)

    # Mapping results to panel positions
    plot_order = [
        #("Centralized Delta0", "a) Centralized\n$\\delta$ = 0", (0, 0)),  # top-left
        ("Decentralized Delta0", "b) Decentralized\n$\\delta$ = 0", (0, 1)),  # top-right
        #("Centralized Delta1", "c) Centralized\n$\\delta$ = 1", (1, 0)),  # bottom-left
        ("Decentralized Delta1", "d) Decentralized\n$\\delta$ = 1", (1, 1))  # bottom-right
    ]

    for key, title, pos in plot_order:
        ax = axes[pos]
        data = results_dict[key]
        ax.boxplot(data, positions=range(1, len(data[0]) + 1),
                   patch_artist=True,
                   boxprops=dict(facecolor='lightgray', color='black'),
                   medianprops=dict(color='black'),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='black'),
                   flierprops=dict(marker='o', markersize=3, markerfacecolor='black', alpha=0.3))

        ax.set_title(title, loc='left', fontsize=12)
        ax.set_xlabel("Farmer Position (Upstream → Downstream)")
        ax.set_xticks(range(1, len(data[0]) + 1))
        ax.set_xticklabels(range(1, len(data[0]) + 1))
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    axes[0, 0].set_ylabel("Annual Irrigation Yield")
    axes[1, 0].set_ylabel("Annual Irrigation Yield")

    plt.tight_layout()
    plt.show()