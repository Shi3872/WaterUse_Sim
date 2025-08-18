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

def farmer_returns_plot(results_by_delta):
    plt.figure(figsize=(8, 6))

    dash_styles = [
        (0, (1, 1)),
        (0, (5, 5)),
        (0, (5, 2, 1, 2)),
        (0, (3, 5, 1, 5)),
        (0, (7, 3)),
        (0, (2, 2, 8, 2)),
    ]

    for i, (delta, returns) in enumerate(results_by_delta.items()):
        style = dash_styles[i % len(dash_styles)]
        plt.plot(range(len(returns)), returns, label=f"δ={delta}", linestyle=style)

    plt.xlabel("Year")
    plt.ylabel("Farmer 9 Budget")
    plt.title("Farmer 9 Returns Over Time for Different δ")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

def fish_plot(deltas,
              central_adults, central_larvae,
              dec_nf_adults, dec_nf_larvae,
              dec_f_adults, dec_f_larvae, dec_f_catch):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a) – Adults and Catch
    axes[0].plot(deltas, central_adults, "ko-", label="centralized")
    axes[0].plot(deltas, dec_nf_adults, "k^--", label="decentralized – no fishing")
    axes[0].plot(deltas, dec_f_adults, "ko--", label="decentralized – fishing")
    axes[0].plot(deltas, dec_f_catch, "kd:", label="mean total catch")
    axes[0].set_xlabel("Delta")
    axes[0].set_ylabel("Average adult fish abundance")
    axes[0].legend()
    axes[0].set_title("a) Adult fish & catch")

    # Panel (b) – Larvae inflow
    axes[1].plot(deltas, central_larvae, "ko-", label="centralized")
    axes[1].plot(deltas, dec_nf_larvae, "k^--", label="decentralized – no fishing")
    axes[1].plot(deltas, dec_f_larvae, "ko--", label="decentralized – fishing")
    axes[1].set_xlabel("Delta")
    axes[1].set_ylabel("Average inflow of larvae")
    axes[1].legend()
    axes[1].set_title("b) Larvae inflow")

    plt.tight_layout()
    plt.show()

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
        ("Centralized Delta0", "a) Centralized\n$\\delta$ = 0", (0, 0)),  # top-left
        ("Decentralized Delta0", "b) Decentralized\n$\\delta$ = 0", (0, 1)),  # top-right
        ("Centralized Delta1", "c) Centralized\n$\\delta$ = 1", (1, 0)),  # bottom-left
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