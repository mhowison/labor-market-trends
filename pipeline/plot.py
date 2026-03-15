import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import DateOffset
from .data import DATES, load_data

from matplotlib import rcParams
rcParams["font.sans-serif"] = "Arial"
rcParams["font.family"] = "sans-serif"

def PlotTrends(source, target, env):
    """
    Plot a comparison of two time series.
    """
    ranges = {
        "S&P 500": [0, 7001, 1000],
        "Russell 3000": [0, 4001, 1000],
        "MSCI World Index": [0, 5001, 1000],
        "Job Openings": [2000, 14001, 1000],
        "Job Openings: Healthcare": [400, 2201, 200],
        "Job Openings: Retail": [100, 1401, 100],
        "Indeed: AI": [140, 421, 20],
    }

    yformatters = {
        "S&P 500": lambda y: "${:,d}".format(y),
        "Russell 3000": lambda y: "${:,d}".format(y),
        "MSCI World Index": lambda y: "${:,d}".format(y),
        "Job Openings": lambda y: "{}M".format(y // 1000),
        "Job Openings: Healthcare": lambda y: "{:,d}K".format(y),
        "Job Openings: Retail": lambda y: "{:,d}K".format(y),
        "Indeed: AI": lambda y: "{:.2f}%".format(y / 100),
    }

    data = load_data(source[0])
    with open(str(source[1])) as f:
        params = json.load(f)
    X = source[2].read()
    Y = source[3].read()
    data = data[[X, Y]].dropna()
    # Use model to transform y
    data[Y] = (data[Y] - params["intercept"]) / params["coefficient"]
    ymin = min(
        ranges[X][0],
        (ranges[Y][0] - params["intercept"]) / params["coefficient"]
    )
    ymax = max(
        ranges[X][1] - 1,
        (ranges[Y][1] - 1 - params["intercept"]) / params["coefficient"]
    )
    # Create plot
    plt.figure(figsize=(10, 6))
    # Date lines
    color="black"
    for date in ["COVID", "ChatGPT"]:
        plt.axvline(x=DATES[date], color=color, lw=0.75, ls="dashed", zorder=-1)
        plt.annotate(date, (DATES[date], ranges[X][1]), ha="center", fontweight="bold", color=color)
    # Plot trend 1
    ax1 = plt.gca()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.set_xlim(data.index[0], data.index[-1] + DateOffset(years=3))
    ax1.set_xticks(
        [datetime(year, 1, 1) for year in range(2001, 2027)],
        labels=[str(year) for year in range(2001, 2027)],
        rotation=90
    )
    ax1.set_ylim(ymin, ymax)
    color = "#808080"
    ax1.set_yticks(
        range(*ranges[X]),
        labels=[yformatters[X](y) for y in range(*ranges[X])],
        color="gray"
    )
    ax1.tick_params(axis="y", color="gray")
    ax1.plot(data.index, data[X], color=color, lw=1.5)
    ax1.annotate(
        X,
        (data.index[-1] + DateOffset(months=3), data[X].iloc[-1]),
        color=color,
        fontweight="bold"
    )
    # Plot trend 2
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.set_ylim(ymin, ymax)
    color="#385CC3"
    ax2.set_yticks(
        [(y - params["intercept"]) / params["coefficient"] for y in range(*ranges[Y])],
        labels=[yformatters[Y](y) for y in range(*ranges[Y])],
        color=color
    )
    ax2.tick_params(axis="y", color=color)
    ax2.plot(data.index, data[Y], color=color, lw=1.5)
    ax2.annotate(
        Y,
        (data.index[-1] + DateOffset(months=3), data[Y].iloc[-1]),
        color=color,
        fontweight="bold"
    )
    # Output
    plt.tight_layout()
    plt.savefig(str(target[0]))
    plt.savefig(str(target[1]), dpi=300)

def PlotAIExposure(source, target, env):
    """
    Pairwise scatterplots and correlations between AI exposure measures.
    """
    data = pd.read_csv(str(source[0]))
    labels = {
        "aioe": "AIOE",
        "anthropic": "Anthropic",
        "tomlinson": "Tomlinson",
        "eisfeldt": "Eisfeldt",
        "eloundou": "Eloundou",
    }
    measures = list(labels.keys())
    n = len(measures)

    fig, axes = plt.subplots(n, n, figsize=(18, 18))

    for i, row_var in enumerate(measures):
        for j, col_var in enumerate(measures):
            ax = axes[i, j]
            if i == j:
                # Diagonal: histogram
                ax.hist(data[row_var].dropna(), bins=10, color="#385CC3", alpha=0.7, edgecolor="white")
                ax.set_ylabel("Count" if j == 0 else "")
            elif i > j:
                # Lower triangle: scatterplot
                mask = data[[row_var, col_var]].dropna().index
                ax.scatter(data.loc[mask, col_var], data.loc[mask, row_var],
                           color="#385CC3", alpha=0.7, s=30, edgecolors="white", linewidths=0.5)
                # Fit line
                x, y = data.loc[mask, col_var], data.loc[mask, row_var]
                m, b = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, m * x_line + b, color="#808080", lw=1, ls="--")
            else:
                # Upper triangle: correlation text
                mask = data[[row_var, col_var]].dropna().index
                r = data.loc[mask, row_var].corr(data.loc[mask, col_var])
                ax.text(0.5, 0.5, f"r = {r:.2f}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=16, fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])

            # Axis labels on edges only
            if i == n - 1:
                ax.set_xlabel(labels[col_var])
            else:
                ax.set_xticklabels([])
            if j == 0 and i != j:
                ax.set_ylabel(labels[row_var])
            elif i != j:
                ax.set_yticklabels([])

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(target[0]))
    plt.savefig(str(target[1]), dpi=300)
