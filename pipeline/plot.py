import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import DateOffset
from matplotlib.patches import Patch
from .data import DATES, load_data
from .util import weighted_correlation


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


def PlotIndeedPostingsAIExposure(source, target, env):
    """
    Scatterplots of Indeed posting index percent change vs each AI exposure measure.
    """
    postings = pd.read_csv(str(source[0]))
    exposure = pd.read_csv(str(source[1]))
    data = pd.merge(postings, exposure, on="sectorName")

    labels = {
        "aioe": "AIOE",
        "anthropic": "Anthropic",
        "tomlinson": "Tomlinson",
        "eisfeldt": "Eisfeldt",
        "eloundou": "Eloundou",
    }
    measures = list(labels.keys())
    n = len(measures)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    for i, measure in enumerate(measures):
        ax = axes[i]
        x = data[measure]
        y = data["pct_change"]

        ax.scatter(x, y, color="none", s=15, edgecolors="black", linewidths=0.75)

        # Fit line
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b, color="#808080", lw=1, ls="--")

        # Correlation
        r = x.corr(y)
        ax.text(0.28, 0.1, f"R\u00B2 = {r:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_xlabel(f"AI Exposure ({labels[measure]})")
        if i == 0:
            ax.set_ylabel("% Change in Pre/Post ChatGPT Posting Index")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(target[0]))
    plt.savefig(str(target[1]), dpi=300)


def PlotIndeedPostingsEventStudy(source, target, env):
    """
    Event-study coefficient plot: monthly betas with 95% cluster CIs,
    ChatGPT and Claude Code launch dates marked.
    """
    ev = pd.read_csv(str(source[0]))
    ev["t"] = pd.PeriodIndex(ev["month"], freq="M").to_timestamp()

    chatgpt = pd.Timestamp(DATES["ChatGPT"])
    claude_code = pd.Timestamp(DATES["ClaudeCode"])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(chatgpt, color="k", ls="--", lw=1)
    ax.axvline(claude_code, color="k", ls="--", lw=1)
    ax.errorbar(
        ev["t"], ev["beta"], yerr=1.96 * ev["se"], fmt="o", ms=3.5,
        lw=1, color="#CC3311", ecolor="#CC331155",
    )
    ax.text(
        chatgpt + pd.Timedelta(days=15), 0.085,
        "ChatGPT\n(Nov 2022)", fontsize=9,
    )
    ax.text(
        claude_code + pd.Timedelta(days=15), 0.085,
        "Claude Code\n(Feb 2025)", fontsize=9,
    )
    ax.set_title(
        "Event study: log postings on AI exposure x month "
        "(ref. Nov 2022), 95% cluster CI"
    )
    ax.set_ylabel("Coef. per 1 SD exposure")
    ax.set_ylim(-0.15, 0.115)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(str(target[0]))
    plt.savefig(str(target[1]), dpi=300)


def PlotIndeedPostingsEventStudyRemote(source, target, env):
    """
    Two-panel figure: (top) split-sample event studies for on-site and remote
    log postings with 95% cluster CI ribbons; (bottom) triple-diff coefficient
    path for the log remote-minus-on-site gap.
    """
    d = pd.read_csv(str(source[0]))
    d["t"] = pd.PeriodIndex(d["month"], freq="M").to_timestamp()
    chatgpt = pd.Timestamp(DATES["ChatGPT"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(chatgpt, color="k", ls="--", lw=1)
    for beta_col, se_col, color, label in [
        ("beta_onsite", "se_onsite", "#004488", "On-site postings"),
        ("beta_remote", "se_remote", "#CC3311", "Remote postings"),
    ]:
        ax.fill_between(
            d["t"], d[beta_col] - 1.96 * d[se_col], d[beta_col] + 1.96 * d[se_col],
            color=color, alpha=0.12,
        )
        ax.plot(d["t"], d[beta_col], color=color, lw=2, label=label)
    ax.set_title("Log postings by AI exposure and remote vs. on-site, 95% cluster CI")
    ax.set_ylabel("Log postings per 1 SD exposure")
    ax.legend(loc="lower left")
    ax.text(chatgpt + pd.Timedelta(days=15), ax.get_ylim()[1] * 0.7,
            "ChatGPT\n(Nov 2022)", fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(chatgpt, color="k", ls="--", lw=1)
    ax.errorbar(
        d["t"], d["beta_gap"], yerr=1.96 * d["se_gap"], fmt="o",
        ms=3.5, lw=1, color="#117733", ecolor="#11773355",
    )
    ax.set_title("Triple difference: log gap in remote vs. on-site postings on "
                 "AI exposure x month (ref. Nov 2022), 95% cluster CI")
    ax.set_ylabel("Differential remote effect per 1 SD")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(target[0]))
    plt.savefig(str(target[1]), dpi=300)


def PlotIndeedPostingsAIExposureHalves(source, target, env):
    """
    Employment-weighted posting index for more vs less AI-exposed halves
    of sectors (median PC1 split), indexed to Nov 2022 = 100, ChatGPT
    launch marked.
    """
    RED, BLUE = "#CC3311", "#4477AA"
    BASE_MONTH = pd.Period("2022-11")
    FIG_START = pd.Period("2022-01")

    postings = pd.read_csv(str(source[0]))
    exposure = pd.read_csv(str(source[1]))

    postings["month"] = pd.to_datetime(postings["dateString"]).dt.to_period("M")
    mp = postings.groupby(["sectorName", "month"], as_index=False)["value"].mean()

    E = exposure.set_index("sectorName")
    E["half"] = np.where(E["pc1"] >= E["pc1"].median(), "high", "low")

    m = mp.merge(E[["half", "weight"]], left_on="sectorName", right_index=True)
    m = m[m["month"] >= FIG_START]
    base = m[m["month"] == BASE_MONTH].set_index("sectorName")["value"]
    m["rebased"] = 100 * m["value"] / m["sectorName"].map(base)

    lines = (m.groupby(["half", "month"], observed=True)
             .apply(lambda g: np.average(g["rebased"], weights=g["weight"]),
                    include_groups=False)
             .reset_index(name="idx"))
    lines["t"] = lines["month"].dt.to_timestamp()

    chatgpt = pd.Timestamp(DATES["ChatGPT"])

    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(10, 6))
    for half, color, label in [("low", BLUE, "Low AI-exposure sectors"),
                               ("high", RED, "High AI-exposure sectors")]:
        g = lines[lines["half"] == half]
        ax.plot(g["t"], g["idx"], color=color, lw=3, label=label)
        ax.text(g["t"].iloc[-1] + pd.Timedelta(days=12), g["idx"].iloc[-1],
                f"{g['idx'].iloc[-1] - 100:+.0f}%", color=color,
                fontweight="bold", va="center", fontsize=14)
    ax.axvline(chatgpt, color="k", ls="--", lw=1.2)
    ax.text(chatgpt + pd.Timedelta(days=15), 108, "ChatGPT\nlaunches",
            fontsize=11)
    ax.axhline(100, color="gray", lw=0.6)
    ax.set_ylabel("Job postings (Nov 2022 = 100)")
    ax.set_title("US job postings since ChatGPT: AI-exposed work fell "
                 "furthest\nIndeed postings, 40 occupational sectors, "
                 "weighted by employment", fontsize=14, loc="left")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(FIG_START.to_timestamp(), pd.Timestamp("2026-12-31"))
    plt.tight_layout()
    plt.savefig(str(target[0]))
    plt.savefig(str(target[1]), dpi=300)


def PlotIndeedPostingsAIExposureLikeForLike(source, target, env):
    """
    Like-for-like bar chart: office-type sectors (2021 remote share above
    sample median), % change in postings Nov 2022 to LATE_START-LATE_END,
    split into more vs less AI-exposed halves with group averages marked.
    """
    RED, BLUE = "#CC3311", "#4477AA"
    BASE_MONTH = pd.Period("2022-11")
    LATE_START, LATE_END = pd.Period("2026-03"), pd.Period("2026-06")

    postings = pd.read_csv(str(source[0]))
    exposure = pd.read_csv(str(source[1]))
    remote = pd.read_csv(str(source[2]))

    postings["month"] = pd.to_datetime(postings["dateString"]).dt.to_period("M")
    remote["month"] = pd.to_datetime(remote["dateString"]).dt.to_period("M")
    mp = postings.groupby(["sectorName", "month"], as_index=False)["value"].mean()
    mr = (remote.groupby(["sectorName", "month"], as_index=False)["value"]
          .mean().rename(columns={"value": "share"}))

    E = exposure.set_index("sectorName")

    base_all = mp[mp["month"] == BASE_MONTH].set_index("sectorName")["value"]
    late = (mp[(mp["month"] >= LATE_START) & (mp["month"] <= LATE_END)]
            .groupby("sectorName")["value"].mean())
    chg = (late / base_all - 1) * 100

    telework_2021 = (mr[(mr["month"] >= pd.Period("2021-01"))
                        & (mr["month"] <= pd.Period("2021-12"))]
                     .groupby("sectorName")["share"].mean())
    df = pd.DataFrame({"chg": chg, "pc1": E["pc1"],
                       "tw": telework_2021}).dropna()

    office = df[df["tw"] >= df["tw"].median()].copy()
    office["grp"] = np.where(office["pc1"] >= office["pc1"].median(), "hi", "lo")
    mean_hi = office.loc[office["grp"] == "hi", "chg"].mean()
    mean_lo = office.loc[office["grp"] == "lo", "chg"].mean()

    office = office.sort_values("chg")
    colors = office["grp"].map({"hi": RED, "lo": BLUE})

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.barh(range(len(office)), office["chg"], color=colors)
    ax.set_yticks(range(len(office)))
    ax.set_yticklabels(office.index)
    for i, v in enumerate(office["chg"]):
        ax.text(v - 0.8, i, f"{v:.0f}%", va="center", ha="right",
                fontsize=10, color="#333")
    ax.axvline(mean_hi, color=RED, ls=":", lw=2)
    ax.axvline(mean_lo, color=BLUE, ls=":", lw=2)
    ax.legend(handles=[
        Patch(color=RED, label=f"High AI-exposure office jobs "
                               f"(avg {mean_hi:.0f}%)"),
        Patch(color=BLUE, label=f"Low AI-exposure office jobs "
                                f"(avg {mean_lo:.0f}%)"),
    ], frameon=False, loc="upper left", fontsize=11)
    ax.set_xlabel("Change in job postings, Nov 2022 to mid-2026 (%)")
    ax.set_title("Comparing like with like: among office-type jobs,\n"
                 "the most AI-exposed fell hardest", fontsize=14, loc="left")
    xmin = min(office["chg"].min(), mean_hi, mean_lo)
    ax.set_xlim(xmin - 8, 0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(str(target[0]))
    plt.savefig(str(target[1]), dpi=300)


def PlotIndeedPostingsAvgAIExposure(source, target, env):
    """
    Scatterplot of Indeed posting index percent change vs average AI exposure.
    """
    postings = pd.read_csv(str(source[0]))
    exposure = pd.read_csv(str(source[1]))
    data = pd.merge(postings, exposure, on="sectorName")

    measures = ["aioe", "anthropic", "tomlinson", "eisfeldt", "eloundou"]
    for col in measures:
        data[col] = (data[col] - data[col].mean()) / data[col].std()
    data["average"] = data[measures].mean(axis=1)

    fig, ax = plt.subplots(figsize=(5, 5))

    x = data["average"]
    y = data["pct_change"]
    w = data["weight"]

    # Scale weights to reasonable circle sizes
    s = w / w.max() * 200

    ax.scatter(x, y, color="none", s=s, edgecolors="black", linewidths=0.75)

    # Fit line
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, color="black", lw=1, ls="--")

    # Correlation
    r = x.corr(y)
    ax.text(0.95, 0.95, f"R\u00B2 = {r:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11)

    # Fit line (weighted)
    m, b = np.polyfit(x, y, 1, w=w)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, color="gray", lw=1, ls="--")

    # Weighted correlation
    r = weighted_correlation(x, y, w)
    ax.text(0.95, 0.90, f"Weighted R\u00B2 = {r:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11, color="gray")

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Average Normalized AI Exposure")
    ax.set_ylabel("% Change in Pre/Post ChatGPT Posting Index")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(target[0]))
    plt.savefig(str(target[1]), dpi=300)
