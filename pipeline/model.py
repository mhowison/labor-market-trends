import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
from linearmodels.panel import PanelOLS
from .data import DATES, load_data

EVENT_STUDY_PANEL_START = pd.Period("2021-01")  # skip 2020 COVID crash
EVENT_STUDY_REF_MONTH = pd.Period("2022-11")    # month before ChatGPT diffusion
EVENT_STUDY_REMOTE_PANEL_END = pd.Period("2026-05")    # remote-share series end


def ModelPreCovidOLS(source, target, env):
    """
    Fit an OLS model in the pre-covid period regressing X on Y.
    """
    data = load_data(source[0])
    X = source[1].read()
    Y = source[2].read()
    train = data.loc[data.index < DATES["COVID"], [X, Y]].dropna()
    model = sm.OLS(train[Y], sm.add_constant(train[[X]])).fit()
    params = {
        "intercept": model.params.iloc[0],
        "coefficient": model.params.iloc[1],
    }
    with open(str(target[0]), "w") as f:
        print(model.summary(), file=f)
    with open(str(target[1]), "w") as f:
        json.dump(params, f)


def ModelIndeedPostingsTWFE(source, target, env):
    """
    Two-way fixed effects model (sector and week-of-year) for Indeed
    posting index by sector.
    """
    data = pd.read_csv(str(source[0]))
    data["date"] = pd.to_datetime(data["dateString"])

    chatgpt = DATES["ChatGPT"]
    start = chatgpt - relativedelta(years=1)
    end = chatgpt + relativedelta(years=1)
    
    data = data[(data["date"] >= start) & (data["date"] < end)]

    data["post_chatgpt"] = (data["date"] > chatgpt).astype(int)

    # Week-of-year for time fixed effects
    data["week"] = data["date"].dt.isocalendar().week.astype(int)

    regressors = ["post_chatgpt"]

    # Set multi-index for panel: entity=sector, time=week-of-year
    data = data.set_index(["sectorName", "week"])

    model = PanelOLS(
        data["value"].astype(float),
        data[regressors],
        entity_effects=True,
        time_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    with open(str(target[0]), "w") as f:
        print(model.summary, file=f)

    results = pd.DataFrame([{
        r: model.params[r] for r in regressors
    }])
    results.to_csv(str(target[1]), index=False)


def ModelIndeedPostingsAIExposureTWFE(source, target, env):
    """
    Two-way fixed effects model (sector and week-of-year) for Indeed
    posting index by sector, with time-varying AI exposure and interactions.
    Also reports partial R² vs the base TWFE model.
    """
    data = pd.read_csv(str(source[0]))
    data["date"] = pd.to_datetime(data["dateString"])
    exposure = pd.read_csv(str(source[1]))

    chatgpt = DATES["ChatGPT"]
    start = chatgpt - relativedelta(years=1)
    end = chatgpt + relativedelta(years=1)
    
    data = data[(data["date"] >= start) & (data["date"] < end)]

    data["post_chatgpt"] = (data["date"] > chatgpt).astype(int)

    # Week-of-year for time fixed effects
    data["week"] = data["date"].dt.isocalendar().week.astype(int)

    # Merge AI exposure scores by sector
    data = pd.merge(data, exposure, on="sectorName", how="left")

    # Normalize each AI exposure measure to zero mean, unit variance
    measures = ["aioe", "eisfeldt", "eloundou", "tomlinson", "anthropic"]
    for col in measures:
        data[col] = (data[col] - data[col].mean()) / data[col].std()

    # Time-varying AI exposure: AIOE pre-ChatGPT, others post-ChatGPT
    post_exposure = data[measures[1:]].mean(axis=1)
    data["ai_exposure"] = (
        data["aioe"] * (1 - data["post_chatgpt"]) +
        post_exposure * data["post_chatgpt"]
    )
    data["post_chatgpt_x_ai_exposure"] = data["post_chatgpt"] * data["ai_exposure"]

    base_regressors = ["post_chatgpt"]
    exposure_regressors = ["ai_exposure", "post_chatgpt_x_ai_exposure"]

    # Set multi-index for panel: entity=sector, time=week-of-year
    data = data.set_index(["sectorName", "week"])

    # Restricted model (base TWFE without exposure)
    restricted = PanelOLS(
        data["value"].astype(float),
        data[base_regressors],
        entity_effects=True,
        time_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    # Full model (with AI exposure and interactions)
    full = PanelOLS(
        data["value"].astype(float),
        data[base_regressors + exposure_regressors],
        entity_effects=True,
        time_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    # Partial R² = (SS_res_restricted - SS_res_full) / SS_res_restricted
    ss_res_restricted = restricted.resid_ss
    ss_res_full = full.resid_ss
    partial_r2 = (ss_res_restricted - ss_res_full) / ss_res_restricted

    with open(str(target[0]), "w") as f:
        print(full.summary, file=f)
        print(f"\nPartial R² (vs base TWFE): {partial_r2:.4f}", file=f)

    results = pd.DataFrame([{
        **{r: full.params[r] for r in base_regressors + exposure_regressors},
        "partial_r2": partial_r2,
    }])
    results.to_csv(str(target[1]), index=False)


def ModelIndeedPostingsEventStudy(source, target, env):
    """
    Event study: regress log sector-level posting index on
    (PC1 AI-exposure x month) interactions with sector and month fixed
    effects. Standard errors clustered by sector.
    """
    postings = pd.read_csv(str(source[0]))
    exposure = pd.read_csv(str(source[1]))

    postings["month"] = pd.to_datetime(postings["dateString"]).dt.to_period("M")
    panel = postings.groupby(
        ["sectorName", "month"], as_index=False
    )["value"].mean()

    panel = panel.merge(exposure[["sectorName", "pc1"]], on="sectorName")
    panel = panel[panel["month"] >= EVENT_STUDY_PANEL_START].copy()
    panel["logy"] = np.log(panel["value"])

    months = sorted(panel["month"].unique())
    inter_months = [m for m in months if m != EVENT_STUDY_REF_MONTH]

    sector_d = pd.get_dummies(panel["sectorName"], drop_first=True, dtype=float)
    month_d = pd.get_dummies(panel["month"].astype(str), drop_first=True, dtype=float)
    month_arr = panel["month"].to_numpy()
    interactions = np.column_stack([
        (month_arr == m).astype(float) * panel["pc1"].to_numpy()
        for m in inter_months
    ])
    X_restricted = np.column_stack([np.ones(len(panel)), sector_d, month_d])
    X = np.column_stack([X_restricted, interactions])
    k_fe = X_restricted.shape[1]

    y = panel["logy"].to_numpy()
    res = sm.OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": panel["sectorName"]}
    )
    restricted = sm.OLS(y, X_restricted).fit()
    partial_r2 = (restricted.ssr - res.ssr) / restricted.ssr

    ev = pd.DataFrame({
        "month": [str(m) for m in inter_months],
        "beta": res.params[k_fe:],
        "se": res.bse[k_fe:],
    }).sort_values("month")

    pre = ev[ev.month <= "2022-10"]["beta"].mean()
    post_avg = ev[ev.month >= "2022-12"]["beta"].mean()

    with open(str(target[0]), "w") as f:
        print(
            f"panel: {panel.sectorName.nunique()} sectors, "
            f"{panel.month.min()} to {panel.month.max()}",
            file=f,
        )
        print(f"pre-period avg beta:  {pre:+.4f}", file=f)
        print(f"post-period avg beta: {post_avg:+.4f}", file=f)
        print(f"Partial R² (AI exposure interactions vs. sector+month FE): {partial_r2:.4f}", file=f)

    ev.to_csv(str(target[1]), index=False)


def ModelIndeedPostingsEventStudyRemote(source, target, env):
    """
    Triple-difference: remote vs on-site postings by AI exposure. Decomposes
    each sector's posting index into remote and on-site components via the
    sector remote-share series, then estimates split-sample event studies
    (PC1 exposure x month) for log on-site, log remote, and the gap
    (log_remote - log_onsite). The gap regression is the triple-diff:
    sector-x-month FE are absorbed by within-cell differencing, sector-level
    shocks cancel by construction.
    """
    postings = pd.read_csv(str(source[0]))
    exposure = pd.read_csv(str(source[1]))
    remote = pd.read_csv(str(source[2]))

    postings["month"] = pd.to_datetime(postings["dateString"]).dt.to_period("M")
    remote["month"] = pd.to_datetime(remote["dateString"]).dt.to_period("M")
    mp = (postings.groupby(["sectorName", "month"], as_index=False)["value"]
          .mean().rename(columns={"value": "idx"}))
    mr = (remote.groupby(["sectorName", "month"], as_index=False)["value"]
          .mean().rename(columns={"value": "share"}))

    dec = mp.merge(mr, on=["sectorName", "month"])
    dec = dec[(dec["month"] >= EVENT_STUDY_PANEL_START) & (dec["month"] <= EVENT_STUDY_REMOTE_PANEL_END)].copy()

    # decompose postings; log odds gap absorbs sector-month FE
    dec["log_onsite"] = np.log(dec["idx"] * (1 - dec["share"] / 100))
    dec["log_remote"] = np.log(dec["idx"] * (dec["share"] / 100))
    dec["gap"] = dec["log_remote"] - dec["log_onsite"]

    # shared design machinery
    months = sorted(dec["month"].unique())
    inter_months = [m for m in months if m != EVENT_STUDY_REF_MONTH]
    month_arr = dec["month"].to_numpy()
    pc1_map = exposure.set_index("sectorName")["pc1"]
    z_exp = dec["sectorName"].map(pc1_map)
    z_exp = ((z_exp - z_exp.mean()) / z_exp.std()).to_numpy()

    FE = np.column_stack([
        np.ones(len(dec)),
        pd.get_dummies(dec["sectorName"], drop_first=True, dtype=float),
        pd.get_dummies(dec["month"].astype(str), drop_first=True, dtype=float),
    ])
    interactions = np.column_stack([
        (month_arr == m).astype(float) * z_exp for m in inter_months
    ])
    X = np.column_stack([FE, interactions])
    k_fe = FE.shape[1]

    def event_study(ycol):
        y = dec[ycol].to_numpy()
        res = sm.OLS(y, X).fit(
            cov_type="cluster", cov_kwds={"groups": dec["sectorName"]}
        )
        restricted = sm.OLS(y, FE).fit()
        partial_r2 = (restricted.ssr - res.ssr) / restricted.ssr
        df = pd.DataFrame({
            "month": [str(m) for m in inter_months],
            "beta": res.params[k_fe:],
            "se": res.bse[k_fe:],
        }).sort_values("month")
        return df, partial_r2

    onsite, r2_onsite = event_study("log_onsite")
    remote_es, r2_remote = event_study("log_remote")
    gap, r2_gap = event_study("gap")

    def avg(d):
        pre = d[d["month"] <= "2022-10"]["beta"].mean()
        post_avg = d[d["month"] >= "2022-12"]["beta"].mean()
        return pre, post_avg

    out = (onsite.merge(remote_es, on="month", suffixes=("_onsite", "_remote"))
           .merge(gap.rename(columns={"beta": "beta_gap", "se": "se_gap"}), on="month"))

    with open(str(target[0]), "w") as f:
        print(
            f"sample: {dec.sectorName.nunique()} sectors, "
            f"{dec.month.min()} to {dec.month.max()}",
            file=f,
        )
        for label, d, r2 in [
            ("on-site", onsite, r2_onsite),
            ("remote", remote_es, r2_remote),
            ("gap (triple-diff)", gap, r2_gap),
        ]:
            pre, post_avg = avg(d)
            print(
                f"{label:20s} pre {pre:+.4f}  post {post_avg:+.4f}  "
                f"partial R² {r2:.4f}",
                file=f,
            )

    out.to_csv(str(target[1]), index=False)
