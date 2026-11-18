import json
import cvxpy as cp
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
from linearmodels.panel import PanelOLS
from scipy.optimize import minimize
from .data import DATES, PANEL_END, load_data

EVENT_STUDY_PANEL_START = pd.Period("2021-01")  # skip 2020 COVID crash
EVENT_STUDY_REF_MONTH = pd.Period("2022-11")    # month before ChatGPT diffusion
# sector weight fit: candidate penalties toward the OEWS prior, chosen by
# out-of-sample error after the training period
WEIGHTS_PENALTIES = [0, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
WEIGHTS_TRAIN_END = pd.Period("2023-12")


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


def ModelIndeedRemoteShareFE(source, target, env):
    """
    Week-of-year fixed effects model for Indeed remote share of postings (%)
    by sector, with the sector's AI exposure (PC1) as the regressor. Reports
    how much of the variation in remote share exposure explains: the
    within-week R² from the fit, and the R² of week effects alone and with
    exposure added.
    """
    data = pd.read_csv(str(source[0]))
    exposure = pd.read_csv(str(source[1]))
    data["date"] = pd.to_datetime(data["dateString"])

    chatgpt = DATES["ChatGPT"]
    start = chatgpt - relativedelta(years=1)
    end = chatgpt + relativedelta(years=1)

    data = data[(data["date"] >= start) & (data["date"] < end)]
    data = data.merge(exposure[["sectorName", "pc1"]], on="sectorName")

    # Week-of-year for time fixed effects
    data["week"] = data["date"].dt.isocalendar().week.astype(int)

    regressors = ["pc1"]

    # Set multi-index for panel: entity=sector, time=week-of-year
    data = data.set_index(["sectorName", "week"])

    model = PanelOLS(
        data["value"].astype(float),
        data[regressors],
        time_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    # share of total variation explained by week effects, alone and with
    # exposure added
    y = data["value"].astype(float)
    week = y.index.get_level_values("week")
    tss = ((y - y.mean()) ** 2).sum()
    r2_week = 1 - ((y - y.groupby(week).transform("mean")) ** 2).sum() / tss
    r2_full = 1 - (model.resids ** 2).sum() / tss

    with open(str(target[0]), "w") as f:
        print(model.summary, file=f)
        print(f"\nR² of week effects alone: {r2_week:.4f}", file=f)
        print(f"R² of week effects + AI exposure: {r2_full:.4f}", file=f)
        print(f"Partial R² of AI exposure (within week): {model.rsquared:.4f}",
              file=f)

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
    panel = panel[(panel["month"] >= EVENT_STUDY_PANEL_START)
                  & (panel["month"] <= PANEL_END)].copy()
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
    dec = dec[(dec["month"] >= EVENT_STUDY_PANEL_START)
              & (dec["month"] <= PANEL_END)].copy()

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


def FitSectorWeights(I, r, I_miss, R, prior, penalty):
    """
    Recover each sector's share of postings at the Feb 2020 baseline from
    the national remote share. Sector volumes are w_s I_st, so
    R_t = sum_s w_s I_st r_st / sum_s w_s I_st, or equivalently
    sum_s w_s I_st (r_st - R_t) = 0 at every month. Sectors without a
    remote-share series (I_miss) get an unknown constant remote share rho_m,
    which enters linearly through u_m = w_m rho_m with 0 <= u_m <= w_m.
    Step 1 solves the linear condition by constrained least squares
    (w >= 0, sum w = 1), with a penalty pulling the weights toward the
    prior in proportion to the prior (a chi-square distance). Step 2
    refines by nonlinear least squares directly on R_t from that start.
    Arguments are T x S arrays (I, r), a T x M array (I_miss), a length-T
    array (R) and a length S + M prior; returns (w, w_miss, rho).
    """
    S, M = I.shape[1], I_miss.shape[1]
    w = cp.Variable(S, nonneg=True)
    w_miss = cp.Variable(M, nonneg=True)
    u = cp.Variable(M, nonneg=True)
    resid = cp.multiply(I, r - R[:, None]) @ w + I_miss @ u \
        - cp.multiply(I_miss, R[:, None]) @ w_miss
    W = cp.hstack([w, w_miss])
    cp.Problem(
        cp.Minimize(cp.sum_squares(resid) / len(R)
                    + penalty * cp.sum(cp.multiply(1 / prior, cp.square(W - prior)))),
        [cp.sum(W) == 1, u <= w_miss],
    ).solve()

    def objective(x):
        W, rho = x[:S + M], x[S + M:]
        fit = ((I * r) @ W[:S] + I_miss @ (W[S:] * rho)) / (I @ W[:S] + I_miss @ W[S:])
        return np.mean((fit - R) ** 2) + penalty * np.sum((W - prior) ** 2 / prior)

    rho0 = np.clip(u.value / np.maximum(w_miss.value, 1e-9), 0, 1)
    x0 = np.r_[np.clip(w.value, 0, 1), np.clip(w_miss.value, 0, 1), rho0]
    x = minimize(objective, x0, method="SLSQP", bounds=[(0, 1)] * (S + 2 * M),
                 constraints=[{"type": "eq", "fun": lambda x: x[:S + M].sum() - 1}],
                 options={"maxiter": 1000, "ftol": 1e-14}).x
    return x[:S], x[S:S + M], x[S + M:]


def ModelIndeedPostingsDecomposeRemote(source, target, env):
    """
    Descriptive breakdown of the national remote share by sector. Indeed's
    sector posting indices all equal 100 in Feb 2020, so they carry no
    information about relative sector sizes; the baseline weights w_s are
    recovered from the national remote share with FitSectorWeights, using
    OEWS employment as the prior. Each sector's share of postings is then
    pi_st = w_s I_st / sum_k w_k I_kt over all sectors, including those
    without a remote-share series, and each sector's contribution to the
    change since the peak month 0 is pi_st r_st - pi_s0 r_s0, which sums to
    the change in the fitted aggregate. Non-seasonally-adjusted posting
    indices are used to match the unadjusted remote-share series. Also
    reports the fixed-mix counterfactual that holds sector posting shares at
    the peak month.
    """
    postings = pd.read_csv(str(source[0]))
    exposure = pd.read_csv(str(source[1]))
    remote = pd.read_csv(str(source[2]))
    national = pd.read_csv(str(source[3]))

    for x in [postings, remote, national]:
        x["month"] = pd.to_datetime(x["dateString"]).dt.to_period("M")
    I = postings.groupby(["month", "sectorName"])["value"].mean().unstack() / 100
    r = remote.groupby(["month", "sectorName"])["value"].mean().unstack() / 100
    R = national.groupby("month")["value"].mean() / 100

    E = exposure.set_index("sectorName")
    months = I.index.intersection(r.index).intersection(R.index)
    months = months[months <= PANEL_END]
    I, r, R = I.loc[months, E.index], r.loc[months], R.loc[months]
    # Indeed does not publish remote shares for some (mostly hands-on)
    # sectors; they enter the fit with a constant remote share
    obs = [s for s in E.index if s in r.columns and r[s].notna().all()]
    miss = [s for s in E.index if s not in obs]
    r = r[obs]
    prior = E.loc[obs + miss, "weight"] / E.loc[obs + miss, "weight"].sum()

    def fit(m, penalty):
        return FitSectorWeights(I.loc[m, obs].values, r.loc[m].values,
                                I.loc[m, miss].values, R.loc[m].values,
                                prior.values, penalty)

    def predict(m, w, w_miss, rho):
        vol = I.loc[m, obs].values @ w + I.loc[m, miss].values @ w_miss
        return ((I.loc[m, obs].values * r.loc[m].values) @ w
                + I.loc[m, miss].values @ (w_miss * rho)) / vol

    def rmse(m, est):
        return 100 * np.sqrt(np.mean((predict(m, *est) - R.loc[m].values) ** 2))

    train = months[months <= WEIGHTS_TRAIN_END]
    test = months[months > WEIGHTS_TRAIN_END]
    validation = {p: rmse(test, fit(train, p)) for p in WEIGHTS_PENALTIES}
    penalty = min(validation, key=validation.get)
    w, w_miss, rho = fit(months, penalty)
    W = pd.Series(np.r_[w, w_miss], index=obs + miss)
    rho = pd.Series(rho, index=miss)

    vol = I * W
    pi = vol.div(vol.sum(axis=1), axis=0)
    remote_postings = pi[obs] * r
    remote_missing = (pi[miss] * rho).sum(axis=1)
    R_fit = remote_postings.sum(axis=1) + remote_missing

    # exposure halves split at the median over all sectors
    half = pd.Series(np.where(E["pc1"] >= E["pc1"].median(), "high", "low"),
                     index=E.index)
    # posting-weighted remote share within each exposure half, over sectors
    # with a remote-share series
    by_half = {f"R_{h}": remote_postings.T.groupby(half[obs]).sum().T[h]
               / pi[obs].T.groupby(half[obs]).sum().T[h] for h in ["low", "high"]}

    base = R.idxmax()
    end = R.index.max()
    contrib = remote_postings.loc[base:] - remote_postings.loc[base]
    # counterfactual holding sector posting shares at the peak month: the
    # fitted mix effect is added to the national series so the two lines
    # differ only by the change in sector mix
    R_fixed_mix_fit = (pi.loc[base, obs] * r.loc[base:]).sum(axis=1) \
        + (pi.loc[base, miss] * rho).sum()
    R_fixed_mix = R.loc[base:] + R_fixed_mix_fit - R_fit.loc[base:]

    with open(str(target[0]), "w") as f:
        print(f"sample: {len(obs)} sectors with remote shares, {len(miss)} "
              f"without, {months.min()} to {months.max()}", file=f)
        print(f"no remote share series: {'; '.join(miss)}", file=f)
        print(f"\nbaseline weights: penalty chosen by fitting through "
              f"{WEIGHTS_TRAIN_END} and predicting {test.min()} to "
              f"{test.max()}", file=f)
        for p, e in validation.items():
            print(f"  penalty {p:g}: out-of-sample RMSE {e:.3f} pp"
                  f"{'  <- chosen' if p == penalty else ''}", file=f)
        print(f"full-sample fit: RMSE {rmse(months, (w, w_miss, rho.values)):.3f} pp, "
              f"correlation with OEWS prior {np.corrcoef(W, prior)[0, 1]:.2f}, "
              f"sectors without remote shares {W[miss].sum():.1%} of baseline "
              f"postings", file=f)
        print(f"\n  {'':40s} {'weight':>7s} {'OEWS':>7s} {'remote':>7s}", file=f)
        for s in W.sort_values(ascending=False).index:
            rs = f"{rho[s]:7.1%}" if s in miss else f"{'':7s}"
            print(f"  {s:40s} {W[s]:7.2%} {prior[s]:7.2%} {rs}", file=f)
        print("  (weight: fitted share of Feb 2020 postings; OEWS: prior; "
              "remote: fitted constant remote share for sectors without a "
              "series)", file=f)
        print(f"\nnational remote share: peak {R.loc[base]:.2%} ({base}), "
              f"latest {R.loc[end]:.2%} ({end}), change "
              f"{100 * (R.loc[end] - R.loc[base]):+.2f} pp", file=f)
        print(f"fitted remote share: peak month {R_fit.loc[base]:.2%}, latest "
              f"{R_fit.loc[end]:.2%}, change "
              f"{100 * (R_fit.loc[end] - R_fit.loc[base]):+.2f} pp", file=f)
        for h in ["low", "high"]:
            Rh = by_half[f"R_{h}"]
            print(f"{h}-exposure half remote share: peak month "
                  f"{Rh.loc[base]:.2%}, latest {Rh.loc[end]:.2%}", file=f)
        print(f"sector mix held at {base}: latest {R_fixed_mix.loc[end]:.2%}",
              file=f)
        print(f"\nsector contributions to the change, {base} to {end} (pp):",
              file=f)
        print(f"  {'':40s} {'PC1':>6s} {'d pi':>8s} {'d remote':>9s} "
              f"{'contrib':>8s}", file=f)
        for s in contrib.loc[end].sort_values().index:
            print(f"  {s:40s} {E.loc[s, 'pc1']:+6.2f} "
                  f"{pi.loc[end, s] / pi.loc[base, s] - 1:+8.1%} "
                  f"{100 * (r.loc[end, s] - r.loc[base, s]):+8.2f}pp "
                  f"{100 * contrib.loc[end, s]:+8.3f}", file=f)
        print(f"  {'Sectors without remote shares':40s} {'':6s} "
              f"{pi.loc[end, miss].sum() / pi.loc[base, miss].sum() - 1:+8.1%} "
              f"{'':10s} "
              f"{100 * (remote_missing.loc[end] - remote_missing.loc[base]):+8.3f}",
              file=f)
        for h in ["high", "low"]:
            c = contrib.loc[end, half[obs] == h]
            print(f"  {h}-exposure half: {100 * c.sum():+.2f} pp "
                  f"({(c < 0).sum()} of {len(c)} negative)", file=f)
        print("  (d pi: relative change in share of postings; d remote: "
              "change in sector remote share)", file=f)

    pd.DataFrame({"R": R, "R_fit": R_fit, "R_fixed_mix": R_fixed_mix,
                  **by_half}).rename_axis("month").reset_index().assign(
        month=lambda x: x["month"].astype(str)).to_csv(str(target[1]), index=False)
    sectors = contrib.rename_axis(index="month", columns="sectorName").stack(
        ).rename("contrib").reset_index()
    sectors["month"] = sectors["month"].astype(str)
    sectors["pc1"] = sectors["sectorName"].map(E["pc1"])
    sectors["half"] = sectors["sectorName"].map(half)
    sectors[["month", "sectorName", "pc1", "half", "contrib"]].to_csv(
        str(target[2]), index=False)
