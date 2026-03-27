import json
import pandas as pd
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
from linearmodels.panel import PanelOLS
from .data import DATES, load_data


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