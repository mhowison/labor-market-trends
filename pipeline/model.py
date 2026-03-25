import json
import pandas as pd
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
from .data import DATES, load_data


def ModelIndeedPostingsAIExposure(source, target, env):
    """
    OLS regression of daily Indeed posting index on post-ChatGPT year indicators
    and a time-varying AI exposure control, pooled across sectors.
    """
    data = pd.read_csv(str(source[0]))
    data["date"] = pd.to_datetime(data["dateString"])
    exposure = pd.read_csv(str(source[1]))

    chatgpt = DATES["ChatGPT"]
    pre_start = chatgpt - relativedelta(years=1)
    year1_end = chatgpt + relativedelta(years=1)
    year2_end = chatgpt + relativedelta(years=2)

    data = data[(data["date"] >= pre_start) & (data["date"] < year2_end)]

    data["post_chatgpt_year1"] = ((data["date"] >= chatgpt) & (data["date"] < year1_end)).astype(int)
    data["post_chatgpt_year2"] = ((data["date"] >= year1_end) & (data["date"] < year2_end)).astype(int)

    # Merge AI exposure scores by sector
    data = pd.merge(data, exposure, on="sectorName", how="left")

    # Normalize each AI exposure measure to zero mean, unit variance
    for col in ["aioe", "eisfeldt", "eloundou", "tomlinson", "anthropic"]:
        data[col] = (data[col] - data[col].mean()) / data[col].std()

    # Time-varying AI exposure: AIOE pre-ChatGPT, Eisfeldt year 1,
    # average of Eloundou/Tomlinson/Anthropic year 2
    exposure_year2 = data[["eloundou", "tomlinson", "anthropic"]].mean(axis=1)
    data["ai_exposure"] = (
        data["aioe"] * (1 - data["post_chatgpt_year1"] - data["post_chatgpt_year2"])
        + data["eisfeldt"] * data["post_chatgpt_year1"]
        + exposure_year2 * data["post_chatgpt_year2"]
    )

    data["year1_x_exposure"] = data["post_chatgpt_year1"] * data["ai_exposure"]
    data["year2_x_exposure"] = data["post_chatgpt_year2"] * data["ai_exposure"]

    regressors = [
        "post_chatgpt_year1", "post_chatgpt_year2",
        "ai_exposure", "year1_x_exposure", "year2_x_exposure",
    ]
    y = data["value"].astype(float)
    X = sm.add_constant(data[regressors])
    model = sm.OLS(y, X).fit()

    with open(str(target[0]), "w") as f:
        print(model.summary(), file=f)

    results = pd.DataFrame([{
        "intercept": model.params["const"],
        **{r: model.params[r] for r in regressors},
    }])
    results.to_csv(str(target[1]), index=False)


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
