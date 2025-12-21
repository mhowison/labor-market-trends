import json
import statsmodels.api as sm
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
