import os
import pandas as pd
from datetime import datetime

DATES = {
    "ChatGPT": datetime(2022, 11, 30),
    "COVID": datetime(2020, 3, 1),
}

def load_data(filename):
    """
    Helper function to load CVS file and convert 'Month' column to datetime.
    """
    data = pd.read_csv(str(filename))
    data["Month"] = pd.to_datetime(data["Month"])
    return data.set_index("Month").sort_index()

def DataIndeed(source, target, env):
    """
    Load a CSV file with job postings data from Indeed Labs.
    """
    data = pd.read_csv(str(source[0]))
    name = source[1].read()
    data["Month"] = pd.to_datetime(data["dateString"])
    data[name] = data["value"].astype(float)
    if name == "Indeed: AI": 
        data[name] *= 100.0
    data.set_index("Month").resample("MS")[name].mean().to_csv(str(target[0]))

def DataFred(source, target, env):
    """
    Load a CSV file with a series from FRED.
    """
    data = pd.read_csv(str(source[0]))
    series = source[1].read()
    name = source[2].read()
    data["Month"] = pd.to_datetime(data["observation_date"])
    data[name] = data[series].astype(float)
    data[["Month", name]].to_csv(str(target[0]), index=False)

def DataStock(source, target, env):
    """
    Load a CSV file with stock history data from Yahoo Finance.
    """
    data = pd.read_csv(str(source[0]), delimiter="\t")
    name = source[1].read()
    data["Month"] = pd.to_datetime(data["Date"])
    data[name] = data["Close"].str.replace(",", "").astype(float)
    data[["Month", name]].to_csv(str(target[0]), index=False)

def DataMerge(source, target, env):
    """
    Merge multiple CSV files on the 'Month' datetime column.
    """
    data = [pd.read_csv(f).set_index("Month") for f in map(str, source)]
    pd.concat(data, axis=1).to_csv(str(target[0]))

def DataAIExposure(source, target, env):
    """
    Merge AI exposure measures from multiple sources at the major occupation level.
    """
    # Load AIOE data
    aioe = pd.read_excel(str(source[0]), sheet_name="Appendix A")
    aioe = aioe.rename(columns={"SOC Code": "occ_code"})
    aioe = aioe[["occ_code", "Occupation Title", "AIOE"]]

    # Load Anthropic data
    anthropic = pd.read_csv(str(source[1]))
    anthropic = anthropic[["occ_code", "title", "observed_exposure"]]

    # Merge AIOE + Anthropic on occupation code
    merged = pd.merge(aioe, anthropic, on="occ_code", how="outer")

    # Load OEWS data
    oews = pd.read_excel(str(source[2]))

    # Get detailed occupation employment
    detailed = oews[oews["O_GROUP"] == "detailed"][["OCC_CODE", "OCC_TITLE", "TOT_EMP"]].copy()
    detailed["TOT_EMP"] = pd.to_numeric(detailed["TOT_EMP"], errors="coerce")

    # Merge with OEWS employment
    merged = pd.merge(merged, detailed, left_on="occ_code", right_on="OCC_CODE", how="inner")

    # Derive major group code
    merged["major_code"] = merged["occ_code"].str[:2] + "-0000"

    # Compute employment-weighted averages
    def weighted_avg(group, col):
        mask = group[col].notna() & group["TOT_EMP"].notna()
        if not mask.any():
            return float("nan")
        return (group.loc[mask, col] * group.loc[mask, "TOT_EMP"]).sum() / group.loc[mask, "TOT_EMP"].sum()

    weighted = merged.groupby("major_code").apply(
        lambda g: pd.Series({
            "AIOE": weighted_avg(g, "AIOE"),
            "observed_exposure": weighted_avg(g, "observed_exposure"),
        })
    ).reset_index()

    # Get major group titles from OEWS
    major = oews[oews["O_GROUP"] == "major"][["OCC_CODE", "OCC_TITLE"]].copy()
    weighted = pd.merge(weighted, major, left_on="major_code", right_on="OCC_CODE", how="left")

    # Normalize titles to match Tomlinson naming
    weighted["Major Group"] = weighted["OCC_TITLE"].str.replace(r" Occupations$", "", regex=True)
    title_fixes = {
        "Arts, Design, Entertainment, Sports, and Media": "Arts, Design, Entertainment, Sports, Media",
        "Building and Grounds Cleaning and Maintenance": "Building, Grounds Cleaning, Maintenance",
    }
    weighted["Major Group"] = weighted["Major Group"].replace(title_fixes)

    # Load Tomlinson data
    tomlinson = pd.read_csv(str(source[3]), sep="\t")

    # Merge with Tomlinson
    columns = {
        "major_code": "soc2_code",
        "OCC_TITLE": "soc2_title",
        "AIOE": "aioe",
        "observed_exposure": "anthropic",
        "Score": "tomlinson",
    }
    result = (pd
        .merge(weighted, tomlinson, on="Major Group", how="outer")
        .rename(columns=columns)
        .loc[:, columns.values()]
    )

    result.to_csv(str(target[0]), index=False)
