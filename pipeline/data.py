import os
import numpy as np
import pandas as pd
from datetime import datetime

DATES = {
    "COVID": datetime(2020, 3, 1),
    "ChatGPT": datetime(2022, 11, 30),
    "ClaudeCode": datetime(2025, 2, 24),
}

AI_EXPOSURE_MEASURES = ["aioe", "anthropic", "tomlinson", "eisfeldt", "eloundou"]

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

def DataIndeedPostingIndex(source, target, env):
    """
    Compute percent change in average posting index pre vs post ChatGPT launch,
    by Indeed sector.
    """
    from dateutil.relativedelta import relativedelta

    data = pd.read_csv(str(source[0]))
    data["date"] = pd.to_datetime(data["dateString"])

    window_years = int(source[1].read()) if len(source) > 1 else 1
    chatgpt = DATES["ChatGPT"]
    pre_start = chatgpt - relativedelta(years=window_years)
    post_end = chatgpt + relativedelta(years=window_years)

    pre = data[(data["date"] >= pre_start) & (data["date"] < chatgpt)]
    post = data[(data["date"] >= chatgpt) & (data["date"] < post_end)]

    pre_avg = pre.groupby("sectorName")["value"].mean()
    post_avg = post.groupby("sectorName")["value"].mean()

    result = pd.DataFrame({
        "sectorName": pre_avg.index,
        "pre_avg": pre_avg.values,
        "post_avg": post_avg.reindex(pre_avg.index).values,
    })
    result["pct_change"] = (result["post_avg"] - result["pre_avg"]) / result["pre_avg"] * 100

    result.sort_values("pct_change", ascending=False).to_csv(str(target[0]), index=False)

def DataAIExposure(source, target, env):
    """
    Merge AI exposure measures from multiple sources, aggregated by Indeed sector.
    """
    # Load SOC 2010→2018 crosswalk
    crosswalk = pd.read_excel(str(source[7]), header=8)
    crosswalk = crosswalk.rename(columns={
        "2010 SOC Code": "soc2010",
        "2018 SOC Code": "soc2018",
    })[["soc2010", "soc2018"]]

    # Load AIOE data (SOC 2010) and map to SOC 2018
    aioe = pd.read_excel(str(source[0]), sheet_name="Appendix A")
    aioe = aioe.rename(columns={"SOC Code": "soc2010"})
    aioe = aioe[["soc2010", "AIOE"]]
    aioe = pd.merge(aioe, crosswalk, on="soc2010", how="left")
    aioe["occ_code"] = aioe["soc2018"].fillna(aioe["soc2010"])
    aioe = aioe.groupby("occ_code")["AIOE"].mean().reset_index()

    # Load Anthropic data
    anthropic = pd.read_csv(str(source[1]))
    anthropic = anthropic[["occ_code", "title", "observed_exposure"]]

    # Load Eisfeldt et al. GenAI exposure data (SOC 2010) and map to SOC 2018
    eisfeldt = pd.read_csv(str(source[2]))
    eisfeldt = eisfeldt.rename(columns={"soc2010": "soc2010"})
    eisfeldt = eisfeldt[["soc2010", "genaiexp_estz_total"]]
    eisfeldt = pd.merge(eisfeldt, crosswalk, on="soc2010", how="left")
    eisfeldt["occ_code"] = eisfeldt["soc2018"].fillna(eisfeldt["soc2010"])
    eisfeldt = eisfeldt.groupby("occ_code")["genaiexp_estz_total"].mean().reset_index()

    # Load Eloundou et al. exposure data (O*NET codes → 6-digit SOC)
    eloundou = pd.read_csv(str(source[3]))
    eloundou["occ_code"] = eloundou["O*NET-SOC Code"].str[:7]
    eloundou = eloundou.groupby("occ_code")["dv_rating_beta"].mean().reset_index()

    # Load Tomlinson et al. AI applicability scores
    tomlinson = pd.read_csv(str(source[4]))
    tomlinson = tomlinson.rename(columns={"SOC Code": "occ_code"})
    tomlinson = tomlinson[["occ_code", "ai_applicability_score"]]

    # Merge all five sources on occupation code
    merged = pd.merge(aioe, anthropic, on="occ_code", how="outer")
    merged = pd.merge(merged, eisfeldt, on="occ_code", how="outer")
    merged = pd.merge(merged, eloundou, on="occ_code", how="outer")
    merged = pd.merge(merged, tomlinson, on="occ_code", how="outer")

    # Load OEWS data
    oews = pd.read_excel(str(source[5]))

    # Get detailed occupation employment
    detailed = oews[oews["O_GROUP"] == "detailed"][["OCC_CODE", "OCC_TITLE", "TOT_EMP"]].copy()
    detailed["TOT_EMP"] = pd.to_numeric(detailed["TOT_EMP"], errors="coerce")

    # Verify all source SOC codes map to an OEWS detailed occupation
    oews_codes = set(detailed["OCC_CODE"])
    for name, codes in [
        ("AIOE", aioe["occ_code"]),
        ("Anthropic", anthropic["occ_code"]),
        ("Eisfeldt", eisfeldt["occ_code"]),
        ("Eloundou", eloundou["occ_code"]),
        ("Tomlinson", tomlinson["occ_code"]),
    ]:
        unmapped = set(codes) - oews_codes
        print(f"{name}: SOC codes not found in OEWS: {sorted(unmapped)}")

    # Merge with OEWS employment
    merged = pd.merge(merged, detailed, left_on="occ_code", right_on="OCC_CODE", how="inner")

    # Load Indeed sector-SOC mapping
    sector_map = pd.read_csv(str(source[6]))
    merged = pd.merge(merged, sector_map[["OCC_CODE", "sectorName"]], on="OCC_CODE", how="left")

    # Compute employment-weighted averages by Indeed sector
    def weighted_avg(group, col):
        mask = group[col].notna() & group["TOT_EMP"].notna()
        if not mask.any():
            return float("nan")
        return (group.loc[mask, col] * group.loc[mask, "TOT_EMP"]).sum() / group.loc[mask, "TOT_EMP"].sum()

    columns = {
        "AIOE": "aioe",
        "observed_exposure": "anthropic",
        "ai_applicability_score": "tomlinson",
        "genaiexp_estz_total": "eisfeldt",
        "dv_rating_beta": "eloundou",
    }

    result = merged.groupby("sectorName", group_keys=False).apply(
        lambda g: pd.Series({
            **{new: weighted_avg(g, old) for old, new in columns.items()},
            "weight": g["TOT_EMP"].sum(),
        }),
        include_groups=False,
    ).reset_index()

    # PC1 composite of AI exposure measures, z-scored, oriented so higher = more exposed
    Z = (result[AI_EXPOSURE_MEASURES] - result[AI_EXPOSURE_MEASURES].mean()) / result[AI_EXPOSURE_MEASURES].std()
    _, S, Vt = np.linalg.svd(Z.values, full_matrices=False)
    pc1 = Z.values @ Vt[0]
    if np.corrcoef(pc1, result["aioe"])[0, 1] < 0:
        pc1 = -pc1
    result["pc1"] = (pc1 - pc1.mean()) / pc1.std()
    print(f"PC1 share of variance: {S[0]**2 / (S**2).sum():.4f}")

    result.to_csv(str(target[0]), index=False)
