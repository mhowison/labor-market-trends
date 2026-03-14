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

    # Load Eisfeldt et al. GenAI exposure data
    eisfeldt = pd.read_csv(str(source[4]))
    eisfeldt = eisfeldt.rename(columns={"soc2010": "occ_code"})
    eisfeldt = eisfeldt[["occ_code", "genaiexp_estz_total"]]

    # Load Eloundou et al. exposure data (O*NET codes → 6-digit SOC)
    eloundou = pd.read_csv(str(source[5]))
    eloundou["occ_code"] = eloundou["O*NET-SOC Code"].str[:7]
    eloundou = eloundou.groupby("occ_code")["dv_rating_beta"].mean().reset_index()

    # Merge AIOE + Anthropic + Eisfeldt + Eloundou on occupation code
    merged = pd.merge(aioe, anthropic, on="occ_code", how="outer")
    merged = pd.merge(merged, eisfeldt, on="occ_code", how="outer")
    merged = pd.merge(merged, eloundou, on="occ_code", how="outer")

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
            "genaiexp_estz_total": weighted_avg(g, "genaiexp_estz_total"),
            "dv_rating_beta": weighted_avg(g, "dv_rating_beta"),
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

    # Load Webb data (occ1990dd codes → SOC major groups via crosswalk)
    webb = pd.read_csv(str(source[6]))
    occ1990dd_to_soc_major = [
        (4, 22, "11-0000"),    # Management
        (23, 37, "13-0000"),   # Business and Financial Operations
        (43, 59, "17-0000"),   # Architecture and Engineering
        (64, 68, "15-0000"),   # Computer and Mathematical
        (69, 83, "19-0000"),   # Life, Physical, and Social Science
        (84, 106, "29-0000"),  # Healthcare Practitioners and Technical
        (154, 159, "25-0000"), # Education, Training, and Library
        (163, 163, "21-0000"), # Community and Social Service
        (164, 165, "25-0000"), # Education (librarians, archivists)
        (166, 169, "19-0000"), # Social Science
        (173, 177, "21-0000"), # Community and Social Service
        (178, 178, "23-0000"), # Legal
        (183, 199, "27-0000"), # Arts, Design, Entertainment, Sports, Media
        (203, 208, "29-0000"), # Healthcare Technical
        (214, 218, "17-0000"), # Engineering technicians
        (223, 224, "19-0000"), # Science technicians
        (226, 228, "53-0000"), # Pilots, ATC, broadcast ops
        (229, 229, "15-0000"), # Computer programmers
        (233, 233, "51-0000"), # CNC programmers
        (234, 234, "23-0000"), # Paralegals
        (235, 235, "15-0000"), # Technicians NEC
        (243, 285, "41-0000"), # Sales
        (303, 389, "43-0000"), # Office and Administrative Support
        (405, 408, "37-0000"), # Building and Grounds Cleaning
        (413, 427, "33-0000"), # Protective Service
        (433, 444, "35-0000"), # Food Preparation and Serving
        (445, 448, "31-0000"), # Healthcare Support
        (450, 472, "39-0000"), # Personal Care and Service
        (473, 498, "45-0000"), # Farming, Fishing, and Forestry
        (503, 549, "49-0000"), # Installation, Maintenance, and Repair
        (558, 617, "47-0000"), # Construction and Extraction
        (628, 799, "51-0000"), # Production
        (803, 889, "53-0000"), # Transportation and Material Moving
    ]
    def map_occ1990dd(code):
        for lo, hi, soc in occ1990dd_to_soc_major:
            if lo <= code <= hi:
                return soc
        return None
    webb["major_code"] = webb["occ1990dd"].apply(map_occ1990dd)
    webb = webb.dropna(subset=["major_code", "pct_ai"])
    webb_agg = webb.groupby("major_code").apply(
        lambda g: (g["pct_ai"] * g["lswt2010"]).sum() / g["lswt2010"].sum()
    ).reset_index(name="webb_pct_ai")

    # Merge with Tomlinson and Webb
    columns = {
        "major_code": "soc2_code",
        "OCC_TITLE": "soc2_title",
        "AIOE": "aioe",
        "observed_exposure": "anthropic",
        "Score": "tomlinson",
        "genaiexp_estz_total": "eisfeldt",
        "dv_rating_beta": "eloundou",
        "webb_pct_ai": "webb",
    }
    result = (pd
        .merge(weighted, tomlinson, on="Major Group", how="outer")
        .merge(webb_agg, on="major_code", how="outer")
        .rename(columns=columns)
        .loc[:, columns.values()]
    )

    result.to_csv(str(target[0]), index=False)
