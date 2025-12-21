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
