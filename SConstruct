import os
from pathlib import Path
from pipeline.data import DataStock, DataFred, DataIndeed, DataMerge
from pipeline.model import ModelPreCovidOLS
from pipeline.plot import PlotTrends

env = Environment(env=os.environ)

input_dir   = Path("input")
scratch_dir = Path("scratch")
output_dir  = Path("output")

names = {
    "sp500": "S&P 500",
    "russell3000": "Russell 3000",
    "msci": "MSCI World Index",
    "JTSJOL": "Job Openings",
    "JTS6200JOL": "Job Openings: Healthcare",
    "JTS4400JOL": "Job Openings: Retail",
    "indeed_ai": "Indeed: AI",
    "indeed_medtech": "Indeed: MedTechs",
    "indeed_sales": "Indeed: Sales",
}
comparisons = [
    ("sp500", "JTSJOL"),
    ("russell3000", "JTSJOL"),
    ("msci", "JTSJOL"),
    ("sp500", "JTS6200JOL"),
    ("sp500", "JTS4400JOL"),
    ("sp500", "indeed_ai"),
]

for stock in ["sp500", "russell3000", "msci"]:
    env.Command(
        source=[input_dir / f"{stock}.tsv", Value(names[stock])],
        target=[scratch_dir / f"data_{stock}.csv"],
        action=DataStock
    )

for series in ["JTSJOL", "JTS6200JOL", "JTS4400JOL"]:
    env.Command(
        source=[input_dir / f"{series}.csv", Value(series), Value(names[series])],
        target=[scratch_dir / f"data_{series}.csv"],
        action=DataFred
    )

for indeed in ["indeed_ai", "indeed_medtech", "indeed_sales"]:
    env.Command(
        source=[input_dir / f"{indeed}.csv", Value(names[indeed])],
        target=[scratch_dir / f"data_{indeed}.csv"],
        action=DataIndeed
    )

env.Command(
    source=[
        scratch_dir / f"data_{x}.csv"
        for x in names
    ],
    target=[output_dir / "data.csv"],
    action=DataMerge
)

for x, y in comparisons:

    env.Command(
        source=[
            output_dir / "data.csv",
            Value(names[x]),
            Value(names[y]),
        ],
        target=[
            output_dir / f"{x}_v_{y}_model.txt",
            scratch_dir / f"{x}_v_{y}_model.json",
        ],
        action=ModelPreCovidOLS
    )

    env.Command(
        source=[
            output_dir / "data.csv",
            scratch_dir / f"{x}_v_{y}_model.json",
            Value(names[x]),
            Value(names[y]),

        ],
        target=[
            output_dir / f"{x}_v_{y}.pdf",
            output_dir / f"{x}_v_{y}.png",
        ],
        action=PlotTrends
    )
