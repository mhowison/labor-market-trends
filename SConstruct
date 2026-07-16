import os
from pathlib import Path
from pipeline.data import (
    DataStock,
    DataFred,
    DataIndeed,
    DataIndeedPostingIndex,
    DataAIExposure,
    DataMerge,
)
from pipeline.model import (
    ModelPreCovidOLS,
    ModelIndeedPostingsTWFE,
    ModelIndeedPostingsAIExposureTWFE,
    ModelIndeedPostingsEventStudy,
    ModelIndeedPostingsEventStudyRemote,
)
from pipeline.plot import(
    PlotTrends,
    PlotAIExposure,
    PlotIndeedPostingsAIExposure,
    PlotIndeedPostingsAvgAIExposure,
    PlotIndeedPostingsEventStudy,
    PlotIndeedPostingsAIExposureHalves,
    PlotIndeedPostingsAIExposureLikeForLike,
    PlotIndeedPostingsEventStudyRemote,
)

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

env.Command(
    source=[input_dir / "indeed-job-postings-sector-index.csv"],
    target=[output_dir / "indeed_postings.csv"],
    action=DataIndeedPostingIndex
)

env.Command(
    source=[
        input_dir / "indeed-job-postings-sector-index.csv",
    ],
    target=[
        output_dir / "indeed_postings_model.txt",
        scratch_dir / "indeed_postings_model.csv",
    ],
    action=ModelIndeedPostingsTWFE
)

env.Command(
    source=[
        input_dir / "indeed-job-postings-sector-index.csv",
        output_dir / "indeed_sector_ai_exposure.csv",
    ],
    target=[
        output_dir / "indeed_postings_ai_exposure_model.txt",
        scratch_dir / "indeed_postings_ai_exposure_model.csv",
    ],
    action=ModelIndeedPostingsAIExposureTWFE
)

env.Command(
    source=[
        input_dir / "AIOE_DataAppendix.xlsx",
        input_dir / "anthropic_economic_index_job_exposure.csv",
        input_dir / "eisfeldt_et_al_genaiexp_estz_occscores.csv",
        input_dir / "eloundou_et_al_occ_level.csv",
        input_dir / "tomlinson_et_al_ai_applicability_scores.csv",
        input_dir / "oews_national_M2024_dl.xlsx",
        input_dir / "indeed-sector-soc-mapping.csv",
        input_dir / "soc_2010_to_2018_crosswalk.xlsx",
    ],
    target=[output_dir / "indeed_sector_ai_exposure.csv"],
    action=DataAIExposure
)

env.Command(
    source=[output_dir / "indeed_sector_ai_exposure.csv"],
    target=[
        output_dir / "indeed_sector_ai_exposure.pdf",
        output_dir / "indeed_sector_ai_exposure.png",
    ],
    action=PlotAIExposure
)

env.Command(
    source=[
        output_dir / "indeed_postings.csv",
        output_dir / "indeed_sector_ai_exposure.csv",
    ],
    target=[
        output_dir / "indeed_postings_ai_exposure.pdf",
        output_dir / "indeed_postings_ai_exposure.png",
    ],
    action=PlotIndeedPostingsAIExposure
)

env.Command(
    source=[
        output_dir / "indeed_postings.csv",
        output_dir / "indeed_sector_ai_exposure.csv",
    ],
    target=[
        output_dir / "indeed_postings_avg_ai_exposure.pdf",
        output_dir / "indeed_postings_avg_ai_exposure.png",
    ],
    action=PlotIndeedPostingsAvgAIExposure
)

env.Command(
    source=[
        input_dir / "indeed-job-postings-sector-index-seasonally-adjusted.csv",
        output_dir / "indeed_sector_ai_exposure.csv",
    ],
    target=[
        output_dir / "indeed_postings_event_study_model.txt",
        scratch_dir / "indeed_postings_event_study_coefficients.csv",
    ],
    action=ModelIndeedPostingsEventStudy
)

env.Command(
    source=[
        scratch_dir / "indeed_postings_event_study_coefficients.csv",
    ],
    target=[
        output_dir / "indeed_postings_event_study.pdf",
        output_dir / "indeed_postings_event_study.png",
    ],
    action=PlotIndeedPostingsEventStudy
)

env.Command(
    source=[
        input_dir / "indeed-job-postings-sector-index-seasonally-adjusted.csv",
        output_dir / "indeed_sector_ai_exposure.csv",
    ],
    target=[
        output_dir / "indeed_postings_ai_exposure_halves.pdf",
        output_dir / "indeed_postings_ai_exposure_halves.png",
    ],
    action=PlotIndeedPostingsAIExposureHalves
)

env.Command(
    source=[
        input_dir / "indeed-job-postings-sector-index-seasonally-adjusted.csv",
        output_dir / "indeed_sector_ai_exposure.csv",
        input_dir / "indeed-job-postings-sector-remote-share.csv",
    ],
    target=[
        output_dir / "indeed_postings_ai_exposure_likeforlike.pdf",
        output_dir / "indeed_postings_ai_exposure_likeforlike.png",
    ],
    action=PlotIndeedPostingsAIExposureLikeForLike
)

env.Command(
    source=[
        input_dir / "indeed-job-postings-sector-index-seasonally-adjusted.csv",
        output_dir / "indeed_sector_ai_exposure.csv",
        input_dir / "indeed-job-postings-sector-remote-share.csv",
    ],
    target=[
        output_dir / "indeed_postings_event_study_remote_model.txt",
        scratch_dir / "indeed_postings_event_study_remote_coefficients.csv",
    ],
    action=ModelIndeedPostingsEventStudyRemote
)

env.Command(
    source=[
        scratch_dir / "indeed_postings_event_study_remote_coefficients.csv",
    ],
    target=[
        output_dir / "indeed_postings_event_study_remote.pdf",
        output_dir / "indeed_postings_event_study_remote.png",
    ],
    action=PlotIndeedPostingsEventStudyRemote
)
