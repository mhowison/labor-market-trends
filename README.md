# Labor Market Trends

Data and scripts to replicate plots of labor market trends.

## Running

Install python 3.12 and then create a virtual environment with (Mac OS/Linux/bash):

    python3.12 -m venv venv
    source activate venv/bin/activate
    pip install -r requirements.txt

Or (Windows/PowerShell):

    python3.12 -m venv venv
    venv/Scripts/Activate.ps1
    pip install -r requirements.txt

Run the analysis pipeline using the `scons` command in the root directory of the repo.

## Citations

* [Job openings data](https://fred.stlouisfed.org/series/JTSJOL)
* [S&P 500 data](https://finance.yahoo.com/quote/%5EGSPC/history/?frequency=1mo&period1=946684800&period2=1765854089)
* [Russell 3000 data](https://finance.yahoo.com/quote/%5ERUA/history/?frequency=1mo&period1=946684800&period2=1765807990)
* [MSCI World data](https://finance.yahoo.com/quote/%5E990100-USD-STRD/history/?frequency=1mo&period1=946684800&period2=1766158482)
* [Indeed Hiring Lab data](https://data.indeed.com/#/postings)
* [Tromlinson et al. 2025](https://github.com/microsoft/working-with-ai/blob/c94a07c52fb1d88ca5d221388f06d10e1bd6d2fe/ai_applicability_scores.csv)
* [Eloundou et al. 2024](https://github.com/openai/GPTs-are-GPTs/blob/main/data/occ_level.csv)
* [Eisfeldt et al. 2024](https://github.com/gschubert/website/blob/gh-pages/genaiexp_estz_occscores.csv)
* [Anthropic Economic Index - Job Exposure](https://huggingface.co/datasets/Anthropic/EconomicIndex/commits/main/labor_market_impacts) (commit 7a1cc3c)
* [AIOE Scores by Occupation](https://github.com/AIOE-Data/AIOE/blob/main/AIOE_DataAppendix.xlsx) (commit 3a89310)
* [OEWS May 2024 National Table](https://www.bls.gov/oes/tables.htm)
* [BLS SOC 2010 to 2018 Crosswalk](https://www.bls.gov/soc/2018/soc_2010_to_2018_crosswalk.xlsx)

## License

[CC BY 4.0](LICENSE.txt): Creative Commons Attribution

Copyright (c) 2025-2026 Mark Howison. All rights reserved.
