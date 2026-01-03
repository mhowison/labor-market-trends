# Labor Market Trends

Data and scripts to replicate plots of labor market trends.

## Running

Install python 3.12 and then create a virtual environment with (Mac OS/Linux/bash):

    python3.12 -m venv venv
    source activate venv
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

## License

[CC BY 4.0](LICENSE.txt): Creative Commons Attribution

Copyright 2025 Mark Howison. All rights reserved.
