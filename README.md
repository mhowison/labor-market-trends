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

## Methodology

### Composite AI Exposure by Sector

Published measures of occupational exposure to AI differ in method and scale but largely agree on which jobs are most exposed. The analyses here combine five of them into a single sector-level index, built by `DataAIExposure` in `pipeline/data.py` and saved to `output/indeed_sector_ai_exposure.csv`:

| Column | Measure |
|---|---|
| `aioe` | Felten et al. AI Occupational Exposure (AIOE) |
| `anthropic` | Anthropic Economic Index observed exposure |
| `tomlinson` | Tomlinson et al. AI applicability score |
| `eisfeldt` | Eisfeldt et al. generative-AI exposure |
| `eloundou` | Eloundou et al. GPT exposure (β rating) |

**Aggregation to sectors.** Each measure is mapped from its source occupation codes to OEWS detailed SOC occupations, and occupations are assigned to Indeed's 47 sectors with `input/indeed-sector-soc-mapping.csv`. A sector's value of each measure is the average over its occupations, weighted by May 2024 OEWS employment.

**Principal components.** Each measure is standardized across the 47 sectors (z-scores), and the principal components are computed by singular value decomposition of the standardized matrix. Anthropic observed exposure is missing for every occupation in two sectors (Personal Care & Home Health and Project Management). Those two cells are imputed at the cross-sector mean (zero in z-scores) so the decomposition is defined, which pulls those sectors' index slightly toward the middle. The first principal component (PC1) is oriented so that higher values mean more exposed and is rescaled to mean 0 and standard deviation 1. It is the `pc1` column used throughout the analyses. The high- and low-exposure halves split the sectors at the median PC1.

**Variance explained.** PC1 explains **81.7%** of the variance across the five standardized measures, and the second component only 9.4%. The measures load almost equally on PC1:

| Measure | PC1 loading | Correlation with PC1 |
|---|---|---|
| Felten et al. AIOE | 0.45 | 0.90 |
| Anthropic observed exposure | 0.40 | 0.84 |
| Tomlinson et al. applicability | 0.44 | 0.88 |
| Eisfeldt et al. generative-AI exposure | 0.48 | 0.96 |
| Eloundou et al. GPT exposure | 0.47 | 0.95 |

The near-equal loadings make PC1 close to a simple average of the standardized measures, and its dominance means the measures share one common dimension of exposure. Anthropic observed exposure, which is based on actual Claude usage rather than on task descriptions, loads least and correlates least with the others. The most exposed sectors are Data & Analytics, Software Development, and Marketing. The least exposed are Cleaning & Sanitation, Construction, and Loading & Stocking. The variance share is printed when the pipeline builds `output/indeed_sector_ai_exposure.csv`.

### Recovering Sector Weights from the National Remote Share

Indeed's sector job postings indices are each normalized to 100 on 1 February 2020, so they show how each sector's postings have grown but not how large sectors are relative to each other. Summing them weights every sector equally at the baseline, which overstates small sectors and understates large ones. The breakdown of the remote share by sector (`output/indeed_postings_decompose_remote.pdf`, described in `REPORT.md`) instead recovers each sector's baseline share of postings from Indeed's national remote share, which is a posting-weighted average of the sector remote shares.

**Model.** Let $w_s$ be sector $s$'s share of postings in February 2020, $I_{st}$ its posting index (divided by 100), $r_{st}$ its remote share, and $R_t$ the national remote share, all as monthly averages of the non-seasonally-adjusted series. Sector volumes are proportional to $w_s I_{st}$, so

$$R_t = \frac{\sum_s w_s I_{st}\, r_{st}}{\sum_s w_s I_{st}}.$$

Multiplying through by the denominator gives a condition that is linear in the weights and holds at every month:

$$\sum_s w_s I_{st} \left(r_{st} - R_t\right) = 0.$$

Indeed does not publish remote shares for nine mostly hands-on sectors, but their postings still count toward the national total. Each gets an unknown constant remote share $\rho_m$. Its term, $w_m I_{mt} (\rho_m - R_t)$, stays linear when written in terms of $w_m$ and $u_m = w_m \rho_m$ with $0 \le u_m \le w_m$.

**Estimation.** `FitSectorWeights` in `pipeline/model.py` estimates the weights in two steps:

1. Constrained least squares (with `cvxpy`) minimizes the squared residuals of the linear condition over all months, subject to $w_s \ge 0$ and $\sum_s w_s = 1$.
2. Nonlinear least squares (with `scipy`) starts from that solution and minimizes the squared error in $R_t$ itself, so the fit is judged on the remote share rather than on the rescaled condition.

**Prior and penalty.** Unconstrained by anything else, the fit tracks the national series almost exactly but sets many weights to zero, including Production & Manufacturing. Sectors with similar posting and remote-share trajectories are identified only jointly, so the optimizer is free to move weight among them. Both steps therefore add a penalty pulling the weights toward a prior $p_s$, each sector's share of May 2024 OEWS employment over its SOC occupations:

$$\lambda \sum_s \frac{(w_s - p_s)^2}{p_s}.$$

Dividing by $p_s$ scales the penalty to each sector's size, so small sectors are not swamped by large ones.

**Choosing the penalty.** The strength $\lambda$ is chosen by out-of-sample validation over time. The weights are fit on February 2020 through December 2023 for each candidate $\lambda$ and used to predict the national remote share from January 2024 onward. The $\lambda$ with the lowest prediction error is then refit on the full sample. The candidates and the training cutoff are `WEIGHTS_PENALTIES` and `WEIGHTS_TRAIN_END` in `pipeline/model.py`.

**Using the weights.** Each sector's share of postings in month $t$ is $\pi_{st} = w_s I_{st} / \sum_k w_k I_{kt}$ over all sectors, including those without remote data. The sector contributions, the fixed-mix counterfactual, and the remote shares within each AI-exposure half are all built from these shares.

**Diagnostics.** `output/indeed_postings_decompose_remote_model.txt` reports:

- the out-of-sample error for each candidate penalty;
- the full-sample fitting error and the correlation of the weights with the prior;
- each sector's fitted weight next to its prior, and the fitted remote share for each sector without remote data.

**Limitations.** The weights are estimates. Weights of sectors that move together are uncertain individually even when their sum is well determined, and the penalty ties the weights toward an employment prior that may differ from posting volumes. Indeed postings not assigned to any sector, if any, have no term in the model and are absorbed into the fitted weights or the residual.

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
