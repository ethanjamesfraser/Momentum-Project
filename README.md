# Momentum: Can Past Winners Predict Future Winners?

## Independent Quantitative Finance Research Project

### Research question
Do stocks with the highest returns over the previous twelve months, excluding the most recent month, continue to outperform over the following month?

### Hypothesis
If cross-sectional momentum exists, stocks ranked highest on past returns should earn higher subsequent returns than those ranked lowest. This would produce a positive winners-minus-losers spread, and a winner portfolio that beats an equal-weighted portfolio of the same stocks.

### Strategy (12–1 month momentum)
1. At the end of each month *t*−1, compute each stock's cumulative return over months *t*−12 to *t*−2. The most recent month is skipped to separate momentum from short-term reversal.
2. Rank the stocks and buy the top 20% (3 of the 19 stocks), equally weighted.
3. Hold the portfolio for month *t*, then rebalance.
4. Deduct trading costs of 10 basis points per dollar traded, with turnover measured after weights drift during the month.

### Comparisons
| Comparison | What it tells you |
|---|---|
| Equal-weighted universe | Whether ranking by momentum adds value beyond simply holding the same 19 stocks. This is the fairest comparison. |
| SPY | Performance against the broad US market. This comparison is inflated by selection bias (see caveats). |
| Winners minus losers (top 20% − bottom 20%) | The standard academic test of the momentum effect. |

### Measures
* **Performance:** average monthly return, annualised geometric return, annualised volatility, Sharpe ratio (excess over the 13-week T-bill), maximum drawdown, positive-month rate, and best and worst months.
* **Trading:** average one-way monthly turnover.
* **Significance:** t-tests on the winners-minus-losers spread and on excess returns over both comparison portfolios, plus CAPM alpha and beta against SPY.
* **Robustness:** lookbacks of 3, 6, 9 and 12 months, each tested with the top 20% and top 30% of stocks.

### Caveats
This is an educational research project, not investment advice.

* **Selection bias.** The universe is a fixed list of companies that are large and successful today, chosen with hindsight. Any long portfolio drawn from it will tend to beat SPY for reasons unrelated to momentum. This is why the equal-weighted universe and the winners-minus-losers spread are the main tests.
* **Small cross-section.** The universe is 19 stocks (Adobe was dropped after its price data failed to download), so each portfolio holds only three stocks and results are noisy.
* **Short sample.** About ten years of monthly data gives limited statistical power.
* **Simplified frictions.** The backtest models a flat trading cost but ignores taxes, market impact, short-selling costs and delistings.
* **Data quality.** `yfinance` is an unofficial interface to Yahoo Finance and does not provide institutional-quality data.

## Project structure
```text
momentum_research_project/
├── momentum_research.ipynb   # step-by-step analysis with commentary
├── momentum_strategy.py      # all functions; runnable as a script
├── requirements.txt
├── README.md
├── outputs/                  # created on run: CSV tables
└── figures/                  # created on run: PNG charts
```

## Reproducing the analysis
```bash
pip install -r requirements.txt
python momentum_strategy.py              # prints results, saves tables and charts
# or
jupyter notebook momentum_research.ipynb
```

The parameters (universe, start date, lookback, portfolio size and cost assumption) are set at the top of `momentum_strategy.py`.

### Data source
Historical prices are retrieved through `yfinance`: https://ranaroussi.github.io/yfinance/
