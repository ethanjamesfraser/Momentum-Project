"""
Momentum: Can Past Winners Predict Future Winners?

Educational quantitative-finance backtest of a long-only 12-1 month
cross-sectional momentum strategy on a fixed universe of large-cap US stocks,
compared against (a) an equal-weighted portfolio of the same universe and
(b) SPY. A winners-minus-losers (WML) spread is also reported as the direct
test of the momentum effect.

Timing convention (all dates are month-ends):
    signal for month t = cumulative return over months t-12 ... t-2
                         (11 months; month t-1 is skipped to avoid
                          short-term reversal)
    portfolio formed at the close of month t-1 and held through month t.

Run:
    python momentum_strategy.py
Tables are written to ./outputs and charts to ./figures.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "AVGO", "JPM", "V", "MA",
    "XOM", "CVX", "COST", "WMT", "HD",
    "NFLX", "CRM", "ORCL", "INTC",
]
# ADBE was in the original 20-stock list but its data failed to download
# in the reported run, so the study uses these 19 stocks.
BENCHMARK = "SPY"
RISK_FREE_TICKER = "^IRX"   # 13-week US T-bill yield, % per annum
START = "2015-01-01"
END = None                  # None = latest available data

LOOKBACK_MONTHS = 12
SKIP_MONTHS = 1
TOP_FRACTION = 0.20
MIN_STOCKS = 5
COST_BPS = 10               # trading cost per dollar traded, basis points

OUTPUT_DIR = Path("outputs")
FIGURE_DIR = Path("figures")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def download_prices(tickers, start=START, end=END) -> pd.DataFrame:
    """Daily dividend- and split-adjusted closing prices."""
    import yfinance as yf

    data = yf.download(tickers, start=start, end=end,
                       auto_adjust=True, progress=False)
    prices = data["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices = prices.dropna(how="all")

    def _missing(df):
        return [t for t in tickers if t not in df.columns or df[t].isna().all()]

    # Yahoo occasionally drops a ticker from a bulk request: retry singly.
    for ticker in _missing(prices):
        for _ in range(3):
            single = yf.download(ticker, start=start, end=end,
                                 auto_adjust=True, progress=False)
            if not single.empty:
                close = single["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                prices[ticker] = close
                break

    missing = _missing(prices)
    if missing:
        raise RuntimeError(
            f"No price data for {missing}. Check your internet connection "
            "and run again; the backtest would otherwise use a smaller "
            "universe than intended.")
    print(f"Downloaded prices for all {len(tickers)} tickers.")
    return prices


def download_risk_free(start=START, end=END) -> pd.Series | None:
    """
    Monthly risk-free return from the 13-week T-bill yield.
    The yield at the end of month t-1 is the return earned during month t.
    Returns None (rf = 0 is then used) if the series cannot be downloaded.
    """
    try:
        import yfinance as yf

        data = yf.download(RISK_FREE_TICKER, start=start, end=end,
                           auto_adjust=False, progress=False)
        irx = data["Close"]
        if isinstance(irx, pd.DataFrame):
            irx = irx.iloc[:, 0]
        irx = irx.dropna()
        if irx.empty:
            raise ValueError("empty risk-free series")
        monthly_yield = irx.resample("ME").last() / 100 / 12
        return monthly_yield.shift(1).rename("RF")
    except Exception as exc:  # network or data problems
        warnings.warn(f"Risk-free rate unavailable ({exc}); using rf = 0.")
        return None


def to_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end prices, dropping the final month if it is incomplete."""
    monthly = prices.resample("ME").last()
    last_obs = prices.index.max()
    if last_obs != last_obs + pd.offsets.BMonthEnd(0):
        monthly = monthly.iloc[:-1]
    return monthly


# ---------------------------------------------------------------------------
# Signal and backtest
# ---------------------------------------------------------------------------
def momentum_signal(monthly_returns: pd.DataFrame,
                    lookback: int = LOOKBACK_MONTHS,
                    skip: int = SKIP_MONTHS) -> pd.DataFrame:
    """
    Signal indexed by the holding month t: cumulative return over months
    t-lookback ... t-skip-1. With lookback=12, skip=1 this is t-12 ... t-2.
    """
    window = lookback - skip
    log_growth = np.log1p(monthly_returns).rolling(
        window, min_periods=window).sum()
    return np.expm1(log_growth).shift(skip + 1)


def run_backtest(monthly_returns: pd.DataFrame,
                 signal: pd.DataFrame,
                 tickers=TICKERS,
                 top_fraction: float = TOP_FRACTION,
                 cost_bps: float = COST_BPS,
                 min_stocks: int = MIN_STOCKS):
    """
    Returns
    -------
    results : DataFrame of monthly returns for the winner portfolio (gross
              and net of costs), loser portfolio, WML spread, equal-weighted
              universe, plus one-way turnover.
    holdings : long-format DataFrame of the stocks held each month.
    """
    stock_returns = monthly_returns[tickers]
    scores_all = signal[tickers]
    cost_rate = cost_bps / 10_000

    rows, holding_rows = [], []
    drifted = pd.Series(0.0, index=tickers)   # weights just before rebalance

    for date in stock_returns.index:
        scores = scores_all.loc[date].dropna()
        if len(scores) < min_stocks:
            continue

        n = max(1, int(np.floor(len(scores) * top_fraction)))
        ranked = scores.sort_values(ascending=False)
        winners, losers = ranked.index[:n], ranked.index[-n:]

        r = stock_returns.loc[date]
        if r[winners].isna().any():
            warnings.warn(f"{date.date()}: missing return for a held stock; "
                          "treated as 0%.")
        r_filled = r.fillna(0.0)

        target = pd.Series(0.0, index=tickers)
        target[winners] = 1.0 / n
        traded = (target - drifted).abs().sum()       # buys + sells

        gross = float((target * r_filled).sum())
        net = gross - cost_rate * traded
        loser_ret = float(r_filled[losers].mean())

        rows.append({
            "date": date,
            "Momentum (gross)": gross,
            "Momentum (net)": net,
            "Losers": loser_ret,
            "WML": gross - loser_ret,
            "EW universe": float(r[scores.index].mean()),
            "Turnover (one-way)": traded / 2,
        })
        for rank, ticker in enumerate(winners, start=1):
            holding_rows.append({"date": date, "rank": rank,
                                 "ticker": ticker,
                                 "signal": float(scores[ticker])})

        grown = target * (1 + r_filled)
        drifted = grown / grown.sum() if grown.sum() > 0 else target

    results = pd.DataFrame(rows).set_index("date")
    holdings = pd.DataFrame(holding_rows)
    return results, holdings


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def performance_stats(returns: pd.Series, rf: pd.Series | None = None,
                      long_short: bool = False) -> pd.Series:
    """Annualised summary statistics from monthly returns.
    The Sharpe ratio uses excess returns over rf, except for a zero-cost
    long-short portfolio, whose return is already an excess return."""
    returns = returns.dropna()
    n = len(returns)
    wealth = (1 + returns).cumprod()

    if rf is not None and not long_short:
        excess = returns - rf.reindex(returns.index).fillna(0.0)
    else:
        excess = returns

    return pd.Series({
        "Months": n,
        "Average monthly return": returns.mean(),
        "Annualised return (geometric)": wealth.iloc[-1] ** (12 / n) - 1,
        "Annualised volatility": returns.std() * np.sqrt(12),
        "Sharpe ratio": excess.mean() / excess.std() * np.sqrt(12),
        "Maximum drawdown": (wealth / wealth.cummax() - 1).min(),
        "Positive-month rate": (returns > 0).mean(),
        "Best month": returns.max(),
        "Worst month": returns.min(),
    })


def mean_test(x: pd.Series) -> pd.Series:
    """t-test that the mean monthly return is zero."""
    x = x.dropna()
    se = x.std(ddof=1) / np.sqrt(len(x))
    return pd.Series({"Mean monthly": x.mean(),
                      "Annualised mean": x.mean() * 12,
                      "t-stat": x.mean() / se,
                      "Months": len(x)})


def capm_regression(returns: pd.Series, market: pd.Series,
                    rf: pd.Series | None = None) -> pd.Series:
    """OLS of portfolio excess returns on market excess returns
    (classical, non-robust standard errors)."""
    rf_ = (rf.reindex(returns.index).fillna(0.0) if rf is not None
           else pd.Series(0.0, index=returns.index))
    df = pd.DataFrame({"y": returns - rf_,
                       "x": market.reindex(returns.index) - rf_}).dropna()
    X = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    y = df["y"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = len(df) - 2
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return pd.Series({"Alpha (monthly)": coef[0],
                      "Alpha (annualised)": coef[0] * 12,
                      "Alpha t-stat": coef[0] / se[0],
                      "Beta": coef[1],
                      "Beta t-stat": coef[1] / se[1],
                      "R-squared": r2})


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------
def analyse(monthly_prices: pd.DataFrame, rf: pd.Series | None = None,
            lookback=LOOKBACK_MONTHS, skip=SKIP_MONTHS,
            top_fraction=TOP_FRACTION, cost_bps=COST_BPS) -> dict:
    monthly_returns = monthly_prices.pct_change(fill_method=None)
    signal = momentum_signal(monthly_returns, lookback, skip)
    results, holdings = run_backtest(monthly_returns, signal,
                                     top_fraction=top_fraction,
                                     cost_bps=cost_bps)
    results[BENCHMARK] = monthly_returns[BENCHMARK].reindex(results.index)
    results = results.dropna(subset=[BENCHMARK])
    if rf is not None:
        rf = rf.reindex(results.index).fillna(0.0)

    stats = pd.DataFrame({
        "Momentum (net)": performance_stats(results["Momentum (net)"], rf),
        "Momentum (gross)": performance_stats(results["Momentum (gross)"], rf),
        "EW universe": performance_stats(results["EW universe"], rf),
        BENCHMARK: performance_stats(results[BENCHMARK], rf),
        "Losers": performance_stats(results["Losers"], rf),
        "WML (gross)": performance_stats(results["WML"], long_short=True),
    })

    tests = pd.DataFrame({
        "WML (winners - losers)": mean_test(results["WML"]),
        "Momentum net - EW universe":
            mean_test(results["Momentum (net)"] - results["EW universe"]),
        f"Momentum net - {BENCHMARK}":
            mean_test(results["Momentum (net)"] - results[BENCHMARK]),
    })

    capm = pd.DataFrame({
        col: capm_regression(results[col], results[BENCHMARK], rf)
        for col in ["Momentum (net)", "EW universe", "WML"]
    })

    turnover = results["Turnover (one-way)"].iloc[1:]   # exclude initial buy
    return {"monthly_returns": monthly_returns, "results": results,
            "holdings": holdings, "stats": stats, "tests": tests,
            "capm": capm, "avg_turnover": turnover.mean(), "rf": rf}


def robustness_grid(monthly_prices, rf=None,
                    lookbacks=(3, 6, 9, 12),
                    top_fractions=(0.20, 0.30),
                    cost_bps=COST_BPS) -> pd.DataFrame:
    """Re-run the strategy under alternative parameter choices."""
    rows = []
    for lb in lookbacks:
        for tf in top_fractions:
            out = analyse(monthly_prices, rf, lookback=lb,
                          top_fraction=tf, cost_bps=cost_bps)
            s, t = out["stats"], out["tests"]
            rows.append({
                "Lookback (months)": lb,
                "Top fraction": tf,
                "Sharpe (net)": s.loc["Sharpe ratio", "Momentum (net)"],
                "Ann. return (net)":
                    s.loc["Annualised return (geometric)", "Momentum (net)"],
                "WML annualised mean":
                    t.loc["Annualised mean", "WML (winners - losers)"],
                "WML t-stat": t.loc["t-stat", "WML (winners - losers)"],
                "Avg one-way turnover": out["avg_turnover"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
COLOURS = {"Momentum (net)": "#1f4e79", "Momentum (gross)": "#1f4e79",
           "EW universe": "#c55a11", BENCHMARK: "#7f7f7f", "WML": "#2e7d32"}


def _style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False,
    })


def make_figures(results: pd.DataFrame, figure_dir=FIGURE_DIR,
                 show: bool = False) -> list[Path]:
    _style()
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    main_cols = ["Momentum (net)", "EW universe", BENCHMARK]

    def _finish(fig, name):
        fig.tight_layout()
        path = figure_dir / name
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
        if show:
            plt.show()
        else:
            plt.close(fig)

    # 1. Cumulative growth of $1 (log scale)
    wealth = (1 + results[main_cols + ["Momentum (gross)"]]).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for col in main_cols:
        ax.plot(wealth.index, wealth[col], label=col,
                color=COLOURS[col], lw=1.8)
    ax.plot(wealth.index, wealth["Momentum (gross)"], ls="--", lw=1,
            color=COLOURS["Momentum (gross)"], label="Momentum (gross)")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title("Growth of $1: 12-1 momentum vs equal-weight universe and SPY")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend()
    _finish(fig, "fig1_cumulative_growth.png")

    # 2. Drawdowns
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for col in main_cols:
        w = (1 + results[col]).cumprod()
        ax.plot(w.index, (w / w.cummax() - 1) * 100, label=col,
                color=COLOURS[col], lw=1.4)
    ax.set_title("Drawdowns from previous peak")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    _finish(fig, "fig2_drawdowns.png")

    # 3. Rolling 12-month winners-minus-losers return
    rolling_wml = (1 + results["WML"]).rolling(12).apply(np.prod, raw=True) - 1
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(rolling_wml.index, rolling_wml * 100, 0,
                    where=rolling_wml >= 0, color=COLOURS["WML"], alpha=0.5)
    ax.fill_between(rolling_wml.index, rolling_wml * 100, 0,
                    where=rolling_wml < 0, color="#b71c1c", alpha=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Rolling 12-month return of winners minus losers")
    ax.set_ylabel("Return (%)")
    _finish(fig, "fig3_rolling_wml.png")

    # 4. Distribution of monthly returns
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = np.linspace(results[main_cols].min().min(),
                       results[main_cols].max().max(), 35) * 100
    for col in main_cols:
        ax.hist(results[col] * 100, bins=bins, alpha=0.45, label=col,
                color=COLOURS[col])
    ax.set_title("Distribution of monthly returns")
    ax.set_xlabel("Monthly return (%)")
    ax.set_ylabel("Number of months")
    ax.legend()
    _finish(fig, "fig4_return_distribution.png")

    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    prices = download_prices(TICKERS + [BENCHMARK])
    rf = download_risk_free()
    monthly_prices = to_monthly(prices)

    out = analyse(monthly_prices, rf)
    grid = robustness_grid(monthly_prices, rf)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out["results"].to_csv(OUTPUT_DIR / "monthly_returns.csv")
    out["holdings"].to_csv(OUTPUT_DIR / "holdings.csv", index=False)
    out["stats"].to_csv(OUTPUT_DIR / "performance_stats.csv")
    out["tests"].to_csv(OUTPUT_DIR / "significance_tests.csv")
    out["capm"].to_csv(OUTPUT_DIR / "capm_regressions.csv")
    grid.to_csv(OUTPUT_DIR / "robustness_grid.csv", index=False)
    make_figures(out["results"])

    r = out["results"]
    pd.set_option("display.width", 140)
    print(f"\nSample: {r.index[0]:%b %Y} to {r.index[-1]:%b %Y} "
          f"({len(r)} months); costs {COST_BPS} bp per dollar traded")
    print("\nPERFORMANCE\n", out["stats"].round(4))
    print("\nSIGNIFICANCE TESTS (H0: mean = 0)\n", out["tests"].round(4))
    print(f"\nCAPM REGRESSIONS vs {BENCHMARK}\n", out["capm"].round(4))
    print(f"\nAverage one-way monthly turnover: {out['avg_turnover']:.1%}")
    print("\nROBUSTNESS\n", grid.round(3).to_string(index=False))
    print(f"\nTables saved to {OUTPUT_DIR}/, charts saved to {FIGURE_DIR}/")


if __name__ == "__main__":
    main()
