"""
test_stablecoin_treasury.py
============================
Paper 4 (Settlement Feedback), Prediction 4:
    Stablecoin market cap flows move US Treasury yields.

Tests whether growth in aggregate stablecoin circulating supply (predominantly
backed by T-bills and money-market instruments) is associated with changes in
short-term US Treasury yields, controlling for VIX and the fed funds rate.

Data sources:
    - DeFiLlama /stablecoincharts/all   (total stablecoin mcap, daily)
    - FRED DGS1MO, DGS3MO, DGS1, DGS2  (T-bill yields, daily)
    - FRED VIXCLS, DFF                   (controls, daily)

Outputs:
    Figures -> /home/jonsmirl/thesis/figures/settlement_feedback/
    Results -> /home/jonsmirl/thesis/thesis_data/stablecoin_treasury_results.txt
"""

import os
import sys
import json
import warnings
from datetime import datetime, timezone
from io import StringIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────────
FIG_DIR = "/home/jonsmirl/thesis/figures/settlement_feedback"
DATA_DIR = "/home/jonsmirl/thesis/thesis_data"
RESULTS_PATH = os.path.join(DATA_DIR, "stablecoin_treasury_results.txt")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    print("ERROR: FRED_API_KEY not set. Source env.sh first.")
    sys.exit(1)

# ── globals for results text ───────────────────────────────────────────────
results_buf = StringIO()


def log(msg=""):
    """Print to stdout and accumulate for results file."""
    print(msg)
    results_buf.write(msg + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  1.  FETCH STABLECOIN DATA (DeFiLlama)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_stablecoin_mcap() -> pd.DataFrame:
    """
    Fetch total stablecoin market-cap time series from DeFiLlama.
    Uses the /stablecoincharts/all endpoint which returns daily data
    with totalCirculatingUSD.peggedUSD = total USD-pegged stablecoin mcap.
    """
    log("=" * 72)
    log("1. FETCHING STABLECOIN MARKET CAP FROM DeFiLlama")
    log("=" * 72)

    url = "https://stablecoins.llama.fi/stablecoincharts/all"
    log(f"   GET {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    log(f"   Received {len(data)} daily observations")

    records = []
    for obs in data:
        ts = int(obs["date"])
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        # totalCirculatingUSD.peggedUSD gives the USD value of all USD-pegged
        # stablecoins in circulation (USDT, USDC, DAI, BUSD, etc.)
        mcap_usd = obs.get("totalCirculatingUSD", {}).get("peggedUSD", np.nan)
        if mcap_usd is not None and mcap_usd > 0:
            records.append({"date": pd.Timestamp(dt), "stablecoin_mcap": float(mcap_usd)})

    df = pd.DataFrame(records).set_index("date").sort_index()
    # Filter to 2020-01-01 onwards (meaningful stablecoin size + yield data)
    df = df.loc["2020-01-01":]
    log(f"   After filtering to 2020+: {len(df)} observations")
    log(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
    log(f"   Mcap range: ${df['stablecoin_mcap'].min()/1e9:.1f}B to ${df['stablecoin_mcap'].max()/1e9:.1f}B")
    log()
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  2.  FETCH FRED SERIES
# ═══════════════════════════════════════════════════════════════════════════

def fetch_fred_series(series_id: str, start: str = "2020-01-01",
                      end: str = "2026-02-17") -> pd.Series:
    """Fetch a single FRED series and return as a pandas Series indexed by date."""
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}"
        f"&api_key={FRED_API_KEY}"
        f"&file_type=json"
        f"&observation_start={start}"
        f"&observation_end={end}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    obs = resp.json()["observations"]
    records = []
    for o in obs:
        val = o["value"]
        if val == ".":
            continue
        records.append({"date": pd.Timestamp(o["date"]), "value": float(val)})
    s = pd.DataFrame(records).set_index("date")["value"]
    s.name = series_id
    return s


def fetch_treasury_and_controls() -> pd.DataFrame:
    """Fetch T-bill yields and control variables from FRED."""
    log("=" * 72)
    log("2. FETCHING TREASURY YIELDS AND CONTROLS FROM FRED")
    log("=" * 72)

    series_ids = {
        "DGS1MO": "1-Month Treasury",
        "DGS3MO": "3-Month Treasury",
        "DGS1":   "1-Year Treasury",
        "DGS2":   "2-Year Treasury",
        "VIXCLS": "VIX",
        "DFF":    "Fed Funds Rate",
    }

    frames = {}
    for sid, label in series_ids.items():
        s = fetch_fred_series(sid)
        frames[sid] = s
        log(f"   {sid} ({label}): {len(s)} obs, {s.index.min().date()} to {s.index.max().date()}")

    df = pd.DataFrame(frames)
    log(f"   Combined FRED panel: {len(df)} rows")
    log()
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  3.  ALIGN TO WEEKLY (FRIDAY)
# ═══════════════════════════════════════════════════════════════════════════

def align_weekly(stbl_df: pd.DataFrame, fred_df: pd.DataFrame) -> pd.DataFrame:
    """Merge stablecoin and FRED data at weekly (Friday) frequency."""
    log("=" * 72)
    log("3. ALIGNING DATA TO WEEKLY (FRIDAY) FREQUENCY")
    log("=" * 72)

    # Resample stablecoin to weekly Friday (last observation in week)
    stbl_w = stbl_df.resample("W-FRI").last()

    # Resample FRED to weekly Friday (last observation in week)
    fred_w = fred_df.resample("W-FRI").last()

    merged = stbl_w.join(fred_w, how="inner").dropna(subset=["stablecoin_mcap"])

    # Compute log stablecoin mcap
    merged["ln_mcap"] = np.log(merged["stablecoin_mcap"])

    # First differences
    merged["d_ln_mcap"] = merged["ln_mcap"].diff()
    for col in ["DGS1MO", "DGS3MO", "DGS1", "DGS2", "VIXCLS", "DFF"]:
        if col in merged.columns:
            merged[f"d_{col}"] = merged[col].diff()

    merged = merged.dropna()

    log(f"   Merged weekly panel: {len(merged)} rows")
    log(f"   Date range: {merged.index.min().date()} to {merged.index.max().date()}")
    log()
    return merged


# ═══════════════════════════════════════════════════════════════════════════
#  4.  REGRESSIONS
# ═══════════════════════════════════════════════════════════════════════════

def run_regressions(df: pd.DataFrame):
    """Run level, first-difference, Granger, and VAR regressions."""
    log("=" * 72)
    log("4. REGRESSION RESULTS")
    log("=" * 72)

    # Use 3-month yield as the primary dependent variable (most T-bill-like)
    yield_col = "DGS3MO"
    d_yield_col = "d_DGS3MO"

    # ── 4a. Level regression ───────────────────────────────────────────
    log("\n--- 4a. Level Regression: DGS3MO ~ ln(stablecoin_mcap) + VIX + DFF ---")
    subset_level = df[["ln_mcap", yield_col, "VIXCLS", "DFF"]].dropna()
    X_level = sm.add_constant(subset_level[["ln_mcap", "VIXCLS", "DFF"]])
    y_level = subset_level[yield_col]
    model_level = sm.OLS(y_level, X_level).fit(cov_type="HC1")
    log(model_level.summary().as_text())
    log()

    # ── 4b. First-difference regression ────────────────────────────────
    log("\n--- 4b. First-Difference: d_DGS3MO ~ d_ln(mcap) + d_VIX + d_DFF ---")
    fd_cols = ["d_ln_mcap", d_yield_col, "d_VIXCLS", "d_DFF"]
    subset_fd = df[fd_cols].dropna()
    X_fd = sm.add_constant(subset_fd[["d_ln_mcap", "d_VIXCLS", "d_DFF"]])
    y_fd = subset_fd[d_yield_col]
    model_fd = sm.OLS(y_fd, X_fd).fit(cov_type="HC1")
    log(model_fd.summary().as_text())
    log()

    # Also run simple bivariate first-difference (no controls)
    log("\n--- 4b2. Bivariate First-Difference: d_DGS3MO ~ d_ln(mcap) ---")
    X_fd_biv = sm.add_constant(subset_fd["d_ln_mcap"])
    model_fd_biv = sm.OLS(y_fd, X_fd_biv).fit(cov_type="HC1")
    log(model_fd_biv.summary().as_text())
    log()

    # ── 4c. First-difference for all yield maturities ──────────────────
    log("\n--- 4c. First-Difference Across Maturities ---")
    for ycol in ["DGS1MO", "DGS3MO", "DGS1", "DGS2"]:
        dcol = f"d_{ycol}"
        if dcol not in df.columns:
            continue
        sub = df[["d_ln_mcap", dcol, "d_VIXCLS", "d_DFF"]].dropna()
        X_ = sm.add_constant(sub[["d_ln_mcap", "d_VIXCLS", "d_DFF"]])
        y_ = sub[dcol]
        m_ = sm.OLS(y_, X_).fit(cov_type="HC1")
        b = m_.params["d_ln_mcap"]
        se = m_.bse["d_ln_mcap"]
        p = m_.pvalues["d_ln_mcap"]
        log(f"   {ycol}: beta(d_ln_mcap) = {b:+.4f} (SE={se:.4f}, p={p:.4f}), R2={m_.rsquared:.4f}")
    log()

    # ── 4d. Granger causality ──────────────────────────────────────────
    log("\n--- 4d. Granger Causality: d_ln(mcap) -> d_DGS3MO ---")
    gc_data = df[["d_ln_mcap", d_yield_col]].dropna()
    # grangercausalitytests expects [y, x] where we test if x Granger-causes y
    gc_array = gc_data[[d_yield_col, "d_ln_mcap"]].values
    max_lag = 8
    granger_results = {}
    log(f"   Testing lags 1 through {max_lag}:")

    # Suppress verbose output from grangercausalitytests
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gc_out = grangercausalitytests(gc_array, maxlag=max_lag, verbose=False)

    for lag in range(1, max_lag + 1):
        f_stat = gc_out[lag][0]["ssr_ftest"][0]
        f_pval = gc_out[lag][0]["ssr_ftest"][1]
        granger_results[lag] = {"F": f_stat, "p": f_pval}
        sig = "***" if f_pval < 0.01 else "**" if f_pval < 0.05 else "*" if f_pval < 0.10 else ""
        log(f"   Lag {lag}: F = {f_stat:.3f}, p = {f_pval:.4f} {sig}")
    log()

    # Also test reverse: yields -> stablecoin mcap
    log("   Reverse: d_DGS3MO -> d_ln(mcap)")
    gc_array_rev = gc_data[["d_ln_mcap", d_yield_col]].values
    gc_out_rev = grangercausalitytests(gc_array_rev, maxlag=max_lag, verbose=False)
    for lag in range(1, max_lag + 1):
        f_stat = gc_out_rev[lag][0]["ssr_ftest"][0]
        f_pval = gc_out_rev[lag][0]["ssr_ftest"][1]
        sig = "***" if f_pval < 0.01 else "**" if f_pval < 0.05 else "*" if f_pval < 0.10 else ""
        log(f"   Lag {lag}: F = {f_stat:.3f}, p = {f_pval:.4f} {sig}")
    log()

    # ── 4e. VAR impulse response ───────────────────────────────────────
    log("\n--- 4e. VAR(4) Impulse Response: d_ln(mcap) shock -> d_DGS3MO ---")
    var_data = df[["d_ln_mcap", d_yield_col]].dropna()
    # Standardize for cleaner IRF interpretation
    var_df = var_data.copy()
    model_var = VAR(var_df)

    # Select lag order
    lag_order_results = model_var.select_order(maxlags=8)
    log(f"   AIC optimal lag: {lag_order_results.aic}")
    log(f"   BIC optimal lag: {lag_order_results.bic}")

    # Fit VAR(4) as specified, but also note optimal
    var_lag = 4
    var_fit = model_var.fit(var_lag)
    log(f"\n   VAR({var_lag}) Summary:")
    log(f"   AIC = {var_fit.aic:.4f}, BIC = {var_fit.bic:.4f}")

    # Impulse response
    irf = var_fit.irf(periods=20)
    # irf.irfs shape: (periods+1, n_vars, n_vars)
    # We want: shock to d_ln_mcap (index 0) -> response in d_DGS3MO (index 1)
    irf_vals = irf.irfs[:, 0, 1]  # shock to var 0, response in var 1

    # Compute 95% CI using asymptotic standard errors
    irf_se = irf.stderr()[:, 0, 1]
    irf_lower = irf_vals - 1.96 * irf_se
    irf_upper = irf_vals + 1.96 * irf_se

    log(f"\n   IRF (shock to d_ln_mcap -> d_DGS3MO response):")
    for h in range(min(13, len(irf_vals))):
        log(f"     h={h:2d}: {irf_vals[h]:+.6f}  [{irf_lower[h]:+.6f}, {irf_upper[h]:+.6f}]")
    log()

    # Cumulative IRF
    cum_irf = np.cumsum(irf_vals)
    log(f"   Cumulative IRF at h=12: {cum_irf[12]:+.6f}")
    log()

    return (model_level, model_fd, model_fd_biv, granger_results,
            var_fit, irf, var_data)


# ═══════════════════════════════════════════════════════════════════════════
#  5.  FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def fig_timeseries(df: pd.DataFrame):
    """Dual-axis time series: stablecoin mcap + 3-month T-bill yield."""
    log("--- Figure 1: Time Series ---")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color_mcap = "#1f77b4"
    color_yield = "#d62728"

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Stablecoin Market Cap ($ billion)", color=color_mcap)
    ax1.plot(df.index, df["stablecoin_mcap"] / 1e9, color=color_mcap,
             linewidth=1.2, alpha=0.85, label="Total Stablecoin Mcap")
    ax1.tick_params(axis="y", labelcolor=color_mcap)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.set_ylabel("3-Month Treasury Yield (%)", color=color_yield)
    ax2.plot(df.index, df["DGS3MO"], color=color_yield,
             linewidth=1.2, alpha=0.85, label="DGS3MO")
    ax2.tick_params(axis="y", labelcolor=color_yield)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()

    fig.suptitle("Stablecoin Market Cap vs. 3-Month Treasury Yield (Weekly)",
                 fontsize=13, fontweight="bold")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_stablecoin_treasury_timeseries.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    log(f"   Saved: {path}\n")


def fig_regression(df: pd.DataFrame, model_fd):
    """Scatter + fitted line: first differences."""
    log("--- Figure 2: First-Difference Regression ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: bivariate scatter d_DGS3MO vs d_ln_mcap
    ax = axes[0]
    x = df["d_ln_mcap"].dropna()
    y = df["d_DGS3MO"].dropna()
    common = x.index.intersection(y.index)
    x, y = x.loc[common], y.loc[common]

    ax.scatter(x, y, alpha=0.3, s=12, color="#1f77b4", edgecolors="none")
    # Fit line
    slope, intercept, r_val, p_val, se = stats.linregress(x, y)
    x_range = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_range, intercept + slope * x_range, "r-", linewidth=2,
            label=f"OLS: b={slope:.3f} (p={p_val:.4f})")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"$\Delta \ln$(Stablecoin Mcap)")
    ax.set_ylabel(r"$\Delta$ DGS3MO (pp)")
    ax.set_title("Bivariate")
    ax.legend(fontsize=9)

    # Right: partial regression plot from multivariate FD
    ax = axes[1]
    # Partial out controls from both d_ln_mcap and d_DGS3MO
    fd_cols = ["d_ln_mcap", "d_DGS3MO", "d_VIXCLS", "d_DFF"]
    sub = df[fd_cols].dropna()
    controls = sm.add_constant(sub[["d_VIXCLS", "d_DFF"]])

    resid_x = sm.OLS(sub["d_ln_mcap"], controls).fit().resid
    resid_y = sm.OLS(sub["d_DGS3MO"], controls).fit().resid

    ax.scatter(resid_x, resid_y, alpha=0.3, s=12, color="#ff7f0e", edgecolors="none")
    slope2, intercept2, r2, p2, se2 = stats.linregress(resid_x, resid_y)
    x_range2 = np.linspace(resid_x.min(), resid_x.max(), 100)
    ax.plot(x_range2, intercept2 + slope2 * x_range2, "r-", linewidth=2,
            label=f"Partial: b={slope2:.3f} (p={p2:.4f})")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"$\Delta \ln$(Mcap) | Controls")
    ax.set_ylabel(r"$\Delta$ DGS3MO | Controls")
    ax.set_title("Partial Regression (controlling VIX, Fed Funds)")
    ax.legend(fontsize=9)

    fig.suptitle("Stablecoin Mcap Growth vs. Treasury Yield Changes (Weekly First Differences)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_stablecoin_treasury_regression.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    log(f"   Saved: {path}\n")


def fig_irf(irf_obj, var_data):
    """VAR impulse response function plot."""
    log("--- Figure 3: VAR Impulse Response ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    periods = 20

    # Left: shock to d_ln_mcap -> response in d_DGS3MO
    ax = axes[0]
    irf_vals = irf_obj.irfs[:, 0, 1]
    irf_se = irf_obj.stderr()[:, 0, 1]
    irf_lower = irf_vals - 1.96 * irf_se
    irf_upper = irf_vals + 1.96 * irf_se
    weeks = np.arange(len(irf_vals))

    ax.plot(weeks, irf_vals, "b-", linewidth=2, label="Point estimate")
    ax.fill_between(weeks, irf_lower, irf_upper, alpha=0.2, color="blue",
                     label="95% CI")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Weeks after shock")
    ax.set_ylabel("Response in d(DGS3MO)")
    ax.set_title(r"Shock: $\Delta\ln$(Stablecoin Mcap) $\to$ $\Delta$(DGS3MO)")
    ax.legend(fontsize=9)

    # Right: cumulative IRF (use cum_effect_stderr for proper CI)
    ax = axes[1]
    cum_irf = np.cumsum(irf_vals)
    cum_se = irf_obj.cum_effect_stderr()[:, 0, 1]
    cum_lower = cum_irf - 1.96 * cum_se
    cum_upper = cum_irf + 1.96 * cum_se

    ax.plot(weeks, cum_irf, "b-", linewidth=2, label="Cumulative point est.")
    ax.fill_between(weeks, cum_lower, cum_upper, alpha=0.2, color="blue",
                     label="95% CI")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Weeks after shock")
    ax.set_ylabel("Cumulative response in DGS3MO (pp)")
    ax.set_title("Cumulative IRF: Mcap Shock -> 3M Yield Level")
    ax.legend(fontsize=9)

    fig.suptitle("VAR(4) Impulse Response Functions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_stablecoin_treasury_irf.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    log(f"   Saved: {path}\n")


def fig_granger(granger_results: dict):
    """Granger causality F-statistics by lag."""
    log("--- Figure 4: Granger Causality ---")
    fig, ax = plt.subplots(figsize=(8, 5))

    lags = sorted(granger_results.keys())
    f_stats = [granger_results[l]["F"] for l in lags]
    p_vals = [granger_results[l]["p"] for l in lags]

    colors = ["#d62728" if p < 0.05 else "#ff7f0e" if p < 0.10 else "#1f77b4"
              for p in p_vals]

    bars = ax.bar(lags, f_stats, color=colors, edgecolor="black", linewidth=0.5)

    # Add p-value annotations
    for i, (lag, f, p) in enumerate(zip(lags, f_stats, p_vals)):
        label = f"p={p:.3f}"
        ax.text(lag, f + 0.05 * max(f_stats), label, ha="center", va="bottom",
                fontsize=8, fontweight="bold" if p < 0.05 else "normal")

    # Significance thresholds
    ax.axhline(y=stats.f.ppf(0.95, 1, 200), color="gray", linestyle="--",
               linewidth=0.8, label="F crit (5%, approx)")

    ax.set_xlabel("Number of Lags")
    ax.set_ylabel("F-Statistic")
    ax.set_title(r"Granger Causality: $\Delta\ln$(Stablecoin Mcap) $\to$ $\Delta$(DGS3MO)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(lags)

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", edgecolor="black", label="p < 0.05"),
        Patch(facecolor="#ff7f0e", edgecolor="black", label="p < 0.10"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="p >= 0.10"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_stablecoin_treasury_granger.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    log(f"   Saved: {path}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  6.  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 72)
    log("TEST: STABLECOIN MARKET CAP -> TREASURY YIELDS")
    log("Paper 4 (Settlement Feedback), Prediction 4")
    log(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 72)
    log()

    # 1. Fetch data
    stbl_df = fetch_stablecoin_mcap()
    fred_df = fetch_treasury_and_controls()

    # 3. Align to weekly
    weekly = align_weekly(stbl_df, fred_df)

    # Save intermediate data
    csv_path = os.path.join(DATA_DIR, "stablecoin_treasury_weekly.csv")
    weekly.to_csv(csv_path)
    log(f"   Saved weekly panel: {csv_path}")
    log()

    # 4. Regressions
    (model_level, model_fd, model_fd_biv, granger_results,
     var_fit, irf_obj, var_data) = run_regressions(weekly)

    # 5. Summary interpretation
    log("=" * 72)
    log("5. SUMMARY & INTERPRETATION")
    log("=" * 72)

    b_fd = model_fd.params.get("d_ln_mcap", np.nan)
    p_fd = model_fd.pvalues.get("d_ln_mcap", np.nan)
    b_biv = model_fd_biv.params.get("d_ln_mcap", np.nan)
    p_biv = model_fd_biv.pvalues.get("d_ln_mcap", np.nan)

    log(f"\n   First-difference coefficient (with controls):")
    log(f"     beta = {b_fd:+.4f}, p = {p_fd:.4f}")
    log(f"     Interpretation: A 1% increase in stablecoin mcap is associated")
    log(f"     with a {b_fd*0.01:+.4f} pp change in the 3-month T-bill yield.")
    log()
    log(f"   First-difference coefficient (bivariate):")
    log(f"     beta = {b_biv:+.4f}, p = {p_biv:.4f}")
    log()

    # Granger significance
    gc_sig_lags = [l for l, r in granger_results.items() if r["p"] < 0.05]
    if gc_sig_lags:
        log(f"   Granger causality significant at 5% for lags: {gc_sig_lags}")
    else:
        gc_sig_lags_10 = [l for l, r in granger_results.items() if r["p"] < 0.10]
        if gc_sig_lags_10:
            log(f"   Granger causality significant at 10% for lags: {gc_sig_lags_10}")
        else:
            log("   Granger causality NOT significant at conventional levels.")
    log()

    # Paper 4 prediction assessment
    log("   Paper 4 Prediction 4 Assessment:")
    log("   ---------------------------------")
    if p_fd < 0.05:
        log("   SUPPORTED: Stablecoin mcap growth significantly predicts")
        log("   T-bill yield changes (p < 0.05), consistent with the settlement")
        log("   feedback mechanism where stablecoin reserves compete for T-bill supply.")
    elif p_fd < 0.10:
        log("   WEAKLY SUPPORTED: Marginal significance (p < 0.10).")
        log("   Direction consistent with settlement feedback but evidence is not robust.")
    else:
        log("   NOT SUPPORTED at conventional significance levels.")
        log("   However, note that the prediction concerns the future regime where")
        log("   stablecoin mcap reaches Treasury-market-moving scale (~$1T+).")
        log("   Current mcap may be below the threshold for detectable yield effects.")
    log()

    # Economic magnitude context
    current_mcap = weekly["stablecoin_mcap"].iloc[-1]
    log(f"   Current stablecoin mcap: ${current_mcap/1e9:.1f}B")
    log(f"   US Treasury bills outstanding: ~$6T")
    log(f"   Stablecoin / T-bill ratio: {current_mcap/6e12*100:.1f}%")
    log(f"   At this scale, yield effects would be expected to be small")
    log(f"   but growing as stablecoin mcap approaches $500B-$1T.")
    log()

    # 6. Figures
    log("=" * 72)
    log("6. GENERATING FIGURES")
    log("=" * 72)
    fig_timeseries(weekly)
    fig_regression(weekly, model_fd)
    fig_irf(irf_obj, var_data)
    fig_granger(granger_results)

    # 7. Save results
    with open(RESULTS_PATH, "w") as f:
        f.write(results_buf.getvalue())
    print(f"\nResults saved to: {RESULTS_PATH}")
    print(f"Figures saved to: {FIG_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
