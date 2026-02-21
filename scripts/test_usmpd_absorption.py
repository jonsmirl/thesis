#!/usr/bin/env python3
"""
test_usmpd_absorption.py
========================
Paper 4 (Settlement Feedback): FOMC Information Absorption Speed Test

Uses the SF Fed U.S. Monetary Policy Event-Study Database (USMPD) to test
whether monetary policy information is being absorbed faster over time.

The USMPD provides high-frequency (30-minute and 70-minute window) yield
changes around FOMC statements and press conferences. This is the direct
measurement of the processing delay tau_H from Definition 4.1 in Paper 4.

Five sub-tests:

  A. Statement concentration: fraction of FOMC-day yield move in the 30-min
     statement window (USMPD vs FRED daily). Increasing = faster absorption.

  B. Surprise-normalized response: |dUST10Y| per unit of monetary policy
     surprise (Acosta et al. 2025). Tests yield sensitivity per unit information.

  C. Press conference incremental content: for meetings with PCs, how much
     does the PC add beyond the 30-min statement? Declining = statement absorbs
     more (faster processing).

  D. Press conference reversal rate: does the PC move reverse the statement
     move? Increasing = hasty initial processing, then correction.

  E. Cross-maturity FG signature: ratio of 2Y to 10Y response. FG operates
     at the short end; if FG degrades, 2Y/10Y ratio should decline.

Data sources:
  USMPD.xlsx (SF Fed, Acosta et al. 2025) — 30-min/70-min window changes
  mps.csv (SF Fed) — normalized monetary policy surprises
  FRED DGS10, DGS2 (daily yields, cached from prior tests)

References:
  Acosta, Ajello, Bauer, Loria, and Miranda-Agrippino (2025),
    "Financial Market Effects of FOMC Communication," SF Fed WP 2025-30.
  Nakamura and Steinsson (2018), "High-Frequency Identification of
    Monetary Non-Neutrality," QJE 133(3).

Outputs:
  thesis_data/usmpd_absorption_results.csv
  thesis_data/usmpd_absorption_results.txt
  thesis_data/usmpd_absorption_table.tex
  figures/settlement_feedback/usmpd_absorption.pdf
"""

import os
import sys
import time
import warnings
from datetime import datetime
from io import StringIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════════════════
#  0.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR = "/home/jonsmirl/thesis"
FIG_DIR = os.path.join(BASE_DIR, "figures/settlement_feedback")
DATA_DIR = os.path.join(BASE_DIR, "thesis_data")
CACHE_DIR = os.path.join(DATA_DIR, "fred_cache")

USMPD_PATH = os.path.join(DATA_DIR, "USMPD.xlsx")
MPS_PATH = os.path.join(DATA_DIR, "mp_surprises/mps.csv")

RESULTS_TXT = os.path.join(DATA_DIR, "usmpd_absorption_results.txt")
RESULTS_CSV = os.path.join(DATA_DIR, "usmpd_absorption_results.csv")
RESULTS_TEX = os.path.join(DATA_DIR, "usmpd_absorption_table.tex")
FIG_PATH = os.path.join(FIG_DIR, "usmpd_absorption.pdf")

for d in [FIG_DIR, DATA_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    print("ERROR: FRED_API_KEY not set. Source env.sh first.")
    sys.exit(1)

if not os.path.exists(USMPD_PATH):
    print(f"ERROR: {USMPD_PATH} not found. Download from https://sffed.us/usmpd")
    sys.exit(1)

results_buf = StringIO()


def log(msg=""):
    print(msg)
    results_buf.write(msg + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def fetch_and_cache(series_id, start="2000-01-01"):
    """Fetch FRED series with local CSV caching."""
    cache_path = os.path.join(CACHE_DIR, f"{series_id}.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df.iloc[:, 0]

    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={FRED_API_KEY}"
        f"&file_type=json&observation_start={start}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log(f"  WARNING: Failed to fetch {series_id}: {e}")
        return None

    dates, values = [], []
    for obs in data.get("observations", []):
        if obs["value"] != ".":
            dates.append(obs["date"])
            values.append(float(obs["value"]))
    if not dates:
        return None

    s = pd.Series(values, index=pd.to_datetime(dates), name=series_id)
    s.to_csv(cache_path)
    time.sleep(0.15)
    return s


def load_all_data():
    """Load USMPD sheets, MPS surprises, and FRED daily yields."""
    log("=" * 72)
    log("1. LOADING DATA")
    log("=" * 72)

    # USMPD sheets
    stmt = pd.read_excel(USMPD_PATH, "Statements")
    pc = pd.read_excel(USMPD_PATH, "Press Conferences")
    me = pd.read_excel(USMPD_PATH, "Monetary Events")

    stmt["Date"] = pd.to_datetime(stmt["Date"])
    pc["Date"] = pd.to_datetime(pc["Date"])
    me["Date"] = pd.to_datetime(me["Date"])

    log(f"  Statements: {len(stmt)} meetings, "
        f"{stmt['Date'].min().date()} to {stmt['Date'].max().date()}")
    log(f"  Press Conferences: {len(pc)} meetings")
    log(f"  Monetary Events: {len(me)} meetings")

    # MPS surprises
    mps = pd.read_csv(MPS_PATH)
    mps["Date"] = pd.to_datetime(mps["Date"])
    log(f"  MPS surprises: {len(mps)} meetings, "
        f"PC non-null: {mps['PC'].notna().sum()}")

    # FRED daily yields (for statement-to-daily comparison)
    log("  Fetching FRED DGS10, DGS2...")
    dgs10 = fetch_and_cache("DGS10", "1994-01-01")
    dgs2 = fetch_and_cache("DGS2", "1994-01-01")

    if dgs10 is not None:
        log(f"    DGS10: {len(dgs10)} obs")
    if dgs2 is not None:
        log(f"    DGS2: {len(dgs2)} obs")

    log()
    return stmt, pc, me, mps, dgs10, dgs2


# ═══════════════════════════════════════════════════════════════════════════
#  2.  TEST A — STATEMENT CONCENTRATION RATIO
# ═══════════════════════════════════════════════════════════════════════════

def test_statement_concentration(stmt, dgs10, dgs2):
    """
    Test A: What fraction of the FOMC-day yield move occurs in the 30-min
    statement window?

    Ratio = |USMPD 30-min ΔUST10Y| / |FRED daily ΔDGS10|

    If absorption is faster, more of the day's move should be concentrated
    in the narrow window.
    """
    log("=" * 72)
    log("2. TEST A: STATEMENT CONCENTRATION RATIO")
    log("=" * 72)

    results = {"test": "A_concentration"}

    if dgs10 is None:
        log("  DGS10 not available")
        return results

    # Daily yield changes from FRED
    daily_chg = dgs10.diff().dropna()

    rows = []
    for _, row in stmt.iterrows():
        d = row["Date"]
        stmt_10y = row.get("UST10Y", np.nan)
        stmt_2y = row.get("UST2Y", np.nan)

        if pd.isna(stmt_10y):
            continue

        # Match to FRED daily: try exact date, then ±1 day
        daily_val = None
        for offset in [0, 1, -1]:
            target = d + pd.Timedelta(days=offset)
            if target in daily_chg.index:
                daily_val = daily_chg.loc[target]
                break

        if daily_val is None or abs(daily_val) < 0.001:
            continue

        # Concentration: |30-min move| / |daily move|
        # USMPD units are percentage points, FRED DGS10 is also percentage points
        conc_10y = abs(stmt_10y) / abs(daily_val)

        rows.append({
            "date": d,
            "year": d.year,
            "year_frac": d.year + d.month / 12,
            "stmt_10y": stmt_10y,
            "daily_10y": daily_val,
            "conc_10y": conc_10y,
            "abs_stmt_10y": abs(stmt_10y),
            "abs_daily_10y": abs(daily_val),
        })

    df = pd.DataFrame(rows)
    # Winsorize concentration ratio (can exceed 1 if intraday reversal)
    df["conc_10y_w"] = df["conc_10y"].clip(0, 5)

    log(f"  Matched meetings: {len(df)}")
    log(f"  Concentration ratio range: [{df['conc_10y'].min():.3f}, {df['conc_10y'].max():.3f}]")
    log(f"  Mean: {df['conc_10y_w'].mean():.3f}, Median: {df['conc_10y_w'].median():.3f}")

    # Kendall tau for trend
    tau, p = stats.kendalltau(df["year_frac"], df["conc_10y_w"])
    log(f"  Kendall tau (concentration vs time): {tau:.4f} (p = {p:.4f})")

    if tau > 0 and p < 0.10:
        log("  => CONSISTENT: 30-min window capturing more of daily move (faster absorption)")
    elif tau > 0:
        log("  => DIRECTIONAL: positive trend but not significant")
    else:
        log("  => NOT CONSISTENT: concentration not increasing")

    results["tau"] = tau
    results["p"] = p
    results["n"] = len(df)
    results["df"] = df

    # Era comparison
    bins = [1993, 2000, 2006, 2012, 2018, 2027]
    labels = ["1994-99", "2000-05", "2006-11", "2012-17", "2018-26"]
    df["era"] = pd.cut(df["year"], bins=bins, labels=labels, right=False)
    era_stats = df.groupby("era", observed=True)["conc_10y_w"].agg(["mean", "median", "count"])
    log(f"\n  By era:")
    for era, row in era_stats.iterrows():
        log(f"    {era}: mean={row['mean']:.3f}, median={row['median']:.3f}, n={int(row['count'])}")

    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  3.  TEST B — SURPRISE-NORMALIZED RESPONSE
# ═══════════════════════════════════════════════════════════════════════════

def test_surprise_response(stmt, me, mps):
    """
    Test B: Yield response per unit of monetary policy surprise.

    |ΔUST10Y| / |MPS surprise| — if FG is degrading, the market should be
    less responsive to each unit of policy surprise communicated through
    statements (the guidance channel weakens).

    Uses the Acosta et al. (2025) normalized surprises from mps.csv.
    """
    log("=" * 72)
    log("3. TEST B: SURPRISE-NORMALIZED YIELD RESPONSE")
    log("=" * 72)

    results = {"test": "B_surprise_response"}

    # Merge statement yields with MPS surprises
    merged = stmt[["Date", "UST2Y", "UST10Y", "UST30Y"]].merge(
        mps[["Date", "STMT", "ME"]], on="Date", how="inner"
    )
    merged["year"] = merged["Date"].dt.year
    merged["year_frac"] = merged["Date"].dt.year + merged["Date"].dt.month / 12

    # Filter to meetings with non-trivial surprises
    for surprise_col, label in [("STMT", "Statement"), ("ME", "Monetary Event")]:
        sub = merged.dropna(subset=[surprise_col, "UST10Y"]).copy()
        sub = sub[sub[surprise_col].abs() > 0.001]  # at least 0.1bp surprise

        if len(sub) < 20:
            log(f"  {label}: too few obs ({len(sub)})")
            continue

        # Yield sensitivity = |ΔUST10Y| / |surprise|
        sub["sensitivity_10y"] = sub["UST10Y"].abs() / sub[surprise_col].abs()
        sub["sensitivity_10y_w"] = sub["sensitivity_10y"].clip(0, 20)

        # Also: signed sensitivity (preserving direction)
        sub["signed_sens_10y"] = sub["UST10Y"] / sub[surprise_col]
        sub["signed_sens_10y_w"] = sub["signed_sens_10y"].clip(-20, 20)

        # 2Y sensitivity for FG comparison
        sub_2y = sub.dropna(subset=["UST2Y"]).copy()
        if len(sub_2y) > 20:
            sub_2y["sensitivity_2y"] = sub_2y["UST2Y"].abs() / sub_2y[surprise_col].abs()
            sub_2y["sensitivity_2y_w"] = sub_2y["sensitivity_2y"].clip(0, 20)

        tau_10, p_10 = stats.kendalltau(sub["year_frac"], sub["sensitivity_10y_w"])
        tau_signed, p_signed = stats.kendalltau(sub["year_frac"], sub["signed_sens_10y_w"])

        log(f"\n  --- {label} surprise (N={len(sub)}) ---")
        log(f"  |ΔUST10Y|/|surprise| mean: {sub['sensitivity_10y_w'].mean():.3f}")
        log(f"  Kendall tau (|sensitivity| vs time): {tau_10:.4f} (p = {p_10:.4f})")
        log(f"  Kendall tau (signed sensitivity vs time): {tau_signed:.4f} (p = {p_signed:.4f})")

        results[f"tau_10y_{surprise_col}"] = tau_10
        results[f"p_10y_{surprise_col}"] = p_10
        results[f"n_{surprise_col}"] = len(sub)
        results[f"df_{surprise_col}"] = sub

    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  4.  TEST C — PRESS CONFERENCE INCREMENTAL CONTENT
# ═══════════════════════════════════════════════════════════════════════════

def test_pc_incremental(stmt, me, pc):
    """
    Test C: How much does the press conference add beyond the 30-min statement?

    For meetings with press conferences:
        PC increment = |ME yield change| - |Statement yield change|
        PC share = |PC yield change| / |ME yield change|

    If the statement absorbs more information over time (faster processing),
    the PC's incremental contribution should decline.
    """
    log("=" * 72)
    log("4. TEST C: PRESS CONFERENCE INCREMENTAL CONTENT")
    log("=" * 72)

    results = {"test": "C_pc_incremental"}

    # Get meetings with press conferences
    pc_dates = set(me[me["PC"] == 1]["Date"].values)
    if not pc_dates:
        log("  No press conference meetings found")
        return results

    # Merge statement and monetary event for PC meetings
    m = stmt[["Date", "UST2Y", "UST10Y"]].merge(
        me[me["PC"] == 1][["Date", "UST2Y", "UST10Y"]],
        on="Date", suffixes=("_stmt", "_me")
    )
    m["year"] = m["Date"].dt.year
    m["year_frac"] = m["Date"].dt.year + m["Date"].dt.month / 12

    # Also merge press conference data directly
    m = m.merge(
        pc[["Date", "UST2Y", "UST10Y"]].rename(
            columns={"UST2Y": "UST2Y_pc", "UST10Y": "UST10Y_pc"}),
        on="Date", how="left"
    )

    m = m.dropna(subset=["UST10Y_stmt", "UST10Y_me"])

    # Concentration ratio: |stmt| / |ME|
    m["abs_me_10y"] = m["UST10Y_me"].abs()
    # Filter tiny moves
    m_sig = m[m["abs_me_10y"] > 0.005].copy()
    m_sig["stmt_share_10y"] = m_sig["UST10Y_stmt"].abs() / m_sig["abs_me_10y"]
    m_sig["stmt_share_10y"] = m_sig["stmt_share_10y"].clip(0, 5)

    # PC reversal: does PC move opposite to statement?
    m_sig["reversal"] = np.sign(m_sig["UST10Y_stmt"]) != np.sign(m_sig["UST10Y_pc"])

    log(f"  PC meetings with |ME_10Y| > 0.5bp: {len(m_sig)}")

    if len(m_sig) < 10:
        log("  Too few meetings for analysis")
        return results

    # Trend in statement share
    tau_share, p_share = stats.kendalltau(m_sig["year_frac"], m_sig["stmt_share_10y"])
    log(f"  Statement share (|stmt|/|ME|) mean: {m_sig['stmt_share_10y'].mean():.3f}")
    log(f"  Kendall tau (stmt share vs time): {tau_share:.4f} (p = {p_share:.4f})")

    if tau_share > 0 and p_share < 0.10:
        log("  => CONSISTENT: statement captures more over time")
    elif tau_share > 0:
        log("  => DIRECTIONAL: positive but not significant")
    else:
        log("  => NOT CONSISTENT: statement share not increasing")

    # Reversal rate by era
    bins = [2010, 2016, 2020, 2027]
    labels = ["2011-15", "2016-19", "2020-26"]
    m_sig["era"] = pd.cut(m_sig["year"], bins=bins, labels=labels, right=False)

    log(f"\n  PC reversal rate by era (PC moves opposite to statement):")
    for era in labels:
        sub = m_sig[m_sig["era"] == era]
        if len(sub) > 0:
            rev_rate = sub["reversal"].mean()
            log(f"    {era}: {rev_rate:.1%} ({sub['reversal'].sum()}/{len(sub)})")

    # Overall reversal trend
    tau_rev, p_rev = stats.kendalltau(m_sig["year_frac"], m_sig["reversal"].astype(float))
    log(f"  Kendall tau (reversal rate vs time): {tau_rev:.4f} (p = {p_rev:.4f})")

    results["tau_share"] = tau_share
    results["p_share"] = p_share
    results["tau_reversal"] = tau_rev
    results["p_reversal"] = p_rev
    results["n"] = len(m_sig)
    results["df"] = m_sig

    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  5.  TEST D — CROSS-MATURITY FG SIGNATURE
# ═══════════════════════════════════════════════════════════════════════════

def test_cross_maturity(stmt, mps):
    """
    Test D: Cross-maturity pattern.

    FG operates primarily at the short end (2Y). QE operates at the long end
    (10Y, 30Y). If FG is degrading, the 2Y response per unit surprise should
    decline relative to 10Y.

    Metric: |ΔUST2Y| / |ΔUST10Y| ratio, tested for trend.
    """
    log("=" * 72)
    log("5. TEST D: CROSS-MATURITY FG SIGNATURE")
    log("=" * 72)

    results = {"test": "D_cross_maturity"}

    df = stmt[["Date", "UST2Y", "UST5Y", "UST10Y", "UST30Y"]].copy()
    df["year"] = df["Date"].dt.year
    df["year_frac"] = df["Date"].dt.year + df["Date"].dt.month / 12
    df = df.dropna(subset=["UST2Y", "UST10Y"])

    # Filter to meetings with non-trivial moves
    df = df[(df["UST2Y"].abs() > 0.001) & (df["UST10Y"].abs() > 0.001)].copy()

    # 2Y/10Y ratio
    df["ratio_2y_10y"] = df["UST2Y"].abs() / df["UST10Y"].abs()
    df["ratio_2y_10y_w"] = df["ratio_2y_10y"].clip(0, 10)

    log(f"  Meetings with valid 2Y and 10Y moves: {len(df)}")

    if len(df) < 20:
        log("  Too few observations")
        return results

    tau, p = stats.kendalltau(df["year_frac"], df["ratio_2y_10y_w"])
    log(f"  |ΔUST2Y|/|ΔUST10Y| mean: {df['ratio_2y_10y_w'].mean():.3f}")
    log(f"  Kendall tau (2Y/10Y ratio vs time): {tau:.4f} (p = {p:.4f})")

    if tau < 0 and p < 0.10:
        log("  => CONSISTENT with FG degradation: 2Y response weakening relative to 10Y")
    elif tau < 0:
        log("  => DIRECTIONAL: 2Y response declining relative to 10Y, not significant")
    else:
        log("  => NOT CONSISTENT: 2Y/10Y ratio not declining")

    # Era comparison
    bins = [1993, 2000, 2006, 2012, 2018, 2027]
    labels = ["1994-99", "2000-05", "2006-11", "2012-17", "2018-26"]
    df["era"] = pd.cut(df["year"], bins=bins, labels=labels, right=False)
    era_stats = df.groupby("era", observed=True)["ratio_2y_10y_w"].agg(["mean", "median", "count"])
    log(f"\n  By era:")
    for era, row in era_stats.iterrows():
        log(f"    {era}: mean={row['mean']:.3f}, median={row['median']:.3f}, n={int(row['count'])}")

    # Also: 30Y/10Y ratio as QE channel proxy
    df_30 = df.dropna(subset=["UST30Y"])
    df_30 = df_30[df_30["UST30Y"].abs() > 0.001].copy()
    if len(df_30) > 20:
        df_30["ratio_30y_10y"] = df_30["UST30Y"].abs() / df_30["UST10Y"].abs()
        df_30["ratio_30y_10y_w"] = df_30["ratio_30y_10y"].clip(0, 10)
        tau30, p30 = stats.kendalltau(df_30["year_frac"], df_30["ratio_30y_10y_w"])
        log(f"\n  |ΔUST30Y|/|ΔUST10Y| ratio:")
        log(f"  Kendall tau: {tau30:.4f} (p = {p30:.4f})")
        results["tau_30y_10y"] = tau30
        results["p_30y_10y"] = p30

    results["tau"] = tau
    results["p"] = p
    results["n"] = len(df)
    results["df"] = df

    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  6.  TEST E — ABSOLUTE RESPONSE MAGNITUDE DECLINE
# ═══════════════════════════════════════════════════════════════════════════

def test_response_magnitude(stmt):
    """
    Test E: Has the absolute 30-min yield response to FOMC statements
    declined over time?

    A declining response could mean: (a) less surprise content per meeting
    (Fed is more transparent / better telegraphed), or (b) information
    is being priced in before the 30-min window (pre-positioning).

    Either interpretation supports the FG degradation story: the statement
    itself is becoming less informationally important.
    """
    log("=" * 72)
    log("6. TEST E: ABSOLUTE RESPONSE MAGNITUDE")
    log("=" * 72)

    results = {"test": "E_magnitude"}

    df = stmt[["Date", "UST2Y", "UST10Y"]].copy()
    df["year"] = df["Date"].dt.year
    df["year_frac"] = df["Date"].dt.year + df["Date"].dt.month / 12
    df["abs_10y"] = df["UST10Y"].abs()
    df["abs_2y"] = df["UST2Y"].abs()
    df = df.dropna(subset=["abs_10y"])

    for col, label in [("abs_10y", "10Y"), ("abs_2y", "2Y")]:
        sub = df.dropna(subset=[col])
        tau, p = stats.kendalltau(sub["year_frac"], sub[col])
        log(f"  |ΔUST{label}| mean: {sub[col].mean()*100:.2f} bp")
        log(f"  Kendall tau (|response| vs time): {tau:.4f} (p = {p:.4f})")
        results[f"tau_{label}"] = tau
        results[f"p_{label}"] = p

    # Era means
    bins = [1993, 2000, 2006, 2012, 2018, 2027]
    labels = ["1994-99", "2000-05", "2006-11", "2012-17", "2018-26"]
    df["era"] = pd.cut(df["year"], bins=bins, labels=labels, right=False)
    log(f"\n  Mean |ΔUST10Y| by era (bp):")
    for era in labels:
        sub = df[df["era"] == era]
        if len(sub) > 0:
            log(f"    {era}: {sub['abs_10y'].mean()*100:.2f} bp (n={len(sub)})")

    results["n"] = len(df)
    results["df"] = df

    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  7.  FIGURES (4-panel PDF)
# ═══════════════════════════════════════════════════════════════════════════

def make_figures(conc_res, surp_res, pc_res, xmat_res, mag_res):
    """Generate 4-panel figure."""
    log("=" * 72)
    log("7. GENERATING FIGURES")
    log("=" * 72)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FOMC Information Absorption Speed (USMPD)\n"
                 "Paper 4: Settlement Feedback",
                 fontsize=13, fontweight="bold")

    # ── Panel (a): Statement concentration ratio over time ──
    ax = axes[0, 0]
    df_c = conc_res.get("df")
    if df_c is not None and len(df_c) > 0:
        ax.scatter(df_c["date"], df_c["conc_10y_w"], s=15, alpha=0.4, color="#1f77b4")

        # Rolling median
        df_c_sorted = df_c.sort_values("date")
        roll = df_c_sorted["conc_10y_w"].rolling(30, min_periods=15, center=True).median()
        ax.plot(df_c_sorted["date"], roll, "-", color="#d62728", linewidth=2,
                label="30-meeting rolling median")

        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        tau_c = conc_res.get("tau", np.nan)
        p_c = conc_res.get("p", np.nan)
        if not np.isnan(tau_c):
            ax.annotate(f"Kendall $\\tau$ = {tau_c:.3f} (p = {p_c:.3f})",
                        xy=(0.02, 0.95), xycoords="axes fraction", fontsize=8,
                        va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    ax.set_ylabel("|30-min stmt| / |daily move|")
    ax.set_title("(a) Statement Concentration Ratio")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 4)

    # ── Panel (b): Surprise-normalized 10Y response ──
    ax = axes[0, 1]
    df_s = surp_res.get("df_STMT")
    if df_s is not None and len(df_s) > 0:
        ax.scatter(df_s["Date"], df_s["sensitivity_10y_w"], s=15, alpha=0.4, color="#2ca02c")

        df_s_sorted = df_s.sort_values("Date")
        roll = df_s_sorted["sensitivity_10y_w"].rolling(30, min_periods=15, center=True).median()
        ax.plot(df_s_sorted["Date"], roll, "-", color="#d62728", linewidth=2,
                label="30-meeting rolling median")

        tau_s = surp_res.get("tau_10y_STMT", np.nan)
        p_s = surp_res.get("p_10y_STMT", np.nan)
        if not np.isnan(tau_s):
            ax.annotate(f"Kendall $\\tau$ = {tau_s:.3f} (p = {p_s:.3f})",
                        xy=(0.02, 0.95), xycoords="axes fraction", fontsize=8,
                        va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    ax.set_ylabel("|$\\Delta$UST10Y| / |surprise|")
    ax.set_title("(b) Surprise-Normalized 10Y Response")
    ax.legend(loc="upper right", fontsize=8)

    # ── Panel (c): Cross-maturity ratio ──
    ax = axes[1, 0]
    df_x = xmat_res.get("df")
    if df_x is not None and len(df_x) > 0:
        ax.scatter(df_x["Date"], df_x["ratio_2y_10y_w"], s=15, alpha=0.4, color="#ff7f0e")

        df_x_sorted = df_x.sort_values("Date")
        roll = df_x_sorted["ratio_2y_10y_w"].rolling(30, min_periods=15, center=True).median()
        ax.plot(df_x_sorted["Date"], roll, "-", color="#d62728", linewidth=2,
                label="30-meeting rolling median")

        tau_x = xmat_res.get("tau", np.nan)
        p_x = xmat_res.get("p", np.nan)
        if not np.isnan(tau_x):
            ax.annotate(f"Kendall $\\tau$ = {tau_x:.3f} (p = {p_x:.3f})",
                        xy=(0.02, 0.95), xycoords="axes fraction", fontsize=8,
                        va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("|$\\Delta$UST2Y| / |$\\Delta$UST10Y|")
    ax.set_title("(c) Cross-Maturity Ratio (FG Channel)")
    ax.legend(loc="upper right", fontsize=8)

    # ── Panel (d): Absolute response magnitude ──
    ax = axes[1, 1]
    df_m = mag_res.get("df")
    if df_m is not None and len(df_m) > 0:
        ax.scatter(df_m["Date"], df_m["abs_10y"] * 100, s=15, alpha=0.4,
                   color="#9467bd", label="10Y")

        df_m_sorted = df_m.sort_values("Date")
        roll_10 = df_m_sorted["abs_10y"].rolling(30, min_periods=15, center=True).median() * 100
        ax.plot(df_m_sorted["Date"], roll_10, "-", color="#d62728", linewidth=2,
                label="10Y rolling median")

        if "abs_2y" in df_m.columns:
            roll_2 = df_m_sorted["abs_2y"].rolling(30, min_periods=15, center=True).median() * 100
            ax.plot(df_m_sorted["Date"], roll_2, "--", color="#1f77b4", linewidth=1.5,
                    label="2Y rolling median")

        tau_m = mag_res.get("tau_10Y", np.nan)
        p_m = mag_res.get("p_10Y", np.nan)
        if not np.isnan(tau_m):
            ax.annotate(f"10Y Kendall $\\tau$ = {tau_m:.3f} (p = {p_m:.3f})",
                        xy=(0.02, 0.95), xycoords="axes fraction", fontsize=8,
                        va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    ax.set_ylabel("|$\\Delta$yield| (bp)")
    ax.set_title("(d) Absolute 30-min Statement Response")
    ax.legend(loc="upper right", fontsize=8)

    for ax_row in axes:
        for ax in ax_row:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator(5))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIG_PATH}")
    log()


# ═══════════════════════════════════════════════════════════════════════════
#  8.  LATEX TABLE AND CSV OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def write_outputs(conc_res, surp_res, pc_res, xmat_res, mag_res):
    """Write LaTeX table and CSV."""
    log("=" * 72)
    log("8. WRITING OUTPUTS")
    log("=" * 72)

    # CSV
    csv_rows = []

    def add_row(test, metric, stat, val, p_val=np.nan):
        csv_rows.append({
            "test": test, "metric": metric, "statistic": stat,
            "value": val, "p_value": p_val,
        })

    add_row("A_concentration", "stmt_daily_ratio_trend", "kendall_tau",
            conc_res.get("tau", np.nan), conc_res.get("p", np.nan))

    for key in ["STMT", "ME"]:
        add_row("B_surprise_response", f"sensitivity_10y_{key}", "kendall_tau",
                surp_res.get(f"tau_10y_{key}", np.nan),
                surp_res.get(f"p_10y_{key}", np.nan))

    add_row("C_pc_incremental", "stmt_share_trend", "kendall_tau",
            pc_res.get("tau_share", np.nan), pc_res.get("p_share", np.nan))
    add_row("C_pc_reversal", "reversal_rate_trend", "kendall_tau",
            pc_res.get("tau_reversal", np.nan), pc_res.get("p_reversal", np.nan))

    add_row("D_cross_maturity", "2y_10y_ratio_trend", "kendall_tau",
            xmat_res.get("tau", np.nan), xmat_res.get("p", np.nan))

    add_row("E_magnitude", "abs_10y_trend", "kendall_tau",
            mag_res.get("tau_10Y", np.nan), mag_res.get("p_10Y", np.nan))
    add_row("E_magnitude", "abs_2y_trend", "kendall_tau",
            mag_res.get("tau_2Y", np.nan), mag_res.get("p_2Y", np.nan))

    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(RESULTS_CSV, index=False)
    log(f"  CSV: {RESULTS_CSV}")

    # LaTeX table
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{FOMC information absorption speed (USMPD, 1994--2026)}")
    lines.append(r"\label{tab:usmpd_absorption}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"Test & Metric & Kendall $\tau$ & $p$-value & FG prediction \\")
    lines.append(r"\midrule")

    def tex_row(label, metric, tau_key, p_key, res, pred_dir):
        tau = res.get(tau_key, np.nan)
        p = res.get(p_key, np.nan)
        if np.isnan(tau):
            return f"{label} & {metric} & --- & --- & --- \\\\"
        check = ""
        if pred_dir == "+" and tau > 0 and p < 0.10:
            check = r"$\checkmark$"
        elif pred_dir == "-" and tau < 0 and p < 0.10:
            check = r"$\checkmark$"
        elif pred_dir == "0" and abs(tau) < 0.10:
            check = r"$\checkmark$"
        return f"{label} & {metric} & {tau:.3f} & {p:.3f} & {check} \\\\"

    lines.append(tex_row("A. Concentration", r"|stmt$_{30\text{min}}$|/|daily|",
                         "tau", "p", conc_res, "+"))
    lines.append(tex_row("B. Surprise resp.", r"|$\Delta$UST10Y|/|surprise|",
                         "tau_10y_STMT", "p_10y_STMT", surp_res, "-"))
    lines.append(tex_row("C. PC increment", r"|stmt|/|ME| share",
                         "tau_share", "p_share", pc_res, "+"))
    lines.append(tex_row("C. PC reversal", "reversal rate",
                         "tau_reversal", "p_reversal", pc_res, "+"))
    lines.append(tex_row("D. Cross-maturity", r"|$\Delta$2Y|/|$\Delta$10Y|",
                         "tau", "p", xmat_res, "-"))
    lines.append(tex_row("E. Magnitude", r"|$\Delta$UST10Y|",
                         "tau_10Y", "p_10Y", mag_res, "-"))
    lines.append(tex_row("E. Magnitude", r"|$\Delta$UST2Y|",
                         "tau_2Y", "p_2Y", mag_res, "-"))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{minipage}{0.95\textwidth}")
    lines.append(r"\vspace{0.5em}")
    lines.append(r"\footnotesize\textit{Notes:} High-frequency yield changes from the SF Fed "
                 r"U.S.\ Monetary Policy Event-Study Database (Acosta et al.\ 2025). "
                 r"Statement windows are 30 minutes; monetary events combine statement and "
                 r"press conference windows. Monetary policy surprises are the Acosta et al.\ "
                 r"normalized surprises from money market futures. "
                 r"FG prediction column indicates consistency with forward guidance degradation "
                 r"hypothesis: concentration and PC share should increase ($\tau > 0$); "
                 r"surprise sensitivity, cross-maturity ratio, and absolute magnitude "
                 r"should decrease ($\tau < 0$). "
                 r"All tests use the full 1994--2026 sample except press conference tests "
                 r"(2011--2026, $N \approx 85$).")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")

    with open(RESULTS_TEX, "w") as f:
        f.write("\n".join(lines))
    log(f"  LaTeX: {RESULTS_TEX}")
    log()


# ═══════════════════════════════════════════════════════════════════════════
#  9.  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 72)
    log("FOMC INFORMATION ABSORPTION SPEED TEST (USMPD)")
    log("Paper 4 (Settlement Feedback)")
    log(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 72)
    log()
    log("Source: Acosta, Ajello, Bauer, Loria, Miranda-Agrippino (2025),")
    log('  "Financial Market Effects of FOMC Communication," SF Fed WP 2025-30.')
    log()

    # Load data
    stmt, pc, me, mps, dgs10, dgs2 = load_all_data()

    # Run tests
    conc_res = test_statement_concentration(stmt, dgs10, dgs2)
    surp_res = test_surprise_response(stmt, me, mps)
    pc_res = test_pc_incremental(stmt, me, pc)
    xmat_res = test_cross_maturity(stmt, mps)
    mag_res = test_response_magnitude(stmt)

    # Figures
    make_figures(conc_res, surp_res, pc_res, xmat_res, mag_res)

    # Output files
    write_outputs(conc_res, surp_res, pc_res, xmat_res, mag_res)

    # Summary
    log("=" * 72)
    log("SUMMARY")
    log("=" * 72)

    tests = [
        ("A. Concentration", conc_res.get("tau", np.nan), conc_res.get("p", np.nan),
         "+", "30-min captures more of daily move"),
        ("B. Surprise resp.", surp_res.get("tau_10y_STMT", np.nan),
         surp_res.get("p_10y_STMT", np.nan), "-", "yield less sensitive per unit surprise"),
        ("C. PC share", pc_res.get("tau_share", np.nan), pc_res.get("p_share", np.nan),
         "+", "statement absorbs more, PC adds less"),
        ("C. PC reversal", pc_res.get("tau_reversal", np.nan), pc_res.get("p_reversal", np.nan),
         "+", "PC increasingly reverses statement"),
        ("D. 2Y/10Y ratio", xmat_res.get("tau", np.nan), xmat_res.get("p", np.nan),
         "-", "FG channel (2Y) weakening vs QE (10Y)"),
        ("E. |response| 10Y", mag_res.get("tau_10Y", np.nan), mag_res.get("p_10Y", np.nan),
         "-", "statement less informationally important"),
    ]

    n_consistent = 0
    n_total = 0
    for label, tau, p, pred_dir, desc in tests:
        if np.isnan(tau):
            continue
        n_total += 1
        consistent = False
        if pred_dir == "+" and tau > 0 and p < 0.10:
            consistent = True
        elif pred_dir == "-" and tau < 0 and p < 0.10:
            consistent = True

        if consistent:
            n_consistent += 1

        sign = "+" if tau > 0 else ""
        star = " *" if p < 0.10 else ""
        chk = " <=" if consistent else ""
        log(f"  {label:20s}: tau={sign}{tau:.3f} (p={p:.3f}){star}{chk}")
        log(f"    Prediction: {desc}")

    log(f"\n  Tests consistent with FG degradation: {n_consistent}/{n_total}")

    if n_consistent >= 4:
        log("  => STRONG SUPPORT for pre-transition FG changes")
    elif n_consistent >= 2:
        log("  => MODERATE SUPPORT — some degradation signatures visible")
    elif n_consistent >= 1:
        log("  => WEAK SUPPORT — isolated signals only")
    else:
        log("  => NO SUPPORT — FG appears intact at phi ~ 0 (expected)")

    log()
    log("Interpretation: At phi ~ 0, we expect limited FG degradation.")
    log("Any degradation visible in 1994-2026 reflects pre-mesh technology")
    log("(algorithmic trading, electronic market-making) that partially")
    log("substitutes for the speed advantage mesh agents will bring.")
    log("The baseline documented here enables future difference-in-differences")
    log("detection as autonomous agent participation increases.")
    log()

    # Save results text
    with open(RESULTS_TXT, "w") as f:
        f.write(results_buf.getvalue())
    log(f"Full results saved to: {RESULTS_TXT}")


if __name__ == "__main__":
    main()
