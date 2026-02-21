#!/usr/bin/env python3
"""
test_mp_degradation.py
======================
Paper 4 (Settlement Feedback), Prediction: Monetary Policy Degradation Sequence

Tests whether monetary policy tools degrade in the predicted order:
  1. Forward guidance FIRST — FOMC days become less special
  2. QE SECOND — yield compression per $100B declines across episodes
  3. Financial repression LAST — real yield vs savings link still intact

Three sub-tests plus a composite sequencing test:
  A. FOMC importance ratio (rolling 3-year windows)
  B. QE multiplier decay across 4 episodes
  C. Financial repression: real yield vs savings correlation stability

Data sources:
  DGS2, DGS10, DFF, VIXCLS (daily); WALCL (weekly); CPIAUCSL, PSAVERT (monthly)
  All from FRED API, cached to thesis_data/fred_cache/

Outputs:
  thesis_data/mp_degradation_results.csv
  thesis_data/mp_degradation_results.txt
  thesis_data/mp_degradation_table.tex
  figures/settlement_feedback/mp_degradation.pdf

Requires: pandas numpy statsmodels scipy matplotlib requests
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

FIG_DIR = "/home/jonsmirl/thesis/figures/settlement_feedback"
DATA_DIR = "/home/jonsmirl/thesis/thesis_data"
CACHE_DIR = os.path.join(DATA_DIR, "fred_cache")
RESULTS_TXT = os.path.join(DATA_DIR, "mp_degradation_results.txt")
RESULTS_CSV = os.path.join(DATA_DIR, "mp_degradation_results.csv")
RESULTS_TEX = os.path.join(DATA_DIR, "mp_degradation_table.tex")
FIG_PATH = os.path.join(FIG_DIR, "mp_degradation.pdf")

for d in [FIG_DIR, DATA_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    print("ERROR: FRED_API_KEY not set. Source env.sh first.")
    sys.exit(1)

results_buf = StringIO()


def log(msg=""):
    """Print to stdout and accumulate for results file."""
    print(msg)
    results_buf.write(msg + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  1.  FOMC MEETING DATES (flat list, ~200 dates 2000-2026)
# ═══════════════════════════════════════════════════════════════════════════

FOMC_DATES = [
    # 2000
    "2000-02-02", "2000-03-21", "2000-05-16", "2000-06-28",
    "2000-08-22", "2000-10-03", "2000-11-15", "2000-12-19",
    # 2001
    "2001-01-03", "2001-01-31", "2001-03-20", "2001-04-18",
    "2001-05-15", "2001-06-27", "2001-08-21", "2001-09-17",
    "2001-10-02", "2001-11-06", "2001-12-11",
    # 2002
    "2002-01-30", "2002-03-19", "2002-05-07", "2002-06-26",
    "2002-08-13", "2002-09-24", "2002-11-06", "2002-12-10",
    # 2003
    "2003-01-29", "2003-03-18", "2003-05-06", "2003-06-25",
    "2003-08-12", "2003-09-16", "2003-10-28", "2003-12-09",
    # 2004
    "2004-01-28", "2004-03-16", "2004-05-04", "2004-06-30",
    "2004-08-10", "2004-09-21", "2004-11-10", "2004-12-14",
    # 2005
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30",
    "2005-08-09", "2005-09-20", "2005-11-01", "2005-12-13",
    # 2006
    "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29",
    "2006-08-08", "2006-09-20", "2006-10-25", "2006-12-12",
    # 2007
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28",
    "2007-08-07", "2007-08-17", "2007-09-18", "2007-10-31",
    "2007-12-11",
    # 2008
    "2008-01-22", "2008-01-30", "2008-03-18", "2008-04-30",
    "2008-06-25", "2008-08-05", "2008-09-16", "2008-10-08",
    "2008-10-29", "2008-12-16",
    # 2009
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24",
    "2009-08-12", "2009-09-23", "2009-11-04", "2009-12-16",
    # 2010
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23",
    "2010-08-10", "2010-09-21", "2010-11-03", "2010-12-14",
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22",
    "2011-08-09", "2011-09-21", "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20",
    "2012-08-01", "2012-09-13", "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19",
    "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18",
    "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17",
    "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-08-27", "2020-09-16",
    "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026
    "2026-01-28",
]

FOMC_TIMESTAMPS = pd.to_datetime(FOMC_DATES)

# ═══════════════════════════════════════════════════════════════════════════
#  2.  QE EPISODE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

QE_EPISODES = [
    {
        "name": "QE1",
        "announce": "2008-11-25",
        "end": "2010-03-31",
        "label_year": 2009,
    },
    {
        "name": "QE2",
        "announce": "2010-11-03",
        "end": "2011-06-30",
        "label_year": 2011,
    },
    {
        "name": "QE3",
        "announce": "2012-09-13",
        "end": "2014-10-29",
        "label_year": 2013,
    },
    {
        "name": "COVID-QE",
        "announce": "2020-03-15",
        "end": "2022-03-16",
        "label_year": 2021,
    },
]

# ═══════════════════════════════════════════════════════════════════════════
#  3.  DATA FETCHING (fetch_and_cache + load_all_data)
# ═══════════════════════════════════════════════════════════════════════════

SERIES_NEEDED = {
    "DGS2": "2-year Treasury yield (daily)",
    "DGS10": "10-year Treasury yield (daily)",
    "DFF": "Fed funds rate (daily)",
    "VIXCLS": "VIX volatility (daily)",
    "WALCL": "Fed total assets (weekly)",
    "CPIAUCSL": "CPI-U index (monthly)",
    "PSAVERT": "Personal savings rate (monthly)",
}


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
        log(f"  WARNING: No data for {series_id}")
        return None

    s = pd.Series(values, index=pd.to_datetime(dates), name=series_id)
    s.to_csv(cache_path)
    time.sleep(0.15)  # rate limit
    return s


def load_all_data():
    """Fetch all required FRED series, return dict of series."""
    log("=" * 72)
    log("3. FETCHING FRED DATA")
    log("=" * 72)

    series = {}
    for sid, desc in SERIES_NEEDED.items():
        log(f"  Fetching {sid} ({desc})...")
        s = fetch_and_cache(sid)
        if s is not None:
            log(f"    {len(s)} obs, {s.index.min().date()} to {s.index.max().date()}")
            series[sid] = s
        else:
            log(f"    MISSING — test will degrade gracefully")
    log()
    return series


# ═══════════════════════════════════════════════════════════════════════════
#  4.  TEST A — FORWARD GUIDANCE IMPORTANCE RATIO
# ═══════════════════════════════════════════════════════════════════════════

def test_forward_guidance(series):
    """
    Test A: FOMC importance ratio.

    Measures what fraction of total yield variation occurs on FOMC days.
    If forward guidance degrades, FOMC days become less "special" (ratio → 1).

    Returns dict with rolling windows, trend test, and robustness results.
    """
    log("=" * 72)
    log("4. TEST A: FORWARD GUIDANCE — FOMC IMPORTANCE RATIO")
    log("=" * 72)

    results = {"test": "A_forward_guidance", "windows": []}

    for yield_id in ["DGS2", "DGS10"]:
        if yield_id not in series:
            log(f"  {yield_id} not available, skipping")
            continue

        y = series[yield_id].copy()
        y = y.sort_index()
        # Daily absolute changes
        dy = y.diff().abs().dropna()

        # Mark FOMC days (match to nearest trading day within ±1 day)
        fomc_set = set()
        trading_dates = dy.index
        for fdate in FOMC_TIMESTAMPS:
            # Find nearest trading day within 1 day
            mask = (trading_dates >= fdate - pd.Timedelta(days=1)) & \
                   (trading_dates <= fdate + pd.Timedelta(days=1))
            candidates = trading_dates[mask]
            if len(candidates) > 0:
                # Pick closest
                diffs = abs(candidates - fdate)
                fomc_set.add(candidates[diffs.argmin()])

        is_fomc = dy.index.isin(fomc_set)
        log(f"\n  --- {yield_id} ---")
        log(f"  Total trading days with Δy: {len(dy)}")
        log(f"  FOMC days matched: {is_fomc.sum()}")

        # Rolling 3-year windows, stepped every 6 months
        window_days = 756  # ~3 years of trading days
        step_days = 126     # ~6 months of trading days
        windows = []

        start_idx = 0
        while start_idx + window_days <= len(dy):
            window = dy.iloc[start_idx:start_idx + window_days]
            fomc_mask = is_fomc[start_idx:start_idx + window_days]

            fomc_moves = window[fomc_mask]
            all_moves = window

            if len(fomc_moves) >= 5 and len(all_moves) >= 100:
                mean_fomc = fomc_moves.mean()
                mean_all = all_moves.mean()
                ratio = mean_fomc / mean_all if mean_all > 0 else np.nan

                center_date = window.index[len(window) // 2]
                windows.append({
                    "center_date": center_date,
                    "year_frac": center_date.year + center_date.month / 12,
                    "importance_ratio": ratio,
                    "n_fomc": len(fomc_moves),
                    "n_total": len(all_moves),
                    "mean_fomc_bp": mean_fomc * 100,
                    "mean_all_bp": mean_all * 100,
                    "series": yield_id,
                })

            start_idx += step_days

        if not windows:
            log(f"  No valid windows for {yield_id}")
            continue

        wdf = pd.DataFrame(windows)
        results["windows"].extend(windows)

        # Kendall τ for declining trend
        tau, p_tau = stats.kendalltau(wdf["year_frac"], wdf["importance_ratio"])

        log(f"  Rolling windows computed: {len(wdf)}")
        log(f"  Importance ratio range: [{wdf['importance_ratio'].min():.3f}, "
            f"{wdf['importance_ratio'].max():.3f}]")
        log(f"  Kendall τ (ratio vs time): {tau:.4f} (p = {p_tau:.4f})")

        if tau < 0 and p_tau < 0.10:
            log(f"  => CONSISTENT: FOMC days becoming less special (τ < 0, p < 0.10)")
        elif tau < 0:
            log(f"  => DIRECTIONAL: negative trend but not significant")
        else:
            log(f"  => INCONSISTENT: ratio not declining")

        results[f"tau_{yield_id}"] = tau
        results[f"p_tau_{yield_id}"] = p_tau
        results[f"n_windows_{yield_id}"] = len(wdf)

    # VIX-conditional robustness (if VIX available)
    if "VIXCLS" in series and "DGS10" in series:
        log("\n  --- VIX-conditional robustness (DGS10) ---")
        vix = series["VIXCLS"]
        y10 = series["DGS10"]

        # Align daily
        common = y10.index.intersection(vix.index)
        dy10 = y10.reindex(common).diff().abs().dropna()
        vix_aligned = vix.reindex(dy10.index).dropna()
        common_idx = dy10.index.intersection(vix_aligned.index)
        dy10 = dy10.reindex(common_idx)
        vix_aligned = vix_aligned.reindex(common_idx)

        vix_median = vix_aligned.median()
        is_fomc_common = common_idx.isin(fomc_set)

        for label, mask in [("Low VIX", vix_aligned <= vix_median),
                            ("High VIX", vix_aligned > vix_median)]:
            sub_dy = dy10[mask]
            sub_fomc = is_fomc_common[mask]
            fomc_mean = sub_dy[sub_fomc].mean()
            all_mean = sub_dy.mean()
            ratio = fomc_mean / all_mean if all_mean > 0 else np.nan
            log(f"    {label} (VIX median={vix_median:.1f}): "
                f"importance ratio = {ratio:.3f} "
                f"(n_fomc={sub_fomc.sum()}, n_total={len(sub_dy)})")
            results[f"vix_ratio_{label.replace(' ', '_').lower()}"] = ratio

    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  5.  TEST B — QE MULTIPLIER DECAY
# ═══════════════════════════════════════════════════════════════════════════

def find_nearest(target, index, direction="forward"):
    """Find nearest date in index on or after/before target."""
    if direction == "forward":
        candidates = index[index >= target]
        return candidates.min() if len(candidates) > 0 else None
    else:
        candidates = index[index <= target]
        return candidates.max() if len(candidates) > 0 else None


def test_qe_multiplier(series):
    """
    Test B: QE multiplier decay.

    Yield compression per $100B of Fed balance sheet expansion, by QE episode.
    Only 4 data points — descriptive comparison, not formal trend test.

    Returns dict with per-episode multipliers.
    """
    log("=" * 72)
    log("5. TEST B: QE MULTIPLIER DECAY")
    log("=" * 72)

    results = {"test": "B_qe_multiplier", "episodes": []}

    if "DGS10" not in series or "WALCL" not in series:
        log("  Missing DGS10 or WALCL — cannot run Test B")
        return results

    y10 = series["DGS10"].sort_index()
    walcl = series["WALCL"].sort_index()

    for ep in QE_EPISODES:
        name = ep["name"]
        announce = pd.Timestamp(ep["announce"])
        end = pd.Timestamp(ep["end"])

        log(f"\n  --- {name}: {ep['announce']} to {ep['end']} ---")

        # Find yield at announcement and end
        y_ann = find_nearest(announce, y10.index, "forward")
        y_end = find_nearest(end, y10.index, "backward")
        # Also get pre-announcement (1 day before)
        y_pre = find_nearest(announce - pd.Timedelta(days=1), y10.index, "backward")

        # WALCL at announcement and end (weekly, so broader search)
        w_ann = find_nearest(announce - pd.Timedelta(days=7), walcl.index, "forward")
        w_end = find_nearest(end, walcl.index, "backward")

        if any(x is None for x in [y_ann, y_end, y_pre, w_ann, w_end]):
            log(f"    Could not find all dates for {name}, skipping")
            continue

        yield_pre = y10.loc[y_pre]
        yield_ann = y10.loc[y_ann]
        yield_end = y10.loc[y_end]

        walcl_ann = walcl.loc[w_ann]
        walcl_end = walcl.loc[w_end]

        # Full episode multiplier
        d_yield_bps = (yield_end - yield_pre) * 100  # convert pp to bp
        d_walcl_B = (walcl_end - walcl_ann) / 1e3    # WALCL in millions → billions
        multiplier = d_yield_bps / (d_walcl_B / 100) if abs(d_walcl_B) > 1 else np.nan

        # Announcement effect (first 2 weeks)
        two_weeks = announce + pd.Timedelta(days=14)
        y_2w = find_nearest(two_weeks, y10.index, "backward")
        if y_2w is not None:
            ann_effect_bps = (y10.loc[y_2w] - yield_pre) * 100
        else:
            ann_effect_bps = np.nan

        # 6-month fixed window variant
        six_months = announce + pd.Timedelta(days=182)
        y_6m = find_nearest(six_months, y10.index, "backward")
        w_6m = find_nearest(six_months, walcl.index, "backward")
        if y_6m is not None and w_6m is not None:
            d_yield_6m = (y10.loc[y_6m] - yield_pre) * 100
            d_walcl_6m = (walcl.loc[w_6m] - walcl_ann) / 1e3
            mult_6m = d_yield_6m / (d_walcl_6m / 100) if abs(d_walcl_6m) > 1 else np.nan
        else:
            mult_6m = np.nan

        episode_data = {
            "name": name,
            "announce": ep["announce"],
            "end": ep["end"],
            "label_year": ep["label_year"],
            "yield_pre_pp": yield_pre,
            "yield_end_pp": yield_end,
            "d_yield_bps": d_yield_bps,
            "walcl_ann_B": walcl_ann / 1e3,
            "walcl_end_B": walcl_end / 1e3,
            "d_walcl_B": d_walcl_B,
            "multiplier": multiplier,
            "ann_effect_bps": ann_effect_bps,
            "mult_6m": mult_6m,
        }
        results["episodes"].append(episode_data)

        log(f"    DGS10: {yield_pre:.2f}% → {yield_end:.2f}% (Δ = {d_yield_bps:+.1f} bp)")
        log(f"    WALCL: ${walcl_ann/1e3:.0f}B → ${walcl_end/1e3:.0f}B (Δ = ${d_walcl_B:+.0f}B)")
        log(f"    Multiplier (bp per $100B): {multiplier:.2f}" if not np.isnan(multiplier) else
            f"    Multiplier: N/A")
        log(f"    Announcement effect (2 weeks): {ann_effect_bps:+.1f} bp" if not np.isnan(ann_effect_bps) else
            f"    Announcement effect: N/A")
        log(f"    6-month multiplier: {mult_6m:.2f}" if not np.isnan(mult_6m) else
            f"    6-month multiplier: N/A")

    # Summary comparison
    if len(results["episodes"]) >= 2:
        log("\n  --- Summary: multiplier comparison ---")
        for ep in results["episodes"]:
            m = ep["multiplier"]
            m_str = f"{m:.2f}" if not np.isnan(m) else "N/A"
            log(f"    {ep['name']:10s}: {m_str} bp/$100B")

        mults = [ep["multiplier"] for ep in results["episodes"] if not np.isnan(ep["multiplier"])]
        if len(mults) >= 3:
            # Check if declining: each episode should have smaller absolute multiplier
            # (yield compression is negative, so multiplier is negative; declining = less negative)
            abs_mults = [abs(m) for m in mults]
            monotone_declining = all(abs_mults[i] >= abs_mults[i+1] for i in range(len(abs_mults)-1))
            log(f"\n    Absolute multiplier declining across episodes: {monotone_declining}")
            results["monotone_declining"] = monotone_declining

        # Note about Operation Twist
        log("\n  Note: Operation Twist (2011-09 to 2012-12) excluded as natural placebo")
        log("  (no net WALCL change — long-end purchased, short-end sold)")
        log("  Note: COVID episode has confounds (pandemic shock, fiscal stimulus)")

    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  6.  TEST C — FINANCIAL REPRESSION STABILITY
# ═══════════════════════════════════════════════════════════════════════════

def test_financial_repression(series):
    """
    Test C: Financial repression — real yield vs savings correlation stability.

    Measures rolling 60-month correlation between real yield and personal savings rate.
    Prediction: correlation is stable (τ ≈ 0) because φ ≈ 0 and S < S_crit.

    Returns dict with rolling correlations and trend test.
    """
    log("=" * 72)
    log("6. TEST C: FINANCIAL REPRESSION — REAL YIELD VS SAVINGS")
    log("=" * 72)

    results = {"test": "C_financial_repression", "windows": []}

    if "DGS10" not in series or "CPIAUCSL" not in series or "PSAVERT" not in series:
        log("  Missing required series (DGS10, CPIAUCSL, or PSAVERT)")
        return results

    y10 = series["DGS10"].sort_index()
    cpi = series["CPIAUCSL"].sort_index()
    psavert = series["PSAVERT"].sort_index()

    # Convert DGS10 daily to monthly average
    y10_monthly = y10.resample("MS").mean().dropna()

    # Trailing 12-month CPI inflation (annualized)
    cpi_inflation = cpi.pct_change(12) * 100  # percent
    cpi_inflation = cpi_inflation.dropna()

    # Align all three to common monthly dates
    common = y10_monthly.index.intersection(cpi_inflation.index).intersection(psavert.index)
    y10_m = y10_monthly.reindex(common)
    inf_m = cpi_inflation.reindex(common)
    sav_m = psavert.reindex(common)

    # Real yield = nominal - inflation
    real_yield = y10_m - inf_m

    log(f"  Monthly obs aligned: {len(common)}")
    log(f"  Date range: {common.min().date()} to {common.max().date()}")
    log(f"  Real yield range: [{real_yield.min():.2f}%, {real_yield.max():.2f}%]")
    log(f"  Savings rate range: [{sav_m.min():.1f}%, {sav_m.max():.1f}%]")

    # Full-sample correlation
    full_corr, full_p = stats.pearsonr(real_yield.dropna(), sav_m.reindex(real_yield.dropna().index))
    log(f"  Full-sample correlation (real yield vs savings): {full_corr:.4f} (p = {full_p:.4f})")

    # Rolling 60-month correlation
    window = 60
    rolling_corrs = []
    for i in range(len(common) - window + 1):
        ry_win = real_yield.iloc[i:i+window]
        sv_win = sav_m.iloc[i:i+window]
        if ry_win.notna().sum() >= 48 and sv_win.notna().sum() >= 48:
            c, _ = stats.pearsonr(ry_win.dropna(), sv_win.reindex(ry_win.dropna().index).dropna())
            center = common[i + window // 2]
            rolling_corrs.append({
                "center_date": center,
                "year_frac": center.year + center.month / 12,
                "correlation": c,
            })

    if not rolling_corrs:
        log("  No valid rolling windows")
        return results

    rdf = pd.DataFrame(rolling_corrs)
    results["windows"] = rolling_corrs

    # Kendall τ for trend
    tau, p_tau = stats.kendalltau(rdf["year_frac"], rdf["correlation"])

    log(f"  Rolling 60-month windows: {len(rdf)}")
    log(f"  Rolling corr range: [{rdf['correlation'].min():.3f}, {rdf['correlation'].max():.3f}]")
    log(f"  Kendall τ (correlation vs time): {tau:.4f} (p = {p_tau:.4f})")

    if abs(tau) < 0.10:
        log(f"  => CONSISTENT: FR correlation stable (|τ| < 0.10)")
    elif tau < -0.10 and p_tau < 0.10:
        log(f"  => FR may be degrading (negative trend, p < 0.10)")
    else:
        log(f"  => AMBIGUOUS: τ = {tau:.3f}")

    results["tau"] = tau
    results["p_tau"] = p_tau
    results["full_corr"] = full_corr
    results["n_windows"] = len(rdf)

    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  7.  SEQUENCING TEST — COMPOSITE NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_sequencing(fg_results, qe_results, fr_results):
    """
    Normalize all three degradation metrics to [0, 1] and test ordering.

    FG: 1 - (ratio_t / ratio_max)       → higher = more degraded
    QE: 1 - (|multiplier| / |QE1 mult|) → higher = more degraded
    FR: 1 - (|corr_t| / |corr_max|)     → higher = more degraded

    Returns dict with composite timeline and ordering assessment.
    """
    log("=" * 72)
    log("7. SEQUENCING TEST: COMPOSITE DEGRADATION INDEX")
    log("=" * 72)

    results = {"test": "sequencing", "fg_norm": [], "qe_norm": [], "fr_norm": []}

    # FG normalization: use DGS10 windows
    fg_windows = [w for w in fg_results.get("windows", []) if w["series"] == "DGS10"]
    if fg_windows:
        fg_df = pd.DataFrame(fg_windows)
        ratio_max = fg_df["importance_ratio"].max()
        if ratio_max > 0:
            fg_df["degradation"] = 1.0 - (fg_df["importance_ratio"] / ratio_max)
            for _, row in fg_df.iterrows():
                results["fg_norm"].append({
                    "year_frac": row["year_frac"],
                    "degradation": row["degradation"],
                })
        log(f"  FG: {len(fg_df)} windows, max ratio = {ratio_max:.3f}")
    else:
        log("  FG: no windows available")

    # QE normalization: use full-episode multiplier
    qe_episodes = qe_results.get("episodes", [])
    if qe_episodes:
        abs_mults = [(ep["label_year"], abs(ep["multiplier"]))
                     for ep in qe_episodes if not np.isnan(ep["multiplier"])]
        if abs_mults:
            qe1_mult = abs_mults[0][1]  # first episode as reference
            if qe1_mult > 0:
                for year, m in abs_mults:
                    deg = 1.0 - (m / qe1_mult)
                    deg = max(0.0, min(1.0, deg))  # clip to [0, 1]
                    results["qe_norm"].append({
                        "year_frac": year,
                        "degradation": deg,
                    })
                log(f"  QE: {len(abs_mults)} episodes, QE1 ref = {qe1_mult:.2f}")
    else:
        log("  QE: no episodes available")

    # FR normalization: use rolling correlations
    fr_windows = fr_results.get("windows", [])
    if fr_windows:
        fr_df = pd.DataFrame(fr_windows)
        corr_max = fr_df["correlation"].abs().max()
        if corr_max > 0:
            fr_df["degradation"] = 1.0 - (fr_df["correlation"].abs() / corr_max)
            for _, row in fr_df.iterrows():
                results["fr_norm"].append({
                    "year_frac": row["year_frac"],
                    "degradation": row["degradation"],
                })
        log(f"  FR: {len(fr_df)} windows, max |corr| = {corr_max:.3f}")
    else:
        log("  FR: no windows available")

    # Ordering assessment
    log("\n  --- Ordering assessment ---")

    # Average degradation in recent period (2018-2026) vs early period (2001-2010)
    def avg_degradation(norm_list, lo, hi):
        vals = [x["degradation"] for x in norm_list if lo <= x["year_frac"] <= hi]
        return np.mean(vals) if vals else np.nan

    recent_fg = avg_degradation(results["fg_norm"], 2018, 2027)
    recent_qe = avg_degradation(results["qe_norm"], 2018, 2027)
    recent_fr = avg_degradation(results["fr_norm"], 2018, 2027)

    log(f"  Recent-period (2018-2026) average degradation:")
    log(f"    FG: {recent_fg:.3f}" if not np.isnan(recent_fg) else "    FG: N/A")
    log(f"    QE: {recent_qe:.3f}" if not np.isnan(recent_qe) else "    QE: N/A")
    log(f"    FR: {recent_fr:.3f}" if not np.isnan(recent_fr) else "    FR: N/A")

    # Check predicted ordering: FG > QE > FR
    if not any(np.isnan(x) for x in [recent_fg, recent_qe, recent_fr]):
        fg_gt_qe = recent_fg > recent_qe
        qe_gt_fr = recent_qe > recent_fr
        full_order = fg_gt_qe and qe_gt_fr

        log(f"\n  Predicted ordering FG > QE > FR:")
        log(f"    FG > QE: {fg_gt_qe} ({recent_fg:.3f} vs {recent_qe:.3f})")
        log(f"    QE > FR: {qe_gt_fr} ({recent_qe:.3f} vs {recent_fr:.3f})")
        log(f"    Full ordering: {'CONFIRMED' if full_order else 'NOT CONFIRMED'}")

        results["fg_gt_qe"] = fg_gt_qe
        results["qe_gt_fr"] = qe_gt_fr
        results["full_order"] = full_order
    else:
        log("  Cannot assess ordering — insufficient data")

    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  8.  FIGURES (4-panel PDF)
# ═══════════════════════════════════════════════════════════════════════════

def make_figures(fg_results, qe_results, fr_results, seq_results):
    """Generate 4-panel figure for mp_degradation.pdf."""
    log("=" * 72)
    log("8. GENERATING FIGURES")
    log("=" * 72)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Monetary Policy Degradation Sequence Test\n"
                 "(Paper 4: Settlement Feedback)",
                 fontsize=13, fontweight="bold")

    # ── Panel (a): FOMC importance ratio over time ──
    ax = axes[0, 0]
    for yield_id, color, marker in [("DGS2", "#1f77b4", "o"), ("DGS10", "#d62728", "s")]:
        wdata = [w for w in fg_results.get("windows", []) if w["series"] == yield_id]
        if wdata:
            wdf = pd.DataFrame(wdata)
            dates = pd.to_datetime([f"{int(y)}-{max(1,int((y%1)*12)):02d}-15"
                                    for y in wdf["year_frac"]])
            ax.plot(dates, wdf["importance_ratio"], f"-{marker}",
                    color=color, markersize=4, alpha=0.8, label=yield_id)

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Importance Ratio\n(mean |Δy| FOMC / mean |Δy| all)")
    ax.set_title("(a) Forward Guidance: FOMC Day Importance")
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))

    # Add trend annotation
    tau_10 = fg_results.get("tau_DGS10", np.nan)
    p_10 = fg_results.get("p_tau_DGS10", np.nan)
    if not np.isnan(tau_10):
        ax.annotate(f"DGS10 Kendall τ = {tau_10:.3f} (p = {p_10:.3f})",
                    xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    # ── Panel (b): QE multiplier by episode ──
    ax = axes[0, 1]
    episodes = qe_results.get("episodes", [])
    if episodes:
        names = [ep["name"] for ep in episodes]
        mults = [ep["multiplier"] for ep in episodes]
        ann_effects = [ep["ann_effect_bps"] for ep in episodes]

        x_pos = np.arange(len(names))
        bars = ax.bar(x_pos, mults, width=0.6, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
                      alpha=0.8, label="Full episode")

        # Add value labels on bars
        for i, (bar, m) in enumerate(zip(bars, mults)):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width()/2, m + (0.3 if m >= 0 else -0.8),
                        f"{m:.1f}", ha="center", fontsize=9, fontweight="bold")

        # Announcement effect as markers
        valid_ann = [(i, a) for i, a in enumerate(ann_effects) if not np.isnan(a)]
        if valid_ann:
            ax.scatter([x_pos[i] for i, _ in valid_ann],
                       [a for _, a in valid_ann],
                       marker="D", color="black", s=60, zorder=5,
                       label="2-week announcement")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel("bp per $100B WALCL expansion")
        ax.set_title("(b) QE Multiplier by Episode")
        ax.legend(loc="best", fontsize=8)
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No QE data available", transform=ax.transAxes,
                ha="center", fontsize=12)
        ax.set_title("(b) QE Multiplier by Episode")

    # ── Panel (c): Real yield vs savings correlation ──
    ax = axes[1, 0]
    fr_windows = fr_results.get("windows", [])
    if fr_windows:
        frdf = pd.DataFrame(fr_windows)
        dates = pd.to_datetime([f"{int(y)}-{max(1,int((y%1)*12)):02d}-15"
                                for y in frdf["year_frac"]])
        ax.plot(dates, frdf["correlation"], "-", color="#2ca02c", linewidth=1.5)
        ax.fill_between(dates, frdf["correlation"], 0, alpha=0.2, color="#2ca02c")
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)

        tau_fr = fr_results.get("tau", np.nan)
        p_fr = fr_results.get("p_tau", np.nan)
        if not np.isnan(tau_fr):
            ax.annotate(f"Kendall τ = {tau_fr:.3f} (p = {p_fr:.3f})",
                        xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    else:
        ax.text(0.5, 0.5, "No FR data available", transform=ax.transAxes,
                ha="center", fontsize=12)

    ax.set_ylabel("Rolling 60-month correlation\n(real yield vs savings rate)")
    ax.set_title("(c) Financial Repression: Real Yield–Savings Link")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))

    # ── Panel (d): Composite degradation index ──
    ax = axes[1, 1]

    for label, norm_data, color, marker in [
        ("Forward Guidance", seq_results.get("fg_norm", []), "#1f77b4", "o"),
        ("QE Effectiveness", seq_results.get("qe_norm", []), "#ff7f0e", "D"),
        ("Financial Repression", seq_results.get("fr_norm", []), "#2ca02c", "s"),
    ]:
        if norm_data:
            ndf = pd.DataFrame(norm_data)
            dates = pd.to_datetime([f"{int(y)}-{max(1,int((y%1)*12)):02d}-15"
                                    for y in ndf["year_frac"]])
            ax.plot(dates, ndf["degradation"], f"-{marker}",
                    color=color, markersize=4, alpha=0.8, label=label,
                    linewidth=1.5 if label != "QE Effectiveness" else 0)
            if label == "QE Effectiveness":
                ax.scatter(dates, ndf["degradation"], color=color, marker=marker,
                           s=80, zorder=5, label=label)

    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(1, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Degradation Index (0 = intact, 1 = fully degraded)")
    ax.set_title("(d) Composite: Predicted Sequencing FG → QE → FR")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(-0.1, 1.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIG_PATH}")
    log()


# ═══════════════════════════════════════════════════════════════════════════
#  9.  LATEX TABLE AND CSV OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def write_latex_table(fg_results, qe_results, fr_results, seq_results):
    """Write publication-ready LaTeX table."""
    log("=" * 72)
    log("9. WRITING OUTPUTS")
    log("=" * 72)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Monetary policy degradation sequence test}")
    lines.append(r"\label{tab:mp_degradation}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Tool & Metric & Statistic & Value & $p$-value & Prediction \\")
    lines.append(r"\midrule")

    # FG rows
    for yield_id in ["DGS2", "DGS10"]:
        tau = fg_results.get(f"tau_{yield_id}", np.nan)
        p = fg_results.get(f"p_tau_{yield_id}", np.nan)
        n = fg_results.get(f"n_windows_{yield_id}", 0)
        if not np.isnan(tau):
            verdict = r"$\checkmark$" if tau < 0 and p < 0.10 else "---"
            lines.append(f"Forward Guidance & {yield_id} importance & "
                         f"Kendall $\\tau$ & {tau:.3f} & {p:.3f} & {verdict} \\\\")

    lines.append(r"\midrule")

    # QE rows
    for ep in qe_results.get("episodes", []):
        m = ep["multiplier"]
        m_str = f"{m:.1f}" if not np.isnan(m) else "---"
        a = ep["ann_effect_bps"]
        a_str = f"{a:+.0f}" if not np.isnan(a) else "---"
        lines.append(f"QE & {ep['name']} multiplier & bp/\\$100B & {m_str} & --- & --- \\\\")

    lines.append(r"\midrule")

    # FR row
    tau_fr = fr_results.get("tau", np.nan)
    p_fr = fr_results.get("p_tau", np.nan)
    full_corr = fr_results.get("full_corr", np.nan)
    if not np.isnan(tau_fr):
        verdict = r"$\checkmark$" if abs(tau_fr) < 0.10 else "---"
        lines.append(f"Fin.\\ Repression & real yield--savings corr & "
                     f"Kendall $\\tau$ & {tau_fr:.3f} & {p_fr:.3f} & {verdict} \\\\")

    lines.append(r"\midrule")

    # Sequencing row
    full_order = seq_results.get("full_order", None)
    if full_order is not None:
        verdict = r"$\checkmark$" if full_order else "---"
        lines.append(f"Sequencing & FG $>$ QE $>$ FR & ordering & "
                     f"{'confirmed' if full_order else 'not confirmed'} & --- & {verdict} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{minipage}{0.95\textwidth}")
    lines.append(r"\vspace{0.5em}")
    lines.append(r"\footnotesize\textit{Notes:} Forward guidance importance ratio = "
                 r"mean $|\Delta y|$ on FOMC days / mean $|\Delta y|$ all days, "
                 r"computed over rolling 3-year windows stepped every 6 months. "
                 r"QE multiplier = yield change (bp) per \$100B of Fed balance sheet expansion. "
                 r"Financial repression metric = rolling 60-month Pearson correlation "
                 r"between real 10-year yield and personal savings rate. "
                 r"Prediction column: $\checkmark$ if consistent with Paper 4 sequencing "
                 r"hypothesis (FG degrades first, then QE, FR last). "
                 r"Data: FRED series DGS2, DGS10, WALCL, CPIAUCSL, PSAVERT, 2000--2026.")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")

    with open(RESULTS_TEX, "w") as f:
        f.write("\n".join(lines))
    log(f"  LaTeX table: {RESULTS_TEX}")

    # CSV with all test statistics
    rows = []

    # FG stats
    for yield_id in ["DGS2", "DGS10"]:
        tau = fg_results.get(f"tau_{yield_id}", np.nan)
        p = fg_results.get(f"p_tau_{yield_id}", np.nan)
        rows.append({
            "test": "A_forward_guidance",
            "series": yield_id,
            "metric": "importance_ratio_trend",
            "statistic": "kendall_tau",
            "value": tau,
            "p_value": p,
        })

    # QE stats
    for ep in qe_results.get("episodes", []):
        rows.append({
            "test": "B_qe_multiplier",
            "series": ep["name"],
            "metric": "multiplier_bp_per_100B",
            "statistic": "point_estimate",
            "value": ep["multiplier"],
            "p_value": np.nan,
        })
        rows.append({
            "test": "B_qe_multiplier",
            "series": ep["name"],
            "metric": "announcement_effect_bps",
            "statistic": "point_estimate",
            "value": ep["ann_effect_bps"],
            "p_value": np.nan,
        })

    # FR stats
    rows.append({
        "test": "C_financial_repression",
        "series": "DGS10-CPIAUCSL-PSAVERT",
        "metric": "correlation_trend",
        "statistic": "kendall_tau",
        "value": fr_results.get("tau", np.nan),
        "p_value": fr_results.get("p_tau", np.nan),
    })
    rows.append({
        "test": "C_financial_repression",
        "series": "DGS10-CPIAUCSL-PSAVERT",
        "metric": "full_sample_correlation",
        "statistic": "pearson_r",
        "value": fr_results.get("full_corr", np.nan),
        "p_value": np.nan,
    })

    # Sequencing
    rows.append({
        "test": "D_sequencing",
        "series": "composite",
        "metric": "ordering_FG_QE_FR",
        "statistic": "boolean",
        "value": 1.0 if seq_results.get("full_order", False) else 0.0,
        "p_value": np.nan,
    })

    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(RESULTS_CSV, index=False)
    log(f"  CSV results: {RESULTS_CSV}")

    log()


# ═══════════════════════════════════════════════════════════════════════════
#  10.  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 72)
    log("MONETARY POLICY DEGRADATION SEQUENCE TEST")
    log("Paper 4 (Settlement Feedback)")
    log(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 72)
    log()
    log("Prediction: monetary policy tools degrade in sequence:")
    log("  1. Forward guidance FIRST (FOMC days less special)")
    log("  2. QE SECOND (lower yield compression per $100B)")
    log("  3. Financial repression LAST (still functional at φ ≈ 0)")
    log()

    # Fetch data
    series = load_all_data()

    # Run three sub-tests
    fg_results = test_forward_guidance(series)
    qe_results = test_qe_multiplier(series)
    fr_results = test_financial_repression(series)

    # Sequencing test
    seq_results = compute_sequencing(fg_results, qe_results, fr_results)

    # Figures
    make_figures(fg_results, qe_results, fr_results, seq_results)

    # Output files
    write_latex_table(fg_results, qe_results, fr_results, seq_results)

    # Summary
    log("=" * 72)
    log("SUMMARY")
    log("=" * 72)

    log("\nTest A (Forward Guidance):")
    tau_2 = fg_results.get("tau_DGS2", np.nan)
    tau_10 = fg_results.get("tau_DGS10", np.nan)
    if not np.isnan(tau_2):
        log(f"  DGS2  importance ratio trend: τ = {tau_2:.4f} "
            f"(p = {fg_results.get('p_tau_DGS2', np.nan):.4f})")
    if not np.isnan(tau_10):
        log(f"  DGS10 importance ratio trend: τ = {tau_10:.4f} "
            f"(p = {fg_results.get('p_tau_DGS10', np.nan):.4f})")

    log("\nTest B (QE Multiplier):")
    for ep in qe_results.get("episodes", []):
        m = ep["multiplier"]
        log(f"  {ep['name']:10s}: {m:.1f} bp/$100B" if not np.isnan(m) else
            f"  {ep['name']:10s}: N/A")

    log("\nTest C (Financial Repression):")
    tau_fr = fr_results.get("tau", np.nan)
    if not np.isnan(tau_fr):
        log(f"  Correlation stability: τ = {tau_fr:.4f} "
            f"(p = {fr_results.get('p_tau', np.nan):.4f})")

    log("\nSequencing:")
    full_order = seq_results.get("full_order", None)
    if full_order is not None:
        log(f"  FG > QE > FR ordering: {'CONFIRMED' if full_order else 'NOT CONFIRMED'}")
    else:
        log("  Insufficient data for ordering assessment")

    # Overall verdict
    log("\n" + "-" * 50)
    fg_degrading = (not np.isnan(tau_10) and tau_10 < 0)
    fr_stable = (not np.isnan(tau_fr) and abs(tau_fr) < 0.15)

    if fg_degrading and fr_stable:
        log("Overall: CONSISTENT with degradation sequence prediction")
        log("  FG showing degradation while FR remains stable")
    elif fg_degrading:
        log("Overall: PARTIALLY CONSISTENT — FG degrading as predicted")
    elif fr_stable:
        log("Overall: PARTIALLY CONSISTENT — FR stable as predicted")
    else:
        log("Overall: AMBIGUOUS — results mixed")
    log()

    # Save results text
    with open(RESULTS_TXT, "w") as f:
        f.write(results_buf.getvalue())
    log(f"Full results saved to: {RESULTS_TXT}")


if __name__ == "__main__":
    main()
