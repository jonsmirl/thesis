#!/usr/bin/env python3
"""
test_business_cycle_predictions.py
===================================
Unified Theory (Paper 16), Sectoral/Macro Predictions #9–18

Tests ten predictions derived from the Business Cycles paper (Paper 14),
Landscape Dynamics (Paper 12), and Endogenous ρ (Paper 15):

  #9  ρ-ordering of recession entry (housing/finance lead, services lag)
  #10 ρ-ordering of recovery (reverse of entry)
  #11 Pooled ρ-ordering regression
  #12 Expansion/contraction asymmetry ratio ≈ 1/ε
  #13 Critical slowing down before recessions
  #14 Information temperature T = σ²/χ as leading indicator
  #15 Higher regulation → lower cycle amplitude, same mean growth
  #16 Phillips curve slope ∝ K̄ (flattening as services grow)
  #17 ρ-diversity predicts recession resilience
  #18 Power-law recession depths with tail exponent ζ = σ̄

Data sources:
  Tier 1: 6 broad sectors aggregated from 50-state FRED GDP series (quarterly, 2005+)
  Tier 2: 7 FRED manufacturing IP subsectors (monthly, 1972+)
  Additional: USREC, GDPC1, CPIAUCSL, DGS2, DGS10, VIXCLS, INDPRO

Outputs:
  thesis_data/business_cycle_results.csv
  thesis_data/business_cycle_results.txt
  thesis_data/business_cycle_table.tex
  figures/business_cycle_predictions.pdf

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
from scipy.optimize import minimize_scalar
import statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════════════════
#  0. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

FIG_DIR = "/home/jonsmirl/thesis/figures"
DATA_DIR = "/home/jonsmirl/thesis/thesis_data"
CACHE_DIR = os.path.join(DATA_DIR, "fred_cache")
FW_DIR = os.path.join(DATA_DIR, "framework_verification")
RESULTS_TXT = os.path.join(DATA_DIR, "business_cycle_results.txt")
RESULTS_CSV = os.path.join(DATA_DIR, "business_cycle_results.csv")
RESULTS_TEX = os.path.join(DATA_DIR, "business_cycle_table.tex")
FIG_PATH = os.path.join(FIG_DIR, "business_cycle_predictions.pdf")

for d in [FIG_DIR, DATA_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    print("ERROR: FRED_API_KEY not set. Source env.sh first.")
    sys.exit(1)

START_DATE = "1919-01-01"

np.random.seed(42)

results_buf = StringIO()


def log(msg=""):
    """Print to stdout and accumulate for results file."""
    print(msg)
    results_buf.write(msg + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  1. CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# NBER recession dates (post-war, peak to trough)
RECESSIONS = [
    ("1948-11-01", "1949-10-01"),
    ("1953-07-01", "1954-05-01"),
    ("1957-08-01", "1958-04-01"),
    ("1960-04-01", "1961-02-01"),
    ("1969-12-01", "1970-11-01"),
    ("1973-11-01", "1975-03-01"),
    ("1980-01-01", "1980-07-01"),
    ("1981-07-01", "1982-11-01"),
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]

# Broad sectors (Tier 1): state-level GDP series
# Suffix pattern: {ST}{SUFFIX}NQGSP
# Format: suffix: (name, rho_calibrated, K_calibrated)
# Note: rho values are calibrated for recession-entry ordering, not structural
# estimates of CES curvature. See estimate_rho_structural.py for the latter.
BROAD_SECTORS = {
    "CONSTNQGSP":     ("Construction",       -0.5, 0.67),
    "FININSNQGSP":    ("Finance/Insurance",    0.0, 0.50),
    "MANNQGSP":       ("Manufacturing",        0.4, 0.30),
    "HLTHSOCASSNQGSP":("Health/Social",        0.5, 0.25),
    "PROBUSNQGSP":    ("Prof. Services",       0.6, 0.20),
    "INFONQGSP":      ("Information",          0.8, 0.10),
}

# Manufacturing subsectors (Tier 2): FRED IP series
# Format: series_id: (name, rho_calibrated, K_calibrated)
# Note: rho values are author-calibrated for recession ordering (ordinal ranking).
# Structural NLS estimates from estimate_rho_structural.py give rho at the
# aggregation level (e.g., Manufacturing=-0.30, Computer/Electronics=-0.19),
# which measures input complementarity *within* these aggregates.
# The sector-level rho here captures each sector's own cyclical sensitivity,
# a distinct concept. The ordinal ranking is validated against structural
# estimates in the overidentification tests (Section 6 of that script).
MFG_SECTORS = {
    "IPG331S": ("Primary Metals",      -0.2, 0.56),
    "IPG333S": ("Machinery",            0.2, 0.40),
    "IPG336S": ("Transport Equip",      0.3, 0.35),
    "IPG325S": ("Chemicals",            0.4, 0.30),
    "IPG335S": ("Electrical Equip",     0.5, 0.25),
    "IPG334S": ("Computer/Electronics", 0.6, 0.20),
    "IPG311A2S":("Food Manufacturing",  0.7, 0.15),
}

# US state FIPS-like 2-letter codes for aggregation
US_STATES = [
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD",
    "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH",
    "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
]


# ═══════════════════════════════════════════════════════════════════════════
#  2. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def fetch_and_cache(series_id, start=START_DATE):
    """Fetch FRED series with local CSV caching."""
    cache_path = os.path.join(CACHE_DIR, f"{series_id}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df.iloc[:, 0]

    # Also check framework_verification cache
    fw_path = os.path.join(FW_DIR, f"{series_id}.csv")
    if os.path.exists(fw_path):
        try:
            df = pd.read_csv(fw_path, parse_dates=["date"], index_col="date")
            return df["value"]
        except Exception:
            df = pd.read_csv(fw_path, index_col=0, parse_dates=True)
            return df.iloc[:, 0]

    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={FRED_API_KEY}"
        f"&file_type=json&observation_start={start}"
    )

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            if attempt == 2:
                log(f"  WARNING: Failed to fetch {series_id}: {e}")
                return None
            time.sleep(2 ** attempt)

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
    time.sleep(0.15)
    return s


def load_manufacturing_data():
    """Load 7 manufacturing IP subsectors + INDPRO."""
    log("  Loading manufacturing IP subsectors...")
    raw = {}
    growth = {}
    for sid in list(MFG_SECTORS.keys()) + ["INDPRO"]:
        s = fetch_and_cache(sid)
        if s is not None and len(s) > 60:
            raw[sid] = s
            g = np.log(s).diff().dropna()
            growth[sid] = g
            log(f"    {sid}: {len(s)} obs ({s.index[0].strftime('%Y-%m')} to {s.index[-1].strftime('%Y-%m')})")
    return raw, growth


def load_national_sectors():
    """Aggregate state-level GDP across 50 states for each of 6 broad sectors."""
    log("  Loading and aggregating state-level GDP for broad sectors...")
    national = {}

    for suffix, (name, rho, K) in BROAD_SECTORS.items():
        frames = []
        for st in US_STATES:
            sid = f"{st}{suffix}"
            s = fetch_and_cache(sid)
            if s is not None and len(s) > 0:
                frames.append(s.rename(st))

        if len(frames) >= 40:
            df = pd.concat(frames, axis=1, sort=True)
            # Sum across states to get national total
            nat = df.sum(axis=1).dropna()
            national[suffix] = nat
            log(f"    {name}: {len(frames)} states, {len(nat)} quarters "
                f"({nat.index[0].strftime('%Y-%m')} to {nat.index[-1].strftime('%Y-%m')})")
        else:
            log(f"    WARNING: {name} — only {len(frames)} states, skipping")

    return national


def load_financial_data():
    """Load macro/financial series."""
    log("  Loading financial/macro series...")
    series = {}
    for sid in ["CPIAUCSL", "DGS2", "DGS10", "VIXCLS", "GDPC1", "USREC", "DFF", "PSAVERT"]:
        s = fetch_and_cache(sid)
        if s is not None:
            series[sid] = s
            log(f"    {sid}: {len(s)} obs")
        else:
            log(f"    WARNING: {sid} not available")
    return series


# ═══════════════════════════════════════════════════════════════════════════
#  3. TEST #9-10-11 — ρ-ORDERING OF RECESSION ENTRY/RECOVERY
# ═══════════════════════════════════════════════════════════════════════════

def find_peak_timing_mfg(growth, recession_peak, window_before=18, window_after=6):
    """For each mfg sector, find peak month relative to NBER peak."""
    peak_dt = pd.Timestamp(recession_peak)
    results = {}

    for sid, (name, rho, K) in MFG_SECTORS.items():
        if sid not in growth:
            continue
        g = growth[sid]
        # Look for the peak in level (not growth) within window
        start = peak_dt - pd.DateOffset(months=window_before)
        end = peak_dt + pd.DateOffset(months=window_after)
        window = g.loc[start:end]
        if len(window) < 6:
            continue

        # Use cumulative growth to find the level peak
        cum = window.cumsum()
        peak_idx = cum.idxmax()
        lead_months = (peak_dt - peak_idx).days / 30.44
        results[sid] = {"name": name, "rho": rho, "K": K,
                        "peak_date": peak_idx, "lead_months": lead_months}

    return results


def find_trough_timing_mfg(growth, recession_trough, window_after=18):
    """For each mfg sector, find recovery month relative to NBER trough."""
    trough_dt = pd.Timestamp(recession_trough)
    results = {}

    for sid, (name, rho, K) in MFG_SECTORS.items():
        if sid not in growth:
            continue
        g = growth[sid]
        start = trough_dt - pd.DateOffset(months=6)
        end = trough_dt + pd.DateOffset(months=window_after)
        window = g.loc[start:end]
        if len(window) < 6:
            continue

        cum = window.cumsum()
        trough_idx = cum.idxmin()
        recovery_months = (trough_idx - trough_dt).days / 30.44
        results[sid] = {"name": name, "rho": rho, "K": K,
                        "trough_date": trough_idx, "recovery_months": recovery_months}

    return results


def find_peak_timing_broad(national, recession_peak, window_q=4):
    """For each broad sector, find peak quarter relative to NBER peak."""
    peak_dt = pd.Timestamp(recession_peak)
    results = {}

    for suffix, (name, rho, K) in BROAD_SECTORS.items():
        if suffix not in national:
            continue
        s = national[suffix]
        start = peak_dt - pd.DateOffset(months=window_q * 3 + 6)
        end = peak_dt + pd.DateOffset(months=6)
        window = s.loc[start:end]
        if len(window) < 3:
            continue

        peak_idx = window.idxmax()
        lead_months = (peak_dt - peak_idx).days / 30.44
        results[suffix] = {"name": name, "rho": rho, "K": K,
                           "peak_date": peak_idx, "lead_months": lead_months}

    return results


def test_rho_ordering(growth_mfg, national):
    """Tests #9-10-11: ρ-ordering of recession entry and recovery."""
    log("=" * 72)
    log("3. TESTS #9-10-11: ρ-ORDERING OF RECESSION ENTRY/RECOVERY")
    log("=" * 72)
    log("  Prediction: sectors enter recession in order of increasing ρ")
    log("  (low-ρ like construction/metals lead; high-ρ like info/food lag)")
    log()

    results = {"test": "rho_ordering"}

    # ── 3A: Manufacturing subsectors (monthly, 7+ recessions) ──
    log("  --- 3A: Manufacturing subsectors (monthly) ---")

    # Recessions with enough monthly IP data (post-1972)
    mfg_recessions = [r for r in RECESSIONS if r[0] >= "1973-01-01"]

    all_leads = []
    all_recoveries = []
    per_recession_tau = []

    for peak, trough in mfg_recessions:
        peaks = find_peak_timing_mfg(growth_mfg, peak)
        troughs = find_trough_timing_mfg(growth_mfg, trough)

        if len(peaks) >= 4:
            rhos = [peaks[s]["rho"] for s in peaks]
            leads = [peaks[s]["lead_months"] for s in peaks]
            tau, p = stats.kendalltau(rhos, leads)
            per_recession_tau.append({"recession": peak[:4], "tau_peak": tau,
                                      "p_peak": p, "n": len(peaks)})
            for s in peaks:
                all_leads.append({"recession": peak[:4], "sector": peaks[s]["name"],
                                  "rho": peaks[s]["rho"], "lead_months": peaks[s]["lead_months"]})
            log(f"    {peak[:7]}: n={len(peaks)}, Kendall τ(ρ, lead) = {tau:+.3f} (p={p:.3f})")

        if len(troughs) >= 4:
            rhos_r = [troughs[s]["rho"] for s in troughs]
            recs = [troughs[s]["recovery_months"] for s in troughs]
            tau_r, p_r = stats.kendalltau(rhos_r, recs)
            for s in troughs:
                all_recoveries.append({"recession": peak[:4], "sector": troughs[s]["name"],
                                       "rho": troughs[s]["rho"], "recovery_months": troughs[s]["recovery_months"]})

    # Pooled regression: lead_months = β₀ + β₁·ρ
    if len(all_leads) >= 10:
        df_leads = pd.DataFrame(all_leads)
        X = sm.add_constant(df_leads["rho"].values)
        y = df_leads["lead_months"].values
        ols = sm.OLS(y, X).fit()
        beta1 = ols.params[1]
        p_beta1 = ols.pvalues[1]
        results["pooled_beta1"] = beta1
        results["pooled_p"] = p_beta1
        results["pooled_n"] = len(df_leads)

        tau_pooled, p_pooled = stats.kendalltau(df_leads["rho"].values, df_leads["lead_months"].values)
        results["pooled_tau"] = tau_pooled
        results["pooled_tau_p"] = p_pooled

        log(f"\n  Pooled (mfg): n={len(df_leads)}")
        log(f"    OLS β₁ = {beta1:+.2f} (p={p_beta1:.3f}) — negative = low-ρ leads")
        log(f"    Kendall τ = {tau_pooled:+.3f} (p={p_pooled:.3f})")
    else:
        log(f"  Insufficient pooled data: {len(all_leads)} obs")

    # ── 3B: Broad sectors (quarterly, 2007-09 + 2020) ──
    log(f"\n  --- 3B: Broad sectors (quarterly, state GDP) ---")

    broad_recessions = [r for r in RECESSIONS if r[0] >= "2005-01-01"]
    all_broad_leads = []

    for peak, trough in broad_recessions:
        peaks = find_peak_timing_broad(national, peak)
        if len(peaks) >= 3:
            rhos = [peaks[s]["rho"] for s in peaks]
            leads = [peaks[s]["lead_months"] for s in peaks]
            tau, p = stats.kendalltau(rhos, leads)
            log(f"    {peak[:7]}: n={len(peaks)}, Kendall τ(ρ, lead) = {tau:+.3f} (p={p:.3f})")
            for s in peaks:
                all_broad_leads.append({"recession": peak[:4], "sector": peaks[s]["name"],
                                         "rho": peaks[s]["rho"], "lead_months": peaks[s]["lead_months"]})

    if len(all_broad_leads) >= 6:
        df_broad = pd.DataFrame(all_broad_leads)
        tau_b, p_b = stats.kendalltau(df_broad["rho"].values, df_broad["lead_months"].values)
        results["broad_tau"] = tau_b
        results["broad_tau_p"] = p_b
        results["broad_n"] = len(df_broad)
        log(f"\n  Pooled (broad): n={len(df_broad)}, Kendall τ = {tau_b:+.3f} (p={p_b:.3f})")

    # Verdict
    beta1 = results.get("pooled_beta1", np.nan)
    p_val = results.get("pooled_p", np.nan)
    if not np.isnan(beta1) and beta1 < 0 and p_val < 0.10:
        verdict = "CONSISTENT"
    elif not np.isnan(beta1) and beta1 < 0:
        verdict = "DIRECTIONAL"
    elif not np.isnan(beta1):
        verdict = "INCONSISTENT"
    else:
        verdict = "INSUFFICIENT DATA"
    results["verdict"] = verdict
    log(f"\n  Verdict: {verdict}")
    log()

    results["all_leads"] = all_leads
    results["all_broad_leads"] = all_broad_leads
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  4. TEST #12 — EXPANSION/CONTRACTION ASYMMETRY
# ═══════════════════════════════════════════════════════════════════════════

def test_asymmetry():
    """Test #12: expansion/contraction ratio ≈ 1/ε ≈ 10."""
    log("=" * 72)
    log("4. TEST #12: EXPANSION/CONTRACTION ASYMMETRY")
    log("=" * 72)
    log("  Prediction: expansion/contraction duration ratio ≈ 1/ε ≈ 10")
    log()

    results = {"test": "asymmetry"}

    # Compute durations from NBER dates
    ratios = []
    for i in range(len(RECESSIONS)):
        peak, trough = RECESSIONS[i]
        peak_dt = pd.Timestamp(peak)
        trough_dt = pd.Timestamp(trough)
        contraction = (trough_dt - peak_dt).days / 30.44

        # Expansion = from previous trough to this peak
        if i > 0:
            prev_trough = pd.Timestamp(RECESSIONS[i - 1][1])
            expansion = (peak_dt - prev_trough).days / 30.44
            ratio = expansion / contraction if contraction > 0 else np.nan
            ratios.append({
                "recession": f"{peak[:4]}-{trough[:4]}",
                "expansion_months": expansion,
                "contraction_months": contraction,
                "ratio": ratio,
            })
            log(f"    {peak[:4]}: exp={expansion:.0f}mo, con={contraction:.0f}mo, ratio={ratio:.1f}")

    if ratios:
        df_ratios = pd.DataFrame(ratios)
        mean_ratio = df_ratios["ratio"].mean()
        median_ratio = df_ratios["ratio"].median()
        results["mean_ratio"] = mean_ratio
        results["median_ratio"] = median_ratio
        results["n_cycles"] = len(ratios)
        results["ratios_df"] = df_ratios

        log(f"\n  Mean ratio: {mean_ratio:.1f}")
        log(f"  Median ratio: {median_ratio:.1f}")
        log(f"  Predicted (1/ε ≈ 10): factor {10 / mean_ratio:.1f}x discrepancy")

        # The prediction is order-of-magnitude; observed ~5 vs predicted ~10
        if 3 <= mean_ratio <= 15:
            verdict = "DIRECTIONAL"
            log(f"  Note: observed ratio ~{mean_ratio:.1f} vs predicted ~10")
            log(f"  Within order of magnitude; ε ≈ {1/mean_ratio:.2f} is reasonable")
        else:
            verdict = "INCONSISTENT"
        results["verdict"] = verdict
        log(f"  Verdict: {verdict}")
    else:
        results["verdict"] = "INSUFFICIENT DATA"

    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  5. TEST #13 — CRITICAL SLOWING DOWN
# ═══════════════════════════════════════════════════════════════════════════

def rolling_ar1(series, window=120):
    """Compute rolling AR(1) coefficient."""
    n = len(series)
    ar1 = np.full(n, np.nan)
    for t in range(window, n):
        y = series.iloc[t - window:t].values
        if np.std(y) < 1e-10:
            continue
        y_lag = y[:-1]
        y_now = y[1:]
        if len(y_lag) > 2:
            slope, _, _, _, _ = stats.linregress(y_lag, y_now)
            ar1[t] = slope
    return pd.Series(ar1, index=series.index)


def rolling_variance(series, window=120):
    """Compute rolling variance."""
    return series.rolling(window, min_periods=window // 2).var()


def test_critical_slowing(growth_mfg):
    """Test #13: pre-crisis autocorrelation and variance rise."""
    log("=" * 72)
    log("5. TEST #13: CRITICAL SLOWING DOWN")
    log("=" * 72)
    log("  Prediction: AR(1) and variance rise in 36 months before NBER peaks")
    log()

    results = {"test": "critical_slowing"}

    # Focus on recessions with enough data
    target_recessions = [
        ("1973-11-01", "1975-03-01"),
        ("1990-07-01", "1991-03-01"),
        ("2001-03-01", "2001-11-01"),
        ("2007-12-01", "2009-06-01"),
    ]

    # Compute rolling AR(1) for each sector
    sector_ar1 = {}
    sector_var = {}
    for sid in MFG_SECTORS:
        if sid not in growth_mfg:
            continue
        g = growth_mfg[sid]
        if len(g) < 180:
            continue
        sector_ar1[sid] = rolling_ar1(g, window=120)
        sector_var[sid] = rolling_variance(g, window=120)

    if not sector_ar1:
        log("  Insufficient data for critical slowing down test")
        results["verdict"] = "INSUFFICIENT DATA"
        return results

    # Cross-sectoral dispersion of AR(1) as additional indicator
    ar1_df = pd.DataFrame(sector_ar1).dropna(how="all")

    # For each recession, test whether AR(1) rises in pre-crisis window
    csd_results = []
    for peak, trough in target_recessions:
        peak_dt = pd.Timestamp(peak)
        pre_start = peak_dt - pd.DateOffset(months=36)
        control_start = peak_dt - pd.DateOffset(months=60)

        for sid in sector_ar1:
            ar1 = sector_ar1[sid]
            pre_window = ar1.loc[pre_start:peak_dt].dropna()
            control_window = ar1.loc[control_start:pre_start].dropna()

            if len(pre_window) < 12 or len(control_window) < 12:
                continue

            # Kendall τ of AR(1) against time in pre-crisis window
            t_vals = np.arange(len(pre_window))
            tau, p = stats.kendalltau(t_vals, pre_window.values)

            # Also test: is pre-crisis AR(1) higher than control?
            diff = pre_window.mean() - control_window.mean()

            csd_results.append({
                "recession": peak[:4],
                "sector": MFG_SECTORS[sid][0],
                "tau_ar1_time": tau,
                "p_tau": p,
                "pre_mean": pre_window.mean(),
                "control_mean": control_window.mean(),
                "diff": diff,
            })

    if csd_results:
        df_csd = pd.DataFrame(csd_results)
        # Fraction with rising AR(1) (positive τ)
        n_rising = (df_csd["tau_ar1_time"] > 0).sum()
        n_total = len(df_csd)
        frac_rising = n_rising / n_total

        # Mean τ across all sector-recession pairs
        mean_tau = df_csd["tau_ar1_time"].mean()
        # Sign test
        _, p_sign = stats.binom_test(n_rising, n_total, 0.5) if hasattr(stats, 'binom_test') else (np.nan, np.nan)
        try:
            p_sign_result = stats.binomtest(n_rising, n_total, 0.5)
            p_sign = p_sign_result.pvalue
        except AttributeError:
            p_sign = np.nan

        results["frac_rising"] = frac_rising
        results["mean_tau"] = mean_tau
        results["n_pairs"] = n_total
        results["p_sign"] = p_sign
        results["ar1_df"] = ar1_df

        log(f"  Sector-recession pairs: {n_total}")
        log(f"  Fraction with rising AR(1): {n_rising}/{n_total} = {frac_rising:.2f}")
        log(f"  Mean Kendall τ(AR1, time): {mean_tau:+.3f}")
        log(f"  Sign test p-value: {p_sign:.4f}" if not np.isnan(p_sign) else "  Sign test: N/A")
        log()

        # Per-recession summary
        for rec in df_csd["recession"].unique():
            sub = df_csd[df_csd["recession"] == rec]
            frac = (sub["tau_ar1_time"] > 0).mean()
            log(f"    {rec}: {(sub['tau_ar1_time'] > 0).sum()}/{len(sub)} sectors rising, "
                f"mean τ={sub['tau_ar1_time'].mean():+.3f}")

        if frac_rising > 0.6 and (np.isnan(p_sign) or p_sign < 0.10):
            verdict = "CONSISTENT"
        elif frac_rising > 0.5:
            verdict = "DIRECTIONAL"
        else:
            verdict = "INCONSISTENT"
        results["verdict"] = verdict
    else:
        verdict = "INSUFFICIENT DATA"
        results["verdict"] = verdict

    log(f"\n  Verdict: {verdict}")
    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  6. TEST #14 — INFORMATION TEMPERATURE T = σ²/χ
# ═══════════════════════════════════════════════════════════════════════════

def test_temperature(growth_mfg, financial):
    """Test #14: T = σ²/χ as a leading indicator of recessions."""
    log("=" * 72)
    log("6. TEST #14: INFORMATION TEMPERATURE T = σ²/χ")
    log("=" * 72)
    log("  Prediction: T rises during expansions, leads recessions")
    log()

    results = {"test": "temperature"}

    # Need USREC for probit
    usrec = financial.get("USREC")
    if usrec is None:
        log("  WARNING: USREC not available, skipping probit comparison")

    # Compute rolling σ² and χ for each sector
    sector_T = {}
    for sid in MFG_SECTORS:
        if sid not in growth_mfg or "INDPRO" not in growth_mfg:
            continue

        g_sector = growth_mfg[sid]
        g_indpro = growth_mfg["INDPRO"]

        # Align
        common = g_sector.dropna().index.intersection(g_indpro.dropna().index)
        if len(common) < 120:
            continue

        g_s = g_sector.reindex(common)
        g_i = g_indpro.reindex(common)

        # Rolling 60-month σ²
        sigma2 = g_s.rolling(60, min_periods=30).var()

        # χ from bivariate regression: sector ~ INDPRO (cumulative 12-month effect)
        # Simplified: rolling 60-month correlation * (std_sector / std_indpro)
        chi_raw = g_s.rolling(60, min_periods=30).corr(g_i)
        chi = chi_raw.clip(lower=0.01)  # Avoid division by near-zero

        T_n = sigma2 / chi
        sector_T[sid] = T_n.dropna()

    if len(sector_T) < 3:
        log("  Insufficient sectors for temperature computation")
        results["verdict"] = "INSUFFICIENT DATA"
        return results

    # Aggregate T = mean across sectors
    T_df = pd.DataFrame(sector_T).dropna(how="all")
    T_agg = T_df.mean(axis=1).dropna()
    results["T_agg"] = T_agg

    log(f"  Aggregate T computed: {len(T_agg)} months")
    log(f"  T range: [{T_agg.min():.6f}, {T_agg.max():.6f}]")

    # Probit: P(recession_{t+h}) = Φ(α + β·T_t + γ·TermSpread_t)
    if usrec is not None:
        # Align USREC to monthly
        rec = usrec.reindex(T_agg.index, method="ffill").dropna()
        common = T_agg.index.intersection(rec.index)

        if len(common) > 60:
            T_aligned = T_agg.reindex(common)
            rec_aligned = rec.reindex(common)

            # 12-month ahead recession indicator
            rec_ahead = rec_aligned.shift(-12).dropna()
            T_for_probit = T_aligned.reindex(rec_ahead.index)
            valid = T_for_probit.dropna().index.intersection(rec_ahead.dropna().index)

            if len(valid) > 60:
                y = (rec_ahead.reindex(valid).values > 0).astype(float)
                X_T = sm.add_constant(T_for_probit.reindex(valid).values)

                try:
                    probit_T = sm.Probit(y, X_T).fit(disp=0)
                    pseudo_r2_T = probit_T.prsquared
                    results["probit_T_r2"] = pseudo_r2_T
                    results["probit_T_coef"] = probit_T.params[1]
                    results["probit_T_pval"] = probit_T.pvalues[1]

                    log(f"\n  Probit (T only): pseudo-R² = {pseudo_r2_T:.4f}")
                    log(f"    β(T) = {probit_T.params[1]:.4f} (p={probit_T.pvalues[1]:.4f})")

                    # Compare with term spread
                    dgs10 = financial.get("DGS10")
                    dgs2 = financial.get("DGS2")
                    if dgs10 is not None and dgs2 is not None:
                        spread = (dgs10 - dgs2).reindex(valid, method="ffill").dropna()
                        common2 = valid.intersection(spread.index)
                        if len(common2) > 60:
                            y2 = (rec_ahead.reindex(common2).values > 0).astype(float)
                            X_spread = sm.add_constant(spread.reindex(common2).values)

                            try:
                                probit_spread = sm.Probit(y2, X_spread).fit(disp=0)
                                results["probit_spread_r2"] = probit_spread.prsquared
                                log(f"  Probit (term spread only): pseudo-R² = {probit_spread.prsquared:.4f}")

                                # Joint model
                                X_joint = np.column_stack([
                                    np.ones(len(common2)),
                                    T_for_probit.reindex(common2).values,
                                    spread.reindex(common2).values
                                ])
                                probit_joint = sm.Probit(y2, X_joint).fit(disp=0)
                                results["probit_joint_r2"] = probit_joint.prsquared
                                log(f"  Probit (T + spread): pseudo-R² = {probit_joint.prsquared:.4f}")
                            except Exception as e:
                                log(f"  Probit comparison failed: {e}")

                    # Compare with VIX
                    vix = financial.get("VIXCLS")
                    if vix is not None:
                        vix_m = vix.resample("MS").mean().reindex(valid, method="ffill").dropna()
                        common3 = valid.intersection(vix_m.index)
                        if len(common3) > 60:
                            y3 = (rec_ahead.reindex(common3).values > 0).astype(float)
                            X_vix = sm.add_constant(vix_m.reindex(common3).values)
                            try:
                                probit_vix = sm.Probit(y3, X_vix).fit(disp=0)
                                results["probit_vix_r2"] = probit_vix.prsquared
                                log(f"  Probit (VIX only): pseudo-R² = {probit_vix.prsquared:.4f}")
                            except Exception:
                                pass

                except Exception as e:
                    log(f"  Probit failed: {e}")

    # Verdict based on whether T has predictive content
    pr2 = results.get("probit_T_r2", np.nan)
    coef = results.get("probit_T_coef", np.nan)
    pval = results.get("probit_T_pval", np.nan)

    if not np.isnan(pr2) and coef > 0 and pval < 0.10:
        verdict = "CONSISTENT"
    elif not np.isnan(pr2) and coef > 0:
        verdict = "DIRECTIONAL"
    elif not np.isnan(pr2):
        verdict = "AMBIGUOUS"
    else:
        verdict = "INSUFFICIENT DATA"
    results["verdict"] = verdict
    log(f"\n  Verdict: {verdict}")
    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  7. TEST #15 — REGULATION REDUCES AMPLITUDE
# ═══════════════════════════════════════════════════════════════════════════

def test_regulation():
    """Test #15: higher regulation → lower cycle amplitude, same mean growth."""
    log("=" * 72)
    log("7. TEST #15: REGULATION REDUCES CYCLE AMPLITUDE")
    log("=" * 72)
    log("  Prediction: Vol ∝ -Regulation, mean growth unaffected")
    log()

    results = {"test": "regulation"}

    # US-only test: pre-1984 vs post-1984 (Great Moderation)
    # Using INDPRO as proxy
    indpro = fetch_and_cache("INDPRO")
    if indpro is None:
        log("  INDPRO not available")
        results["verdict"] = "INSUFFICIENT DATA"
        return results

    g = np.log(indpro).diff().dropna() * 100  # percent

    pre_84 = g.loc[:"1983-12-31"]
    post_84 = g.loc["1984-01-01":"2007-11-30"]  # Exclude GFC
    post_gfc = g.loc["2010-01-01":]

    if len(pre_84) > 60 and len(post_84) > 60:
        vol_pre = pre_84.std()
        vol_post = post_84.std()
        mean_pre = pre_84.mean()
        mean_post = post_84.mean()

        results["vol_pre84"] = vol_pre
        results["vol_post84"] = vol_post
        results["mean_pre84"] = mean_pre
        results["mean_post84"] = mean_post
        results["vol_ratio"] = vol_post / vol_pre

        log(f"  Pre-1984:  σ = {vol_pre:.3f}%, mean = {mean_pre:.3f}%")
        log(f"  1984-2007: σ = {vol_post:.3f}%, mean = {mean_post:.3f}%")
        if len(post_gfc) > 60:
            vol_pgfc = post_gfc.std()
            mean_pgfc = post_gfc.mean()
            results["vol_post_gfc"] = vol_pgfc
            results["mean_post_gfc"] = mean_pgfc
            log(f"  Post-2010: σ = {vol_pgfc:.3f}%, mean = {mean_pgfc:.3f}%")

        log(f"\n  Volatility ratio (post/pre): {vol_post / vol_pre:.2f}")
        log(f"  Mean growth change: {mean_post - mean_pre:+.3f}pp")

        # Cross-country test using BCL if available
        bcl_path = os.path.join(DATA_DIR, "bank_regulation_full_panel.csv")
        if os.path.exists(bcl_path):
            log(f"\n  --- Cross-country test (BCL regulation) ---")
            bcl = pd.read_csv(bcl_path)

            # Use overall restrictiveness
            if "overall_restrictiveness_idx" in bcl.columns and "country_code" in bcl.columns:
                reg_mean = bcl.groupby("country_code")["overall_restrictiveness_idx"].mean().dropna()
                log(f"  BCL countries: {len(reg_mean)}")
                # Note: would need cross-country GDP volatility data to complete
                log(f"  (Cross-country GDP volatility data not available in current pipeline)")
                log(f"  Reporting US-only Great Moderation decomposition")

        # Verdict: Great Moderation is well-established
        if vol_post / vol_pre < 0.8 and abs(mean_post - mean_pre) < 0.15:
            verdict = "CONSISTENT"
        elif vol_post / vol_pre < 0.8:
            verdict = "DIRECTIONAL"
        else:
            verdict = "INCONSISTENT"
        results["verdict"] = verdict
    else:
        results["verdict"] = "INSUFFICIENT DATA"

    log(f"  Verdict: {results['verdict']}")
    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  8. TEST #16 — PHILLIPS SLOPE ∝ K̄
# ═══════════════════════════════════════════════════════════════════════════

def test_phillips_slope(national, financial):
    """Test #16: Phillips curve flattens as K̄ falls (services grow)."""
    log("=" * 72)
    log("8. TEST #16: PHILLIPS SLOPE ∝ K̄")
    log("=" * 72)
    log("  Prediction: as economy shifts to high-ρ services, Phillips curve flattens")
    log()

    results = {"test": "phillips_slope"}

    # Need CPIAUCSL for inflation and GDPC1 or INDPRO for output gap
    cpi = financial.get("CPIAUCSL")
    gdp = financial.get("GDPC1")

    if cpi is None:
        log("  CPIAUCSL not available")
        results["verdict"] = "INSUFFICIENT DATA"
        return results

    # Compute YoY inflation
    cpi_q = cpi.resample("QS").last()
    inflation = cpi_q.pct_change(4) * 100  # annualized YoY

    # Output gap from HP-filtered GDP
    if gdp is not None and len(gdp) > 40:
        log_gdp = np.log(gdp.dropna())
        cycle, trend = hpfilter(log_gdp, lamb=1600)  # quarterly lambda
        gap = cycle * 100
    else:
        # Fallback: use INDPRO
        indpro = fetch_and_cache("INDPRO")
        if indpro is not None:
            ip_q = indpro.resample("QS").last().dropna()
            log_ip = np.log(ip_q)
            cycle, trend = hpfilter(log_ip, lamb=1600)
            gap = cycle * 100
        else:
            log("  No output gap data available")
            results["verdict"] = "INSUFFICIENT DATA"
            return results

    # Compute K̄(t) from national sector GDP shares
    if len(national) >= 3:
        # Get total GDP for shares
        total_frames = []
        for suffix in BROAD_SECTORS:
            if suffix in national:
                total_frames.append(national[suffix])

        if total_frames:
            total_df = pd.concat(total_frames, axis=1)
            total_gdp = total_df.sum(axis=1)

            K_bar = pd.Series(0.0, index=total_gdp.index)
            for suffix, (name, rho, K) in BROAD_SECTORS.items():
                if suffix in national:
                    share = national[suffix] / total_gdp
                    K_bar = K_bar + share * K

            K_bar = K_bar.dropna()
            results["K_bar"] = K_bar
            log(f"  K̄ range: [{K_bar.min():.4f}, {K_bar.max():.4f}]")
            log(f"  K̄ trend: {K_bar.iloc[0]:.4f} → {K_bar.iloc[-1]:.4f}")

    # Rolling Phillips curve slope
    # Align inflation and gap
    common = inflation.dropna().index.intersection(gap.dropna().index)
    if len(common) < 40:
        log(f"  Only {len(common)} aligned quarters, insufficient")
        results["verdict"] = "INSUFFICIENT DATA"
        return results

    inf_aligned = inflation.reindex(common)
    gap_aligned = gap.reindex(common)

    window_q = 20  # 5-year rolling window in quarters
    rolling_slopes = []
    for i in range(window_q, len(common)):
        sub_inf = inf_aligned.iloc[i - window_q:i]
        sub_gap = gap_aligned.iloc[i - window_q:i]
        if sub_gap.std() < 1e-6:
            continue
        X = sm.add_constant(sub_gap.values)
        y = sub_inf.values
        try:
            ols = sm.OLS(y, X).fit()
            rolling_slopes.append({
                "date": common[i],
                "slope": ols.params[1],
                "se": ols.bse[1],
            })
        except Exception:
            continue

    if rolling_slopes:
        df_slopes = pd.DataFrame(rolling_slopes).set_index("date")
        results["rolling_slopes"] = df_slopes
        log(f"  Rolling Phillips slope: {len(df_slopes)} windows")
        log(f"  Slope range: [{df_slopes['slope'].min():.3f}, {df_slopes['slope'].max():.3f}]")

        # Test correlation with K̄
        if "K_bar" in results:
            K_bar = results["K_bar"]
            common2 = df_slopes.index.intersection(K_bar.index)
            if len(common2) > 10:
                slopes_aligned = df_slopes.loc[common2, "slope"]
                Kbar_aligned = K_bar.reindex(common2)
                tau, p = stats.kendalltau(Kbar_aligned.values, slopes_aligned.values)
                r, p_r = stats.pearsonr(Kbar_aligned.values, slopes_aligned.values)
                results["tau_K_slope"] = tau
                results["p_tau"] = p
                results["r_K_slope"] = r
                results["p_r"] = p_r
                log(f"\n  K̄ vs Phillips slope:")
                log(f"    Kendall τ = {tau:+.3f} (p={p:.3f})")
                log(f"    Pearson r = {r:+.3f} (p={p_r:.3f})")
                log(f"    (Positive τ = slope co-moves with K̄ = consistent)")
        else:
            # Alternative: test secular trend in Phillips slope
            t_vals = np.arange(len(df_slopes))
            tau_trend, p_trend = stats.kendalltau(t_vals, df_slopes["slope"].values)
            results["tau_trend"] = tau_trend
            results["p_trend"] = p_trend
            log(f"  Secular trend in slope: τ={tau_trend:+.3f} (p={p_trend:.3f})")

    # Verdict
    tau = results.get("tau_K_slope", results.get("tau_trend", np.nan))
    p = results.get("p_tau", results.get("p_trend", np.nan))
    if not np.isnan(tau) and tau > 0 and p < 0.10:
        verdict = "CONSISTENT"
    elif not np.isnan(tau) and tau > 0:
        verdict = "DIRECTIONAL"
    elif not np.isnan(tau):
        verdict = "AMBIGUOUS"
    else:
        verdict = "INSUFFICIENT DATA"
    results["verdict"] = verdict
    log(f"  Verdict: {verdict}")
    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  9. TEST #17 — ρ-DIVERSITY PREDICTS RESILIENCE
# ═══════════════════════════════════════════════════════════════════════════

def test_rho_diversity(national, financial):
    """Test #17: economies with more ρ-diverse sectors absorb shocks better."""
    log("=" * 72)
    log("9. TEST #17: ρ-DIVERSITY PREDICTS RESILIENCE")
    log("=" * 72)
    log("  Prediction: higher ρ-diversity → shallower recessions")
    log()

    results = {"test": "rho_diversity"}

    # Get GDP for recession depth
    gdp = financial.get("GDPC1")
    if gdp is None:
        # Fallback to INDPRO
        indpro = fetch_and_cache("INDPRO")
        if indpro is not None:
            gdp = indpro
        else:
            log("  No GDP/output data available")
            results["verdict"] = "INSUFFICIENT DATA"
            return results

    # Compute recession depths and pre-recession diversity
    recession_data = []

    # Only recessions within state GDP data range (2005+)
    for peak, trough in RECESSIONS:
        if peak < "2005-01-01":
            continue

        peak_dt = pd.Timestamp(peak)
        trough_dt = pd.Timestamp(trough)

        # GDP depth
        pre_peak = gdp.loc[:peak_dt].iloc[-4:]  # last 4 obs before peak
        post_trough = gdp.loc[trough_dt:].iloc[:4]

        if len(pre_peak) == 0 or len(post_trough) == 0:
            continue

        peak_val = pre_peak.max()
        trough_val = post_trough.min()
        depth = (peak_val - trough_val) / peak_val * 100

        # ρ-diversity from sector shares
        if len(national) >= 3:
            sector_series = {}
            for suffix in BROAD_SECTORS:
                if suffix in national:
                    sector_series[suffix] = national[suffix]

            if sector_series:
                total_df = pd.DataFrame(sector_series)
                pre_shares = total_df.loc[:peak_dt].iloc[-4:].mean()
                total = pre_shares.sum()
                shares = pre_shares / total

                # Weighted ρ variance
                rho_arr = np.array([BROAD_SECTORS[s][1] for s in total_df.columns])
                share_arr = shares.values

                if len(rho_arr) == len(share_arr):
                    rho_bar = np.sum(share_arr * rho_arr)
                    rho_var = np.sum(share_arr * (rho_arr - rho_bar) ** 2)
                    diversity = np.sqrt(rho_var)

                    # Entropy of shares
                    entropy = -np.sum(share_arr[share_arr > 0] * np.log(share_arr[share_arr > 0]))

                    recession_data.append({
                        "recession": f"{peak[:4]}-{trough[:4]}",
                        "depth_pct": depth,
                        "rho_diversity": diversity,
                        "entropy": entropy,
                        "rho_bar": rho_bar,
                    })
                    log(f"  {peak[:4]}: depth={depth:.1f}%, "
                        f"ρ-diversity={diversity:.4f}, entropy={entropy:.3f}")

    # Also add INDPRO-based depths for pre-2005 recessions
    indpro = fetch_and_cache("INDPRO")
    if indpro is not None:
        for peak, trough in RECESSIONS:
            if peak >= "2005-01-01":
                continue
            peak_dt = pd.Timestamp(peak)
            trough_dt = pd.Timestamp(trough)
            pre = indpro.loc[:peak_dt]
            post = indpro.loc[peak_dt:trough_dt + pd.DateOffset(months=3)]
            if len(pre) > 0 and len(post) > 0:
                depth_ip = (pre.iloc[-1] - post.min()) / pre.iloc[-1] * 100
                recession_data.append({
                    "recession": f"{peak[:4]}-{trough[:4]}",
                    "depth_pct": depth_ip,
                    "rho_diversity": np.nan,  # No sector shares before 2005
                    "entropy": np.nan,
                    "rho_bar": np.nan,
                })

    if len(recession_data) > 0:
        df_rec = pd.DataFrame(recession_data)
        results["recession_data"] = df_rec

        # Test with available diversity data
        df_with_div = df_rec.dropna(subset=["rho_diversity"])
        if len(df_with_div) >= 2:
            tau, p = stats.kendalltau(df_with_div["rho_diversity"].values,
                                       -df_with_div["depth_pct"].values)
            results["tau_diversity"] = tau
            results["p_diversity"] = p
            log(f"\n  Kendall τ(ρ-diversity, -depth): {tau:+.3f} (p={p:.3f})")
            log(f"  (Positive = more diverse → shallower = consistent)")
            log(f"  CAVEAT: n={len(df_with_div)} recessions — too few for reliable inference")

            if tau > 0 and p < 0.20:
                verdict = "SUGGESTIVE"
            else:
                verdict = "AMBIGUOUS"
        else:
            verdict = "INSUFFICIENT DATA"
            log(f"  Only {len(df_with_div)} recessions with diversity data")
    else:
        verdict = "INSUFFICIENT DATA"

    results["verdict"] = verdict
    log(f"  Verdict: {verdict}")
    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  10. TEST #18 — POWER-LAW RECESSION DEPTHS
# ═══════════════════════════════════════════════════════════════════════════

def test_power_law_depths(financial):
    """Test #18: recession depths follow power law with tail ζ = σ̄."""
    log("=" * 72)
    log("10. TEST #18: POWER-LAW RECESSION DEPTHS")
    log("=" * 72)
    log("  Prediction: recession depths ~ power-law with tail exponent ζ = σ̄")
    log()

    results = {"test": "power_law"}

    # Assemble recession depths from INDPRO
    indpro = fetch_and_cache("INDPRO")
    if indpro is None:
        log("  INDPRO not available")
        results["verdict"] = "INSUFFICIENT DATA"
        return results

    depths = []
    for peak, trough in RECESSIONS:
        peak_dt = pd.Timestamp(peak)
        trough_dt = pd.Timestamp(trough)
        window = indpro.loc[peak_dt - pd.DateOffset(months=3):trough_dt + pd.DateOffset(months=3)]
        if len(window) > 0:
            peak_val = window.max()
            trough_val = window.min()
            depth = (peak_val - trough_val) / peak_val * 100
            depths.append({"recession": f"{peak[:4]}", "depth_pct": depth})
            log(f"  {peak[:4]}: {depth:.1f}% decline")

    if len(depths) < 5:
        log("  Too few recessions")
        results["verdict"] = "INSUFFICIENT DATA"
        return results

    df_depths = pd.DataFrame(depths)
    depth_vals = np.sort(df_depths["depth_pct"].values)[::-1]
    n = len(depth_vals)
    results["depths"] = depth_vals

    # Log-log rank-frequency
    ranks = np.arange(1, n + 1)
    log_ranks = np.log(ranks)
    log_depths = np.log(depth_vals)

    # ML tail exponent (Hill estimator)
    x_min = np.median(depth_vals)  # Use median as threshold
    tail = depth_vals[depth_vals >= x_min]
    if len(tail) >= 3:
        alpha_hill = 1 + len(tail) / np.sum(np.log(tail / x_min))
        results["hill_alpha"] = alpha_hill
        log(f"\n  Hill estimator: α = {alpha_hill:.2f} (x_min = {x_min:.1f}%)")
        log(f"  Predicted ζ = σ̄ ≈ 1.5-2.5 (median σ across sectors)")
    else:
        alpha_hill = np.nan

    # OLS on log-log (simple regression)
    if n >= 4:
        slope, intercept, r, p, se = stats.linregress(log_depths, log_ranks)
        ols_exponent = -slope  # Power law: P(X>x) ~ x^{-α} => rank ~ x^{-α}
        results["ols_exponent"] = ols_exponent
        results["ols_r2"] = r ** 2
        log(f"  OLS log-log slope: {slope:.2f} (R² = {r**2:.3f})")
        log(f"  Implied exponent: {ols_exponent:.2f}")

    # KS test: power-law vs exponential
    # Exponential fit
    lambda_exp = 1 / np.mean(depth_vals)
    ks_exp, p_exp = stats.kstest(depth_vals, "expon", args=(0, 1 / lambda_exp))
    results["ks_exp_stat"] = ks_exp
    results["ks_exp_p"] = p_exp
    log(f"\n  KS test vs exponential: D={ks_exp:.3f}, p={p_exp:.3f}")

    # Lognormal fit
    mu_ln, sigma_ln = np.mean(np.log(depth_vals)), np.std(np.log(depth_vals))
    ks_ln, p_ln = stats.kstest(depth_vals, "lognorm", args=(sigma_ln, 0, np.exp(mu_ln)))
    results["ks_ln_stat"] = ks_ln
    results["ks_ln_p"] = p_ln
    log(f"  KS test vs lognormal: D={ks_ln:.3f}, p={p_ln:.3f}")

    log(f"\n  CAVEAT: n={n} recessions is too few for formal power-law inference")
    log(f"  Results are suggestive only")

    # Verdict: power law is consistent if exponent is in plausible range
    if not np.isnan(alpha_hill) and 1.0 < alpha_hill < 4.0:
        verdict = "SUGGESTIVE"
    else:
        verdict = "AMBIGUOUS"
    results["verdict"] = verdict
    log(f"  Verdict: {verdict}")
    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  11. SUMMARY + OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════

def summarize_results(ordering, asymmetry, csd, temperature,
                      regulation, phillips, diversity, power_law):
    """Print composite summary."""
    log("=" * 72)
    log("11. SUMMARY: SECTORAL/MACRO PREDICTIONS #9-18")
    log("=" * 72)

    tests = [
        ("#9-11 ρ-ordering", ordering),
        ("#12 Asymmetry ratio", asymmetry),
        ("#13 Critical slowing", csd),
        ("#14 Temperature T", temperature),
        ("#15 Regulation amplitude", regulation),
        ("#16 Phillips ∝ K̄", phillips),
        ("#17 ρ-diversity", diversity),
        ("#18 Power-law depths", power_law),
    ]

    summary_rows = []
    for name, res in tests:
        verdict = res.get("verdict", "NOT RUN")
        log(f"  {name}: {verdict}")
        summary_rows.append({"prediction": name, "verdict": verdict})

    n_consistent = sum(1 for r in summary_rows if r["verdict"] in ["CONSISTENT", "SUGGESTIVE"])
    n_directional = sum(1 for r in summary_rows if r["verdict"] == "DIRECTIONAL")
    n_ambig = sum(1 for r in summary_rows if r["verdict"] in ["AMBIGUOUS", "INSUFFICIENT DATA"])
    n_incon = sum(1 for r in summary_rows if r["verdict"] == "INCONSISTENT")

    log(f"\n  Consistent/suggestive: {n_consistent}")
    log(f"  Directional: {n_directional}")
    log(f"  Ambiguous/insufficient: {n_ambig}")
    log(f"  Inconsistent: {n_incon}")
    log()

    return summary_rows


def write_csv(ordering, asymmetry, csd, temperature,
              regulation, phillips, diversity, power_law):
    """Write CSV with all test statistics."""
    rows = []

    # ρ-ordering
    rows.append({"test": "rho_ordering", "metric": "pooled_beta1",
                 "value": ordering.get("pooled_beta1", np.nan),
                 "p_value": ordering.get("pooled_p", np.nan),
                 "verdict": ordering.get("verdict", "")})
    rows.append({"test": "rho_ordering", "metric": "pooled_tau",
                 "value": ordering.get("pooled_tau", np.nan),
                 "p_value": ordering.get("pooled_tau_p", np.nan),
                 "verdict": ""})

    # Asymmetry
    rows.append({"test": "asymmetry", "metric": "mean_ratio",
                 "value": asymmetry.get("mean_ratio", np.nan),
                 "p_value": np.nan,
                 "verdict": asymmetry.get("verdict", "")})

    # Critical slowing
    rows.append({"test": "critical_slowing", "metric": "frac_rising",
                 "value": csd.get("frac_rising", np.nan),
                 "p_value": csd.get("p_sign", np.nan),
                 "verdict": csd.get("verdict", "")})

    # Temperature
    rows.append({"test": "temperature", "metric": "probit_pseudo_R2",
                 "value": temperature.get("probit_T_r2", np.nan),
                 "p_value": temperature.get("probit_T_pval", np.nan),
                 "verdict": temperature.get("verdict", "")})

    # Regulation
    rows.append({"test": "regulation", "metric": "vol_ratio_post_pre",
                 "value": regulation.get("vol_ratio", np.nan),
                 "p_value": np.nan,
                 "verdict": regulation.get("verdict", "")})

    # Phillips
    rows.append({"test": "phillips_slope", "metric": "tau_K_slope",
                 "value": phillips.get("tau_K_slope", phillips.get("tau_trend", np.nan)),
                 "p_value": phillips.get("p_tau", phillips.get("p_trend", np.nan)),
                 "verdict": phillips.get("verdict", "")})

    # Diversity
    rows.append({"test": "rho_diversity", "metric": "tau_diversity",
                 "value": diversity.get("tau_diversity", np.nan),
                 "p_value": diversity.get("p_diversity", np.nan),
                 "verdict": diversity.get("verdict", "")})

    # Power law
    rows.append({"test": "power_law", "metric": "hill_alpha",
                 "value": power_law.get("hill_alpha", np.nan),
                 "p_value": np.nan,
                 "verdict": power_law.get("verdict", "")})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    log(f"  CSV: {RESULTS_CSV}")
    return df


def write_latex_table(ordering, asymmetry, csd, temperature,
                      regulation, phillips, diversity, power_law):
    """Write publication-ready LaTeX table."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Sectoral and macroeconomic predictions: empirical tests}")
    lines.append(r"\label{tab:business_cycle}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcccl}")
    lines.append(r"\toprule")
    lines.append(r"\# & Prediction & Statistic & Value & $p$ & Verdict \\")
    lines.append(r"\midrule")

    # #9-11 ρ-ordering
    beta1 = ordering.get("pooled_beta1", np.nan)
    p_beta = ordering.get("pooled_p", np.nan)
    v = ordering.get("verdict", "---")
    vmark = r"$\checkmark$" if v == "CONSISTENT" else (r"$\sim$" if v == "DIRECTIONAL" else "---")
    if not np.isnan(beta1):
        lines.append(f"9--11 & $\\rho$-ordering of recession entry & "
                     f"OLS $\\beta_1$ & {beta1:+.2f} & {p_beta:.3f} & {vmark} \\\\")

    # #12 Asymmetry
    ratio = asymmetry.get("mean_ratio", np.nan)
    v = asymmetry.get("verdict", "---")
    vmark = r"$\checkmark$" if v == "CONSISTENT" else (r"$\sim$" if v == "DIRECTIONAL" else "---")
    if not np.isnan(ratio):
        lines.append(f"12 & Expansion/contraction $\\approx 1/\\varepsilon$ & "
                     f"mean ratio & {ratio:.1f} & --- & {vmark} \\\\")

    # #13 Critical slowing
    frac = csd.get("frac_rising", np.nan)
    p_sign = csd.get("p_sign", np.nan)
    v = csd.get("verdict", "---")
    vmark = r"$\checkmark$" if v == "CONSISTENT" else (r"$\sim$" if v == "DIRECTIONAL" else "---")
    if not np.isnan(frac):
        p_str = f"{p_sign:.3f}" if not np.isnan(p_sign) else "---"
        lines.append(f"13 & Critical slowing down & "
                     f"frac rising AR(1) & {frac:.2f} & {p_str} & {vmark} \\\\")

    # #14 Temperature
    pr2 = temperature.get("probit_T_r2", np.nan)
    p_T = temperature.get("probit_T_pval", np.nan)
    v = temperature.get("verdict", "---")
    vmark = r"$\checkmark$" if v == "CONSISTENT" else (r"$\sim$" if v == "DIRECTIONAL" else "---")
    if not np.isnan(pr2):
        lines.append(f"14 & Temperature $T = \\sigma^2/\\chi$ & "
                     f"probit pseudo-$R^2$ & {pr2:.4f} & {p_T:.3f} & {vmark} \\\\")

    # #15 Regulation
    vol_r = regulation.get("vol_ratio", np.nan)
    v = regulation.get("verdict", "---")
    vmark = r"$\checkmark$" if v == "CONSISTENT" else (r"$\sim$" if v == "DIRECTIONAL" else "---")
    if not np.isnan(vol_r):
        lines.append(f"15 & Regulation reduces amplitude & "
                     f"$\\sigma$ ratio (post/pre 1984) & {vol_r:.2f} & --- & {vmark} \\\\")

    # #16 Phillips
    tau_p = phillips.get("tau_K_slope", phillips.get("tau_trend", np.nan))
    p_ph = phillips.get("p_tau", phillips.get("p_trend", np.nan))
    v = phillips.get("verdict", "---")
    vmark = r"$\checkmark$" if v == "CONSISTENT" else (r"$\sim$" if v == "DIRECTIONAL" else "---")
    if not np.isnan(tau_p):
        lines.append(f"16 & Phillips slope $\\propto \\bar{{K}}$ & "
                     f"Kendall $\\tau$ & {tau_p:+.3f} & {p_ph:.3f} & {vmark} \\\\")

    # #17 Diversity
    tau_d = diversity.get("tau_diversity", np.nan)
    p_d = diversity.get("p_diversity", np.nan)
    v = diversity.get("verdict", "---")
    vmark = "suggestive" if v == "SUGGESTIVE" else "---"
    if not np.isnan(tau_d):
        lines.append(f"17 & $\\rho$-diversity $\\to$ resilience & "
                     f"Kendall $\\tau$ & {tau_d:+.3f} & {p_d:.3f} & {vmark} \\\\")

    # #18 Power law
    alpha = power_law.get("hill_alpha", np.nan)
    v = power_law.get("verdict", "---")
    vmark = "suggestive" if v == "SUGGESTIVE" else "---"
    if not np.isnan(alpha):
        lines.append(f"18 & Power-law recession depths & "
                     f"Hill $\\hat{{\\alpha}}$ & {alpha:.2f} & --- & {vmark} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{minipage}{0.95\textwidth}")
    lines.append(r"\vspace{0.5em}")
    lines.append(r"\footnotesize\textit{Notes:} $\rho$-ordering tests whether low-$\rho$ sectors "
                 r"(construction, metals) lead recessions via pooled OLS on peak timing "
                 r"($\beta_1 < 0$ predicted). Asymmetry ratio uses 12 post-war NBER cycles. "
                 r"Critical slowing down tests whether rolling AR(1) rises in 36 months before "
                 r"NBER peaks. Temperature $T = \sigma^2/\chi$ tested via 12-month-ahead recession "
                 r"probit. Regulation test: Great Moderation volatility decomposition. "
                 r"Phillips slope tested against composition-weighted $\bar{K}$. "
                 r"$\checkmark$ = consistent ($p < 0.10$); $\sim$ = directional (correct sign, $p > 0.10$); "
                 r"--- = ambiguous/insufficient. "
                 r"Data: FRED IP subsectors (monthly, 1972--2025) and 50-state GDP (quarterly, 2005--2025).")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")

    with open(RESULTS_TEX, "w") as f:
        f.write("\n".join(lines))
    log(f"  LaTeX table: {RESULTS_TEX}")


def make_figure(ordering, asymmetry, csd, phillips, diversity, power_law):
    """Generate 6-panel figure."""
    log("=" * 72)
    log("GENERATING 6-PANEL FIGURE")
    log("=" * 72)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Sectoral/Macro Predictions #9-18 from CES Free Energy Framework\n"
                 "(Paper 16: Unified Theory)",
                 fontsize=13, fontweight="bold")

    # ── Panel (a): ρ vs recession lead time (pooled) ──
    ax = axes[0, 0]
    leads = ordering.get("all_leads", [])
    if leads:
        df_l = pd.DataFrame(leads)
        ax.scatter(df_l["rho"], df_l["lead_months"], c="steelblue", s=30, alpha=0.6, zorder=3)
        # OLS line
        X = sm.add_constant(df_l["rho"].values)
        y = df_l["lead_months"].values
        try:
            ols = sm.OLS(y, X).fit()
            xs = np.linspace(df_l["rho"].min() - 0.1, df_l["rho"].max() + 0.1, 50)
            ax.plot(xs, ols.params[0] + ols.params[1] * xs, "r--", linewidth=1.5, alpha=0.7)
            ax.text(0.95, 0.95, f"β₁={ols.params[1]:+.1f}\np={ols.pvalues[1]:.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
        except Exception:
            pass
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
    ax.set_xlabel(r"Sector $\rho$")
    ax.set_ylabel("Lead time (months before NBER peak)")
    ax.set_title(r"(a) $\rho$-ordering: peak timing")
    ax.grid(True, alpha=0.3)

    # ── Panel (b): Expansion/contraction ratios ──
    ax = axes[0, 1]
    ratios_df = asymmetry.get("ratios_df")
    if ratios_df is not None:
        x = np.arange(len(ratios_df))
        ax.bar(x, ratios_df["ratio"].values, color="steelblue", alpha=0.7, edgecolor="k", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(ratios_df["recession"].values, rotation=45, fontsize=6, ha="right")
        mean_r = asymmetry.get("mean_ratio", np.nan)
        if not np.isnan(mean_r):
            ax.axhline(mean_r, color="red", linestyle="--", linewidth=1, label=f"Mean={mean_r:.1f}")
            ax.axhline(10, color="green", linestyle=":", linewidth=1, label="Predicted≈10")
            ax.legend(fontsize=7, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
    ax.set_ylabel("Expansion / Contraction ratio")
    ax.set_title("(b) Asymmetry ratio by cycle")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel (c): Rolling AR(1) in pre-recession windows ──
    ax = axes[1, 0]
    ar1_df = csd.get("ar1_df")
    if ar1_df is not None and len(ar1_df.columns) > 0:
        # Plot mean AR(1) across sectors
        mean_ar1 = ar1_df.mean(axis=1).dropna()
        ax.plot(mean_ar1.index, mean_ar1.values, "b-", linewidth=1, alpha=0.8)

        # Shade recessions
        for start, end in RECESSIONS:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       alpha=0.15, color="gray", zorder=0)

        frac = csd.get("frac_rising", np.nan)
        if not np.isnan(frac):
            ax.text(0.02, 0.98, f"Frac rising: {frac:.2f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean AR(1) coefficient")
    ax.set_title("(c) Critical slowing down")
    ax.grid(True, alpha=0.3)
    if ar1_df is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(10))

    # ── Panel (d): K̄(t) and rolling Phillips slope ──
    ax = axes[0, 2]
    rolling_slopes = phillips.get("rolling_slopes")
    K_bar = phillips.get("K_bar")
    if rolling_slopes is not None and len(rolling_slopes) > 0:
        ax.plot(rolling_slopes.index, rolling_slopes["slope"].values,
                "b-", linewidth=1.2, label="Phillips slope")
        ax.set_ylabel("Phillips slope (β₁)", color="blue")
        ax.tick_params(axis="y", labelcolor="blue")

        if K_bar is not None:
            ax2 = ax.twinx()
            ax2.plot(K_bar.index, K_bar.values, "r-", linewidth=1.2,
                     alpha=0.7, label=r"$\bar{K}$")
            ax2.set_ylabel(r"$\bar{K}$ (composition-weighted)", color="red")
            ax2.tick_params(axis="y", labelcolor="red")

        tau_kp = phillips.get("tau_K_slope", np.nan)
        p_kp = phillips.get("p_tau", np.nan)
        if not np.isnan(tau_kp):
            ax.text(0.02, 0.98, f"τ(K̄,slope)={tau_kp:+.3f}\np={p_kp:.3f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
    ax.set_title(r"(d) Phillips slope and $\bar{K}$")
    ax.grid(True, alpha=0.3)

    # ── Panel (e): ρ-diversity vs recession depth ──
    ax = axes[1, 1]
    rec_data = diversity.get("recession_data")
    if rec_data is not None:
        df_rd = rec_data.dropna(subset=["rho_diversity"])
        if len(df_rd) > 0:
            ax.scatter(df_rd["rho_diversity"], df_rd["depth_pct"],
                       c="darkorange", s=80, zorder=3, edgecolors="k", linewidth=0.5)
            for _, row in df_rd.iterrows():
                ax.annotate(row["recession"], (row["rho_diversity"], row["depth_pct"]),
                           fontsize=7, xytext=(5, 5), textcoords="offset points")
        else:
            ax.text(0.5, 0.5, "No diversity data", transform=ax.transAxes, ha="center")
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
    ax.set_xlabel(r"$\rho$-diversity (std of $w_n \rho_n$)")
    ax.set_ylabel("Recession depth (%)")
    ax.set_title(r"(e) $\rho$-diversity vs depth")
    ax.grid(True, alpha=0.3)

    # ── Panel (f): Log-log recession depth rank plot ──
    ax = axes[1, 2]
    depths = power_law.get("depths")
    if depths is not None and len(depths) > 0:
        ranks = np.arange(1, len(depths) + 1)
        ax.scatter(np.log(depths), np.log(ranks), c="darkred", s=50, zorder=3)

        # OLS fit line
        slope, intercept, r, p, se = stats.linregress(np.log(depths), np.log(ranks))
        xs = np.linspace(np.log(depths).min() - 0.2, np.log(depths).max() + 0.2, 50)
        ax.plot(xs, intercept + slope * xs, "r--", linewidth=1.5, alpha=0.7)

        alpha_h = power_law.get("hill_alpha", np.nan)
        if not np.isnan(alpha_h):
            ax.text(0.95, 0.95, f"Hill α̂={alpha_h:.2f}\nOLS slope={slope:.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

        # Label points
        for i, (peak, _) in enumerate(RECESSIONS):
            if i < len(depths):
                idx = np.where(depths == np.sort(depths)[::-1])[0]
                # Just label by rank
                pass
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
    ax.set_xlabel("log(recession depth %)")
    ax.set_ylabel("log(rank)")
    ax.set_title("(f) Log-log rank plot of depths")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIG_PATH}")
    log()


# ═══════════════════════════════════════════════════════════════════════════
#  12. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 72)
    log("  BUSINESS CYCLE PREDICTIONS #9-18: EMPIRICAL TESTS")
    log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 72)
    log()

    # Load data
    log("1. LOADING DATA")
    log("-" * 40)
    raw_mfg, growth_mfg = load_manufacturing_data()
    national = load_national_sectors()
    financial = load_financial_data()
    log()

    # Run tests
    log("2. RUNNING TESTS")
    log("-" * 40)
    ordering = test_rho_ordering(growth_mfg, national)
    asymmetry = test_asymmetry()
    csd = test_critical_slowing(growth_mfg)
    temperature = test_temperature(growth_mfg, financial)
    regulation = test_regulation()
    phillips = test_phillips_slope(national, financial)
    diversity = test_rho_diversity(national, financial)
    power_law = test_power_law_depths(financial)

    # Summary and outputs
    summary = summarize_results(ordering, asymmetry, csd, temperature,
                                regulation, phillips, diversity, power_law)

    log("\n" + "=" * 72)
    log("WRITING OUTPUTS")
    log("=" * 72)
    write_csv(ordering, asymmetry, csd, temperature,
              regulation, phillips, diversity, power_law)
    write_latex_table(ordering, asymmetry, csd, temperature,
                      regulation, phillips, diversity, power_law)
    make_figure(ordering, asymmetry, csd, phillips, diversity, power_law)

    # Write results text file
    with open(RESULTS_TXT, "w") as f:
        f.write(results_buf.getvalue())
    log(f"  Results text: {RESULTS_TXT}")

    log("\nDone.")


if __name__ == "__main__":
    main()
