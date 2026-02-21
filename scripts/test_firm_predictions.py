#!/usr/bin/env python3
"""
test_firm_predictions.py
========================
Unified Theory (Paper 16), Firm-Level Predictions Empirical Tests

Tests four firm-level predictions derived from the CES free energy framework:

  #4 FDT (Fluctuation-Dissipation Theorem): T = σ²/χ — sector volatility and
     susceptibility should be linearly related
  #6 Equicorrelation: CES compound symmetry implies off-diagonal correlations
     are approximately equal
  #7 Onsager Reciprocity: Cross-sector IRF matrix should be symmetric (L_ij ≈ L_ji)
  #8 Cyclical ρ: CES complementarity parameter should be countercyclical

Data sources:
  30 FRED Industrial Production series, cached in thesis_data/fred_cache/
  Monthly data starting 1972-01-01

Outputs:
  thesis_data/firm_predictions_results.csv
  thesis_data/firm_predictions_results.txt
  thesis_data/firm_predictions_table.tex
  figures/firm_predictions.pdf

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
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════════════════
#  0. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

FIG_DIR = "/home/jonsmirl/thesis/figures"
DATA_DIR = "/home/jonsmirl/thesis/thesis_data"
CACHE_DIR = os.path.join(DATA_DIR, "fred_cache")
RESULTS_TXT = os.path.join(DATA_DIR, "firm_predictions_results.txt")
RESULTS_CSV = os.path.join(DATA_DIR, "firm_predictions_results.csv")
RESULTS_TEX = os.path.join(DATA_DIR, "firm_predictions_table.tex")
FIG_PATH = os.path.join(FIG_DIR, "firm_predictions.pdf")

for d in [FIG_DIR, DATA_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    print("ERROR: FRED_API_KEY not set. Source env.sh first.")
    sys.exit(1)

START_DATE = "1972-01-01"

np.random.seed(42)

results_buf = StringIO()


def log(msg=""):
    """Print to stdout and accumulate for results file."""
    print(msg)
    results_buf.write(msg + "\n")


# NAICS hierarchy of FRED IP series (from test_ces_emergence.py)
HIERARCHY = {
    'INDPRO': {
        'IPMAN': {
            'IPDMAN': {
                'IPG331S': {},   # Primary metals
                'IPG332S': {},   # Fabricated metal products
                'IPG333S': {},   # Machinery
                'IPG334S': {     # Computer & electronic products
                    'IPG3341S': {},  # Computer & peripheral equipment
                    'IPG3342S': {},  # Communications equipment
                    'IPG3343S': {},  # Audio & video equipment
                    'IPG3344S': {},  # Semiconductors
                    'IPG3345S': {},  # Electronic instruments
                },
                'IPG335S': {},   # Electrical equipment
                'IPG336S': {     # Transportation equipment
                    'IPG3361T3S': {},  # Motor vehicles & parts
                    'IPG3364T9S': {},  # Aerospace & other transport
                },
                'IPG337S': {},   # Furniture
                'IPG339S': {},   # Miscellaneous manufacturing
            },
            'IPNMAN': {
                'IPG311A2S': {},  # Food, beverage, tobacco
                'IPG313A4S': {},  # Textiles & products
                'IPG315A6S': {},  # Apparel & leather
                'IPG321S': {},    # Wood products
                'IPG322S': {},    # Paper
                'IPG323S': {},    # Printing
                'IPG324S': {},    # Petroleum & coal products
                'IPG325S': {},    # Chemicals
                'IPG326S': {},    # Plastics & rubber
            },
        },
        'IPMINE': {},
        'IPUTIL': {},
    }
}

LABELS = {
    'INDPRO': 'Total IP',
    'IPMAN': 'Manufacturing',
    'IPDMAN': 'Durable Goods',
    'IPNMAN': 'Nondurable Goods',
    'IPMINE': 'Mining',
    'IPUTIL': 'Utilities',
    'IPG331S': 'Primary Metals',
    'IPG332S': 'Fabricated Metals',
    'IPG333S': 'Machinery',
    'IPG334S': 'Computer/Elec',
    'IPG335S': 'Electrical Equip',
    'IPG336S': 'Transport Equip',
    'IPG337S': 'Furniture',
    'IPG339S': 'Misc Mfg',
    'IPG3341S': 'Computers',
    'IPG3342S': 'Comms Equip',
    'IPG3343S': 'Audio/Video',
    'IPG3344S': 'Semiconductors',
    'IPG3345S': 'Elec Instruments',
    'IPG3361T3S': 'Motor Vehicles',
    'IPG3364T9S': 'Aerospace',
    'IPG311A2S': 'Food/Bev/Tobacco',
    'IPG313A4S': 'Textiles',
    'IPG315A6S': 'Apparel/Leather',
    'IPG321S': 'Wood Products',
    'IPG322S': 'Paper',
    'IPG323S': 'Printing',
    'IPG324S': 'Petroleum/Coal',
    'IPG325S': 'Chemicals',
    'IPG326S': 'Plastics/Rubber',
}

# 17 leaf sectors for correlation/FDT tests
LEAF_SECTORS = [
    # Durable (8)
    'IPG331S', 'IPG332S', 'IPG333S', 'IPG334S',
    'IPG335S', 'IPG336S', 'IPG337S', 'IPG339S',
    # Nondurable (9)
    'IPG311A2S', 'IPG313A4S', 'IPG315A6S', 'IPG321S',
    'IPG322S', 'IPG323S', 'IPG324S', 'IPG325S', 'IPG326S',
]

# 8 durable manufacturing sectors for Onsager VAR / cyclical ρ
DURABLE_SECTORS = [
    'IPG331S', 'IPG332S', 'IPG333S', 'IPG334S',
    'IPG335S', 'IPG336S', 'IPG337S', 'IPG339S',
]

# NBER recession dates for panel (d) shading
RECESSIONS = [
    ('1973-11', '1975-03'), ('1980-01', '1980-07'), ('1981-07', '1982-11'),
    ('1990-07', '1991-03'), ('2001-03', '2001-11'), ('2007-12', '2009-06'),
    ('2020-02', '2020-04'),
]


# ═══════════════════════════════════════════════════════════════════════════
#  1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def fetch_and_cache(series_id):
    """Fetch FRED series with local CSV caching."""
    cache_path = os.path.join(CACHE_DIR, f"{series_id}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df.iloc[:, 0]

    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={FRED_API_KEY}"
        f"&file_type=json&observation_start={START_DATE}"
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
    time.sleep(0.15)
    return s


def collect_all_series(hierarchy, collected=None):
    """Recursively collect all series IDs from hierarchy."""
    if collected is None:
        collected = set()
    for parent, children in hierarchy.items():
        collected.add(parent)
        if children:
            collect_all_series(children, collected)
    return collected


def load_all_data():
    """Fetch all 30 series and compute log-first-differences (growth rates)."""
    log("=" * 72)
    log("1. LOADING FRED IP DATA")
    log("=" * 72)

    all_ids = collect_all_series(HIERARCHY)
    log(f"  Fetching {len(all_ids)} FRED IP series...")

    raw = {}
    for sid in sorted(all_ids):
        s = fetch_and_cache(sid)
        if s is not None and len(s) > 0:
            raw[sid] = s
            log(f"    {sid}: {len(s)} obs ({s.index[0].year}-{s.index[-1].year})")
        else:
            log(f"    {sid}: MISSING")

    log(f"  Loaded {len(raw)} series")

    # Compute growth rates (log first-differences)
    growth = {}
    for sid, s in raw.items():
        g = np.log(s).diff().dropna()
        growth[sid] = g

    # Build aligned dataframe of leaf sector growth rates
    leaf_df = pd.DataFrame({sid: growth[sid] for sid in LEAF_SECTORS if sid in growth})
    leaf_df = leaf_df.dropna()
    log(f"  Leaf sector panel: {leaf_df.shape[0]} months x {leaf_df.shape[1]} sectors")
    log(f"  Date range: {leaf_df.index[0].date()} to {leaf_df.index[-1].date()}")

    # Durable sector growth rates
    dur_df = pd.DataFrame({sid: growth[sid] for sid in DURABLE_SECTORS if sid in growth})
    dur_df = dur_df.dropna()
    log(f"  Durable panel: {dur_df.shape[0]} months x {dur_df.shape[1]} sectors")

    log()
    return raw, growth, leaf_df, dur_df


# ═══════════════════════════════════════════════════════════════════════════
#  2. TEST #4: FLUCTUATION-DISSIPATION THEOREM (FDT)
# ═══════════════════════════════════════════════════════════════════════════

def test_fdt(growth, leaf_df):
    """
    Test the Fluctuation-Dissipation Theorem: T = σ²/χ.

    For each leaf sector j:
      σ²_j = variance of growth rate (rolling 120-month, then average)
      χ_j  = cumulative IRF response to INDPRO shock (bivariate VAR, h=1..12)

    OLS: σ²_j = α + T·χ_j + ε across 17 sectors.
    Prediction: R² > 0.3, slope T > 0.
    """
    log("=" * 72)
    log("2. TEST #4: FLUCTUATION-DISSIPATION THEOREM (FDT)")
    log("=" * 72)
    log("  Theory: T = σ²/χ — sector volatility and susceptibility linearly related")
    log()

    results = {"test": "FDT"}

    if 'INDPRO' not in growth:
        log("  ERROR: INDPRO not available")
        return results

    indpro_g = growth['INDPRO']
    sectors = [s for s in LEAF_SECTORS if s in leaf_df.columns]
    log(f"  Testing {len(sectors)} leaf sectors against INDPRO")

    sector_stats = []

    for sid in sectors:
        sector_g = growth[sid]

        # Align sector and INDPRO growth
        combined = pd.DataFrame({'sector': sector_g, 'indpro': indpro_g}).dropna()
        if len(combined) < 150:
            log(f"    {LABELS.get(sid, sid)}: insufficient data ({len(combined)} obs), skipping")
            continue

        # σ² = average of rolling 120-month variance
        rolling_var = combined['sector'].rolling(120, min_periods=60).var()
        sigma2 = rolling_var.mean()

        # χ = cumulative IRF from bivariate VAR
        try:
            # Order: [sector, indpro] — shock to indpro (col 1), response of sector (row 0)
            var_data = combined[['sector', 'indpro']].values
            model = VAR(var_data)
            # BIC lag selection, maxlags=8
            lag_order = model.select_order(maxlags=8)
            p = lag_order.bic
            if p < 1:
                p = 1

            var_fit = model.fit(p)
            irf = var_fit.irf(12)

            # Cumulative response of sector (variable 0) to indpro shock (shock 1)
            # irf.irfs shape: (horizons+1, nvars, nvars) — irfs[h][response][shock]
            chi = np.sum(irf.irfs[1:13, 0, 1])  # h=1..12, response=sector, shock=indpro

            sector_stats.append({
                'sector': sid,
                'label': LABELS.get(sid, sid),
                'sigma2': sigma2,
                'chi': chi,
                'n_obs': len(combined),
                'var_lags': p,
            })
            log(f"    {LABELS.get(sid, sid):20s}: σ²={sigma2:.6f}, χ={chi:.4f}, "
                f"lags={p}, n={len(combined)}")

        except Exception as e:
            log(f"    {LABELS.get(sid, sid)}: VAR failed: {e}")
            continue

    if len(sector_stats) < 5:
        log(f"  ERROR: only {len(sector_stats)} sectors succeeded, need >= 5")
        return results

    sdf = pd.DataFrame(sector_stats)
    results['sector_stats'] = sdf

    # OLS: σ² = α + T·χ + ε
    X = sm.add_constant(sdf['chi'].values)
    y = sdf['sigma2'].values
    ols = sm.OLS(y, X).fit()

    T_est = ols.params[1]
    intercept = ols.params[0]
    r2 = ols.rsquared
    p_slope = ols.pvalues[1]
    se_slope = ols.bse[1]

    results['T_estimate'] = T_est
    results['intercept'] = intercept
    results['R2'] = r2
    results['p_slope'] = p_slope
    results['se_slope'] = se_slope
    results['n_sectors'] = len(sdf)

    log(f"\n  OLS: σ² = {intercept:.6f} + {T_est:.6f} · χ")
    log(f"  R²  = {r2:.4f}")
    log(f"  T̂   = {T_est:.6f} (SE = {se_slope:.6f}, p = {p_slope:.4f})")
    log(f"  N   = {len(sdf)} sectors")

    if r2 > 0.3 and T_est > 0 and p_slope < 0.05:
        log(f"  => CONSISTENT: R² > 0.3, T > 0, p < 0.05")
    elif T_est > 0 and p_slope < 0.10:
        log(f"  => PARTIALLY CONSISTENT: T > 0 (p < 0.10) but R² = {r2:.3f}")
    elif T_est > 0:
        log(f"  => DIRECTIONAL: T > 0 but not significant (p = {p_slope:.3f})")
    else:
        log(f"  => INCONSISTENT: T ≤ 0")

    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  3. TEST #7: ONSAGER RECIPROCITY
# ═══════════════════════════════════════════════════════════════════════════

def test_onsager(dur_df):
    """
    Test Onsager reciprocity: L_ij ≈ L_ji in cross-sector IRF matrix.

    Estimate 8-variable VAR on durable manufacturing sectors.
    Extract IRF matrix L at horizon h=12: L_ij = cumulative response of i to shock j.
    Pearson correlation of {L_ij} vs {L_ji} across 28 off-diagonal pairs.
    Prediction: r > 0.5, slope near 1.0.
    """
    log("=" * 72)
    log("3. TEST #7: ONSAGER RECIPROCITY")
    log("=" * 72)
    log("  Theory: Cross-sector IRF matrix should be symmetric (L_ij ≈ L_ji)")
    log()

    results = {"test": "Onsager"}

    n_vars = dur_df.shape[1]
    if n_vars < 3:
        log(f"  ERROR: only {n_vars} durable sectors, need >= 3")
        return results

    log(f"  Fitting {n_vars}-variable VAR on durable manufacturing sectors")
    log(f"  Sectors: {', '.join(LABELS.get(s, s) for s in dur_df.columns)}")
    log(f"  Observations: {len(dur_df)}")

    try:
        model = VAR(dur_df.values)
        lag_order = model.select_order(maxlags=6)
        p = lag_order.bic
        if p < 1:
            p = 1
        log(f"  BIC-selected lag order: {p}")

        var_fit = model.fit(p)
        irf = var_fit.irf(12)

        # Cumulative IRF matrix at h=12
        # irf.irfs shape: (13, n_vars, n_vars) — irfs[h][response][shock]
        L = np.sum(irf.irfs[1:13], axis=0)  # sum h=1..12, shape (n_vars, n_vars)

        # Extract 28 off-diagonal pairs (i < j)
        pairs_ij = []
        pairs_ji = []
        pair_labels = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                pairs_ij.append(L[i, j])
                pairs_ji.append(L[j, i])
                si = dur_df.columns[i]
                sj = dur_df.columns[j]
                pair_labels.append(f"{LABELS.get(si, si)[:8]}-{LABELS.get(sj, sj)[:8]}")

        pairs_ij = np.array(pairs_ij)
        pairs_ji = np.array(pairs_ji)

        results['L_matrix'] = L
        results['pairs_ij'] = pairs_ij
        results['pairs_ji'] = pairs_ji
        results['pair_labels'] = pair_labels
        results['var_lags'] = p
        results['sector_names'] = list(dur_df.columns)

        # Pearson correlation
        r_pearson, p_pearson = stats.pearsonr(pairs_ij, pairs_ji)

        # Regression: L_ji = a + b·L_ij
        X = sm.add_constant(pairs_ij)
        ols = sm.OLS(pairs_ji, X).fit()
        slope = ols.params[1]
        slope_se = ols.bse[1]

        # Symmetry measure: mean |L_ij - L_ji| / mean |L_ij|
        asym = np.mean(np.abs(pairs_ij - pairs_ji))
        scale = np.mean(np.abs(np.concatenate([pairs_ij, pairs_ji])))
        rel_asym = asym / scale if scale > 0 else np.nan

        results['r_pearson'] = r_pearson
        results['p_pearson'] = p_pearson
        results['slope'] = slope
        results['slope_se'] = slope_se
        results['n_pairs'] = len(pairs_ij)
        results['relative_asymmetry'] = rel_asym

        log(f"  Off-diagonal pairs: {len(pairs_ij)}")
        log(f"  Pearson r(L_ij, L_ji)  = {r_pearson:.4f} (p = {p_pearson:.4f})")
        log(f"  Regression slope       = {slope:.4f} (SE = {slope_se:.4f})")
        log(f"  Relative asymmetry     = {rel_asym:.4f}")

        if r_pearson > 0.5 and p_pearson < 0.05:
            log(f"  => CONSISTENT: r > 0.5, significant (p < 0.05)")
        elif r_pearson > 0.3 and p_pearson < 0.10:
            log(f"  => PARTIALLY CONSISTENT: r > 0.3 (p < 0.10)")
        elif r_pearson > 0:
            log(f"  => DIRECTIONAL: r > 0 but weak")
        else:
            log(f"  => INCONSISTENT: r ≤ 0")

    except Exception as e:
        log(f"  ERROR: VAR estimation failed: {e}")

    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  4. TEST #6: EQUICORRELATION (COMPOUND SYMMETRY)
# ═══════════════════════════════════════════════════════════════════════════

def test_equicorrelation(leaf_df):
    """
    Test CES compound symmetry: off-diagonal correlations approximately equal.

    Compute correlation matrix of 17 leaf sector growth rates.
    Measure CV of off-diagonal correlations and Frobenius distance from equicorr.
    Compare to null via permutation (1000 shuffles).
    Prediction: CV < null median; Frobenius ratio < 0.5.
    """
    log("=" * 72)
    log("4. TEST #6: EQUICORRELATION (COMPOUND SYMMETRY)")
    log("=" * 72)
    log("  Theory: CES implies Σ = (σ²−γ)I + γ11ᵀ — equal off-diagonal correlations")
    log()

    results = {"test": "Equicorrelation"}

    n = leaf_df.shape[1]
    log(f"  Computing {n}x{n} correlation matrix ({n*(n-1)//2} off-diagonal pairs)")

    corr_matrix = leaf_df.corr().values
    results['corr_matrix'] = corr_matrix
    results['sector_names'] = list(leaf_df.columns)

    # Extract off-diagonal correlations
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            off_diag.append(corr_matrix[i, j])
    off_diag = np.array(off_diag)

    mean_corr = np.mean(off_diag)
    std_corr = np.std(off_diag)
    cv_corr = std_corr / abs(mean_corr) if abs(mean_corr) > 1e-10 else np.inf

    # Frobenius distance from equicorrelation matrix
    equicorr = np.full_like(corr_matrix, mean_corr)
    np.fill_diagonal(equicorr, 1.0)
    frob_actual = np.linalg.norm(corr_matrix, 'fro')
    frob_diff = np.linalg.norm(corr_matrix - equicorr, 'fro')
    frob_ratio = frob_diff / frob_actual if frob_actual > 0 else np.nan

    results['mean_corr'] = mean_corr
    results['std_corr'] = std_corr
    results['cv_corr'] = cv_corr
    results['frob_ratio'] = frob_ratio
    results['n_pairs'] = len(off_diag)

    log(f"  Mean off-diagonal correlation: {mean_corr:.4f}")
    log(f"  Std of off-diagonal corr:      {std_corr:.4f}")
    log(f"  CV of off-diagonal corr:        {cv_corr:.4f}")
    log(f"  Frobenius ratio ||Σ-Σ_eq||/||Σ||: {frob_ratio:.4f}")

    # Permutation null: shuffle time indices to break correlations, recompute CV
    log(f"\n  Running 1000-permutation null distribution...")
    null_cvs = []
    null_frobs = []
    T = len(leaf_df)
    data_vals = leaf_df.values.copy()

    for perm in range(1000):
        # Independently shuffle each column
        shuffled = data_vals.copy()
        for col in range(n):
            np.random.shuffle(shuffled[:, col])

        perm_corr = np.corrcoef(shuffled, rowvar=False)
        perm_offdiag = []
        for i in range(n):
            for j in range(i + 1, n):
                perm_offdiag.append(perm_corr[i, j])
        perm_offdiag = np.array(perm_offdiag)

        perm_mean = np.mean(perm_offdiag)
        perm_std = np.std(perm_offdiag)
        perm_cv = perm_std / abs(perm_mean) if abs(perm_mean) > 1e-10 else np.inf
        null_cvs.append(perm_cv)

        perm_equicorr = np.full_like(perm_corr, perm_mean)
        np.fill_diagonal(perm_equicorr, 1.0)
        perm_frob = np.linalg.norm(perm_corr - perm_equicorr, 'fro') / np.linalg.norm(perm_corr, 'fro')
        null_frobs.append(perm_frob)

    null_cvs = np.array(null_cvs)
    null_frobs = np.array(null_frobs)

    # Percentile rank (lower = more equicorrelated than null)
    cv_pctile = np.mean(null_cvs <= cv_corr) * 100
    frob_pctile = np.mean(null_frobs <= frob_ratio) * 100

    results['null_cv_median'] = np.median(null_cvs)
    results['null_cv_mean'] = np.mean(null_cvs)
    results['cv_percentile'] = cv_pctile
    results['frob_percentile'] = frob_pctile

    log(f"  Null CV median: {np.median(null_cvs):.4f} (actual: {cv_corr:.4f})")
    log(f"  CV percentile rank: {cv_pctile:.1f}% (lower = more equicorrelated)")
    log(f"  Null Frobenius median: {np.median(null_frobs):.4f} (actual: {frob_ratio:.4f})")
    log(f"  Frobenius percentile rank: {frob_pctile:.1f}%")

    # Bartlett's test of sphericity (H0: Σ = I)
    # Test statistic: -(T-1-(2n+5)/6) * log(det(R))
    det_R = np.linalg.det(corr_matrix)
    if det_R > 0:
        bartlett_stat = -(T - 1 - (2 * n + 5) / 6.0) * np.log(det_R)
        bartlett_df = n * (n - 1) / 2
        bartlett_p = 1 - stats.chi2.cdf(bartlett_stat, bartlett_df)
        results['bartlett_stat'] = bartlett_stat
        results['bartlett_p'] = bartlett_p
        log(f"\n  Bartlett's test of sphericity: χ²={bartlett_stat:.1f}, "
            f"df={bartlett_df:.0f}, p={bartlett_p:.2e}")
        log(f"  (H0: Σ=I rejected)" if bartlett_p < 0.05 else
            f"  (H0: Σ=I not rejected)")
    else:
        log(f"  Bartlett's test: det(R) ≤ 0, matrix near-singular")

    if cv_corr < np.median(null_cvs) and frob_ratio < 0.5:
        log(f"\n  => CONSISTENT: CV < null median and Frobenius ratio < 0.5")
    elif cv_corr < np.median(null_cvs):
        log(f"\n  => PARTIALLY CONSISTENT: CV < null median but Frobenius ratio = {frob_ratio:.3f}")
    else:
        log(f"\n  => INCONSISTENT: CV ≥ null median")

    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  5. TEST #8: CYCLICAL ρ
# ═══════════════════════════════════════════════════════════════════════════

def log_ces_function(X_matrix, log_A, rho, *log_weights):
    """Log CES aggregation for NLS estimation (from test_ces_emergence.py)."""
    K = X_matrix.shape[1]
    lw = np.array(log_weights[:K])
    w = np.exp(lw - lw.max())
    w = w / w.sum()

    rho_safe = np.clip(rho, -2.0, 2.0)
    if abs(rho_safe) < 0.01:
        return log_A + np.sum(w[np.newaxis, :] * np.log(np.maximum(X_matrix, 1e-10)), axis=1)

    inner = np.sum(w[np.newaxis, :] * np.power(np.maximum(X_matrix, 1e-10), rho_safe), axis=1)
    inner = np.maximum(inner, 1e-10)
    return log_A + (1.0 / rho_safe) * np.log(inner)


def estimate_ces_rolling(y, X_dict, window=120, step=12):
    """Rolling-window CES NLS estimation. Returns list of (center_date, rho) tuples."""
    names = sorted(X_dict.keys())
    K = len(names)

    df = pd.DataFrame({'y': y})
    for n in names:
        df[n] = X_dict[n]
    df = df.dropna()
    df = df[(df > 0).all(axis=1)]

    results = []
    T = len(df)

    for start in range(0, T - window + 1, step):
        end = start + window
        sub = df.iloc[start:end]
        center_date = sub.index[window // 2]

        y_vals = sub['y'].values
        X_matrix = sub[names].values
        log_y = np.log(y_vals)

        X_scales = X_matrix.mean(axis=0)
        X_norm = X_matrix / X_scales[np.newaxis, :]

        def model(X, *params):
            return log_ces_function(X, params[0], params[1], *params[2:])

        best_rho = np.nan
        best_cost = np.inf

        for rho_init in [0.5, 0.0, -0.5, 0.8, -1.0]:
            p0 = [np.mean(log_y), rho_init] + [0.0] * K
            bounds_low = [-10.0, -1.99] + [-10.0] * K
            bounds_high = [20.0, 0.99] + [10.0] * K
            try:
                popt, _ = curve_fit(
                    model, X_norm, log_y,
                    p0=p0, bounds=(bounds_low, bounds_high),
                    maxfev=10000, method='trf'
                )
                residuals = log_y - model(X_norm, *popt)
                cost = np.sum(residuals ** 2)
                if cost < best_cost:
                    best_cost = cost
                    best_rho = popt[1]
            except Exception:
                continue

        if not np.isnan(best_rho):
            results.append({'center_date': center_date, 'rho': best_rho})

    return results


def test_cyclical_rho(raw, growth, dur_df):
    """
    Test that CES ρ is countercyclical (rises in contractions, falls in expansions).

    Method A: Rolling 60-month average pairwise correlation as ρ proxy.
    Method B: Rolling 120-month CES NLS estimation of ρ.
    Both correlated with INDPRO growth cycle indicator.
    Prediction #8: ρ rises in contractions → τ(ρ̂, growth) < 0.
    """
    log("=" * 72)
    log("5. TEST #8: CYCLICAL ρ")
    log("=" * 72)
    log("  Theory: ρ is countercyclical — rises in contractions, falls in expansions")
    log()

    results = {"test": "Cyclical_rho"}

    # ── Method A: Rolling correlation proxy ──
    log("  --- Method A: Rolling correlation proxy ---")

    n_dur = dur_df.shape[1]
    T = len(dur_df)
    window_a = 60

    rolling_corr_proxy = []
    for start in range(0, T - window_a + 1, 6):  # step every 6 months
        end = start + window_a
        sub = dur_df.iloc[start:end]
        center_date = sub.index[window_a // 2]

        # Average pairwise correlation
        corr_mat = sub.corr().values
        off_diag = []
        for i in range(n_dur):
            for j in range(i + 1, n_dur):
                off_diag.append(corr_mat[i, j])
        avg_corr = np.mean(off_diag)
        rolling_corr_proxy.append({'center_date': center_date, 'avg_corr': avg_corr})

    proxy_df = pd.DataFrame(rolling_corr_proxy)
    proxy_df = proxy_df.set_index('center_date')

    # INDPRO growth as cycle indicator (rolling 60-month mean)
    if 'INDPRO' not in growth:
        log("  ERROR: INDPRO growth not available")
        return results

    indpro_cycle = growth['INDPRO'].rolling(60, min_periods=30).mean()

    # Align
    common = proxy_df.index.intersection(indpro_cycle.dropna().index)
    proxy_aligned = proxy_df.loc[common, 'avg_corr']
    cycle_aligned = indpro_cycle.reindex(common)

    if len(common) < 10:
        log(f"  Only {len(common)} aligned observations, skipping Method A")
    else:
        tau_a, p_a = stats.kendalltau(proxy_aligned.values, cycle_aligned.values)
        results['tau_A'] = tau_a
        results['p_A'] = p_a
        results['n_windows_A'] = len(common)
        results['proxy_df'] = proxy_df

        log(f"  Windows: {len(common)}")
        log(f"  Kendall τ(avg_corr, INDPRO_growth): {tau_a:.4f} (p = {p_a:.4f})")
        log(f"  (Negative τ = correlation rises in downturns = ρ rises = countercyclical)")

        if tau_a < 0 and p_a < 0.05:
            log(f"  => CONSISTENT: countercyclical (τ < 0, p < 0.05)")
        elif tau_a < 0 and p_a < 0.10:
            log(f"  => PARTIALLY CONSISTENT: τ < 0, p < 0.10")
        elif tau_a < 0:
            log(f"  => DIRECTIONAL: τ < 0 but not significant")
        else:
            log(f"  => INCONSISTENT: τ ≥ 0")

    # ── Method B: Rolling CES NLS estimation ──
    log(f"\n  --- Method B: Rolling CES NLS (120-month window) ---")

    if 'IPDMAN' in raw:
        dur_raw = {sid: raw[sid] for sid in DURABLE_SECTORS if sid in raw}
        ces_results = estimate_ces_rolling(raw['IPDMAN'], dur_raw, window=120, step=12)
        log(f"  CES windows estimated: {len(ces_results)}")

        if len(ces_results) >= 5:
            ces_df = pd.DataFrame(ces_results).set_index('center_date')
            results['ces_df'] = ces_df

            # Align with INDPRO cycle
            cycle_120 = growth['INDPRO'].rolling(120, min_periods=60).mean()
            common_b = ces_df.index.intersection(cycle_120.dropna().index)

            if len(common_b) >= 5:
                rho_aligned = ces_df.loc[common_b, 'rho']
                cycle_b = cycle_120.reindex(common_b)

                tau_b, p_b = stats.kendalltau(rho_aligned.values, cycle_b.values)
                results['tau_B'] = tau_b
                results['p_B'] = p_b
                results['n_windows_B'] = len(common_b)

                log(f"  Aligned windows: {len(common_b)}")
                log(f"  ρ̂ range: [{rho_aligned.min():.3f}, {rho_aligned.max():.3f}]")
                log(f"  Kendall τ(ρ̂, INDPRO_growth): {tau_b:.4f} (p = {p_b:.4f})")
                log(f"  (Negative τ = ρ rises when growth falls = countercyclical)")

                if tau_b < 0 and p_b < 0.05:
                    log(f"  => CONSISTENT: ρ countercyclical (τ < 0, p < 0.05)")
                elif tau_b < 0 and p_b < 0.10:
                    log(f"  => PARTIALLY CONSISTENT: τ < 0, p < 0.10")
                elif tau_b < 0:
                    log(f"  => DIRECTIONAL: τ < 0 but not significant")
                else:
                    log(f"  => INCONSISTENT: τ ≥ 0")
            else:
                log(f"  Only {len(common_b)} aligned windows, insufficient for Method B test")
        else:
            log(f"  Only {len(ces_results)} CES windows converged, insufficient")
    else:
        log("  IPDMAN not available, skipping Method B")

    log()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  6. SUMMARY AND COMPOSITE RESULTS
# ═══════════════════════════════════════════════════════════════════════════

def summarize_results(fdt, onsager, equicorr, cyclical):
    """Print composite summary of all four tests."""
    log("=" * 72)
    log("6. SUMMARY: FIRM-LEVEL PREDICTIONS")
    log("=" * 72)

    tests = [
        ("#4 FDT", fdt),
        ("#6 Equicorrelation", equicorr),
        ("#7 Onsager Reciprocity", onsager),
        ("#8 Cyclical ρ", cyclical),
    ]

    n_consistent = 0
    n_partial = 0
    n_tested = 0

    for name, res in tests:
        log(f"\n  {name}:")
        if name == "#4 FDT":
            r2 = res.get('R2', np.nan)
            T = res.get('T_estimate', np.nan)
            p = res.get('p_slope', np.nan)
            if not np.isnan(r2):
                n_tested += 1
                verdict = "CONSISTENT" if (r2 > 0.3 and T > 0 and p < 0.05) else \
                          "PARTIAL" if (T > 0 and p < 0.10) else "WEAK/INCONSISTENT"
                log(f"    R²={r2:.3f}, T̂={T:.4f}, p={p:.4f} → {verdict}")
                if verdict == "CONSISTENT":
                    n_consistent += 1
                elif verdict == "PARTIAL":
                    n_partial += 1

        elif name == "#6 Equicorrelation":
            cv = res.get('cv_corr', np.nan)
            frob = res.get('frob_ratio', np.nan)
            cv_pct = res.get('cv_percentile', np.nan)
            null_med = res.get('null_cv_median', np.nan)
            if not np.isnan(cv):
                n_tested += 1
                verdict = "CONSISTENT" if (cv < null_med and frob < 0.5) else \
                          "PARTIAL" if (cv < null_med) else "INCONSISTENT"
                log(f"    CV={cv:.3f} (null median={null_med:.3f}), "
                    f"Frobenius={frob:.3f} → {verdict}")
                if verdict == "CONSISTENT":
                    n_consistent += 1
                elif verdict == "PARTIAL":
                    n_partial += 1

        elif name == "#7 Onsager Reciprocity":
            r = res.get('r_pearson', np.nan)
            p = res.get('p_pearson', np.nan)
            if not np.isnan(r):
                n_tested += 1
                verdict = "CONSISTENT" if (r > 0.5 and p < 0.05) else \
                          "PARTIAL" if (r > 0.3 and p < 0.10) else "WEAK/INCONSISTENT"
                log(f"    r={r:.3f}, p={p:.4f} → {verdict}")
                if verdict == "CONSISTENT":
                    n_consistent += 1
                elif verdict == "PARTIAL":
                    n_partial += 1

        elif name == "#8 Cyclical ρ":
            tau_a = res.get('tau_A', np.nan)
            p_a = res.get('p_A', np.nan)
            tau_b = res.get('tau_B', np.nan)
            p_b = res.get('p_B', np.nan)
            if not np.isnan(tau_a):
                n_tested += 1
                # Both methods: negative τ = countercyclical ρ (rises in contractions)
                a_ok = tau_a < 0 and p_a < 0.10
                b_ok = not np.isnan(tau_b) and tau_b < 0 and p_b < 0.10
                if a_ok and b_ok:
                    verdict = "CONSISTENT"
                elif a_ok or b_ok:
                    verdict = "PARTIAL"
                else:
                    verdict = "WEAK/INCONSISTENT"
                log(f"    Method A: τ={tau_a:.3f} (p={p_a:.3f})")
                if not np.isnan(tau_b):
                    log(f"    Method B: τ={tau_b:.3f} (p={p_b:.3f})")
                log(f"    → {verdict}")
                if verdict == "CONSISTENT":
                    n_consistent += 1
                elif verdict == "PARTIAL":
                    n_partial += 1

    log(f"\n  Overall: {n_consistent} consistent, {n_partial} partial, "
        f"{n_tested - n_consistent - n_partial} weak/inconsistent out of {n_tested} tested")

    log()
    return n_consistent, n_partial, n_tested


# ═══════════════════════════════════════════════════════════════════════════
#  7. FOUR-PANEL FIGURE
# ═══════════════════════════════════════════════════════════════════════════

def make_figure(fdt, onsager, equicorr, cyclical, growth):
    """Generate 4-panel figure."""
    log("=" * 72)
    log("7. GENERATING FIGURE")
    log("=" * 72)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Firm-Level Predictions from CES Free Energy Framework\n"
                 "(Paper 16: Unified Theory)",
                 fontsize=13, fontweight="bold")

    # ── Panel (a): FDT scatter — σ² vs χ ──
    ax = axes[0, 0]
    sdf = fdt.get('sector_stats')
    if sdf is not None and len(sdf) > 0:
        ax.scatter(sdf['chi'], sdf['sigma2'], c='steelblue', s=50, zorder=3)

        # Label each point
        for _, row in sdf.iterrows():
            ax.annotate(row['label'], (row['chi'], row['sigma2']),
                       fontsize=5.5, alpha=0.8, ha='center', va='bottom',
                       xytext=(0, 4), textcoords='offset points')

        # Regression line
        r2 = fdt.get('R2', np.nan)
        T_est = fdt.get('T_estimate', np.nan)
        intercept = fdt.get('intercept', np.nan)
        p_slope = fdt.get('p_slope', np.nan)
        if not np.isnan(T_est):
            x_range = np.linspace(sdf['chi'].min() - 0.005, sdf['chi'].max() + 0.005, 50)
            ax.plot(x_range, intercept + T_est * x_range, 'r--', linewidth=1.5, alpha=0.7)
            ax.text(0.95, 0.95, f'R²={r2:.3f}\nT̂={T_est:.4f}\np={p_slope:.3f}',
                   transform=ax.transAxes, ha='right', va='top', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, "No FDT data", transform=ax.transAxes, ha='center')

    ax.set_xlabel('Susceptibility χ (cumulative IRF to INDPRO)')
    ax.set_ylabel('Fluctuation σ² (growth rate variance)')
    ax.set_title('(a) FDT: σ² vs χ across sectors')
    ax.grid(True, alpha=0.3)

    # ── Panel (b): Equicorrelation heatmap ──
    ax = axes[0, 1]
    corr_mat = equicorr.get('corr_matrix')
    sector_names = equicorr.get('sector_names', [])
    if corr_mat is not None:
        # Sort: durable first, then nondurable
        dur_in_leaf = [s for s in DURABLE_SECTORS if s in sector_names]
        nondur_in_leaf = [s for s in sector_names if s not in DURABLE_SECTORS]
        sorted_sectors = dur_in_leaf + nondur_in_leaf
        sorted_idx = [sector_names.index(s) for s in sorted_sectors]
        sorted_corr = corr_mat[np.ix_(sorted_idx, sorted_idx)]
        sorted_labels = [LABELS.get(s, s)[:10] for s in sorted_sectors]

        im = ax.imshow(sorted_corr, cmap='RdBu_r', vmin=-0.5, vmax=1.0, aspect='auto')
        ax.set_xticks(range(len(sorted_labels)))
        ax.set_yticks(range(len(sorted_labels)))
        ax.set_xticklabels(sorted_labels, rotation=90, fontsize=5)
        ax.set_yticklabels(sorted_labels, fontsize=5)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation')

        # Add divider between durable and nondurable
        n_dur = len(dur_in_leaf)
        ax.axhline(n_dur - 0.5, color='black', linewidth=1.5)
        ax.axvline(n_dur - 0.5, color='black', linewidth=1.5)

        cv = equicorr.get('cv_corr', np.nan)
        frob = equicorr.get('frob_ratio', np.nan)
        ax.set_title(f'(b) Equicorrelation (CV={cv:.2f}, Frob={frob:.2f})')
    else:
        ax.text(0.5, 0.5, "No correlation data", transform=ax.transAxes, ha='center')
        ax.set_title('(b) Equicorrelation')

    # ── Panel (c): Onsager scatter — L_ij vs L_ji ──
    ax = axes[1, 0]
    pairs_ij = onsager.get('pairs_ij')
    pairs_ji = onsager.get('pairs_ji')
    if pairs_ij is not None and pairs_ji is not None:
        ax.scatter(pairs_ij, pairs_ji, c='darkorange', s=30, alpha=0.7, zorder=3)

        # 45-degree line
        all_vals = np.concatenate([pairs_ij, pairs_ji])
        lo, hi = all_vals.min() - 0.01, all_vals.max() + 0.01
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.5, label='45° line')

        # Regression line
        slope = onsager.get('slope', np.nan)
        r_p = onsager.get('r_pearson', np.nan)
        p_p = onsager.get('p_pearson', np.nan)
        if not np.isnan(r_p):
            ax.text(0.05, 0.95, f'r={r_p:.3f}\np={p_p:.3f}\nslope={slope:.2f}',
                   transform=ax.transAxes, ha='left', va='top', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

        ax.legend(loc='lower right', fontsize=8)
    else:
        ax.text(0.5, 0.5, "No Onsager data", transform=ax.transAxes, ha='center')

    ax.set_xlabel(r'$L_{ij}$ (response of $i$ to shock $j$)')
    ax.set_ylabel(r'$L_{ji}$ (response of $j$ to shock $i$)')
    ax.set_title('(c) Onsager Reciprocity: IRF symmetry')
    ax.grid(True, alpha=0.3)

    # ── Panel (d): Cyclical ρ — dual axis with recession shading ──
    ax = axes[1, 1]
    ces_df = cyclical.get('ces_df')
    proxy_df = cyclical.get('proxy_df')

    # Use Method B (CES estimates) if available, otherwise Method A (correlation proxy)
    if ces_df is not None and len(ces_df) > 0:
        ax.plot(ces_df.index, ces_df['rho'], 'b-o', markersize=3, linewidth=1.2,
                label=r'$\hat{\rho}$ (CES NLS)', zorder=3)
        ax.set_ylabel(r'$\hat{\rho}$ (CES curvature)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        # Right axis: INDPRO growth (inverted)
        if 'INDPRO' in growth:
            ax2 = ax.twinx()
            cycle = growth['INDPRO'].rolling(120, min_periods=60).mean()
            cycle = cycle.dropna()
            ax2.plot(cycle.index, cycle.values, 'r-', linewidth=0.8, alpha=0.6,
                     label='INDPRO growth (120m)')
            ax2.set_ylabel('INDPRO growth rate (120m avg)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.invert_yaxis()

    elif proxy_df is not None and len(proxy_df) > 0:
        ax.plot(proxy_df.index, proxy_df['avg_corr'], 'b-', linewidth=1.2,
                label='Avg pairwise corr (60m)', zorder=3)
        ax.set_ylabel('Average pairwise correlation', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        if 'INDPRO' in growth:
            ax2 = ax.twinx()
            cycle = growth['INDPRO'].rolling(60, min_periods=30).mean()
            cycle = cycle.dropna()
            ax2.plot(cycle.index, cycle.values, 'r-', linewidth=0.8, alpha=0.6,
                     label='INDPRO growth (60m)')
            ax2.set_ylabel('INDPRO growth rate (60m avg)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No cyclical ρ data", transform=ax.transAxes, ha='center')

    # NBER recession shading
    for start, end in RECESSIONS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.15, color='gray', zorder=0)

    tau_a = cyclical.get('tau_A', np.nan)
    p_a = cyclical.get('p_A', np.nan)
    tau_b = cyclical.get('tau_B', np.nan)
    p_b = cyclical.get('p_B', np.nan)
    stat_text = ""
    if not np.isnan(tau_a):
        stat_text += f"τ_A={tau_a:.3f} (p={p_a:.3f})"
    if not np.isnan(tau_b):
        stat_text += f"\nτ_B={tau_b:.3f} (p={p_b:.3f})"
    if stat_text:
        ax.text(0.02, 0.02, stat_text, transform=ax.transAxes, fontsize=7,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5),
               va='bottom')

    ax.set_title(r'(d) Cyclical $\rho$: CES complementarity vs business cycle')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(10))

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  Saved: {FIG_PATH}")
    log()


# ═══════════════════════════════════════════════════════════════════════════
#  8. LATEX TABLE AND CSV OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def write_outputs(fdt, onsager, equicorr, cyclical):
    """Write LaTeX table and CSV results."""
    log("=" * 72)
    log("8. WRITING OUTPUTS")
    log("=" * 72)

    # ── LaTeX table ──
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Firm-level predictions from CES free energy framework}")
    lines.append(r"\label{tab:firm_predictions}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Prediction & Method & Statistic & Value & $p$-value & Verdict \\")
    lines.append(r"\midrule")

    # FDT row
    r2 = fdt.get('R2', np.nan)
    T_est = fdt.get('T_estimate', np.nan)
    p_fdt = fdt.get('p_slope', np.nan)
    if not np.isnan(r2):
        v = r"$\checkmark$" if (r2 > 0.3 and T_est > 0 and p_fdt < 0.05) else "---"
        lines.append(f"\\#4 FDT ($T = \\sigma^2/\\chi$) & bivariate VAR + OLS & "
                     f"$R^2$ & {r2:.3f} & {p_fdt:.3f} & {v} \\\\")

    lines.append(r"\midrule")

    # Equicorrelation row
    cv = equicorr.get('cv_corr', np.nan)
    frob = equicorr.get('frob_ratio', np.nan)
    cv_pct = equicorr.get('cv_percentile', np.nan)
    null_med = equicorr.get('null_cv_median', np.nan)
    if not np.isnan(cv):
        v = r"$\checkmark$" if (cv < null_med and frob < 0.5) else "---"
        lines.append(f"\\#6 Equicorrelation & correlation matrix & "
                     f"CV & {cv:.3f} & pctl={cv_pct:.0f}\\% & {v} \\\\")
        lines.append(f" & & Frobenius ratio & {frob:.3f} & --- & \\\\")

    lines.append(r"\midrule")

    # Onsager row
    r_ons = onsager.get('r_pearson', np.nan)
    p_ons = onsager.get('p_pearson', np.nan)
    slope_ons = onsager.get('slope', np.nan)
    if not np.isnan(r_ons):
        v = r"$\checkmark$" if (r_ons > 0.5 and p_ons < 0.05) else "---"
        lines.append(f"\\#7 Onsager ($L_{{ij}} \\approx L_{{ji}}$) & 8-var VAR IRF & "
                     f"Pearson $r$ & {r_ons:.3f} & {p_ons:.3f} & {v} \\\\")

    lines.append(r"\midrule")

    # Cyclical ρ rows
    tau_a = cyclical.get('tau_A', np.nan)
    p_a = cyclical.get('p_A', np.nan)
    tau_b = cyclical.get('tau_B', np.nan)
    p_b = cyclical.get('p_B', np.nan)
    if not np.isnan(tau_a):
        v = r"$\checkmark$" if (tau_a < 0 and p_a < 0.10) else "---"
        lines.append(f"\\#8 Cyclical $\\rho$ (A) & rolling correlation & "
                     f"Kendall $\\tau$ & {tau_a:.3f} & {p_a:.3f} & {v} \\\\")
    if not np.isnan(tau_b):
        v = r"$\checkmark$" if (tau_b < 0 and p_b < 0.10) else "---"
        lines.append(f"\\#8 Cyclical $\\rho$ (B) & rolling CES NLS & "
                     f"Kendall $\\tau$ & {tau_b:.3f} & {p_b:.3f} & {v} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{minipage}{0.95\textwidth}")
    lines.append(r"\vspace{0.5em}")
    lines.append(r"\footnotesize\textit{Notes:} FDT tests whether sector variance "
                 r"($\sigma^2$) and susceptibility ($\chi$) are linearly related. "
                 r"Equicorrelation tests whether off-diagonal correlations are uniform "
                 r"(CV and Frobenius distance from equicorrelation matrix; percentile "
                 r"against 1000 permutations). Onsager tests IRF matrix symmetry "
                 r"($L_{ij} \approx L_{ji}$) via 8-variable VAR on durable sectors. "
                 r"Cyclical $\rho$ tests countercyclicality of CES complementarity: "
                 r"Method A uses rolling pairwise correlation as $\rho$ proxy; "
                 r"Method B uses rolling CES NLS estimation. "
                 r"Data: FRED Industrial Production indices, monthly, 1972--present.")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")

    with open(RESULTS_TEX, "w") as f:
        f.write("\n".join(lines))
    log(f"  LaTeX table: {RESULTS_TEX}")

    # ── CSV ──
    rows = []

    # FDT
    rows.append({
        "test": "4_FDT", "method": "bivariate_VAR_OLS",
        "metric": "R2", "value": fdt.get('R2', np.nan),
        "p_value": fdt.get('p_slope', np.nan),
    })
    rows.append({
        "test": "4_FDT", "method": "bivariate_VAR_OLS",
        "metric": "T_estimate", "value": fdt.get('T_estimate', np.nan),
        "p_value": fdt.get('p_slope', np.nan),
    })

    # Equicorrelation
    rows.append({
        "test": "6_Equicorrelation", "method": "correlation_matrix",
        "metric": "CV", "value": equicorr.get('cv_corr', np.nan),
        "p_value": np.nan,
    })
    rows.append({
        "test": "6_Equicorrelation", "method": "correlation_matrix",
        "metric": "Frobenius_ratio", "value": equicorr.get('frob_ratio', np.nan),
        "p_value": np.nan,
    })
    rows.append({
        "test": "6_Equicorrelation", "method": "permutation_test",
        "metric": "CV_percentile", "value": equicorr.get('cv_percentile', np.nan),
        "p_value": np.nan,
    })

    # Onsager
    rows.append({
        "test": "7_Onsager", "method": "8var_VAR_IRF",
        "metric": "Pearson_r", "value": onsager.get('r_pearson', np.nan),
        "p_value": onsager.get('p_pearson', np.nan),
    })
    rows.append({
        "test": "7_Onsager", "method": "8var_VAR_IRF",
        "metric": "slope", "value": onsager.get('slope', np.nan),
        "p_value": np.nan,
    })

    # Cyclical ρ
    rows.append({
        "test": "8_Cyclical_rho", "method": "rolling_correlation",
        "metric": "Kendall_tau_A", "value": cyclical.get('tau_A', np.nan),
        "p_value": cyclical.get('p_A', np.nan),
    })
    rows.append({
        "test": "8_Cyclical_rho", "method": "rolling_CES_NLS",
        "metric": "Kendall_tau_B", "value": cyclical.get('tau_B', np.nan),
        "p_value": cyclical.get('p_B', np.nan),
    })

    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(RESULTS_CSV, index=False)
    log(f"  CSV results: {RESULTS_CSV}")

    log()


# ═══════════════════════════════════════════════════════════════════════════
#  9. MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 72)
    log("FIRM-LEVEL PREDICTIONS EMPIRICAL TEST")
    log("Paper 16 (Unified Theory)")
    log(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 72)
    log()
    log("Testing four predictions from the CES free energy framework:")
    log("  #4 FDT: T = σ²/χ (fluctuation-dissipation theorem)")
    log("  #6 Equicorrelation: off-diagonal correlations approximately equal")
    log("  #7 Onsager Reciprocity: cross-sector IRF symmetry (L_ij ≈ L_ji)")
    log("  #8 Cyclical ρ: CES complementarity countercyclical")
    log()

    # Load data
    raw, growth, leaf_df, dur_df = load_all_data()

    # Run four tests
    fdt_results = test_fdt(growth, leaf_df)
    onsager_results = test_onsager(dur_df)
    equicorr_results = test_equicorrelation(leaf_df)
    cyclical_results = test_cyclical_rho(raw, growth, dur_df)

    # Summary
    summarize_results(fdt_results, onsager_results, equicorr_results, cyclical_results)

    # Figure
    make_figure(fdt_results, onsager_results, equicorr_results, cyclical_results, growth)

    # Outputs
    write_outputs(fdt_results, onsager_results, equicorr_results, cyclical_results)

    # Save results text
    with open(RESULTS_TXT, "w") as f:
        f.write(results_buf.getvalue())
    log(f"Full results saved to: {RESULTS_TXT}")

    log()
    log("=" * 72)
    log("DONE")
    log("=" * 72)


if __name__ == "__main__":
    main()
