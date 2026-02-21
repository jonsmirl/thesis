#!/usr/bin/env python3
"""
CES Emergence Under Aggregation — Empirical Test
=================================================

Tests the theoretical prediction (Paper 17) that CES is the unique
RG fixed point of hierarchical aggregation. Non-CES (translog interaction)
terms should vanish at higher aggregation depths.

Uses the NAICS hierarchy of FRED Industrial Production indices to provide
6+ aggregation events across 4 depth levels, with three complementary tests:

  A. Translog interaction decay (F-test for β=0)
  B. Direct CES estimation via NLS
  C. Model comparison (AIC/BIC)

Key predictions (depth = tree depth from root; agg_levels = max_depth - depth):
  1. interaction_ratio decreases with agg_levels (Kendall τ < 0)
     Equivalently, interaction_ratio is lowest at the root (most aggregated)
  2. F-test p-value for β=0 increases with agg_levels
  3. Estimated ρ stabilizes across depths (converges to fixed point)
  4. BIC favors CES over translog at higher agg_levels

Usage:
  source .venv/bin/activate && source env.sh
  python scripts/test_ces_emergence.py
"""

import os
import sys
import time
import warnings
import requests
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.stats import kendalltau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

FRED_API_KEY = os.environ.get('FRED_API_KEY')
if not FRED_API_KEY:
    print("Error: FRED_API_KEY environment variable not set.")
    print("Run: source env.sh")
    sys.exit(1)

START_DATE = '1972-01-01'
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'thesis_data', 'fred_cache')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'thesis_data')
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

np.random.seed(42)

# NAICS hierarchy of FRED IP series
# Each key is a parent series, value is dict of child series
# Leaf nodes have empty dict as value
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

# Human-readable labels
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
    'IPG334S': 'Computer/Electronics',
    'IPG335S': 'Electrical Equipment',
    'IPG336S': 'Transport Equipment',
    'IPG337S': 'Furniture',
    'IPG339S': 'Misc Manufacturing',
    'IPG3341S': 'Computers',
    'IPG3342S': 'Communications Equip',
    'IPG3343S': 'Audio/Video',
    'IPG3344S': 'Semiconductors',
    'IPG3345S': 'Electronic Instruments',
    'IPG3361T3S': 'Motor Vehicles',
    'IPG3364T9S': 'Aerospace/Other',
    'IPG311A2S': 'Food/Beverage/Tobacco',
    'IPG313A4S': 'Textiles',
    'IPG315A6S': 'Apparel/Leather',
    'IPG321S': 'Wood Products',
    'IPG322S': 'Paper',
    'IPG323S': 'Printing',
    'IPG324S': 'Petroleum/Coal',
    'IPG325S': 'Chemicals',
    'IPG326S': 'Plastics/Rubber',
}


# ---------------------------------------------------------------------------
# 2. Data fetching with caching
# ---------------------------------------------------------------------------

def fetch_and_cache(series_id):
    """Fetch FRED series with local CSV caching."""
    cache_path = os.path.join(CACHE_DIR, f'{series_id}.csv')

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
        print(f"  WARNING: Failed to fetch {series_id}: {e}")
        return None

    dates, values = [], []
    for obs in data.get('observations', []):
        if obs['value'] != '.':
            dates.append(obs['date'])
            values.append(float(obs['value']))

    if not dates:
        print(f"  WARNING: No data for {series_id}")
        return None

    s = pd.Series(values, index=pd.to_datetime(dates), name=series_id)
    s.to_csv(cache_path)
    time.sleep(0.15)  # rate limit
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


def fetch_all_data(hierarchy):
    """Fetch all series and return as dict of {series_id: pd.Series}."""
    all_ids = collect_all_series(hierarchy)
    print(f"Fetching {len(all_ids)} FRED IP series...")

    data = {}
    for sid in sorted(all_ids):
        s = fetch_and_cache(sid)
        if s is not None and len(s) > 0:
            data[sid] = s
            print(f"  {sid}: {len(s)} obs ({s.index[0].year}-{s.index[-1].year})")
        else:
            print(f"  {sid}: MISSING")

    return data


# ---------------------------------------------------------------------------
# 3. Test A: Translog interaction test
# ---------------------------------------------------------------------------

def test_translog(y, X_dict):
    """
    Fit translog model and test whether interaction terms are zero.

    ln Y = c + Σ αⱼ ln Xⱼ + Σᵢ≤ⱼ βᵢⱼ ln Xᵢ ln Xⱼ + γ·t + ε

    Returns dict with F-stat, p-value, interaction_ratio, R² improvement.
    """
    names = sorted(X_dict.keys())
    K = len(names)

    # Build dataframe with aligned, positive data
    df = pd.DataFrame({'y': y})
    for n in names:
        df[n] = X_dict[n]
    df = df.dropna()
    df = df[(df > 0).all(axis=1)]

    if len(df) < 2 * (1 + K + K * (K + 1) // 2 + 1):
        return None  # not enough observations

    # Log transform
    log_y = np.log(df['y'].values)
    log_X = np.log(df[names].values)
    T = len(df)
    trend = np.arange(T, dtype=float) / T

    # Restricted model: Cobb-Douglas (no interactions)
    Z_r = np.column_stack([np.ones(T), log_X, trend.reshape(-1, 1)])
    try:
        res_r = sm.OLS(log_y, Z_r).fit()
    except Exception:
        return None

    # Unrestricted model: Translog (with interactions)
    interaction_cols = []
    interaction_names = []
    for i in range(K):
        for j in range(i, K):
            interaction_cols.append(log_X[:, i] * log_X[:, j])
            interaction_names.append(f'{names[i]}x{names[j]}')

    interactions = np.column_stack(interaction_cols) if interaction_cols else np.empty((T, 0))
    Z_u = np.column_stack([np.ones(T), log_X, interactions, trend.reshape(-1, 1)])

    try:
        res_u = sm.OLS(log_y, Z_u).fit()
    except Exception:
        return None

    # F-test for H₀: all βᵢⱼ = 0
    n_restrictions = len(interaction_cols)
    if n_restrictions == 0:
        return None

    RSS_r = res_r.ssr
    RSS_u = res_u.ssr
    df_u = T - Z_u.shape[1]

    if RSS_u <= 0 or df_u <= 0:
        return None

    F_stat = ((RSS_r - RSS_u) / n_restrictions) / (RSS_u / df_u)
    from scipy.stats import f as f_dist
    p_value = 1 - f_dist.cdf(F_stat, n_restrictions, df_u)

    # Interaction ratio: ||β|| / ||α||
    # α coefficients are indices 1..K, β coefficients are K+1..K+n_restrictions
    alpha_coefs = res_u.params[1:K + 1]
    beta_coefs = res_u.params[K + 1:K + 1 + n_restrictions]

    norm_alpha = np.linalg.norm(alpha_coefs)
    norm_beta = np.linalg.norm(beta_coefs)

    interaction_ratio = norm_beta / norm_alpha if norm_alpha > 1e-10 else np.nan

    # R² improvement
    r2_cd = res_r.rsquared
    r2_translog = res_u.rsquared

    return {
        'F_stat': F_stat,
        'p_value': p_value,
        'interaction_ratio': interaction_ratio,
        'norm_alpha': norm_alpha,
        'norm_beta': norm_beta,
        'R2_cd': r2_cd,
        'R2_translog': r2_translog,
        'R2_improvement': r2_translog - r2_cd,
        'n_obs': T,
        'n_inputs': K,
        'n_interactions': n_restrictions,
    }


# ---------------------------------------------------------------------------
# 4. Test B: CES NLS estimation
# ---------------------------------------------------------------------------

def log_ces_function(X_matrix, log_A, rho, *log_weights):
    """Log CES aggregation: log Y = log A + (1/ρ) log(Σ wⱼ Xⱼ^ρ)

    Parameters are in log space for numerical stability:
    - log_A: log of scale parameter
    - rho: CES curvature parameter
    - log_weights: log of (unnormalized) weights → softmax normalization
    """
    K = X_matrix.shape[1]
    lw = np.array(log_weights[:K])
    w = np.exp(lw - lw.max())  # softmax for numerical stability
    w = w / w.sum()

    rho_safe = np.clip(rho, -2.0, 2.0)
    if abs(rho_safe) < 0.01:
        # Near Cobb-Douglas limit: geometric mean
        return log_A + np.sum(w[np.newaxis, :] * np.log(np.maximum(X_matrix, 1e-10)), axis=1)

    inner = np.sum(w[np.newaxis, :] * np.power(np.maximum(X_matrix, 1e-10), rho_safe), axis=1)
    inner = np.maximum(inner, 1e-10)
    return log_A + (1.0 / rho_safe) * np.log(inner)


def estimate_ces(y, X_dict):
    """
    Estimate CES parameters via NLS in log space.

    Fits: log Y = log A + (1/ρ) log(Σ wⱼ Xⱼ^ρ) + ε
    This makes RSS directly comparable to OLS on log Y (CD, translog).

    Returns dict with rho, R2, confidence intervals, A, weights, RSS.
    """
    names = sorted(X_dict.keys())
    K = len(names)

    df = pd.DataFrame({'y': y})
    for n in names:
        df[n] = X_dict[n]
    df = df.dropna()
    df = df[(df > 0).all(axis=1)]

    if len(df) < K + 3:
        return None

    y_vals = df['y'].values
    X_matrix = df[names].values
    log_y = np.log(y_vals)
    T = len(df)

    # Normalize X for numerical stability
    X_scales = X_matrix.mean(axis=0)
    X_norm = X_matrix / X_scales[np.newaxis, :]

    def model(X, *params):
        return log_ces_function(X, params[0], params[1], *params[2:])

    best_result = None
    best_cost = np.inf

    # Try multiple starting points for rho
    for rho_init in [0.5, 0.0, -0.5, 0.8, -1.0, 1.5]:
        p0_trial = [np.mean(log_y), rho_init] + [0.0] * K  # log-weights start at 0 (equal)
        bounds_low = [-10.0, -1.99] + [-10.0] * K
        bounds_high = [20.0, 0.99] + [10.0] * K
        try:
            popt, pcov = curve_fit(
                model, X_norm, log_y,
                p0=p0_trial, bounds=(bounds_low, bounds_high),
                maxfev=20000, method='trf'
            )
            residuals = log_y - model(X_norm, *popt)
            cost = np.sum(residuals ** 2)
            if cost < best_cost:
                best_cost = cost
                best_result = (popt, pcov)
        except Exception:
            continue

    if best_result is None:
        return None

    popt, pcov = best_result
    log_A_est = popt[0]
    rho_est = popt[1]
    lw = np.array(popt[2:K + 2])
    w_est = np.exp(lw - lw.max())
    w_est = w_est / w_est.sum()

    # R² in log space
    log_y_pred = model(X_norm, *popt)
    ss_res = np.sum((log_y - log_y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Confidence interval for rho
    try:
        se = np.sqrt(np.diag(pcov))
        rho_se = se[1]
        rho_ci = (rho_est - 1.96 * rho_se, rho_est + 1.96 * rho_se)
    except Exception:
        rho_se = np.nan
        rho_ci = (np.nan, np.nan)

    sigma_est = 1.0 / (1.0 - rho_est) if abs(1.0 - rho_est) > 0.01 else np.inf

    return {
        'rho': rho_est,
        'sigma': sigma_est,
        'rho_se': rho_se,
        'rho_ci_low': rho_ci[0],
        'rho_ci_high': rho_ci[1],
        'A': np.exp(log_A_est),
        'weights': dict(zip(names, w_est)),
        'R2_ces': R2,
        'RSS_ces': ss_res,
        'n_obs': T,
        'n_params': K + 2,
    }


# ---------------------------------------------------------------------------
# 5. Test C: Model comparison (AIC/BIC)
# ---------------------------------------------------------------------------

def compare_models(y, X_dict):
    """
    Compare CES, Cobb-Douglas, and Translog via AIC/BIC.

    All models estimated in log space for comparable likelihoods:
    - CD:       log Y = c + Σ αⱼ log Xⱼ + γt             (OLS)
    - Translog: log Y = c + Σ αⱼ log Xⱼ + Σ βᵢⱼ ... + γt (OLS)
    - CES:      log Y = log A + (1/ρ) log(Σ wⱼ Xⱼ^ρ)     (NLS)

    BIC = T·log(RSS/T) + k·log(T) for all three.
    """
    names = sorted(X_dict.keys())
    K = len(names)

    df = pd.DataFrame({'y': y})
    for n in names:
        df[n] = X_dict[n]
    df = df.dropna()
    df = df[(df > 0).all(axis=1)]

    T = len(df)
    if T < 2 * (1 + K + K * (K + 1) // 2 + 1):
        return None

    log_y = np.log(df['y'].values)
    log_X = np.log(df[names].values)
    trend = np.arange(T, dtype=float) / T

    def compute_bic_aic(rss, k, T):
        """BIC and AIC from RSS assuming normal errors in log space."""
        sigma2 = rss / T
        if sigma2 <= 0:
            return np.nan, np.nan
        ll = -T / 2 * (np.log(2 * np.pi * sigma2) + 1)
        aic = -2 * ll + 2 * k
        bic = -2 * ll + np.log(T) * k
        return aic, bic

    # --- Cobb-Douglas: ln Y = c + Σ αⱼ ln Xⱼ + γ·t ---
    Z_cd = np.column_stack([np.ones(T), log_X, trend.reshape(-1, 1)])
    k_cd = Z_cd.shape[1]
    try:
        res_cd = sm.OLS(log_y, Z_cd).fit()
        rss_cd = res_cd.ssr
        r2_cd = res_cd.rsquared
        aic_cd, bic_cd = compute_bic_aic(rss_cd, k_cd, T)
    except Exception:
        return None

    # --- Translog: add interaction terms ---
    interaction_cols = []
    for i in range(K):
        for j in range(i, K):
            interaction_cols.append(log_X[:, i] * log_X[:, j])
    interactions = np.column_stack(interaction_cols) if interaction_cols else np.empty((T, 0))
    Z_tl = np.column_stack([np.ones(T), log_X, interactions, trend.reshape(-1, 1)])
    k_tl = Z_tl.shape[1]
    try:
        res_tl = sm.OLS(log_y, Z_tl).fit()
        rss_tl = res_tl.ssr
        r2_tl = res_tl.rsquared
        aic_tl, bic_tl = compute_bic_aic(rss_tl, k_tl, T)
    except Exception:
        return None

    # --- CES: already estimated in log space by estimate_ces ---
    ces_result = estimate_ces(df['y'], {n: df[n] for n in names})
    if ces_result is not None and 'RSS_ces' in ces_result:
        rss_ces = ces_result['RSS_ces']
        k_ces = ces_result['n_params']
        r2_ces = ces_result['R2_ces']
        aic_ces, bic_ces = compute_bic_aic(rss_ces, k_ces, T)
    else:
        aic_ces = np.nan
        bic_ces = np.nan
        r2_ces = np.nan
        k_ces = K + 2

    return {
        'AIC_cd': aic_cd,
        'BIC_cd': bic_cd,
        'R2_cd': r2_cd,
        'k_cd': k_cd,
        'AIC_translog': aic_tl,
        'BIC_translog': bic_tl,
        'R2_translog': r2_tl,
        'k_translog': k_tl,
        'AIC_ces': aic_ces,
        'BIC_ces': bic_ces,
        'R2_ces': r2_ces,
        'k_ces': k_ces,
        'BIC_ces_minus_translog': bic_ces - bic_tl if not np.isnan(bic_ces) else np.nan,
        'BIC_ces_minus_cd': bic_ces - bic_cd if not np.isnan(bic_ces) else np.nan,
        'n_obs': T,
    }


# ---------------------------------------------------------------------------
# 6. Walk the hierarchy
# ---------------------------------------------------------------------------

def compute_depth(hierarchy, target, current_depth=0):
    """Find the depth of a parent node in the hierarchy."""
    for parent, children in hierarchy.items():
        if parent == target:
            return current_depth
        if children:
            d = compute_depth(children, target, current_depth + 1)
            if d is not None:
                return d
    return None


def find_aggregation_events(hierarchy, data, depth=0):
    """
    Walk hierarchy and identify aggregation events.

    An aggregation event is a parent with 2+ children that all have data.
    Returns list of dicts with parent, children, depth.
    """
    events = []
    for parent, children in hierarchy.items():
        if children and parent in data:
            # Check which children have data
            available = {c: data[c] for c in children if c in data}
            if len(available) >= 2:
                events.append({
                    'parent': parent,
                    'children': available,
                    'depth': depth,
                    'n_children': len(available),
                    'label': LABELS.get(parent, parent),
                })
            # Recurse into children
            for child, grandchildren in children.items():
                if grandchildren:
                    events.extend(
                        find_aggregation_events({child: grandchildren}, data, depth + 1)
                    )
    return events


def align_series(parent_series, child_dict):
    """Align parent and child series to common date range."""
    df = pd.DataFrame({'parent': parent_series})
    for name, series in child_dict.items():
        df[name] = series
    df = df.dropna()
    df = df[(df > 0).all(axis=1)]
    y = df['parent']
    X = {c: df[c] for c in child_dict if c in df.columns}
    return y, X


# ---------------------------------------------------------------------------
# 7. Main analysis
# ---------------------------------------------------------------------------

def run_analysis():
    """Run all three tests at every aggregation event."""

    # Fetch data
    data = fetch_all_data(HIERARCHY)
    print(f"\nSuccessfully loaded {len(data)} series.\n")

    # Find aggregation events
    events = find_aggregation_events(HIERARCHY, data)
    print(f"Found {len(events)} aggregation events:\n")

    for ev in events:
        children_str = ', '.join(LABELS.get(c, c) for c in ev['children'])
        print(f"  Depth {ev['depth']}: {ev['label']} = f({children_str})")

    print("\n" + "=" * 80)
    print("RUNNING TESTS")
    print("=" * 80)

    # Find max depth for computing agg_levels
    max_depth = max(ev['depth'] for ev in events)

    results = []

    for ev in events:
        parent_id = ev['parent']
        label = ev['label']
        depth = ev['depth']
        agg_levels = max_depth - depth  # how many aggregation levels applied

        print(f"\n--- {label} (depth={depth}, agg_levels={agg_levels}, {ev['n_children']} inputs) ---")

        y, X = align_series(data[parent_id], ev['children'])
        if len(y) < 30:
            print(f"  Skipping: only {len(y)} observations after alignment")
            continue

        row = {
            'parent': parent_id,
            'label': label,
            'depth': depth,
            'agg_levels': agg_levels,
            'n_inputs': len(X),
            'n_obs': len(y),
        }

        # Test A: Translog
        tl = test_translog(y, X)
        if tl:
            row.update({
                'interaction_ratio': tl['interaction_ratio'],
                'F_stat': tl['F_stat'],
                'F_pvalue': tl['p_value'],
                'R2_cd': tl['R2_cd'],
                'R2_translog': tl['R2_translog'],
                'R2_improvement': tl['R2_improvement'],
            })
            sig = "***" if tl['p_value'] < 0.001 else "**" if tl['p_value'] < 0.01 else "*" if tl['p_value'] < 0.05 else "ns"
            print(f"  Test A: interaction_ratio={tl['interaction_ratio']:.4f}, "
                  f"F={tl['F_stat']:.2f}, p={tl['p_value']:.4f} {sig}")
            print(f"          R²(CD)={tl['R2_cd']:.6f}, R²(TL)={tl['R2_translog']:.6f}, "
                  f"ΔR²={tl['R2_improvement']:.6f}")
        else:
            print("  Test A: FAILED (insufficient data)")

        # Test B: CES NLS
        ces = estimate_ces(y, X)
        if ces:
            row.update({
                'rho': ces['rho'],
                'sigma': ces['sigma'],
                'rho_se': ces['rho_se'],
                'rho_ci_low': ces['rho_ci_low'],
                'rho_ci_high': ces['rho_ci_high'],
                'R2_ces': ces['R2_ces'],
            })
            print(f"  Test B: ρ={ces['rho']:.4f} (σ={ces['sigma']:.2f}), "
                  f"95% CI=[{ces['rho_ci_low']:.3f}, {ces['rho_ci_high']:.3f}], "
                  f"R²={ces['R2_ces']:.6f}")
            # Print top weights
            sorted_w = sorted(ces['weights'].items(), key=lambda x: -x[1])
            top3 = ', '.join(f"{LABELS.get(k, k)}={v:.3f}" for k, v in sorted_w[:3])
            print(f"          Top weights: {top3}")
        else:
            print("  Test B: FAILED (CES estimation did not converge)")

        # Test C: Model comparison
        mc = compare_models(y, X)
        if mc:
            row.update({
                'AIC_cd': mc['AIC_cd'],
                'BIC_cd': mc['BIC_cd'],
                'AIC_translog': mc['AIC_translog'],
                'BIC_translog': mc['BIC_translog'],
                'AIC_ces': mc['AIC_ces'],
                'BIC_ces': mc['BIC_ces'],
                'BIC_ces_minus_translog': mc['BIC_ces_minus_translog'],
                'BIC_ces_minus_cd': mc.get('BIC_ces_minus_cd', np.nan),
            })
            # Parameter-count-adjusted normalization
            # Extra translog params beyond CES = K(K+1)/2 interaction terms
            K = len(X)
            n_extra = K * (K + 1) // 2
            row['n_extra_translog_params'] = n_extra
            dbic = mc['BIC_ces_minus_translog']
            row['BIC_per_extra_param'] = dbic / n_extra if (not np.isnan(dbic) and n_extra > 0) else np.nan
            # Identify BIC winner
            bics = {'CD': mc['BIC_cd'], 'Translog': mc['BIC_translog']}
            if not np.isnan(mc['BIC_ces']):
                bics['CES'] = mc['BIC_ces']
            winner = min(bics, key=bics.get)
            print(f"  Test C: BIC winner={winner}")
            print(f"          BIC(CD)={mc['BIC_cd']:.1f}, BIC(TL)={mc['BIC_translog']:.1f}, "
                  f"BIC(CES)={mc['BIC_ces']:.1f}")
            if not np.isnan(dbic):
                sign = "+" if dbic > 0 else ""
                print(f"          BIC(CES−TL)={sign}{dbic:.1f}  "
                      f"(normalized: {sign}{dbic/n_extra:.1f} per extra param, "
                      f"Δk={n_extra})")
        else:
            print("  Test C: FAILED (model comparison failed)")

        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 8. Summary statistics and hypothesis tests
# ---------------------------------------------------------------------------

def summarize_results(df):
    """Compute summary statistics and formal hypothesis tests.

    Uses agg_levels = max_depth - tree_depth, so:
    - agg_levels=3: root (INDPRO) — most aggregated, should be closest to CES fixed point
    - agg_levels=0: subsectors — least aggregated, farthest from fixed point

    RG prediction: CES emerges under repeated aggregation → interaction terms
    should DECREASE with agg_levels (τ < 0 for interaction_ratio vs agg_levels).
    """
    print("\n" + "=" * 80)
    print("SUMMARY: PREDICTIONS vs RESULTS")
    print("=" * 80)
    print("\nConvention: agg_levels = number of aggregation levels applied")
    print("  agg_levels=3 → root (INDPRO, most aggregated, RG fixed point)")
    print("  agg_levels=0 → subsectors (least aggregated)")

    # Prediction 1: interaction_ratio decreases with agg_levels
    valid = df.dropna(subset=['interaction_ratio', 'agg_levels'])
    if len(valid) >= 3:
        tau, p_tau = kendalltau(valid['agg_levels'], valid['interaction_ratio'])
        print(f"\n1. Interaction ratio vs aggregation levels:")
        print(f"   Kendall τ = {tau:.4f}, p = {p_tau:.4f}")
        print(f"   RG prediction: τ < 0 (interactions vanish at fixed point)")
        result = "CONSISTENT" if tau < 0 else "INCONSISTENT"
        sig = f" (p={p_tau:.4f})" if p_tau < 0.1 else " (not significant)"
        print(f"   Result: {result}{sig}")

        for d in sorted(valid['agg_levels'].unique()):
            subset = valid[valid['agg_levels'] == d]
            mean_ir = subset['interaction_ratio'].mean()
            std_ir = subset['interaction_ratio'].std() if len(subset) > 1 else 0
            labels = ', '.join(subset['label'].values)
            print(f"   agg_levels={d}: mean={mean_ir:.4f} ± {std_ir:.4f} (n={len(subset)}) [{labels}]")
    else:
        print("\n1. Insufficient data for Kendall τ test")

    # Prediction 2: R² improvement from translog decreases with agg_levels
    valid_r2 = df.dropna(subset=['R2_improvement', 'agg_levels'])
    if len(valid_r2) >= 3:
        tau_r2, p_tau_r2 = kendalltau(valid_r2['agg_levels'], valid_r2['R2_improvement'])
        print(f"\n2. R² improvement (translog over CD) vs aggregation levels:")
        print(f"   Kendall τ = {tau_r2:.4f}, p = {p_tau_r2:.4f}")
        print(f"   RG prediction: τ < 0 (translog adds less at higher aggregation)")
        result = "CONSISTENT" if tau_r2 < 0 else "INCONSISTENT"
        print(f"   Result: {result}")

        for d in sorted(valid_r2['agg_levels'].unique()):
            subset = valid_r2[valid_r2['agg_levels'] == d]
            mean_imp = subset['R2_improvement'].mean()
            print(f"   agg_levels={d}: mean ΔR²={mean_imp:.6f}")
    else:
        print("\n2. Insufficient data for R² improvement trend")

    # Prediction 3: ρ stabilizes across depths
    valid_rho = df.dropna(subset=['rho', 'agg_levels'])
    if len(valid_rho) >= 3:
        print(f"\n3. CES ρ estimates across aggregation levels:")
        for d in sorted(valid_rho['agg_levels'].unique()):
            subset = valid_rho[valid_rho['agg_levels'] == d]
            mean_rho = subset['rho'].mean()
            std_rho = subset['rho'].std() if len(subset) > 1 else 0
            labels = ', '.join(subset['label'].values)
            print(f"   agg_levels={d}: ρ = {mean_rho:.4f} ± {std_rho:.4f} (n={len(subset)}) [{labels}]")
        print(f"   Overall: ρ = {valid_rho['rho'].mean():.4f} ± {valid_rho['rho'].std():.4f}")
        print(f"   RG prediction: ρ converges (variance shrinks at high agg_levels)")
    else:
        print("\n3. Insufficient data for ρ stability analysis")

    # Prediction 4: BIC(CES−Translog) decreases with agg_levels
    valid_bic = df.dropna(subset=['BIC_ces_minus_translog', 'agg_levels'])
    if len(valid_bic) >= 3:
        tau_bic, p_tau_bic = kendalltau(valid_bic['agg_levels'], valid_bic['BIC_ces_minus_translog'])
        print(f"\n4a. BIC(CES−Translog) vs aggregation levels (raw):")
        print(f"    Kendall τ = {tau_bic:.4f}, p = {p_tau_bic:.4f}")
        print(f"    RG prediction: τ < 0 (CES increasingly favored at fixed point)")
        result = "CONSISTENT" if tau_bic < 0 else "INCONSISTENT"
        print(f"    Result: {result}")

        for d in sorted(valid_bic['agg_levels'].unique()):
            subset = valid_bic[valid_bic['agg_levels'] == d]
            mean_diff = subset['BIC_ces_minus_translog'].mean()
            n_favor_ces = (subset['BIC_ces_minus_translog'] < 0).sum()
            print(f"    agg_levels={d}: mean ΔBIC={mean_diff:.1f}, CES wins {n_favor_ces}/{len(subset)}")
    else:
        print("\n4a. Insufficient data for BIC trend")

    # Prediction 4b: Normalized ΔBIC per extra translog parameter
    valid_norm = df.dropna(subset=['BIC_per_extra_param', 'agg_levels'])
    if len(valid_norm) >= 3:
        tau_norm, p_tau_norm = kendalltau(valid_norm['agg_levels'], valid_norm['BIC_per_extra_param'])
        print(f"\n4b. ΔBIC per extra translog parameter vs aggregation levels:")
        print(f"    Kendall τ = {tau_norm:.4f}, p = {p_tau_norm:.4f}")
        print(f"    Normalizes for translog's parameter advantage at high K")
        print(f"    (translog gets K(K+1)/2 extra params; CES gets only ρ)")
        result = "CONSISTENT" if tau_norm < 0 else "INCONSISTENT"
        print(f"    Result: {result}")

        for d in sorted(valid_norm['agg_levels'].unique()):
            subset = valid_norm[valid_norm['agg_levels'] == d]
            mean_norm = subset['BIC_per_extra_param'].mean()
            labels = ', '.join(f"{r['label']}({int(r['n_extra_translog_params'])})" for _, r in subset.iterrows())
            print(f"    agg_levels={d}: mean={mean_norm:.1f}/param [{labels}]")
    else:
        print("\n4b. Insufficient data for normalized BIC trend")

    # Prediction 5: CES beats Cobb-Douglas everywhere (CES is a better model)
    valid_cd = df.dropna(subset=['BIC_ces_minus_cd', 'agg_levels'])
    if len(valid_cd) >= 1:
        n_ces_beats_cd = (valid_cd['BIC_ces_minus_cd'] < 0).sum()
        print(f"\n5. CES vs Cobb-Douglas (BIC):")
        print(f"   CES wins: {n_ces_beats_cd}/{len(valid_cd)} events")
        print(f"   RG prediction: CES should outperform CD everywhere (ρ ≠ 0)")
        for _, row in valid_cd.iterrows():
            winner = "CES" if row['BIC_ces_minus_cd'] < 0 else "CD"
            print(f"   {row['label']}: ΔBIC={row['BIC_ces_minus_cd']:.1f} → {winner}")


# ---------------------------------------------------------------------------
# 9. Output: CSV, LaTeX, Figure
# ---------------------------------------------------------------------------

def save_outputs(df):
    """Save CSV, LaTeX table, and 4-panel figure."""

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, 'ces_emergence_results.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\nSaved: {csv_path}")

    # LaTeX table
    tex_path = os.path.join(OUTPUT_DIR, 'ces_emergence_table.tex')
    write_latex_table(df, tex_path)
    print(f"Saved: {tex_path}")

    # Figure
    fig_path = os.path.join(FIGURE_DIR, 'ces_emergence.pdf')
    plot_results(df, fig_path)
    print(f"Saved: {fig_path}")


def write_latex_table(df, path):
    """Write publication-ready LaTeX table."""
    cols = ['label', 'agg_levels', 'n_inputs', 'interaction_ratio', 'F_pvalue',
            'rho', 'rho_ci_low', 'rho_ci_high', 'R2_ces', 'BIC_per_extra_param']

    available = [c for c in cols if c in df.columns]
    sub = df[available].copy()

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{CES emergence under aggregation: FRED IP hierarchy}')
    lines.append(r'\label{tab:ces_emergence}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{llccccccc}')
    lines.append(r'\toprule')
    lines.append(r'Aggregate & Agg. & $K$ & $\|\beta\|/\|\alpha\|$ & $F$-test $p$ '
                 r'& $\hat\rho$ & 95\% CI & $R^2_{\text{CES}}$ & $\Delta$BIC/$\Delta k$ \\')
    lines.append(r'\midrule')

    for _, row in sub.iterrows():
        label = row.get('label', '')
        depth = int(row.get('agg_levels', 0))
        K = int(row.get('n_inputs', 0))
        ir = f"{row['interaction_ratio']:.4f}" if pd.notna(row.get('interaction_ratio')) else '--'
        fp = f"{row['F_pvalue']:.4f}" if pd.notna(row.get('F_pvalue')) else '--'
        rho = f"{row['rho']:.3f}" if pd.notna(row.get('rho')) else '--'
        ci_lo = row.get('rho_ci_low', np.nan)
        ci_hi = row.get('rho_ci_high', np.nan)
        ci = f"[{ci_lo:.2f}, {ci_hi:.2f}]" if pd.notna(ci_lo) and pd.notna(ci_hi) else '--'
        r2 = f"{row['R2_ces']:.4f}" if pd.notna(row.get('R2_ces')) else '--'
        dbic_norm = row.get('BIC_per_extra_param', np.nan)
        dbic_str = f"{dbic_norm:.1f}" if pd.notna(dbic_norm) else '--'

        lines.append(f'{label} & {depth} & {K} & {ir} & {fp} & {rho} & {ci} & {r2} & {dbic_str} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\begin{minipage}{0.95\textwidth}')
    lines.append(r'\vspace{0.5em}')
    lines.append(r'\footnotesize\textit{Notes:} $\|\beta\|/\|\alpha\|$ is the ratio of translog '
                 r'interaction coefficients to first-order coefficients. $F$-test $p$ tests '
                 r'$H_0$: all interaction terms zero (CES restriction). $\hat\rho$ is the CES '
                 r'curvature parameter estimated via NLS. $\Delta$BIC/$\Delta k$ = '
                 r'[BIC(CES) $-$ BIC(Translog)] / (extra translog parameters); '
                 r'normalizes for translog parameter advantage at high $K$. '
                 r'Negative values favor CES. Data: FRED Industrial Production indices, monthly.')
    lines.append(r'\end{minipage}')
    lines.append(r'\end{table}')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def plot_results(df, path):
    """Create 4-panel figure."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('CES Emergence Under Aggregation: FRED IP Hierarchy', fontsize=13, fontweight='bold')

    depth_jitter = 0.05

    # (a) Interaction ratio vs agg_levels
    ax = axes[0, 0]
    valid = df.dropna(subset=['interaction_ratio', 'agg_levels'])
    if len(valid) > 0:
        al = valid['agg_levels'].values + np.random.uniform(-depth_jitter, depth_jitter, len(valid))
        ax.scatter(al, valid['interaction_ratio'].values, c='steelblue', s=50, zorder=3)
        for _, row in valid.iterrows():
            ax.annotate(row['label'], (row['agg_levels'], row['interaction_ratio']),
                       fontsize=6, alpha=0.7, ha='center', va='bottom',
                       xytext=(0, 4), textcoords='offset points')
        if len(valid) >= 3:
            z = np.polyfit(valid['agg_levels'].values, valid['interaction_ratio'].values, 1)
            x_line = np.linspace(valid['agg_levels'].min() - 0.2, valid['agg_levels'].max() + 0.2, 50)
            ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.5, linewidth=1)
            tau, p = kendalltau(valid['agg_levels'], valid['interaction_ratio'])
            ax.text(0.95, 0.95, f'τ={tau:.3f}\np={p:.3f}',
                   transform=ax.transAxes, ha='right', va='top', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Aggregation Levels (0=raw, 3=root)')
    ax.set_ylabel(r'$\|\beta\| / \|\alpha\|$')
    ax.set_title('(a) Interaction Ratio vs Aggregation')
    ax.grid(True, alpha=0.3)

    # (b) CES R² vs agg_levels
    ax = axes[0, 1]
    valid = df.dropna(subset=['R2_ces', 'agg_levels'])
    if len(valid) > 0:
        al = valid['agg_levels'].values + np.random.uniform(-depth_jitter, depth_jitter, len(valid))
        ax.scatter(al, valid['R2_ces'].values, c='forestgreen', s=50, zorder=3)
        for _, row in valid.iterrows():
            ax.annotate(row['label'], (row['agg_levels'], row['R2_ces']),
                       fontsize=6, alpha=0.7, ha='center', va='bottom',
                       xytext=(0, 4), textcoords='offset points')
    ax.set_xlabel('Aggregation Levels (0=raw, 3=root)')
    ax.set_ylabel(r'$R^2$ (CES)')
    ax.set_title('(b) CES Fit Quality vs Aggregation')
    ax.grid(True, alpha=0.3)

    # (c) Estimated ρ vs agg_levels with CI
    ax = axes[1, 0]
    valid = df.dropna(subset=['rho', 'agg_levels'])
    if len(valid) > 0:
        al = valid['agg_levels'].values
        rhos = valid['rho'].values
        ci_lo = valid['rho_ci_low'].values if 'rho_ci_low' in valid.columns else np.full_like(rhos, np.nan)
        ci_hi = valid['rho_ci_high'].values if 'rho_ci_high' in valid.columns else np.full_like(rhos, np.nan)

        ax.scatter(al + np.random.uniform(-depth_jitter, depth_jitter, len(valid)),
                  rhos, c='darkorange', s=50, zorder=3)

        has_ci = ~(np.isnan(ci_lo) | np.isnan(ci_hi))
        if has_ci.any():
            errs = np.array([rhos[has_ci] - ci_lo[has_ci], ci_hi[has_ci] - rhos[has_ci]])
            ax.errorbar(al[has_ci], rhos[has_ci], yerr=errs,
                       fmt='none', ecolor='darkorange', alpha=0.4, capsize=3)

        for _, row in valid.iterrows():
            ax.annotate(row['label'], (row['agg_levels'], row['rho']),
                       fontsize=6, alpha=0.7, ha='center', va='bottom',
                       xytext=(0, 4), textcoords='offset points')

        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label=r'$\rho$=0 (Cobb-Douglas)')
    ax.set_xlabel('Aggregation Levels (0=raw, 3=root)')
    ax.set_ylabel(r'$\hat{\rho}$')
    ax.set_title(r'(c) Estimated $\rho$ vs Aggregation')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # (d) Normalized ΔBIC per extra translog parameter vs agg_levels
    ax = axes[1, 1]
    valid = df.dropna(subset=['BIC_per_extra_param', 'agg_levels'])
    if len(valid) > 0:
        al = valid['agg_levels'].values + np.random.uniform(-depth_jitter, depth_jitter, len(valid))
        vals = valid['BIC_per_extra_param'].values
        colors = ['forestgreen' if v < 0 else 'firebrick' for v in vals]
        ax.scatter(al, vals, c=colors, s=50, zorder=3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        for _, row in valid.iterrows():
            K = int(row.get('n_inputs', 0))
            n_extra = K * (K + 1) // 2
            ax.annotate(f"{row['label']} (Δk={n_extra})",
                       (row['agg_levels'], row['BIC_per_extra_param']),
                       fontsize=5.5, alpha=0.7, ha='center', va='bottom',
                       xytext=(0, 4), textcoords='offset points')
        if len(valid) >= 3:
            z = np.polyfit(valid['agg_levels'].values, vals, 1)
            x_line = np.linspace(valid['agg_levels'].min() - 0.2, valid['agg_levels'].max() + 0.2, 50)
            ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.5, linewidth=1)
            tau, p = kendalltau(valid['agg_levels'], valid['BIC_per_extra_param'])
            ax.text(0.95, 0.95, f'τ={tau:.3f}\np={p:.3f}',
                   transform=ax.transAxes, ha='right', va='top', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Aggregation Levels (0=raw, 3=root)')
    ax.set_ylabel(r'$\Delta$BIC / $\Delta k$  (per extra param)')
    ax.set_title(r'(d) Normalized $\Delta$BIC vs Aggregation')
    ax.text(0.05, 0.05, 'CES favored', transform=ax.transAxes,
           fontsize=7, color='forestgreen', va='bottom')
    ax.text(0.05, 0.95, 'Translog favored', transform=ax.transAxes,
           fontsize=7, color='firebrick', va='top')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 10. Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("CES Emergence Under Aggregation — Empirical Test")
    print("=" * 80)
    print()

    df = run_analysis()

    if len(df) == 0:
        print("\nNo aggregation events could be tested. Check FRED API key and data availability.")
        sys.exit(1)

    summarize_results(df)
    save_outputs(df)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
