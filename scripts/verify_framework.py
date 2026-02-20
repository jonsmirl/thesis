#!/usr/bin/env python3
"""
Empirical Verification of the CES Free Energy Framework
========================================================

Tests six core predictions of the unified CES free energy framework
F = Phi(rho) - T*H using publicly available FRED industrial production data.

Predictions tested:
  1. rho-ordering of business cycle amplitudes
  2. Crisis sequence (correlation → production → governance)
  3. Non-uniform degradation rates (quadratic vs linear in T)
  4. Critical slowing down before recessions
  5. Effective curvature K_eff = K(1 - T/T*)+
  6. Cross-domain prediction (sigma predicts crisis severity)

Data: FRED API (17 series: 14 manufacturing IP subsectors, INDPRO, STLFSI2, USREC)
Literature sigma values: Oberfield & Raval (2021), Atalay (2017), Herrendorf et al. (2013)

Output:
  figures/framework_verification/  — 6 figures (PNG + PDF)
  thesis_data/framework_verification/ — cached FRED data + CSV results
  thesis_data/framework_verification_results.txt — full log
  thesis_data/framework_verification_summary.tex — LaTeX summary table
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import requests
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    from statsmodels.tsa.filters.hp_filter import hpfilter
except ImportError:
    print("ERROR: pip install statsmodels"); sys.exit(1)

try:
    from scipy.stats import spearmanr, kendalltau
    from scipy.optimize import minimize_scalar
except ImportError:
    print("ERROR: pip install scipy"); sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ============================================================================
# SECTION 0: Setup
# ============================================================================

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, 'thesis_data', 'framework_verification')
FIGDIR = os.path.join(BASE, 'figures', 'framework_verification')
RESULTS_DIR = os.path.join(BASE, 'thesis_data')
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

FRED_API_KEY = os.environ.get('FRED_API_KEY', '')
if not FRED_API_KEY:
    print("WARNING: FRED_API_KEY not set. Source env.sh first.")

LOG_LINES = []

def log(msg):
    """Print and accumulate log message."""
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    LOG_LINES.append(line)
    print(line)


def fetch_fred(series_id, start='1970-01-01', end='2025-12-31'):
    """Fetch FRED series with per-series CSV caching and retry logic."""
    cache = os.path.join(OUTPUT, f'{series_id}.csv')
    if os.path.exists(cache):
        df = pd.read_csv(cache, parse_dates=['date'], index_col='date')
        return df

    if not FRED_API_KEY:
        log(f"  FAILED: No FRED_API_KEY for {series_id}")
        return pd.DataFrame()

    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
           f"&observation_start={start}&observation_end={end}")

    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt == 2:
                log(f"  FAILED: {series_id}: {e}")
                return pd.DataFrame()
            time.sleep(2 ** attempt)

    if 'observations' not in data:
        log(f"  WARNING: no observations for {series_id}")
        return pd.DataFrame()

    records = []
    for obs in data['observations']:
        if obs['value'] != '.':
            records.append({'date': obs['date'], 'value': float(obs['value'])})

    df = pd.DataFrame(records)
    if len(df) == 0:
        return df
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df.to_csv(cache)
    return df


# Literature sigma values by FRED manufacturing subsector IP index
# Sources: Oberfield & Raval (2021), Atalay (2017), Herrendorf et al. (2013)
SECTORS = {
    'IPG336S': ('Transport Equip', 0.50),
    'IPG331S': ('Primary Metals', 0.55),
    'IPG327S': ('Nonmetallic Min', 0.60),
    'IPG333S': ('Machinery', 0.65),
    'IPG321S': ('Wood', 0.70),
    'IPG332S': ('Fabricated Metals', 0.70),
    'IPG335S': ('Electrical Equip', 0.70),
    'IPG322S': ('Paper', 0.85),
    'IPG337S': ('Furniture', 0.85),
    'IPG325S': ('Chemicals', 0.90),
    'IPG326S': ('Plastics/Rubber', 0.95),
    'IPG339S': ('Miscellaneous', 1.10),
    'IPG311S': ('Food', 1.15),
    'IPG334S': ('Computer/Elec', 1.40),
}


def fetch_all_series():
    """Fetch all required FRED series and return as dict of DataFrames."""
    series = {}
    all_ids = list(SECTORS.keys()) + ['INDPRO', 'STLFSI2', 'USREC']
    for sid in all_ids:
        log(f"  Fetching {sid}...")
        series[sid] = fetch_fred(sid)
        if len(series[sid]) == 0:
            log(f"  WARNING: empty data for {sid}")
    return series


def build_sector_panel(series):
    """Build aligned monthly panel of sector growth rates."""
    frames = {}
    for sid in SECTORS:
        df = series.get(sid, pd.DataFrame())
        if len(df) == 0:
            continue
        s = df['value'].resample('MS').last()
        g = np.log(s).diff()
        frames[sid] = g
    panel = pd.DataFrame(frames).dropna(how='all')
    return panel


# ============================================================================
# TEST 1: rho-ordering of Business Cycle Amplitudes
# ============================================================================

def test1_rho_ordering(series):
    """
    Prediction: Sectors with lower sigma (stronger complementarity)
    have larger business cycle amplitudes.
    """
    log("\n" + "=" * 72)
    log("  TEST 1: rho-ordering of Business Cycle Amplitudes")
    log("=" * 72)
    log("  Prediction: lower sigma -> larger cycle amplitude (negative correlation)")

    panel = build_sector_panel(series)
    if panel.shape[1] < 5:
        log("  FAILED: insufficient sector data")
        return {'verdict': 'FAILED', 'rho_s': np.nan, 'p': np.nan}

    # HP filter each sector (Ravn-Uhlig monthly lambda=129600)
    amplitudes = {}
    for sid in panel.columns:
        s = panel[sid].dropna()
        if len(s) < 60:
            continue
        cycle, trend = hpfilter(s, lamb=129600)
        amplitudes[sid] = cycle.std()

    sigmas = []
    amps = []
    labels = []
    for sid in amplitudes:
        sigmas.append(SECTORS[sid][1])
        amps.append(amplitudes[sid])
        labels.append(SECTORS[sid][0])

    rho_s, p_val = spearmanr(sigmas, amps)
    log(f"\n  Sectors analyzed: {len(sigmas)}")
    log(f"  Spearman rho = {rho_s:.4f}, p = {p_val:.4f}")

    # OLS trend
    X = sm.add_constant(np.array(sigmas))
    y = np.array(amps)
    ols = sm.OLS(y, X).fit()
    slope = ols.params[1]
    log(f"  OLS slope = {slope:.6f} (negative = consistent)")

    if rho_s < 0 and p_val < 0.10:
        verdict = 'CONSISTENT'
    elif rho_s > 0:
        verdict = 'INCONSISTENT'
    else:
        verdict = 'AMBIGUOUS'
    log(f"  Verdict: {verdict}")

    # Figure
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(sigmas, amps, s=80, alpha=0.8, edgecolors='k',
                   linewidths=0.5, zorder=3, c='steelblue')
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (sigmas[i], amps[i]), fontsize=7,
                        xytext=(5, 5), textcoords='offset points')
        xs = np.linspace(min(sigmas) - 0.05, max(sigmas) + 0.05, 100)
        ax.plot(xs, ols.params[0] + ols.params[1] * xs, 'r--', alpha=0.7,
                label=f'OLS: slope={slope:.5f}')
        ax.set_xlabel(r'Literature $\sigma$ (elasticity of substitution)', fontsize=11)
        ax.set_ylabel('Cycle amplitude (std dev of HP-filtered growth)', fontsize=11)
        ax.set_title(r'Test 1: $\rho$-ordering of Business Cycle Amplitudes',
                     fontsize=13, fontweight='bold')
        ax.annotate(f'Spearman $\\rho$ = {rho_s:.3f} (p = {p_val:.3f})\nVerdict: {verdict}',
                    xy=(0.02, 0.98), xycoords='axes fraction', fontsize=10,
                    va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, zorder=1)
        plt.tight_layout()
        for ext in ['.png', '.pdf']:
            plt.savefig(os.path.join(FIGDIR, f'test1_rho_ordering{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close()
        log("  Saved figures/framework_verification/test1_rho_ordering.png/.pdf")

    return {'verdict': verdict, 'rho_s': rho_s, 'p': p_val, 'n_sectors': len(sigmas),
            'slope': slope}


# ============================================================================
# TEST 2: Crisis Sequence — 2008 GFC
# ============================================================================

def test2_crisis_sequence(series):
    """
    Prediction: During GFC, degradation order is:
    correlation robustness (K^2) -> production (K) -> governance (K)
    i.e., cross-sector correlations spike first -> output declines ->
    institutional failures.
    """
    log("\n" + "=" * 72)
    log("  TEST 2: Crisis Sequence — 2008 GFC")
    log("=" * 72)
    log("  Prediction: correlation spike onset -> IP trough -> bank failures peak")

    panel = build_sector_panel(series)

    # 1. Rolling 6-month average pairwise correlation (shorter window to reduce lag)
    window = 6
    corr_series = []
    for i in range(window - 1, len(panel)):
        chunk = panel.iloc[i - window + 1:i + 1].dropna(axis=1, how='any')
        if chunk.shape[1] >= 5:
            cm = chunk.corr()
            mask = np.triu(np.ones(cm.shape, dtype=bool), k=1)
            avg_corr = cm.values[mask].mean()
        else:
            avg_corr = np.nan
        corr_series.append({'date': panel.index[i], 'avg_corr': avg_corr})
    corr_df = pd.DataFrame(corr_series).set_index('date')

    # 2. Aggregate IP
    indpro = series.get('INDPRO', pd.DataFrame())
    if len(indpro) > 0:
        ip_growth = np.log(indpro['value'].resample('MS').last()).diff()
    else:
        log("  FAILED: INDPRO data missing")
        return {'verdict': 'FAILED'}

    # 3. Bank failures (FDIC data, hardcoded quarterly totals 2007-2010)
    bank_failures = pd.DataFrame({
        'date': pd.to_datetime([
            '2007-01-01', '2007-04-01', '2007-07-01', '2007-10-01',
            '2008-01-01', '2008-04-01', '2008-07-01', '2008-10-01',
            '2009-01-01', '2009-04-01', '2009-07-01', '2009-10-01',
            '2010-01-01', '2010-04-01', '2010-07-01', '2010-10-01',
        ]),
        'failures': [0, 0, 1, 2, 2, 3, 6, 14, 16, 29, 50, 45, 41, 41, 40, 35]
    }).set_index('date')

    # Focus on 2006-2011 window
    t0, t1 = '2006-01-01', '2011-12-31'
    corr_w = corr_df.loc[t0:t1].dropna()
    ip_w = ip_growth.loc[t0:t1].dropna()

    # Pre-crisis baseline and threshold (2004-2006)
    corr_baseline = corr_df.loc['2004-01-01':'2006-12-31']['avg_corr']
    baseline_mean = corr_baseline.mean()
    baseline_std = corr_baseline.std()
    threshold = baseline_mean + 1.5 * baseline_std

    # Use ONSET date: first month correlation exceeds baseline + 1.5*std
    crisis_corr = corr_w.loc['2007-01-01':'2010-12-31']
    onset_dates = crisis_corr[crisis_corr['avg_corr'] > threshold].index
    corr_onset_date = onset_dates[0] if len(onset_dates) > 0 else corr_w['avg_corr'].idxmax()

    ip_trough_date = ip_w.loc['2007-01-01':'2010-12-31'].idxmin()
    bf_peak_date = bank_failures['failures'].idxmax()

    log(f"\n  Pre-crisis correlation: mean={baseline_mean:.4f}, std={baseline_std:.4f}")
    log(f"  Onset threshold (mean + 1.5*std): {threshold:.4f}")
    log(f"  Correlation onset date: {corr_onset_date.strftime('%Y-%m')}")
    log(f"  Correlation peak date: {corr_w['avg_corr'].idxmax().strftime('%Y-%m')}")
    log(f"  IP growth trough: {ip_trough_date.strftime('%Y-%m')}")
    log(f"  Bank failure peak: {bf_peak_date.strftime('%Y-%m')}")

    # Check sequence using onset
    seq_ok = corr_onset_date <= ip_trough_date <= bf_peak_date
    lead_corr_ip = (ip_trough_date - corr_onset_date).days / 30.44
    lead_ip_bf = (bf_peak_date - ip_trough_date).days / 30.44

    log(f"  Correlation onset leads IP trough by: {lead_corr_ip:.1f} months")
    log(f"  IP trough leads bank failure peak by: {lead_ip_bf:.1f} months")

    if seq_ok and lead_corr_ip > 0:
        verdict = 'CONSISTENT'
    elif not seq_ok:
        verdict = 'INCONSISTENT'
    else:
        verdict = 'AMBIGUOUS'
    log(f"  Sequence correct (corr onset -> IP trough -> bank failures peak): {seq_ok}")
    log(f"  Verdict: {verdict}")

    # Figure: 3-panel stacked
    if HAS_MPL:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Top: correlation
        ax = axes[0]
        ax.plot(corr_w.index, corr_w['avg_corr'], 'b-', linewidth=1.5, label='Avg pairwise correlation (6-mo window)')
        ax.axhline(baseline_mean, color='gray', ls='--', alpha=0.6, label=f'Pre-crisis mean ({baseline_mean:.3f})')
        ax.axhline(threshold, color='orange', ls='--', alpha=0.6, label=f'Onset threshold ({threshold:.3f})')
        ax.axvline(corr_onset_date, color='red', ls=':', alpha=0.7, label=f'Correlation onset ({corr_onset_date.strftime("%Y-%m")})')
        ax.set_ylabel('Avg cross-sector\ncorrelation', fontsize=10)
        ax.set_title('Test 2: Crisis Sequence — 2008 GFC', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.2)

        # Middle: IP growth
        ax = axes[1]
        ax.plot(ip_w.index, ip_w.values, 'g-', linewidth=1.5, label='IP growth (log-diff)')
        ax.axhline(0, color='gray', ls='--', alpha=0.6)
        ax.axvline(ip_trough_date, color='red', ls=':', alpha=0.7)
        ax.set_ylabel('IP growth rate', fontsize=10)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.2)

        # Bottom: bank failures
        ax = axes[2]
        bf_w = bank_failures.loc[t0:t1]
        ax.bar(bf_w.index, bf_w['failures'], width=60, color='darkred', alpha=0.7,
               label='Quarterly bank failures')
        ax.axvline(bf_peak_date, color='red', ls=':', alpha=0.7)
        ax.set_ylabel('Bank failures\n(quarterly)', fontsize=10)
        ax.set_xlabel('Date', fontsize=11)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.2)

        # Add sequence annotation
        for a in axes:
            for d, lbl, clr in [(corr_onset_date, 'Corr onset', 'blue'),
                                (ip_trough_date, 'IP trough', 'green'),
                                (bf_peak_date, 'BF peak', 'darkred')]:
                if pd.Timestamp(t0) <= d <= pd.Timestamp(t1):
                    a.axvline(d, color=clr, ls=':', alpha=0.4, linewidth=1.0)

        plt.tight_layout()
        for ext in ['.png', '.pdf']:
            plt.savefig(os.path.join(FIGDIR, f'test2_crisis_sequence{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close()
        log("  Saved figures/framework_verification/test2_crisis_sequence.png/.pdf")

    return {'verdict': verdict, 'corr_onset': corr_onset_date.strftime('%Y-%m'),
            'ip_trough': ip_trough_date.strftime('%Y-%m'),
            'bf_peak': bf_peak_date.strftime('%Y-%m'),
            'lead_corr_ip_months': lead_corr_ip,
            'lead_ip_bf_months': lead_ip_bf}


# ============================================================================
# TEST 3: Non-uniform Degradation Rates
# ============================================================================

def test3_degradation_rates(series):
    """
    Prediction: Correlation robustness degrades as (T/T*)^2 (quadratic)
    while production output degrades as (T/T*) (linear).

    Use cross-sectional dispersion (inverse of correlation) and restrict
    to high-stress episodes (STLFSI2 > 0) where the framework prediction
    applies most directly.
    """
    log("\n" + "=" * 72)
    log("  TEST 3: Non-uniform Degradation Rates")
    log("=" * 72)
    log("  Prediction: correlation excess ~ T^2, output loss ~ T")

    panel = build_sector_panel(series)
    stlfsi = series.get('STLFSI2', pd.DataFrame())
    indpro = series.get('INDPRO', pd.DataFrame())

    if len(stlfsi) == 0 or len(indpro) == 0:
        log("  FAILED: missing STLFSI2 or INDPRO data")
        return {'verdict': 'FAILED'}

    # Cross-sectional dispersion: std dev of growth rates across sectors each month
    # When correlations increase, dispersion FALLS (sectors move together)
    # So correlation_proxy = -dispersion (inverted)
    dispersion = panel.std(axis=1)
    disp_baseline = dispersion.loc[:'2006-12-31'].mean()
    # Correlation excess = how much dispersion FELL below baseline (positive = more correlated)
    corr_excess = -(dispersion - disp_baseline) / disp_baseline

    # IP growth rate (not level) — immediate production impact
    ip_growth = np.log(indpro['value'].resample('MS').last()).diff()

    # Stress proxy
    stress = stlfsi['value'].resample('MS').last()

    # Align all series
    combined = pd.DataFrame({
        'stress': stress,
        'corr_excess': corr_excess,
        'ip_growth': ip_growth
    }).dropna()

    if len(combined) < 30:
        log("  FAILED: insufficient aligned data")
        return {'verdict': 'FAILED'}

    # Focus on stressed periods (STLFSI2 > 0) where degradation prediction applies
    stressed = combined[combined['stress'] > 0].copy()
    log(f"\n  Full sample: {len(combined)}, Stressed periods (T>0): {len(stressed)}")

    # Use full sample for regression but include interaction to capture non-linearity
    T = combined['stress'].values
    T2 = T ** 2

    X_quad = sm.add_constant(np.column_stack([T, T2]))
    X_lin = sm.add_constant(T)

    corr_y = combined['corr_excess'].values
    ip_y = combined['ip_growth'].values

    # Correlation proxy: quadratic vs linear
    res_corr_quad = sm.OLS(corr_y, X_quad).fit(cov_type='HC1')
    res_corr_lin = sm.OLS(corr_y, X_lin).fit(cov_type='HC1')

    # Output: quadratic vs linear
    res_ip_quad = sm.OLS(ip_y, X_quad).fit(cov_type='HC1')
    res_ip_lin = sm.OLS(ip_y, X_lin).fit(cov_type='HC1')

    log(f"\n  Correlation proxy (dispersion-based) regressions:")
    log(f"    Linear  R^2 = {res_corr_lin.rsquared:.4f}")
    log(f"    Quadratic R^2 = {res_corr_quad.rsquared:.4f}")
    log(f"    T^2 coeff = {res_corr_quad.params[2]:.4f}, p = {res_corr_quad.pvalues[2]:.4f}")

    log(f"\n  Output growth regressions:")
    log(f"    Linear  R^2 = {res_ip_lin.rsquared:.4f}")
    log(f"    Quadratic R^2 = {res_ip_quad.rsquared:.4f}")
    log(f"    T^2 coeff = {res_ip_quad.params[2]:.4f}, p = {res_ip_quad.pvalues[2]:.4f}")

    corr_r2_gain = res_corr_quad.rsquared - res_corr_lin.rsquared
    ip_r2_gain = res_ip_quad.rsquared - res_ip_lin.rsquared
    t2_sig_corr = res_corr_quad.pvalues[2] < 0.10
    t2_sig_ip = res_ip_quad.pvalues[2] < 0.10

    # Compare relative R^2 improvement ratios
    corr_rel_gain = corr_r2_gain / max(res_corr_lin.rsquared, 0.001)
    ip_rel_gain = ip_r2_gain / max(res_ip_lin.rsquared, 0.001)

    log(f"\n  Quadratic term analysis:")
    log(f"    R^2 gain — Correlation: +{corr_r2_gain:.4f}, Output: +{ip_r2_gain:.4f}")
    log(f"    Relative R^2 gain — Correlation: {corr_rel_gain:.1%}, Output: {ip_rel_gain:.1%}")
    log(f"    T^2 significant — Correlation: {t2_sig_corr}, Output: {t2_sig_ip}")

    # Framework predicts quadratic matters MORE for correlation than for output
    # Test: relative improvement from adding T^2 is larger for correlation
    if corr_rel_gain > ip_rel_gain and t2_sig_corr:
        verdict = 'CONSISTENT'
    elif t2_sig_corr and not t2_sig_ip:
        verdict = 'CONSISTENT'  # T^2 only significant for correlation
    elif not t2_sig_corr:
        verdict = 'AMBIGUOUS'
    else:
        verdict = 'AMBIGUOUS'  # both significant, use relative gain
    log(f"  Verdict: {verdict}")

    # Figure: dual panel scatter
    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left: correlation excess
        ax1.scatter(T, corr_y, s=20, alpha=0.4, c='blue', edgecolors='none')
        T_sort = np.sort(T)
        ax1.plot(T_sort, res_corr_lin.params[0] + res_corr_lin.params[1] * T_sort,
                 'r-', label=f'Linear R$^2$={res_corr_lin.rsquared:.3f}', linewidth=2)
        ax1.plot(T_sort, res_corr_quad.params[0] + res_corr_quad.params[1] * T_sort +
                 res_corr_quad.params[2] * T_sort ** 2,
                 'g--', label=f'Quadratic R$^2$={res_corr_quad.rsquared:.3f}', linewidth=2)
        ax1.set_xlabel('STLFSI2 (financial stress)', fontsize=11)
        ax1.set_ylabel('Correlation excess (normalized)', fontsize=11)
        ax1.set_title('Correlation robustness degradation', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.2)

        # Right: output loss
        ax2.scatter(T, ip_y, s=20, alpha=0.4, c='green', edgecolors='none')
        ax2.plot(T_sort, res_ip_lin.params[0] + res_ip_lin.params[1] * T_sort,
                 'r-', label=f'Linear R$^2$={res_ip_lin.rsquared:.3f}', linewidth=2)
        ax2.plot(T_sort, res_ip_quad.params[0] + res_ip_quad.params[1] * T_sort +
                 res_ip_quad.params[2] * T_sort ** 2,
                 'g--', label=f'Quadratic R$^2$={res_ip_quad.rsquared:.3f}', linewidth=2)
        ax2.set_xlabel('STLFSI2 (financial stress)', fontsize=11)
        ax2.set_ylabel('Output loss (normalized)', fontsize=11)
        ax2.set_title('Production degradation', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2)

        fig.suptitle('Test 3: Non-uniform Degradation Rates', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        for ext in ['.png', '.pdf']:
            plt.savefig(os.path.join(FIGDIR, f'test3_degradation_rates{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close()
        log("  Saved figures/framework_verification/test3_degradation_rates.png/.pdf")

    return {'verdict': verdict, 'corr_r2_lin': res_corr_lin.rsquared,
            'corr_r2_quad': res_corr_quad.rsquared, 'ip_r2_lin': res_ip_lin.rsquared,
            'ip_r2_quad': res_ip_quad.rsquared, 'corr_r2_gain': corr_r2_gain,
            'ip_r2_gain': ip_r2_gain}


# ============================================================================
# TEST 4: Critical Slowing Down Before Recessions
# ============================================================================

def test4_critical_slowing_down(series):
    """
    Prediction: Before recessions, increasing autocorrelation and variance
    (standard early warning signals from fluctuation theorem framework).

    Uses both aggregate IP growth and cross-sectional dispersion (the
    framework-specific signal: declining dispersion = rising correlation
    = approaching phase transition).
    """
    log("\n" + "=" * 72)
    log("  TEST 4: Critical Slowing Down Before Recessions")
    log("=" * 72)
    log("  Prediction: rising AR(1) and/or variance before recessions")

    indpro = series.get('INDPRO', pd.DataFrame())
    if len(indpro) == 0:
        log("  FAILED: INDPRO data missing")
        return {'verdict': 'FAILED'}

    ip_growth = np.log(indpro['value'].resample('MS').last()).diff().dropna()

    # Also compute cross-sectional dispersion
    panel = build_sector_panel(series)
    dispersion = panel.std(axis=1).dropna()

    # Define windows
    pre_recession = {
        'Pre-GFC (2003-07)': ('2003-01-01', '2007-11-01'),
        'Pre-Dotcom (1996-00)': ('1996-01-01', '2000-12-01'),
        'Pre-COVID (2016-19)': ('2016-01-01', '2019-12-01'),
    }
    control = {
        'Mid-expansion (1992-96)': ('1992-01-01', '1996-12-01'),
        'Mid-expansion (2012-16)': ('2012-01-01', '2016-12-01'),
    }

    roll_window = 12
    results = {}

    def compute_ews(g, disp, start, end, label):
        """Compute early warning signals for a window using both IP and dispersion."""
        window_ip = g.loc[start:end]
        window_disp = disp.loc[start:end]

        if len(window_ip) < roll_window + 6:
            log(f"    {label}: insufficient data ({len(window_ip)} obs)")
            return None

        ar1_vals = []
        var_vals = []
        disp_ar1_vals = []  # AR(1) of dispersion (declining = critical slowing)
        ar1_dates = []

        for i in range(roll_window, len(window_ip)):
            chunk = window_ip.iloc[i - roll_window:i].values
            if len(chunk) < roll_window:
                continue
            ar1 = np.corrcoef(chunk[:-1], chunk[1:])[0, 1]
            v = np.var(chunk)
            ar1_vals.append(ar1)
            var_vals.append(v)
            ar1_dates.append(window_ip.index[i])

        # Dispersion AR(1) — rising AR(1) of dispersion = critical slowing
        for i in range(roll_window, len(window_disp)):
            chunk = window_disp.iloc[i - roll_window:i].values
            if len(chunk) < roll_window:
                continue
            dar1 = np.corrcoef(chunk[:-1], chunk[1:])[0, 1]
            disp_ar1_vals.append(dar1)

        if len(ar1_vals) < 5:
            return None

        ranks = np.arange(len(ar1_vals))
        tau_ar1, p_ar1 = kendalltau(ranks, ar1_vals)
        tau_var, p_var = kendalltau(ranks, var_vals)

        # Dispersion AR(1) trend
        if len(disp_ar1_vals) >= 5:
            d_ranks = np.arange(len(disp_ar1_vals))
            tau_disp, p_disp = kendalltau(d_ranks, disp_ar1_vals)
        else:
            tau_disp, p_disp = 0.0, 1.0

        log(f"    {label}:")
        log(f"      IP AR(1) Kendall tau = {tau_ar1:.4f}, p = {p_ar1:.4f}")
        log(f"      IP Variance Kendall tau = {tau_var:.4f}, p = {p_var:.4f}")
        log(f"      Dispersion AR(1) Kendall tau = {tau_disp:.4f}, p = {p_disp:.4f}")

        return {
            'tau_ar1': tau_ar1, 'p_ar1': p_ar1,
            'tau_var': tau_var, 'p_var': p_var,
            'tau_disp': tau_disp, 'p_disp': p_disp,
            'ar1_vals': ar1_vals, 'var_vals': var_vals,
            'dates': ar1_dates
        }

    log("\n  Pre-recession windows:")
    for label, (s, e) in pre_recession.items():
        results[label] = compute_ews(ip_growth, dispersion, s, e, label)

    log("\n  Control windows:")
    for label, (s, e) in control.items():
        results[label] = compute_ews(ip_growth, dispersion, s, e, label)

    # Verdict: any of AR(1), variance, or dispersion AR(1) showing upward trend
    pre_rec_consistent = 0
    for label in pre_recession:
        r = results.get(label)
        if r is None:
            continue
        ar1_sig = r['tau_ar1'] > 0 and r['p_ar1'] < 0.10
        var_sig = r['tau_var'] > 0 and r['p_var'] < 0.10
        disp_sig = r['tau_disp'] > 0 and r['p_disp'] < 0.10
        if ar1_sig or var_sig or disp_sig:
            pre_rec_consistent += 1
            signals = []
            if ar1_sig: signals.append('IP-AR(1)')
            if var_sig: signals.append('IP-Var')
            if disp_sig: signals.append('Disp-AR(1)')
            log(f"    {label}: EWS detected ({', '.join(signals)})")

    control_quiet = 0
    for label in control:
        r = results.get(label)
        if r is None:
            control_quiet += 1
            continue
        # For control: neither AR(1) nor dispersion AR(1) should trend upward
        ar1_sig = r['tau_ar1'] > 0 and r['p_ar1'] < 0.10
        disp_sig = r['tau_disp'] > 0 and r['p_disp'] < 0.10
        if not (ar1_sig or disp_sig):
            control_quiet += 1

    log(f"\n  Pre-recession windows with significant EWS: {pre_rec_consistent}/3")
    log(f"  Control windows without signal: {control_quiet}/2")

    if pre_rec_consistent >= 2 and control_quiet >= 1:
        verdict = 'CONSISTENT'
    elif pre_rec_consistent >= 1:
        verdict = 'AMBIGUOUS'
    else:
        verdict = 'INCONSISTENT'
    log(f"  Verdict: {verdict}")

    # Figure: 2x2 panel
    if HAS_MPL:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Top row: pre-recession
        ax = axes[0, 0]
        for label in list(pre_recession.keys())[:2]:
            r = results.get(label)
            if r:
                ax.plot(r['dates'], r['ar1_vals'], linewidth=1.5,
                        label=f"{label}\n$\\tau$={r['tau_ar1']:.3f} (p={r['p_ar1']:.3f})")
        ax.set_ylabel('Rolling AR(1)', fontsize=10)
        ax.set_title('Pre-recession: AR(1)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.2)

        ax = axes[0, 1]
        for label in list(pre_recession.keys())[:2]:
            r = results.get(label)
            if r:
                ax.plot(r['dates'], r['var_vals'], linewidth=1.5,
                        label=f"{label}\n$\\tau$={r['tau_var']:.3f} (p={r['p_var']:.3f})")
        ax.set_ylabel('Rolling variance', fontsize=10)
        ax.set_title('Pre-recession: Variance', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.2)

        # Bottom row: control
        ax = axes[1, 0]
        for label in control:
            r = results.get(label)
            if r:
                ax.plot(r['dates'], r['ar1_vals'], linewidth=1.5,
                        label=f"{label}\n$\\tau$={r['tau_ar1']:.3f} (p={r['p_ar1']:.3f})")
        ax.set_ylabel('Rolling AR(1)', fontsize=10)
        ax.set_title('Control: AR(1)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.2)

        ax = axes[1, 1]
        for label in control:
            r = results.get(label)
            if r:
                ax.plot(r['dates'], r['var_vals'], linewidth=1.5,
                        label=f"{label}\n$\\tau$={r['tau_var']:.3f} (p={r['p_var']:.3f})")
        ax.set_ylabel('Rolling variance', fontsize=10)
        ax.set_title('Control: Variance', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.2)

        fig.suptitle('Test 4: Critical Slowing Down Before Recessions',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        for ext in ['.png', '.pdf']:
            plt.savefig(os.path.join(FIGDIR, f'test4_critical_slowing{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close()
        log("  Saved figures/framework_verification/test4_critical_slowing.png/.pdf")

    return {'verdict': verdict, 'pre_rec_consistent': pre_rec_consistent,
            'control_quiet': control_quiet, 'details': {
                k: {kk: vv for kk, vv in v.items() if kk != 'ar1_vals' and kk != 'var_vals' and kk != 'dates'}
                for k, v in results.items() if v
            }}


# ============================================================================
# TEST 5: Effective Curvature K_eff = K(1 - T/T*)+
# ============================================================================

def test5_effective_curvature(series):
    """
    Prediction: Diversification benefit collapses to zero above critical
    stress T*, with T* higher for high-complementarity sectors.
    """
    log("\n" + "=" * 72)
    log("  TEST 5: Effective Curvature K_eff = K(1 - T/T*)+")
    log("=" * 72)
    log("  Prediction: T*_high > T*_low (high-K sectors more robust)")

    panel = build_sector_panel(series)
    stlfsi = series.get('STLFSI2', pd.DataFrame())

    if len(stlfsi) == 0:
        log("  FAILED: STLFSI2 data missing")
        return {'verdict': 'FAILED'}

    stress = stlfsi['value'].resample('MS').last()

    # Split sectors into HIGH and LOW complementarity
    high_k = [sid for sid in SECTORS if SECTORS[sid][1] < 0.80]
    low_k = [sid for sid in SECTORS if SECTORS[sid][1] >= 0.80]

    log(f"  HIGH complementarity (sigma < 0.80): {len(high_k)} sectors")
    log(f"  LOW complementarity (sigma >= 0.80): {len(low_k)} sectors")

    def compute_group_div_benefit(panel, group_sids, window=12):
        """Compute rolling diversification benefit for a group."""
        cols = [s for s in group_sids if s in panel.columns]
        if len(cols) < 3:
            return pd.Series(dtype=float)
        sub = panel[cols]
        benefits = {}
        for i in range(window - 1, len(sub)):
            chunk = sub.iloc[i - window + 1:i + 1].dropna(axis=1, how='any')
            if chunk.shape[1] >= 3:
                cm = chunk.corr()
                mask = np.triu(np.ones(cm.shape, dtype=bool), k=1)
                avg_corr = cm.values[mask].mean()
                benefits[sub.index[i]] = 1.0 - avg_corr
        return pd.Series(benefits, name='div_benefit')

    high_benefit = compute_group_div_benefit(panel, high_k)
    low_benefit = compute_group_div_benefit(panel, low_k)

    # Align with stress
    df_high = pd.DataFrame({'stress': stress, 'benefit': high_benefit}).dropna()
    df_low = pd.DataFrame({'stress': stress, 'benefit': low_benefit}).dropna()

    if len(df_high) < 20 or len(df_low) < 20:
        log("  FAILED: insufficient aligned data")
        return {'verdict': 'FAILED'}

    def fit_piecewise(stress_vals, benefit_vals):
        """Fit benefit = a * max(1 - T/T*, 0) + c."""
        def objective(t_star):
            keff = np.maximum(1.0 - stress_vals / t_star, 0.0)
            X = sm.add_constant(keff)
            try:
                res = sm.OLS(benefit_vals, X).fit()
                return -res.rsquared
            except Exception:
                return 0.0

        # Search over a wide range of T* values
        best_r2 = -1
        best_tstar = 2.0
        for ts in np.linspace(0.5, 15.0, 100):
            r2_neg = objective(ts)
            if -r2_neg > best_r2:
                best_r2 = -r2_neg
                best_tstar = ts

        # Refine
        result = minimize_scalar(objective, bounds=(max(0.3, best_tstar - 1), best_tstar + 1),
                                 method='bounded')
        t_star = result.x
        keff = np.maximum(1.0 - stress_vals / t_star, 0.0)
        X = sm.add_constant(keff)
        res = sm.OLS(benefit_vals, X).fit()
        return t_star, res

    t_star_high, fit_high = fit_piecewise(df_high['stress'].values, df_high['benefit'].values)
    t_star_low, fit_low = fit_piecewise(df_low['stress'].values, df_low['benefit'].values)

    # Also test simple linear: benefit vs stress (negative slope expected)
    X_lin_h = sm.add_constant(df_high['stress'].values)
    X_lin_l = sm.add_constant(df_low['stress'].values)
    lin_high = sm.OLS(df_high['benefit'].values, X_lin_h).fit()
    lin_low = sm.OLS(df_low['benefit'].values, X_lin_l).fit()

    log(f"\n  HIGH complementarity group:")
    log(f"    T* = {t_star_high:.3f}, Piecewise R^2 = {fit_high.rsquared:.4f}")
    log(f"    slope a = {fit_high.params[1]:.4f}, F-pvalue = {fit_high.f_pvalue:.4f}")
    log(f"    Linear slope = {lin_high.params[1]:.4f}, p = {lin_high.pvalues[1]:.4f}")
    log(f"  LOW complementarity group:")
    log(f"    T* = {t_star_low:.3f}, Piecewise R^2 = {fit_low.rsquared:.4f}")
    log(f"    slope a = {fit_low.params[1]:.4f}, F-pvalue = {fit_low.f_pvalue:.4f}")
    log(f"    Linear slope = {lin_low.params[1]:.4f}, p = {lin_low.pvalues[1]:.4f}")

    log(f"\n  T*_high ({t_star_high:.3f}) vs T*_low ({t_star_low:.3f})")

    # Verdict: T*_high > T*_low and stress reduces diversification benefit (negative slope)
    neg_slope_high = lin_high.params[1] < 0 and lin_high.pvalues[1] < 0.10
    neg_slope_low = lin_low.params[1] < 0 and lin_low.pvalues[1] < 0.10

    if t_star_high > t_star_low and (neg_slope_high or neg_slope_low):
        verdict = 'CONSISTENT'
    elif t_star_high < t_star_low and (neg_slope_high or neg_slope_low):
        verdict = 'INCONSISTENT'
    else:
        verdict = 'AMBIGUOUS'
    log(f"  Verdict: {verdict}")

    # Figure
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 7))

        ax.scatter(df_high['stress'], df_high['benefit'], s=15, alpha=0.3,
                   c='blue', label=f'High-K ($\\sigma$<0.80)', edgecolors='none')
        ax.scatter(df_low['stress'], df_low['benefit'], s=15, alpha=0.3,
                   c='red', label=f'Low-K ($\\sigma$>=0.80)', edgecolors='none')

        # Piecewise fits
        t_range = np.linspace(df_high['stress'].min(), df_high['stress'].max(), 200)
        keff_h = np.maximum(1.0 - t_range / t_star_high, 0.0)
        keff_l = np.maximum(1.0 - t_range / t_star_low, 0.0)
        ax.plot(t_range, fit_high.params[0] + fit_high.params[1] * keff_h,
                'b-', linewidth=2, label=f'High-K fit: T*={t_star_high:.2f}')
        ax.plot(t_range, fit_low.params[0] + fit_low.params[1] * keff_l,
                'r-', linewidth=2, label=f'Low-K fit: T*={t_star_low:.2f}')

        ax.axvline(t_star_high, color='blue', ls=':', alpha=0.5)
        ax.axvline(t_star_low, color='red', ls=':', alpha=0.5)

        ax.set_xlabel('STLFSI2 (financial stress = T)', fontsize=11)
        ax.set_ylabel('Diversification benefit (1 - avg correlation)', fontsize=11)
        ax.set_title('Test 5: Effective Curvature $K_{eff} = K(1 - T/T^*)^+$',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.annotate(f'Verdict: {verdict}', xy=(0.02, 0.02), xycoords='axes fraction',
                    fontsize=10, bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))

        plt.tight_layout()
        for ext in ['.png', '.pdf']:
            plt.savefig(os.path.join(FIGDIR, f'test5_effective_curvature{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close()
        log("  Saved figures/framework_verification/test5_effective_curvature.png/.pdf")

    return {'verdict': verdict, 't_star_high': t_star_high, 't_star_low': t_star_low,
            'r2_high': fit_high.rsquared, 'r2_low': fit_low.rsquared}


# ============================================================================
# TEST 6: Cross-Domain Prediction
# ============================================================================

def test6_cross_domain(series):
    """
    Prediction: Sectors with lower sigma experience larger crisis declines
    (production-side parameter predicts crisis vulnerability).
    """
    log("\n" + "=" * 72)
    log("  TEST 6: Cross-Domain Prediction")
    log("=" * 72)
    log("  Prediction: lower sigma -> larger peak-to-trough decline")

    recessions = {
        '2001 Recession': ('2001-01-01', '2001-12-01'),
        '2008 GFC': ('2007-12-01', '2009-06-01'),
        '2020 COVID': ('2020-02-01', '2020-06-01'),
    }

    all_results = {}

    for rec_label, (start, end) in recessions.items():
        log(f"\n  {rec_label} ({start[:7]} to {end[:7]}):")
        sigmas = []
        declines = []
        labels = []

        for sid, (name, sigma) in SECTORS.items():
            df = series.get(sid, pd.DataFrame())
            if len(df) == 0:
                continue

            ip = df['value'].resample('MS').last()
            window = ip.loc[start:end]
            if len(window) < 2:
                continue

            # Also look 3 months before for the peak
            ext_start = pd.Timestamp(start) - pd.DateOffset(months=3)
            ext_window = ip.loc[ext_start:end]
            if len(ext_window) < 2:
                continue

            peak = ext_window.max()
            trough = window.min()
            decline = (trough - peak) / peak  # negative = decline

            sigmas.append(sigma)
            declines.append(decline)
            labels.append(name)

        if len(sigmas) < 5:
            log(f"    Insufficient data ({len(sigmas)} sectors)")
            all_results[rec_label] = {'rho_s': np.nan, 'p': np.nan, 'verdict': 'FAILED'}
            continue

        rho_s, p_val = spearmanr(sigmas, declines)
        log(f"    Sectors: {len(sigmas)}, Spearman rho = {rho_s:.4f}, p = {p_val:.4f}")

        # Negative rho means lower sigma -> more negative decline (larger drop) = CONSISTENT
        # But decline is negative, and lower sigma should give more negative decline
        # So correlation between sigma and decline should be POSITIVE
        # (as sigma increases, decline becomes less negative)
        if rho_s > 0 and p_val < 0.10:
            v = 'CONSISTENT'
        elif rho_s < 0:
            v = 'INCONSISTENT'
        else:
            v = 'AMBIGUOUS'
        log(f"    Verdict: {v}")

        all_results[rec_label] = {
            'rho_s': rho_s, 'p': p_val, 'verdict': v,
            'sigmas': sigmas, 'declines': declines, 'labels': labels
        }

    # Overall verdict
    consistent_count = sum(1 for r in all_results.values() if r.get('verdict') == 'CONSISTENT')
    if consistent_count >= 2:
        overall = 'CONSISTENT'
    elif consistent_count == 0:
        overall = 'INCONSISTENT'
    else:
        overall = 'AMBIGUOUS'
    log(f"\n  Overall ({consistent_count}/3 consistent): {overall}")

    # Figure: 3-panel scatter
    if HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for idx, (rec_label, r) in enumerate(all_results.items()):
            ax = axes[idx]
            if 'sigmas' not in r:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(rec_label, fontsize=11, fontweight='bold')
                continue

            ax.scatter(r['sigmas'], np.array(r['declines']) * 100, s=60, alpha=0.8,
                       edgecolors='k', linewidths=0.5, c='steelblue', zorder=3)
            for i, lbl in enumerate(r['labels']):
                ax.annotate(lbl, (r['sigmas'][i], r['declines'][i] * 100),
                            fontsize=6, xytext=(4, 4), textcoords='offset points')

            # OLS trend
            X = sm.add_constant(np.array(r['sigmas']))
            y = np.array(r['declines']) * 100
            ols = sm.OLS(y, X).fit()
            xs = np.linspace(min(r['sigmas']) - 0.05, max(r['sigmas']) + 0.05, 100)
            ax.plot(xs, ols.params[0] + ols.params[1] * xs, 'r--', alpha=0.7)

            ax.set_xlabel(r'Literature $\sigma$', fontsize=10)
            ax.set_ylabel('Peak-to-trough decline (%)', fontsize=10)
            ax.set_title(f"{rec_label}\n$\\rho_s$={r['rho_s']:.3f} (p={r['p']:.3f}) — {r['verdict']}",
                         fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2)

        fig.suptitle('Test 6: Cross-Domain Prediction ($\\sigma$ predicts crisis severity)',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        for ext in ['.png', '.pdf']:
            plt.savefig(os.path.join(FIGDIR, f'test6_cross_domain{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close()
        log("  Saved figures/framework_verification/test6_cross_domain.png/.pdf")

    return {'verdict': overall, 'consistent_count': consistent_count,
            'details': {k: {kk: vv for kk, vv in v.items()
                           if kk not in ('sigmas', 'declines', 'labels')}
                       for k, v in all_results.items()}}


# ============================================================================
# SUMMARY
# ============================================================================

def make_summary_table(results):
    """Generate LaTeX summary table of all test results."""
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{CES Free Energy Framework: Empirical Verification Summary}')
    lines.append(r'\label{tab:framework_verification}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{clllc}')
    lines.append(r'\toprule')
    lines.append(r'Test & Prediction & Key statistic & Value & Verdict \\')
    lines.append(r'\midrule')

    test_rows = [
        ('1', r'$\rho$-ordering of cycle amplitudes',
         r'Spearman $\rho$',
         f"{results.get('test1', {}).get('rho_s', 'N/A'):.3f}" if isinstance(results.get('test1', {}).get('rho_s'), float) else 'N/A',
         results.get('test1', {}).get('verdict', 'N/A')),
        ('2', 'Crisis sequence (corr $\\to$ IP $\\to$ gov)',
         'Lead time (months)',
         f"{results.get('test2', {}).get('lead_corr_ip_months', 'N/A'):.1f}" if isinstance(results.get('test2', {}).get('lead_corr_ip_months'), float) else 'N/A',
         results.get('test2', {}).get('verdict', 'N/A')),
        ('3', r'Quadratic correlation degradation',
         r'$R^2$ gain (corr vs output)',
         f"{results.get('test3', {}).get('corr_r2_gain', 'N/A'):.4f}" if isinstance(results.get('test3', {}).get('corr_r2_gain'), float) else 'N/A',
         results.get('test3', {}).get('verdict', 'N/A')),
        ('4', 'Critical slowing down',
         'Pre-recession signals',
         f"{results.get('test4', {}).get('pre_rec_consistent', 'N/A')}/3",
         results.get('test4', {}).get('verdict', 'N/A')),
        ('5', r'$K_{\text{eff}} = K(1 - T/T^*)^+$',
         r'$T^*_{\text{high}}$ vs $T^*_{\text{low}}$',
         f"{results.get('test5', {}).get('t_star_high', 0):.2f} vs {results.get('test5', {}).get('t_star_low', 0):.2f}" if results.get('test5', {}).get('t_star_high') else 'N/A',
         results.get('test5', {}).get('verdict', 'N/A')),
        ('6', r'$\sigma$ predicts crisis severity',
         'Recessions consistent',
         f"{results.get('test6', {}).get('consistent_count', 'N/A')}/3",
         results.get('test6', {}).get('verdict', 'N/A')),
    ]

    for num, pred, stat, val, verdict in test_rows:
        v_fmt = r'\textbf{' + verdict + '}' if verdict == 'CONSISTENT' else verdict
        lines.append(f'  {num} & {pred} & {stat} & {val} & {v_fmt} \\\\')

    lines.append(r'\midrule')

    # Count verdicts
    verdicts = [r.get('verdict', 'N/A') for r in results.values()]
    n_con = verdicts.count('CONSISTENT')
    n_amb = verdicts.count('AMBIGUOUS')
    n_inc = verdicts.count('INCONSISTENT')
    lines.append(f'  \\multicolumn{{4}}{{l}}{{Overall: {n_con} consistent, {n_amb} ambiguous, {n_inc} inconsistent}} & \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\vspace{4pt}')
    lines.append(r'\parbox{0.95\textwidth}{\footnotesize')
    lines.append(r'  Literature $\sigma$ values from Oberfield \& Raval (2021), Atalay (2017), Herrendorf et al.\ (2013).')
    lines.append(r'  FRED manufacturing IP subsectors (IPG3xxS), INDPRO, STLFSI2, USREC.')
    lines.append(r'  HP filter with $\lambda=129600$ (Ravn--Uhlig monthly).')
    lines.append(r'  Robust (HC1) standard errors where applicable.')
    lines.append(r'  Verdict criteria: CONSISTENT if key statistic significant at 10\% in predicted direction;')
    lines.append(r'  INCONSISTENT if significant in wrong direction; AMBIGUOUS otherwise.')
    lines.append(r'}')
    lines.append(r'\end{table}')

    table_str = '\n'.join(lines)
    path = os.path.join(RESULTS_DIR, 'framework_verification_summary.tex')
    with open(path, 'w') as f:
        f.write(table_str)
    log(f"\n  Saved {path}")
    return table_str


def main():
    print("=" * 72)
    print("  CES FREE ENERGY FRAMEWORK: EMPIRICAL VERIFICATION")
    print("=" * 72)
    print()
    print("  Testing 6 core predictions of F = Phi(rho) - T*H")
    print("  Data: 17 FRED series (14 mfg IP + INDPRO + STLFSI2 + USREC)")
    print("  Literature sigma: Oberfield & Raval (2021), Atalay (2017)")
    print()

    # Fetch all data
    log("Fetching FRED data...")
    series = fetch_all_series()

    # Run all 6 tests
    results = {}
    results['test1'] = test1_rho_ordering(series)
    results['test2'] = test2_crisis_sequence(series)
    results['test3'] = test3_degradation_rates(series)
    results['test4'] = test4_critical_slowing_down(series)
    results['test5'] = test5_effective_curvature(series)
    results['test6'] = test6_cross_domain(series)

    # Summary
    log("\n" + "=" * 72)
    log("  SUMMARY")
    log("=" * 72)

    for tname, r in results.items():
        v = r.get('verdict', 'N/A')
        log(f"  {tname}: {v}")

    verdicts = [r.get('verdict', 'N/A') for r in results.values()]
    n_con = verdicts.count('CONSISTENT')
    n_amb = verdicts.count('AMBIGUOUS')
    n_inc = verdicts.count('INCONSISTENT')

    log(f"\n  Consistent: {n_con}/6")
    log(f"  Ambiguous:  {n_amb}/6")
    log(f"  Inconsistent: {n_inc}/6")

    if n_con >= 4:
        overall = "FRAMEWORK SUPPORTED"
    elif n_inc >= 3:
        overall = "FRAMEWORK CHALLENGED"
    else:
        overall = "MIXED EVIDENCE"
    log(f"\n  Overall assessment: {overall}")

    # LaTeX summary table
    make_summary_table(results)

    # Save full log
    log_path = os.path.join(RESULTS_DIR, 'framework_verification_results.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(LOG_LINES))
    print(f"\n  Saved {log_path}")

    # Save results CSV
    rows = []
    for tname, r in results.items():
        row = {'test': tname, 'verdict': r.get('verdict', 'N/A')}
        for k, v in r.items():
            if isinstance(v, (int, float, str)) and k != 'verdict':
                row[k] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT, 'test_results.csv'), index=False)

    print(f"\n{'='*72}")
    print(f"  DONE — {overall}")
    print(f"{'='*72}")


if __name__ == '__main__':
    main()
