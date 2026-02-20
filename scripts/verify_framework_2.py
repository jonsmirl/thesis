#!/usr/bin/env python3
"""
Empirical Verification of the CES Free Energy Framework — Round 2
==================================================================

Six additional tests probing deeper structural predictions of the
CES free energy framework F = Phi(rho) - T*H.

Tests 7-12 focus on:
  7. Hysteresis: correlation spikes faster than it recovers
  8. Eigenvalue concentration: first PC explains more during crises
  9. Granger causality: stress → correlation, not reverse
  10. Recovery ordering: high-sigma sectors recover first
  11. Hierarchical lead-lag: tech sector leads aggregate
  12. PC1 loading ordering: lower sigma → higher PC1 loading

Data: Same 17 FRED series cached by verify_framework.py
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import requests
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.tsa.filters.hp_filter import hpfilter
except ImportError:
    print("ERROR: pip install statsmodels"); sys.exit(1)

try:
    from scipy.stats import spearmanr, kendalltau
    from scipy.linalg import eigh
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
# SECTION 0: Setup (shared with verify_framework.py)
# ============================================================================

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, 'thesis_data', 'framework_verification')
FIGDIR = os.path.join(BASE, 'figures', 'framework_verification')
RESULTS_DIR = os.path.join(BASE, 'thesis_data')
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

FRED_API_KEY = os.environ.get('FRED_API_KEY', '')

LOG_LINES = []

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    LOG_LINES.append(line)
    print(line)


def fetch_fred(series_id, start='1970-01-01', end='2025-12-31'):
    """Fetch FRED series with CSV caching and retry logic."""
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
    series = {}
    all_ids = list(SECTORS.keys()) + ['INDPRO', 'STLFSI2', 'USREC']
    for sid in all_ids:
        log(f"  Fetching {sid}...")
        series[sid] = fetch_fred(sid)
    return series


def build_sector_panel(series):
    frames = {}
    for sid in SECTORS:
        df = series.get(sid, pd.DataFrame())
        if len(df) == 0:
            continue
        s = df['value'].resample('MS').last()
        g = np.log(s).diff()
        frames[sid] = g
    return pd.DataFrame(frames).dropna(how='all')


def build_level_panel(series):
    """Build panel of IP index levels (not growth rates)."""
    frames = {}
    for sid in SECTORS:
        df = series.get(sid, pd.DataFrame())
        if len(df) == 0:
            continue
        frames[sid] = df['value'].resample('MS').last()
    return pd.DataFrame(frames).dropna(how='all')


def rolling_avg_correlation(panel, window=6):
    """Compute rolling average pairwise correlation."""
    corr_dict = {}
    for i in range(window - 1, len(panel)):
        chunk = panel.iloc[i - window + 1:i + 1].dropna(axis=1, how='any')
        if chunk.shape[1] >= 5:
            cm = chunk.corr()
            mask = np.triu(np.ones(cm.shape, dtype=bool), k=1)
            corr_dict[panel.index[i]] = cm.values[mask].mean()
    return pd.Series(corr_dict, name='avg_corr')


# ============================================================================
# TEST 7: Hysteresis — Asymmetric Correlation Dynamics
# ============================================================================

def test7_hysteresis(series):
    """
    Prediction: The free energy landscape has a steep descent (fast
    correlation spike during crisis) but gradual ascent (slow recovery).
    Correlation should rise faster than it falls.
    """
    log("\n" + "=" * 72)
    log("  TEST 7: Hysteresis — Asymmetric Correlation Dynamics")
    log("=" * 72)
    log("  Prediction: correlation spikes faster than it recovers")

    panel = build_sector_panel(series)
    corr_s = rolling_avg_correlation(panel, window=6)

    # Analyze around each recession
    recessions = {
        '2001': {'pre': '1999-01-01', 'start': '2001-03-01',
                 'trough': '2001-11-01', 'post': '2004-01-01'},
        '2008': {'pre': '2006-01-01', 'start': '2007-12-01',
                 'trough': '2009-06-01', 'post': '2012-01-01'},
        '2020': {'pre': '2019-01-01', 'start': '2020-02-01',
                 'trough': '2020-04-01', 'post': '2022-01-01'},
    }

    rise_rates = []
    fall_rates = []
    rec_labels = []

    for label, dates in recessions.items():
        pre_corr = corr_s.loc[dates['pre']:dates['start']]
        crisis_corr = corr_s.loc[dates['start']:dates['trough']]
        recovery_corr = corr_s.loc[dates['trough']:dates['post']]

        if len(pre_corr) < 3 or len(crisis_corr) < 2 or len(recovery_corr) < 3:
            log(f"  {label}: insufficient data, skipping")
            continue

        baseline = pre_corr.mean()
        peak = corr_s.loc[dates['start']:dates['post']].max()

        # Rise: baseline to peak
        rise_months = (corr_s.loc[dates['start']:dates['post']].idxmax() -
                       pd.Timestamp(dates['start'])).days / 30.44
        rise_amplitude = peak - baseline

        # Fall: peak to return-to-baseline
        post_peak = corr_s.loc[corr_s.loc[dates['start']:dates['post']].idxmax():dates['post']]
        returned = post_peak[post_peak <= baseline + 0.5 * rise_amplitude]
        if len(returned) > 0:
            fall_months = (returned.index[0] - post_peak.index[0]).days / 30.44
        else:
            fall_months = (pd.Timestamp(dates['post']) - post_peak.index[0]).days / 30.44

        rise_rate = rise_amplitude / max(rise_months, 0.5)
        fall_rate = rise_amplitude / max(fall_months, 0.5)

        rise_rates.append(rise_rate)
        fall_rates.append(fall_rate)
        rec_labels.append(label)

        log(f"\n  {label} recession:")
        log(f"    Baseline corr: {baseline:.4f}, Peak corr: {peak:.4f}")
        log(f"    Rise: {rise_amplitude:.4f} in {rise_months:.1f} months (rate={rise_rate:.4f}/mo)")
        log(f"    Fall: {rise_amplitude:.4f} in {fall_months:.1f} months (rate={fall_rate:.4f}/mo)")
        log(f"    Asymmetry ratio (rise/fall): {rise_rate/fall_rate:.2f}")

    if len(rise_rates) == 0:
        log("  FAILED: no recession data")
        return {'verdict': 'FAILED'}

    # Test: rise rate > fall rate consistently
    asymmetry_ratios = [r / f for r, f in zip(rise_rates, fall_rates)]
    avg_ratio = np.mean(asymmetry_ratios)
    n_faster_rise = sum(1 for r in asymmetry_ratios if r > 1.0)

    log(f"\n  Average asymmetry ratio: {avg_ratio:.2f}")
    log(f"  Recessions with faster rise: {n_faster_rise}/{len(asymmetry_ratios)}")

    if n_faster_rise >= 2 and avg_ratio > 1.5:
        verdict = 'CONSISTENT'
    elif n_faster_rise == 0:
        verdict = 'INCONSISTENT'
    else:
        verdict = 'AMBIGUOUS'
    log(f"  Verdict: {verdict}")

    # Figure
    if HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for idx, (label, dates) in enumerate(recessions.items()):
            ax = axes[idx]
            window = corr_s.loc[dates['pre']:dates['post']]
            if len(window) == 0:
                continue
            ax.plot(window.index, window.values, 'b-', linewidth=1.5)
            ax.axvline(pd.Timestamp(dates['start']), color='red', ls='--', alpha=0.7,
                       label='Recession start')
            ax.axvline(pd.Timestamp(dates['trough']), color='green', ls='--', alpha=0.7,
                       label='Recession trough')

            pre_corr = corr_s.loc[dates['pre']:dates['start']]
            if len(pre_corr) > 0:
                ax.axhline(pre_corr.mean(), color='gray', ls=':', alpha=0.5,
                           label=f'Baseline ({pre_corr.mean():.3f})')

            ax.set_title(f'{label} Recession', fontsize=11, fontweight='bold')
            ax.set_ylabel('Avg pairwise correlation', fontsize=10)
            ax.legend(fontsize=7, loc='upper left')
            ax.grid(True, alpha=0.2)

            if idx < len(asymmetry_ratios):
                ax.annotate(f'Asym. ratio: {asymmetry_ratios[idx]:.2f}',
                            xy=(0.98, 0.02), xycoords='axes fraction', fontsize=9,
                            ha='right', bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))

        fig.suptitle('Test 7: Hysteresis — Correlation Spike vs Recovery',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        for ext in ['.png', '.pdf']:
            plt.savefig(os.path.join(FIGDIR, f'test7_hysteresis{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close()
        log("  Saved figures/framework_verification/test7_hysteresis.png/.pdf")

    return {'verdict': verdict, 'avg_asymmetry': avg_ratio,
            'n_faster_rise': n_faster_rise, 'n_recessions': len(asymmetry_ratios)}


# ============================================================================
# TEST 8: Eigenvalue Concentration During Crises
# ============================================================================

def test8_eigenvalue_concentration(series):
    """
    Prediction: During crises, the first eigenvalue of the sector
    correlation matrix absorbs a larger share of total variance
    (all sectors move together = loss of diversification = K_eff → 0).
    """
    log("\n" + "=" * 72)
    log("  TEST 8: Eigenvalue Concentration During Crises")
    log("=" * 72)
    log("  Prediction: first eigenvalue share spikes during crises")

    panel = build_sector_panel(series)
    stlfsi = series.get('STLFSI2', pd.DataFrame())

    if len(stlfsi) == 0:
        log("  FAILED: STLFSI2 data missing")
        return {'verdict': 'FAILED'}

    stress = stlfsi['value'].resample('MS').last()

    # Rolling PCA: compute first eigenvalue share over 12-month windows
    window = 12
    ev_share = {}
    for i in range(window - 1, len(panel)):
        chunk = panel.iloc[i - window + 1:i + 1].dropna(axis=1, how='any')
        if chunk.shape[1] < 5:
            continue
        # Correlation matrix eigenvalues
        cm = chunk.corr().values
        try:
            eigenvalues = np.sort(np.linalg.eigvalsh(cm))[::-1]
            share = eigenvalues[0] / eigenvalues.sum()
            ev_share[panel.index[i]] = share
        except np.linalg.LinAlgError:
            continue

    ev_series = pd.Series(ev_share, name='ev1_share')

    # Align with stress
    combined = pd.DataFrame({'stress': stress, 'ev1_share': ev_series}).dropna()

    if len(combined) < 30:
        log("  FAILED: insufficient aligned data")
        return {'verdict': 'FAILED'}

    # Regression: eigenvalue share on stress
    X = sm.add_constant(combined['stress'].values)
    y = combined['ev1_share'].values
    res = sm.OLS(y, X).fit(cov_type='HC1')

    log(f"\n  N observations: {len(combined)}")
    log(f"  Mean EV1 share: {y.mean():.4f}")
    log(f"  Stress coefficient: {res.params[1]:.4f} (p = {res.pvalues[1]:.4f})")
    log(f"  R^2: {res.rsquared:.4f}")

    # Compare crisis vs calm periods
    calm = combined[combined['stress'] < 0]['ev1_share']
    crisis = combined[combined['stress'] > 1.0]['ev1_share']
    if len(calm) > 5 and len(crisis) > 5:
        log(f"\n  Calm periods (T<0): mean EV1 share = {calm.mean():.4f} (N={len(calm)})")
        log(f"  Crisis periods (T>1): mean EV1 share = {crisis.mean():.4f} (N={len(crisis)})")
        log(f"  Increase: {(crisis.mean() - calm.mean()) / calm.mean():.1%}")

    if res.params[1] > 0 and res.pvalues[1] < 0.05:
        verdict = 'CONSISTENT'
    elif res.params[1] <= 0:
        verdict = 'INCONSISTENT'
    else:
        verdict = 'AMBIGUOUS'
    log(f"  Verdict: {verdict}")

    # Figure
    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left: time series
        ax1.plot(ev_series.index, ev_series.values, 'b-', linewidth=1, alpha=0.7)
        ax1.axhline(y.mean(), color='gray', ls='--', alpha=0.5, label=f'Mean ({y.mean():.3f})')
        # Shade recession periods
        usrec = series.get('USREC', pd.DataFrame())
        if len(usrec) > 0:
            rec = usrec['value'].resample('MS').last()
            rec_aligned = rec.reindex(ev_series.index).fillna(0)
            ax1.fill_between(ev_series.index,
                             ev_series.index.map(lambda x: 0),
                             ev_series.index.map(lambda x: 1),
                             where=rec_aligned.reindex(ev_series.index).fillna(0) > 0.5,
                             alpha=0.15, color='red', label='NBER recession')
        ax1.set_ylabel('First eigenvalue share', fontsize=10)
        ax1.set_title('EV1 share over time', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.2)

        # Right: scatter vs stress
        ax2.scatter(combined['stress'], combined['ev1_share'], s=15, alpha=0.4,
                    c='steelblue', edgecolors='none')
        T_sort = np.sort(combined['stress'].values)
        ax2.plot(T_sort, res.params[0] + res.params[1] * T_sort, 'r-', linewidth=2,
                 label=f'OLS: $\\beta$={res.params[1]:.4f} (p={res.pvalues[1]:.4f})')
        ax2.set_xlabel('STLFSI2 (financial stress)', fontsize=10)
        ax2.set_ylabel('First eigenvalue share', fontsize=10)
        ax2.set_title('EV1 share vs stress', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2)

        fig.suptitle('Test 8: Eigenvalue Concentration During Crises',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        for ext in ['.png', '.pdf']:
            plt.savefig(os.path.join(FIGDIR, f'test8_eigenvalue{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close()
        log("  Saved figures/framework_verification/test8_eigenvalue.png/.pdf")

    return {'verdict': verdict, 'stress_beta': res.params[1],
            'stress_p': res.pvalues[1], 'r2': res.rsquared}


# ============================================================================
# TEST 9: Granger Causality — Stress → Correlation
# ============================================================================

def test9_granger_causality(series):
    """
    Prediction: Financial stress (T) drives correlation changes, not
    reverse. In the framework, T is the exogenous "information temperature"
    that modulates K_eff = K(1 - T/T*)+.
    """
    log("\n" + "=" * 72)
    log("  TEST 9: Granger Causality — Stress → Correlation")
    log("=" * 72)
    log("  Prediction: stress Granger-causes correlation, not reverse")

    panel = build_sector_panel(series)
    stlfsi = series.get('STLFSI2', pd.DataFrame())

    if len(stlfsi) == 0:
        log("  FAILED: STLFSI2 data missing")
        return {'verdict': 'FAILED'}

    corr_s = rolling_avg_correlation(panel, window=6)
    stress = stlfsi['value'].resample('MS').last()

    combined = pd.DataFrame({'stress': stress, 'correlation': corr_s}).dropna()

    if len(combined) < 50:
        log("  FAILED: insufficient aligned data")
        return {'verdict': 'FAILED'}

    log(f"\n  N observations: {len(combined)}")

    # Test both directions at multiple lags
    max_lag = 6
    stress_causes_corr_pvals = []
    corr_causes_stress_pvals = []

    for lag in range(1, max_lag + 1):
        try:
            # Stress → Correlation
            gc1 = grangercausalitytests(
                combined[['correlation', 'stress']].values, maxlag=lag, verbose=False)
            p1 = gc1[lag][0]['ssr_ftest'][1]
            stress_causes_corr_pvals.append(p1)

            # Correlation → Stress
            gc2 = grangercausalitytests(
                combined[['stress', 'correlation']].values, maxlag=lag, verbose=False)
            p2 = gc2[lag][0]['ssr_ftest'][1]
            corr_causes_stress_pvals.append(p2)

            log(f"  Lag {lag}: Stress→Corr p={p1:.4f}, Corr→Stress p={p2:.4f}")
        except Exception as e:
            log(f"  Lag {lag}: error — {e}")
            stress_causes_corr_pvals.append(1.0)
            corr_causes_stress_pvals.append(1.0)

    # Verdict: stress should Granger-cause correlation at some lag
    min_p_forward = min(stress_causes_corr_pvals) if stress_causes_corr_pvals else 1.0
    min_p_reverse = min(corr_causes_stress_pvals) if corr_causes_stress_pvals else 1.0

    forward_sig = min_p_forward < 0.05
    reverse_sig = min_p_reverse < 0.05

    log(f"\n  Min p-value (Stress→Corr): {min_p_forward:.4f} {'***' if min_p_forward < 0.001 else '**' if min_p_forward < 0.01 else '*' if min_p_forward < 0.05 else ''}")
    log(f"  Min p-value (Corr→Stress): {min_p_reverse:.4f} {'***' if min_p_reverse < 0.001 else '**' if min_p_reverse < 0.01 else '*' if min_p_reverse < 0.05 else ''}")

    if forward_sig and not reverse_sig:
        verdict = 'CONSISTENT'
        log("  Direction: unidirectional Stress → Correlation")
    elif forward_sig and reverse_sig:
        verdict = 'AMBIGUOUS'
        log("  Direction: bidirectional (feedback)")
    elif not forward_sig and reverse_sig:
        verdict = 'INCONSISTENT'
        log("  Direction: reverse Correlation → Stress")
    else:
        verdict = 'AMBIGUOUS'
        log("  Direction: no significant Granger causality")
    log(f"  Verdict: {verdict}")

    # Figure
    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        lags = list(range(1, max_lag + 1))
        ax1.bar(np.array(lags) - 0.15, stress_causes_corr_pvals, width=0.3,
                label='Stress → Corr', color='steelblue', alpha=0.8)
        ax1.bar(np.array(lags) + 0.15, corr_causes_stress_pvals, width=0.3,
                label='Corr → Stress', color='coral', alpha=0.8)
        ax1.axhline(0.05, color='red', ls='--', alpha=0.7, label='p = 0.05')
        ax1.set_xlabel('Lag (months)', fontsize=10)
        ax1.set_ylabel('p-value (F-test)', fontsize=10)
        ax1.set_title('Granger Causality p-values', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.set_xticks(lags)
        ax1.grid(True, alpha=0.2)

        # Time series comparison
        ax2.plot(combined.index, combined['stress'] / combined['stress'].std(),
                 'r-', alpha=0.6, linewidth=1, label='Stress (normalized)')
        ax2.plot(combined.index, combined['correlation'] / combined['correlation'].std(),
                 'b-', alpha=0.6, linewidth=1, label='Correlation (normalized)')
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylabel('Normalized value', fontsize=10)
        ax2.set_title('Stress vs Correlation time series', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2)

        fig.suptitle('Test 9: Granger Causality — Stress → Correlation',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        for ext in ['.png', '.pdf']:
            plt.savefig(os.path.join(FIGDIR, f'test9_granger{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close()
        log("  Saved figures/framework_verification/test9_granger.png/.pdf")

    return {'verdict': verdict, 'min_p_forward': min_p_forward,
            'min_p_reverse': min_p_reverse,
            'forward_pvals': stress_causes_corr_pvals,
            'reverse_pvals': corr_causes_stress_pvals}


# ============================================================================
# TEST 10: Recovery Ordering by Sigma
# ============================================================================

def test10_recovery_ordering(series):
    """
    Prediction: Post-crisis, high-sigma (low complementarity) sectors
    recover faster because they require less coordination to restart.
    Low-complementarity sectors can operate more independently.
    """
    log("\n" + "=" * 72)
    log("  TEST 10: Recovery Ordering by Sigma")
    log("=" * 72)
    log("  Prediction: higher sigma -> faster recovery (less coordination needed)")

    levels = build_level_panel(series)

    recessions = {
        '2001': {'trough': '2001-11-01', 'search_end': '2004-12-01'},
        '2008': {'trough': '2009-06-01', 'search_end': '2014-12-01'},
        '2020': {'trough': '2020-04-01', 'search_end': '2022-12-01'},
    }

    all_results = {}

    for rec_label, dates in recessions.items():
        log(f"\n  {rec_label} recession:")
        sigmas = []
        recovery_months = []
        labels = []

        for sid, (name, sigma) in SECTORS.items():
            if sid not in levels.columns:
                continue

            ip = levels[sid]
            trough_date = pd.Timestamp(dates['trough'])

            # Find pre-recession peak (12 months before trough)
            pre_window = ip.loc[trough_date - pd.DateOffset(months=18):trough_date]
            if len(pre_window) < 3:
                continue
            pre_peak = pre_window.max()

            # Find trough value
            trough_val = ip.loc[trough_date - pd.DateOffset(months=3):
                                trough_date + pd.DateOffset(months=3)].min()

            # Recovery target: 90% of pre-peak level
            target = trough_val + 0.9 * (pre_peak - trough_val)

            # Find first month above target after trough
            post = ip.loc[trough_date:dates['search_end']]
            recovered = post[post >= target]

            if len(recovered) > 0:
                months = (recovered.index[0] - trough_date).days / 30.44
            else:
                months = (pd.Timestamp(dates['search_end']) - trough_date).days / 30.44

            sigmas.append(sigma)
            recovery_months.append(months)
            labels.append(name)

        if len(sigmas) < 5:
            log(f"    Insufficient data ({len(sigmas)} sectors)")
            all_results[rec_label] = {'rho_s': np.nan, 'p': np.nan, 'verdict': 'FAILED'}
            continue

        # Negative Spearman = higher sigma recovers faster (fewer months)
        rho_s, p_val = spearmanr(sigmas, recovery_months)
        log(f"    Sectors: {len(sigmas)}")
        log(f"    Spearman rho(sigma, recovery_months) = {rho_s:.4f}, p = {p_val:.4f}")
        log(f"    (negative = higher sigma recovers faster = CONSISTENT)")

        if rho_s < 0 and p_val < 0.10:
            v = 'CONSISTENT'
        elif rho_s > 0 and p_val < 0.10:
            v = 'INCONSISTENT'
        else:
            v = 'AMBIGUOUS'
        log(f"    Verdict: {v}")

        all_results[rec_label] = {
            'rho_s': rho_s, 'p': p_val, 'verdict': v,
            'sigmas': sigmas, 'months': recovery_months, 'labels': labels
        }

    consistent_count = sum(1 for r in all_results.values() if r.get('verdict') == 'CONSISTENT')
    if consistent_count >= 2:
        overall = 'CONSISTENT'
    elif consistent_count == 0 and any(r.get('verdict') == 'INCONSISTENT' for r in all_results.values()):
        overall = 'INCONSISTENT'
    else:
        overall = 'AMBIGUOUS'
    log(f"\n  Overall ({consistent_count}/3 consistent): {overall}")

    # Figure
    if HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for idx, (rec_label, r) in enumerate(all_results.items()):
            ax = axes[idx]
            if 'sigmas' not in r:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title(rec_label, fontsize=11)
                continue

            ax.scatter(r['sigmas'], r['months'], s=60, alpha=0.8,
                       edgecolors='k', linewidths=0.5, c='steelblue', zorder=3)
            for i, lbl in enumerate(r['labels']):
                ax.annotate(lbl, (r['sigmas'][i], r['months'][i]),
                            fontsize=6, xytext=(4, 4), textcoords='offset points')

            # OLS trend
            X = sm.add_constant(np.array(r['sigmas']))
            y = np.array(r['months'])
            ols = sm.OLS(y, X).fit()
            xs = np.linspace(min(r['sigmas']) - 0.05, max(r['sigmas']) + 0.05, 100)
            ax.plot(xs, ols.params[0] + ols.params[1] * xs, 'r--', alpha=0.7)

            ax.set_xlabel(r'Literature $\sigma$', fontsize=10)
            ax.set_ylabel('Months to 90% recovery', fontsize=10)
            ax.set_title(f"{rec_label}\n$\\rho_s$={r['rho_s']:.3f} (p={r['p']:.3f}) — {r['verdict']}",
                         fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2)

        fig.suptitle('Test 10: Recovery Ordering by $\\sigma$',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        for ext in ['.png', '.pdf']:
            plt.savefig(os.path.join(FIGDIR, f'test10_recovery{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close()
        log("  Saved figures/framework_verification/test10_recovery.png/.pdf")

    return {'verdict': overall, 'consistent_count': consistent_count,
            'details': {k: {kk: vv for kk, vv in v.items()
                           if kk not in ('sigmas', 'months', 'labels')}
                       for k, v in all_results.items()}}


# ============================================================================
# TEST 11: Hierarchical Lead-Lag
# ============================================================================

def test11_hierarchical_leadlag(series):
    """
    Prediction: In the four-level hierarchy, faster levels lead slower
    ones. Computer/Electronics (highest sigma = 1.40, most substitutable,
    fastest adjustment) should lead heavy industry (lowest sigma, slowest).
    The tech sector's growth rate changes should precede aggregate changes.
    """
    log("\n" + "=" * 72)
    log("  TEST 11: Hierarchical Lead-Lag")
    log("=" * 72)
    log("  Prediction: high-sigma sectors lead low-sigma sectors")

    panel = build_sector_panel(series)

    # Sort sectors by sigma
    sorted_sids = sorted(SECTORS.keys(), key=lambda s: SECTORS[s][1])

    # Group into fast (top 4 sigma) and slow (bottom 4 sigma)
    fast_sids = [s for s in sorted_sids[-4:] if s in panel.columns]
    slow_sids = [s for s in sorted_sids[:4] if s in panel.columns]

    if len(fast_sids) < 2 or len(slow_sids) < 2:
        log("  FAILED: insufficient sector data")
        return {'verdict': 'FAILED'}

    fast_names = [SECTORS[s][0] for s in fast_sids]
    slow_names = [SECTORS[s][0] for s in slow_sids]
    log(f"  Fast group (high sigma): {fast_names}")
    log(f"  Slow group (low sigma): {slow_names}")

    # Average growth rate for each group
    fast_avg = panel[fast_sids].mean(axis=1).dropna()
    slow_avg = panel[slow_sids].mean(axis=1).dropna()

    # Align
    aligned = pd.DataFrame({'fast': fast_avg, 'slow': slow_avg}).dropna()

    if len(aligned) < 60:
        log("  FAILED: insufficient aligned data")
        return {'verdict': 'FAILED'}

    # Cross-correlation function
    max_lag = 12
    ccf_vals = []
    lags = list(range(-max_lag, max_lag + 1))

    for lag in lags:
        if lag > 0:
            # Fast leads slow: correlate fast[:-lag] with slow[lag:]
            c = np.corrcoef(aligned['fast'].values[:-lag],
                            aligned['slow'].values[lag:])[0, 1]
        elif lag < 0:
            # Slow leads fast
            c = np.corrcoef(aligned['fast'].values[-lag:],
                            aligned['slow'].values[:lag])[0, 1]
        else:
            c = np.corrcoef(aligned['fast'].values, aligned['slow'].values)[0, 1]
        ccf_vals.append(c)

    peak_lag = lags[np.argmax(ccf_vals)]
    peak_corr = max(ccf_vals)

    log(f"\n  Cross-correlation peak at lag = {peak_lag} months")
    log(f"  Peak correlation = {peak_corr:.4f}")
    log(f"  (positive lag = fast leads slow)")

    # Also test individual sector pairs with Granger causality
    # Computer/Elec (IPG334S, sigma=1.40) → Transport (IPG336S, sigma=0.50)
    tech_sid = 'IPG334S'
    heavy_sid = 'IPG336S'
    if tech_sid in panel.columns and heavy_sid in panel.columns:
        pair = pd.DataFrame({
            'heavy': panel[heavy_sid],
            'tech': panel[tech_sid]
        }).dropna()

        if len(pair) > 50:
            try:
                gc_forward = grangercausalitytests(
                    pair[['heavy', 'tech']].values, maxlag=3, verbose=False)
                gc_reverse = grangercausalitytests(
                    pair[['tech', 'heavy']].values, maxlag=3, verbose=False)

                p_fwd = min(gc_forward[lag][0]['ssr_ftest'][1] for lag in range(1, 4))
                p_rev = min(gc_reverse[lag][0]['ssr_ftest'][1] for lag in range(1, 4))

                log(f"\n  Granger: Computer/Elec → Transport: min p = {p_fwd:.4f}")
                log(f"  Granger: Transport → Computer/Elec: min p = {p_rev:.4f}")
            except Exception as e:
                log(f"  Granger test error: {e}")
                p_fwd, p_rev = 1.0, 1.0
        else:
            p_fwd, p_rev = 1.0, 1.0
    else:
        p_fwd, p_rev = 1.0, 1.0

    # Verdict: peak lag should be positive (fast leads slow)
    if peak_lag > 0:
        verdict = 'CONSISTENT'
    elif peak_lag == 0:
        verdict = 'AMBIGUOUS'
    else:
        verdict = 'INCONSISTENT'
    log(f"  Verdict: {verdict}")

    # Figure
    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.bar(lags, ccf_vals, color='steelblue', alpha=0.7, width=0.8)
        ax1.axvline(peak_lag, color='red', ls='--', alpha=0.7,
                    label=f'Peak at lag={peak_lag}')
        ax1.axvline(0, color='gray', ls='-', alpha=0.3)
        ax1.set_xlabel('Lag (months, positive = fast leads)', fontsize=10)
        ax1.set_ylabel('Cross-correlation', fontsize=10)
        ax1.set_title('Fast vs Slow sector cross-correlation', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.2)

        # Smoothed time series comparison
        fast_smooth = fast_avg.rolling(6).mean()
        slow_smooth = slow_avg.rolling(6).mean()
        ax2.plot(fast_smooth.index, fast_smooth.values, 'b-', linewidth=1.5,
                 alpha=0.7, label=f'Fast group (6-mo avg)')
        ax2.plot(slow_smooth.index, slow_smooth.values, 'r-', linewidth=1.5,
                 alpha=0.7, label=f'Slow group (6-mo avg)')
        ax2.axhline(0, color='gray', ls='--', alpha=0.3)
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylabel('Avg growth rate', fontsize=10)
        ax2.set_title('Fast vs Slow sector growth rates', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2)

        fig.suptitle('Test 11: Hierarchical Lead-Lag (High-$\\sigma$ leads Low-$\\sigma$)',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        for ext in ['.png', '.pdf']:
            plt.savefig(os.path.join(FIGDIR, f'test11_leadlag{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close()
        log("  Saved figures/framework_verification/test11_leadlag.png/.pdf")

    return {'verdict': verdict, 'peak_lag': peak_lag, 'peak_corr': peak_corr,
            'granger_p_fwd': p_fwd, 'granger_p_rev': p_rev}


# ============================================================================
# TEST 12: PC1 Loading Ordered by Sigma
# ============================================================================

def test12_pc1_loading(series):
    """
    Prediction: Sectors with lower sigma (higher complementarity) should
    have higher loadings on the first principal component, because
    complementary sectors are more tightly coupled to the aggregate.
    """
    log("\n" + "=" * 72)
    log("  TEST 12: PC1 Loading Ordered by Sigma")
    log("=" * 72)
    log("  Prediction: lower sigma -> higher PC1 loading (more coupled)")

    panel = build_sector_panel(series)
    clean = panel.dropna()

    if clean.shape[1] < 8:
        log("  FAILED: insufficient sector data")
        return {'verdict': 'FAILED'}

    # Standardize
    standardized = (clean - clean.mean()) / clean.std()

    # PCA via correlation matrix
    corr_matrix = standardized.corr().values
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # PC1 loadings (absolute value — sign is arbitrary)
    pc1_loadings = np.abs(eigenvectors[:, 0])
    sector_ids = list(clean.columns)

    log(f"\n  Sectors in PCA: {len(sector_ids)}")
    log(f"  Variance explained by PC1: {eigenvalues[0] / eigenvalues.sum():.1%}")
    log(f"  Variance explained by PC1-2: {eigenvalues[:2].sum() / eigenvalues.sum():.1%}")

    sigmas = []
    loadings = []
    labels = []
    for i, sid in enumerate(sector_ids):
        if sid in SECTORS:
            sigmas.append(SECTORS[sid][1])
            loadings.append(pc1_loadings[i])
            labels.append(SECTORS[sid][0])

    if len(sigmas) < 5:
        log("  FAILED: insufficient matched sectors")
        return {'verdict': 'FAILED'}

    rho_s, p_val = spearmanr(sigmas, loadings)
    log(f"\n  Spearman rho(sigma, |PC1 loading|) = {rho_s:.4f}, p = {p_val:.4f}")
    log(f"  (negative = lower sigma has higher loading = CONSISTENT)")

    # Print sector loadings
    log(f"\n  {'Sector':<20} {'sigma':>6} {'|PC1|':>7}")
    log(f"  {'-'*35}")
    sorted_idx = np.argsort(sigmas)
    for i in sorted_idx:
        log(f"  {labels[i]:<20} {sigmas[i]:>6.2f} {loadings[i]:>7.4f}")

    if rho_s < 0 and p_val < 0.10:
        verdict = 'CONSISTENT'
    elif rho_s > 0 and p_val < 0.10:
        verdict = 'INCONSISTENT'
    else:
        verdict = 'AMBIGUOUS'
    log(f"  Verdict: {verdict}")

    # Also check during crisis vs calm periods
    calm_data = clean.loc[:'2007-01-01']
    crisis_data = clean.loc['2008-01-01':'2009-12-01']

    crisis_results = {}
    for period_label, period_data in [('Calm (pre-2007)', calm_data),
                                       ('Crisis (2008-09)', crisis_data)]:
        if len(period_data) < 12:
            continue
        std_p = (period_data - period_data.mean()) / period_data.std()
        cm = std_p.corr().values
        ev, evec = np.linalg.eigh(cm)
        idx_p = np.argsort(ev)[::-1]
        evec = evec[:, idx_p]
        pc1_p = np.abs(evec[:, 0])

        p_sigmas = [SECTORS[s][1] for s in period_data.columns if s in SECTORS]
        p_loads = [pc1_p[list(period_data.columns).index(s)]
                   for s in period_data.columns if s in SECTORS]

        if len(p_sigmas) >= 5:
            rho_p, p_p = spearmanr(p_sigmas, p_loads)
            log(f"\n  {period_label}: Spearman rho = {rho_p:.4f}, p = {p_p:.4f}")
            crisis_results[period_label] = {'rho': rho_p, 'p': p_p}

    # Figure
    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.scatter(sigmas, loadings, s=80, alpha=0.8, edgecolors='k',
                    linewidths=0.5, c='steelblue', zorder=3)
        for i, lbl in enumerate(labels):
            ax1.annotate(lbl, (sigmas[i], loadings[i]), fontsize=7,
                         xytext=(5, 5), textcoords='offset points')

        X = sm.add_constant(np.array(sigmas))
        ols = sm.OLS(np.array(loadings), X).fit()
        xs = np.linspace(min(sigmas) - 0.05, max(sigmas) + 0.05, 100)
        ax1.plot(xs, ols.params[0] + ols.params[1] * xs, 'r--', alpha=0.7)

        ax1.set_xlabel(r'Literature $\sigma$', fontsize=10)
        ax1.set_ylabel('|PC1 loading|', fontsize=10)
        ax1.set_title(f'Full sample: $\\rho_s$={rho_s:.3f} (p={p_val:.3f})',
                       fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.2)

        # Eigenvalue scree plot
        n_ev = min(10, len(eigenvalues))
        ax2.bar(range(1, n_ev + 1), eigenvalues[:n_ev] / eigenvalues.sum() * 100,
                color='steelblue', alpha=0.8)
        ax2.set_xlabel('Principal component', fontsize=10)
        ax2.set_ylabel('Variance explained (%)', fontsize=10)
        ax2.set_title('Eigenvalue scree plot', fontsize=11, fontweight='bold')
        ax2.set_xticks(range(1, n_ev + 1))
        ax2.grid(True, alpha=0.2)

        fig.suptitle('Test 12: PC1 Loading Ordered by $\\sigma$',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        for ext in ['.png', '.pdf']:
            plt.savefig(os.path.join(FIGDIR, f'test12_pc1_loading{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close()
        log("  Saved figures/framework_verification/test12_pc1_loading.png/.pdf")

    return {'verdict': verdict, 'rho_s': rho_s, 'p': p_val,
            'var_explained_pc1': eigenvalues[0] / eigenvalues.sum(),
            'crisis_results': crisis_results}


# ============================================================================
# SUMMARY
# ============================================================================

def make_summary_table(results, prev_results=None):
    """Generate combined LaTeX summary table for tests 7-12 (and optionally 1-6)."""
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{CES Free Energy Framework: Extended Verification (Tests 7--12)}')
    lines.append(r'\label{tab:framework_verification_2}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{clllc}')
    lines.append(r'\toprule')
    lines.append(r'Test & Prediction & Key statistic & Value & Verdict \\')
    lines.append(r'\midrule')

    def fmt(val, fmt_str='.3f'):
        return f'{val:{fmt_str}}' if isinstance(val, float) else str(val)

    test_rows = [
        ('7', 'Hysteresis (fast spike, slow recovery)',
         'Avg asymmetry ratio',
         fmt(results.get('test7', {}).get('avg_asymmetry', 'N/A')),
         results.get('test7', {}).get('verdict', 'N/A')),
        ('8', 'EV1 concentration in crises',
         r'Stress $\beta$',
         fmt(results.get('test8', {}).get('stress_beta', 'N/A'), '.4f'),
         results.get('test8', {}).get('verdict', 'N/A')),
        ('9', r'Stress $\to$ correlation (Granger)',
         r'Min $p$ forward',
         fmt(results.get('test9', {}).get('min_p_forward', 'N/A'), '.4f'),
         results.get('test9', {}).get('verdict', 'N/A')),
        ('10', r'High-$\sigma$ recovers faster',
         'Recessions consistent',
         f"{results.get('test10', {}).get('consistent_count', 'N/A')}/3",
         results.get('test10', {}).get('verdict', 'N/A')),
        ('11', r'High-$\sigma$ leads low-$\sigma$',
         'CCF peak lag (months)',
         fmt(results.get('test11', {}).get('peak_lag', 'N/A'), '.0f') if isinstance(results.get('test11', {}).get('peak_lag'), (int, float)) else 'N/A',
         results.get('test11', {}).get('verdict', 'N/A')),
        ('12', r'Lower $\sigma$ $\to$ higher PC1 loading',
         r'Spearman $\rho$',
         fmt(results.get('test12', {}).get('rho_s', 'N/A')),
         results.get('test12', {}).get('verdict', 'N/A')),
    ]

    for num, pred, stat, val, verdict in test_rows:
        v_fmt = r'\textbf{' + verdict + '}' if verdict == 'CONSISTENT' else verdict
        lines.append(f'  {num} & {pred} & {stat} & {val} & {v_fmt} \\\\')

    lines.append(r'\midrule')
    verdicts = [r.get('verdict', 'N/A') for r in results.values()]
    n_con = verdicts.count('CONSISTENT')
    n_amb = verdicts.count('AMBIGUOUS')
    n_inc = verdicts.count('INCONSISTENT')
    lines.append(f'  \\multicolumn{{4}}{{l}}{{Tests 7--12: {n_con} consistent, {n_amb} ambiguous, {n_inc} inconsistent}} & \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\vspace{4pt}')
    lines.append(r'\parbox{0.95\textwidth}{\footnotesize')
    lines.append(r'  Same 14 FRED manufacturing IP subsectors as Tests 1--6.')
    lines.append(r'  Granger causality uses F-test (SSR) at lags 1--6.')
    lines.append(r'  PCA computed on standardized monthly growth rates.')
    lines.append(r'  Recovery measured as months from trough to 90\% of pre-crisis peak.')
    lines.append(r'}')
    lines.append(r'\end{table}')

    table_str = '\n'.join(lines)
    path = os.path.join(RESULTS_DIR, 'framework_verification_2_summary.tex')
    with open(path, 'w') as f:
        f.write(table_str)
    log(f"\n  Saved {path}")
    return table_str


def main():
    print("=" * 72)
    print("  CES FREE ENERGY FRAMEWORK: EXTENDED VERIFICATION (Tests 7-12)")
    print("=" * 72)
    print()
    print("  Testing 6 additional structural predictions")
    print("  Data: Same 17 FRED series (cached from Round 1)")
    print()

    log("Fetching FRED data...")
    series = fetch_all_series()

    results = {}
    results['test7'] = test7_hysteresis(series)
    results['test8'] = test8_eigenvalue_concentration(series)
    results['test9'] = test9_granger_causality(series)
    results['test10'] = test10_recovery_ordering(series)
    results['test11'] = test11_hierarchical_leadlag(series)
    results['test12'] = test12_pc1_loading(series)

    # Summary
    log("\n" + "=" * 72)
    log("  SUMMARY (Tests 7-12)")
    log("=" * 72)

    for tname, r in results.items():
        v = r.get('verdict', 'N/A')
        log(f"  {tname}: {v}")

    verdicts = [r.get('verdict', 'N/A') for r in results.values()]
    n_con = verdicts.count('CONSISTENT')
    n_amb = verdicts.count('AMBIGUOUS')
    n_inc = verdicts.count('INCONSISTENT')

    log(f"\n  Round 2: {n_con} consistent, {n_amb} ambiguous, {n_inc} inconsistent")

    # Load Round 1 results if available
    r1_path = os.path.join(OUTPUT, 'test_results.csv')
    if os.path.exists(r1_path):
        r1 = pd.read_csv(r1_path)
        r1_verdicts = r1['verdict'].tolist()
        all_verdicts = r1_verdicts + verdicts
        total_con = all_verdicts.count('CONSISTENT')
        total_amb = all_verdicts.count('AMBIGUOUS')
        total_inc = all_verdicts.count('INCONSISTENT')
        log(f"\n  Combined (Tests 1-12): {total_con} consistent, {total_amb} ambiguous, {total_inc} inconsistent out of {len(all_verdicts)}")

        if total_con >= 8:
            overall = "STRONG SUPPORT"
        elif total_con >= 6:
            overall = "MODERATE SUPPORT"
        elif total_inc >= 4:
            overall = "FRAMEWORK CHALLENGED"
        else:
            overall = "MIXED EVIDENCE"
        log(f"  Overall assessment: {overall}")
    else:
        if n_con >= 4:
            overall = "FRAMEWORK SUPPORTED"
        elif n_inc >= 3:
            overall = "FRAMEWORK CHALLENGED"
        else:
            overall = "MIXED EVIDENCE"
        log(f"  Overall assessment: {overall}")

    make_summary_table(results)

    # Save full log
    log_path = os.path.join(RESULTS_DIR, 'framework_verification_2_results.txt')
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
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT, 'test_results_2.csv'), index=False)

    print(f"\n{'='*72}")
    print(f"  DONE — {overall}")
    print(f"{'='*72}")


if __name__ == '__main__':
    main()
