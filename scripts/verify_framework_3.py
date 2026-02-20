#!/usr/bin/env python3
"""
Empirical Test: Does Technical Complexity Determine Hierarchy Depth?
====================================================================

The CES free energy framework currently assumes a fixed 4-level hierarchy.
This script tests whether the NUMBER of effective dynamical levels is
endogenous — determined by technical complexity rather than assumed.

Operationalization:
  - Technical complexity proxy: R&D intensity (NSF BRDIS, % of sales)
  - Effective hierarchy depth: number of distinct timescales in the
    dynamical system, measured via Singular Spectrum Analysis (SSA)

Six tests:
  13. Effective dimensionality (SSA) vs R&D intensity
  14. Autocorrelation decay structure vs R&D intensity
  15. Spectral peak count vs R&D intensity
  16. Within-sector hierarchy depth (4-digit NAICS eigenvalues)
  17. Timescale span (ratio of slowest to fastest) vs R&D intensity
  18. Temporal emergence — did Computer/Elec gain levels over time?

Data: FRED API (14 existing + ~30 new 4-digit NAICS IP series)
R&D intensity: NSF NCSES Business R&D Survey 2019
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import requests
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
except ImportError:
    print("ERROR: pip install statsmodels"); sys.exit(1)

try:
    from scipy.stats import spearmanr
    from scipy.signal import find_peaks
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

LOG_LINES = []

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    LOG_LINES.append(line)
    print(line)


def fetch_fred(series_id, start='1972-01-01', end='2025-12-31'):
    cache = os.path.join(OUTPUT, f'{series_id}.csv')
    if os.path.exists(cache):
        df = pd.read_csv(cache, parse_dates=['date'], index_col='date')
        return df

    if not FRED_API_KEY:
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


# Literature sigma values and R&D intensity (NSF NCSES BRDIS 2019)
SECTORS = {
    'IPG311S': ('Food', 0.4, 1.15),
    'IPG321S': ('Wood', 0.2, 0.70),
    'IPG322S': ('Paper', 1.0, 0.85),
    'IPG325S': ('Chemicals', 9.5, 0.90),
    'IPG326S': ('Plastics/Rubber', 1.2, 0.95),
    'IPG327S': ('Nonmetallic Min', 0.7, 0.60),
    'IPG331S': ('Primary Metals', 0.7, 0.55),
    'IPG332S': ('Fabricated Metals', 1.1, 0.70),
    'IPG333S': ('Machinery', 3.8, 0.65),
    'IPG334S': ('Computer/Elec', 14.8, 1.40),
    'IPG335S': ('Electrical Equip', 4.2, 0.70),
    'IPG336S': ('Transport Equip', 4.6, 0.50),
    'IPG337S': ('Furniture', 0.6, 0.85),
    'IPG339S': ('Miscellaneous', 8.2, 1.10),
}
# Format: series_id -> (name, R&D_pct, sigma)

# 4-digit NAICS subsectors for within-sector analysis
SUBSECTORS = {
    'Computer/Elec (334)': {
        'IPG3341S': 'Computer/Peripheral',
        'IPG3342S': 'Communications Equip',
        'IPG3344S': 'Semiconductor',
        'IPG3345S': 'Instruments',
    },
    'Chemicals (325)': {
        'IPG3251S': 'Basic Chemical',
        'IPG3252S': 'Resin/Synthetic',
        'IPG3254S': 'Pharmaceutical',
        'IPG3256S': 'Soap/Cleaning',
    },
    'Transport (336)': {
        'IPG3361S': 'Motor Vehicle',
        'IPG3363S': 'Motor Vehicle Parts',
        'IPG3364S': 'Aerospace',
    },
    'Food (311)': {
        'IPG3111S': 'Animal Food',
        'IPG3112S': 'Grain/Oilseed',
        'IPG3113S': 'Sugar/Confect',
        'IPG3114S': 'Fruit/Veg Preserv',
        'IPG3115S': 'Dairy',
        'IPG3116S': 'Animal Slaughter',
        'IPG3118S': 'Bakeries',
    },
    'Machinery (333)': {
        'IPG3331S': 'Ag/Construct/Mining Mach',
        'IPG3332S': 'Industrial Machinery',
        'IPG3334S': 'HVAC',
        'IPG3336S': 'Engine/Turbine',
    },
}

# R&D intensity for sector groups (used in Test 16)
SECTOR_GROUP_RD = {
    'Computer/Elec (334)': 14.8,
    'Chemicals (325)': 9.5,
    'Transport (336)': 4.6,
    'Machinery (333)': 3.8,
    'Food (311)': 0.4,
}


def fetch_all_series():
    """Fetch all required series (3-digit + 4-digit)."""
    series = {}
    # 3-digit sectors
    for sid in SECTORS:
        series[sid] = fetch_fred(sid)
    # 4-digit subsectors
    for group_name, sids in SUBSECTORS.items():
        for sid in sids:
            if sid not in series:
                series[sid] = fetch_fred(sid)
    return series


def get_growth_rate(series, sid):
    """Get monthly log-difference growth rate for a FRED series."""
    df = series.get(sid, pd.DataFrame())
    if len(df) == 0:
        return pd.Series(dtype=float)
    s = df['value'].resample('MS').last()
    return np.log(s).diff().dropna()


# ============================================================================
# SSA Helpers
# ============================================================================

def hankel_matrix(x, L):
    """Construct trajectory (Hankel) matrix for SSA. L = window length."""
    N = len(x)
    K = N - L + 1
    H = np.zeros((L, K))
    for i in range(K):
        H[:, i] = x[i:i + L]
    return H


def marchenko_pastur_threshold(N, L, noise_var=None):
    """Upper edge of the Marchenko-Pastur distribution for random matrices."""
    gamma = L / N
    if gamma > 1:
        gamma = 1.0 / gamma
    if noise_var is None:
        noise_var = 1.0
    return noise_var * (1 + np.sqrt(gamma)) ** 2


def ssa_effective_dim(x, L=None):
    """
    Compute effective dimensionality via Singular Spectrum Analysis.
    Returns (n_significant, singular_values, threshold).
    """
    x = np.array(x, dtype=float)
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    N = len(x)

    if L is None:
        L = min(N // 3, 48)  # ~4 years for monthly data

    if L < 5 or N < 2 * L:
        return 0, np.array([]), 0.0

    H = hankel_matrix(x, L)
    try:
        U, s, Vt = np.linalg.svd(H, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0, np.array([]), 0.0

    # Eigenvalues of the covariance matrix = s^2 / K
    K = H.shape[1]
    eigenvalues = s ** 2 / K

    # Marchenko-Pastur threshold for noise
    noise_var = np.median(eigenvalues[-max(1, L // 4):])
    mp_upper = marchenko_pastur_threshold(K, L, noise_var)

    n_sig = np.sum(eigenvalues > mp_upper)
    return int(n_sig), eigenvalues, mp_upper


def count_acf_timescales(x, max_lag=48):
    """
    Count distinct timescales from the autocorrelation function.
    A timescale is detected at each zero-crossing or sign change
    of the ACF derivative (local extremum).
    """
    x = np.array(x, dtype=float)
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    N = len(x)

    if N < max_lag + 10:
        return 0, np.array([])

    acf = np.correlate(x, x, mode='full')[N - 1:N - 1 + max_lag + 1]
    acf = acf / acf[0]

    # Find peaks and troughs in the ACF (each corresponds to a timescale)
    peaks, _ = find_peaks(acf, height=0.05, distance=3)
    troughs, _ = find_peaks(-acf, height=0.05, distance=3)

    # Each peak represents a resonant timescale
    n_timescales = len(peaks)

    return n_timescales, acf


def count_spectral_peaks(x, fs=12.0):
    """
    Count significant peaks in the power spectrum.
    fs = 12 for monthly data (12 samples/year).
    """
    x = np.array(x, dtype=float)
    x = (x - np.mean(x))
    N = len(x)

    if N < 60:
        return 0, np.array([]), np.array([])

    # FFT
    fft = np.fft.rfft(x * np.hanning(N))
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # Exclude DC and very high frequencies
    mask = (freqs > 0.05) & (freqs < 4.0)  # ~3 months to ~20 years
    power_m = power[mask]
    freqs_m = freqs[mask]

    if len(power_m) < 5:
        return 0, freqs, power

    # Significance threshold: 3x median power (white noise floor)
    threshold = 3.0 * np.median(power_m)
    peaks, properties = find_peaks(power_m, height=threshold, distance=3,
                                   prominence=threshold * 0.5)

    return len(peaks), freqs, power


# ============================================================================
# TEST 13: Effective Dimensionality (SSA) vs R&D Intensity
# ============================================================================

def test13_ssa_vs_rd(series):
    """
    Prediction: Sectors with higher R&D intensity have higher effective
    dimensionality (more distinct timescales in their dynamics).
    """
    log("\n" + "=" * 72)
    log("  TEST 13: Effective Dimensionality (SSA) vs R&D Intensity")
    log("=" * 72)
    log("  Prediction: higher R&D -> more effective dynamical dimensions")

    rd_vals = []
    dim_vals = []
    labels = []

    log(f"\n  {'Sector':<22} {'R&D%':>5} {'N_eff':>5} {'SV above MP':>15}")
    log(f"  {'-'*50}")

    for sid, (name, rd_pct, sigma) in SECTORS.items():
        g = get_growth_rate(series, sid)
        if len(g) < 120:
            continue

        n_sig, eigenvalues, threshold = ssa_effective_dim(g.values, L=48)

        rd_vals.append(rd_pct)
        dim_vals.append(n_sig)
        labels.append(name)
        log(f"  {name:<22} {rd_pct:>5.1f} {n_sig:>5} {threshold:>10.4f}")

    if len(rd_vals) < 5:
        log("  FAILED: insufficient sector data")
        return {'verdict': 'FAILED'}

    rho_s, p_val = spearmanr(rd_vals, dim_vals)
    log(f"\n  Spearman rho(R&D%, N_eff) = {rho_s:.4f}, p = {p_val:.4f}")

    if rho_s > 0 and p_val < 0.10:
        verdict = 'CONSISTENT'
    elif rho_s < 0 and p_val < 0.10:
        verdict = 'INCONSISTENT'
    else:
        verdict = 'AMBIGUOUS'
    log(f"  Verdict: {verdict}")

    # Figure
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(rd_vals, dim_vals, s=80, alpha=0.8, edgecolors='k',
                   linewidths=0.5, c='steelblue', zorder=3)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (rd_vals[i], dim_vals[i]), fontsize=7,
                        xytext=(5, 5), textcoords='offset points')

        X = sm.add_constant(np.array(rd_vals))
        ols = sm.OLS(np.array(dim_vals, dtype=float), X).fit()
        xs = np.linspace(0, max(rd_vals) + 1, 100)
        ax.plot(xs, ols.params[0] + ols.params[1] * xs, 'r--', alpha=0.7,
                label=f'OLS: slope={ols.params[1]:.3f}')

        ax.set_xlabel('R&D intensity (% of sales, NSF BRDIS 2019)', fontsize=11)
        ax.set_ylabel('SSA effective dimensionality (N_eff)', fontsize=11)
        ax.set_title('Test 13: Effective Dimensionality vs R&D Intensity',
                     fontsize=13, fontweight='bold')
        ax.annotate(f'Spearman $\\rho$ = {rho_s:.3f} (p = {p_val:.3f})\nVerdict: {verdict}',
                    xy=(0.02, 0.98), xycoords='axes fraction', fontsize=10,
                    va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, zorder=1)
        plt.tight_layout()
        for ext in ['.png', '.pdf']:
            plt.savefig(os.path.join(FIGDIR, f'test13_ssa_rd{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close()
        log("  Saved figures/framework_verification/test13_ssa_rd.png/.pdf")

    return {'verdict': verdict, 'rho_s': rho_s, 'p': p_val,
            'sectors': list(zip(labels, rd_vals, dim_vals))}


# ============================================================================
# TEST 14: Autocorrelation Timescale Count vs R&D Intensity
# ============================================================================

def test14_acf_timescales(series):
    """
    Prediction: More complex sectors show more distinct timescale
    peaks in their autocorrelation function.
    """
    log("\n" + "=" * 72)
    log("  TEST 14: ACF Timescale Count vs R&D Intensity")
    log("=" * 72)
    log("  Prediction: higher R&D -> more ACF peaks (more timescales)")

    rd_vals = []
    ts_vals = []
    labels = []
    acf_data = {}

    for sid, (name, rd_pct, sigma) in SECTORS.items():
        g = get_growth_rate(series, sid)
        if len(g) < 120:
            continue

        n_ts, acf = count_acf_timescales(g.values, max_lag=48)

        rd_vals.append(rd_pct)
        ts_vals.append(n_ts)
        labels.append(name)
        acf_data[name] = acf

    if len(rd_vals) < 5:
        log("  FAILED: insufficient data")
        return {'verdict': 'FAILED'}

    log(f"\n  {'Sector':<22} {'R&D%':>5} {'ACF peaks':>10}")
    log(f"  {'-'*40}")
    for i in range(len(labels)):
        log(f"  {labels[i]:<22} {rd_vals[i]:>5.1f} {ts_vals[i]:>10}")

    rho_s, p_val = spearmanr(rd_vals, ts_vals)
    log(f"\n  Spearman rho(R&D%, ACF_peaks) = {rho_s:.4f}, p = {p_val:.4f}")

    if rho_s > 0 and p_val < 0.10:
        verdict = 'CONSISTENT'
    elif rho_s < 0 and p_val < 0.10:
        verdict = 'INCONSISTENT'
    else:
        verdict = 'AMBIGUOUS'
    log(f"  Verdict: {verdict}")

    # Figure: ACFs for highest and lowest R&D sectors
    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left: scatter
        ax1.scatter(rd_vals, ts_vals, s=80, alpha=0.8, edgecolors='k',
                    linewidths=0.5, c='steelblue', zorder=3)
        for i, lbl in enumerate(labels):
            ax1.annotate(lbl, (rd_vals[i], ts_vals[i]), fontsize=7,
                         xytext=(5, 3), textcoords='offset points')
        ax1.set_xlabel('R&D intensity (%)', fontsize=10)
        ax1.set_ylabel('Number of ACF peaks', fontsize=10)
        ax1.set_title(f'ACF peaks vs R&D: $\\rho_s$={rho_s:.3f} (p={p_val:.3f})',
                      fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.2)

        # Right: example ACFs
        high_rd = max(SECTORS.items(), key=lambda x: x[1][1])
        low_rd = min(SECTORS.items(), key=lambda x: x[1][1])
        for sid, color, ls in [(high_rd[0], 'blue', '-'), (low_rd[0], 'red', '--')]:
            name = SECTORS[sid][0]
            if name in acf_data:
                acf = acf_data[name]
                ax2.plot(range(len(acf)), acf, color=color, ls=ls, linewidth=1.5,
                         label=f'{name} (R&D={SECTORS[sid][1]}%)')
        ax2.axhline(0, color='gray', ls=':', alpha=0.5)
        ax2.set_xlabel('Lag (months)', fontsize=10)
        ax2.set_ylabel('Autocorrelation', fontsize=10)
        ax2.set_title('ACF: highest vs lowest R&D', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2)

        fig.suptitle('Test 14: Autocorrelation Timescale Count',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        for ext in ['.png', '.pdf']:
            plt.savefig(os.path.join(FIGDIR, f'test14_acf_timescales{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close()
        log("  Saved figures/framework_verification/test14_acf_timescales.png/.pdf")

    return {'verdict': verdict, 'rho_s': rho_s, 'p': p_val}


# ============================================================================
# TEST 15: Spectral Peak Count vs R&D Intensity
# ============================================================================

def test15_spectral_peaks(series):
    """
    Prediction: More complex sectors have more distinct frequency peaks
    in their power spectrum (more characteristic oscillation frequencies).
    """
    log("\n" + "=" * 72)
    log("  TEST 15: Spectral Peak Count vs R&D Intensity")
    log("=" * 72)
    log("  Prediction: higher R&D -> more spectral peaks")

    rd_vals = []
    peak_vals = []
    labels = []

    for sid, (name, rd_pct, sigma) in SECTORS.items():
        g = get_growth_rate(series, sid)
        if len(g) < 120:
            continue

        n_peaks, freqs, power = count_spectral_peaks(g.values)

        rd_vals.append(rd_pct)
        peak_vals.append(n_peaks)
        labels.append(name)

    if len(rd_vals) < 5:
        log("  FAILED: insufficient data")
        return {'verdict': 'FAILED'}

    log(f"\n  {'Sector':<22} {'R&D%':>5} {'Spectral peaks':>15}")
    log(f"  {'-'*45}")
    for i in range(len(labels)):
        log(f"  {labels[i]:<22} {rd_vals[i]:>5.1f} {peak_vals[i]:>15}")

    rho_s, p_val = spearmanr(rd_vals, peak_vals)
    log(f"\n  Spearman rho(R&D%, spectral_peaks) = {rho_s:.4f}, p = {p_val:.4f}")

    if rho_s > 0 and p_val < 0.10:
        verdict = 'CONSISTENT'
    elif rho_s < 0 and p_val < 0.10:
        verdict = 'INCONSISTENT'
    else:
        verdict = 'AMBIGUOUS'
    log(f"  Verdict: {verdict}")

    # Figure
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(rd_vals, peak_vals, s=80, alpha=0.8, edgecolors='k',
                   linewidths=0.5, c='steelblue', zorder=3)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (rd_vals[i], peak_vals[i]), fontsize=7,
                        xytext=(5, 3), textcoords='offset points')
        ax.set_xlabel('R&D intensity (%)', fontsize=11)
        ax.set_ylabel('Number of spectral peaks', fontsize=11)
        ax.set_title(f'Test 15: Spectral Peaks vs R&D ($\\rho_s$={rho_s:.3f}, p={p_val:.3f})',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        for ext in ['.png', '.pdf']:
            plt.savefig(os.path.join(FIGDIR, f'test15_spectral{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close()
        log("  Saved figures/framework_verification/test15_spectral.png/.pdf")

    return {'verdict': verdict, 'rho_s': rho_s, 'p': p_val}


# ============================================================================
# TEST 16: Within-Sector Hierarchy Depth (4-digit NAICS)
# ============================================================================

def test16_within_sector_depth(series):
    """
    Prediction: Sectors with higher R&D intensity show more distinct
    eigenvalue layers in their within-sector correlation structure
    (deeper internal hierarchy).
    """
    log("\n" + "=" * 72)
    log("  TEST 16: Within-Sector Hierarchy Depth (4-digit NAICS)")
    log("=" * 72)
    log("  Prediction: higher R&D -> more significant eigenvalues within sector")

    group_results = {}

    for group_name, sids in SUBSECTORS.items():
        log(f"\n  {group_name} (R&D = {SECTOR_GROUP_RD.get(group_name, '?')}%):")

        # Build growth rate panel for this group's subsectors
        frames = {}
        for sid, sub_name in sids.items():
            g = get_growth_rate(series, sid)
            if len(g) > 60:
                frames[sub_name] = g
                log(f"    {sub_name}: {len(g)} obs")
            else:
                log(f"    {sub_name}: no data or too short")

        if len(frames) < 3:
            log(f"    Skipping: only {len(frames)} subsectors available")
            continue

        panel = pd.DataFrame(frames).dropna()
        if len(panel) < 60:
            log(f"    Skipping: only {len(panel)} overlapping months")
            continue

        # Correlation matrix eigenvalues
        cm = panel.corr().values
        eigenvalues = np.sort(np.linalg.eigvalsh(cm))[::-1]

        # Marchenko-Pastur threshold
        N_obs = len(panel)
        p_vars = panel.shape[1]
        gamma = p_vars / N_obs
        noise_est = eigenvalues[-1]  # smallest eigenvalue as noise proxy
        mp_upper = noise_est * (1 + np.sqrt(gamma)) ** 2

        n_sig = int(np.sum(eigenvalues > mp_upper))
        ev1_share = eigenvalues[0] / eigenvalues.sum()

        log(f"    Subsectors: {panel.shape[1]}, Months: {len(panel)}")
        log(f"    Eigenvalues: {np.round(eigenvalues, 4)}")
        log(f"    MP threshold: {mp_upper:.4f}")
        log(f"    Significant eigenvalues: {n_sig}")
        log(f"    EV1 share: {ev1_share:.1%}")

        group_results[group_name] = {
            'rd_pct': SECTOR_GROUP_RD.get(group_name, 0),
            'n_subsectors': panel.shape[1],
            'n_sig_ev': n_sig,
            'ev1_share': ev1_share,
            'eigenvalues': eigenvalues,
        }

    if len(group_results) < 3:
        log("  FAILED: insufficient sector groups")
        return {'verdict': 'FAILED'}

    # Compare
    rd_vals = [v['rd_pct'] for v in group_results.values()]
    sig_vals = [v['n_sig_ev'] for v in group_results.values()]
    group_names = list(group_results.keys())

    rho_s, p_val = spearmanr(rd_vals, sig_vals)
    log(f"\n  Spearman rho(R&D%, n_sig_eigenvalues) = {rho_s:.4f}, p = {p_val:.4f}")

    # Alternative: compare EV1 share (lower = more distributed = deeper hierarchy)
    ev1_shares = [v['ev1_share'] for v in group_results.values()]
    rho_ev1, p_ev1 = spearmanr(rd_vals, ev1_shares)
    log(f"  Spearman rho(R&D%, EV1_share) = {rho_ev1:.4f}, p = {p_ev1:.4f}")
    log(f"  (negative = higher R&D has more distributed eigenvalues = deeper hierarchy)")

    if (rho_s > 0 and p_val < 0.10) or (rho_ev1 < 0 and p_ev1 < 0.10):
        verdict = 'CONSISTENT'
    elif (rho_s < 0 and p_val < 0.10) or (rho_ev1 > 0 and p_ev1 < 0.10):
        verdict = 'INCONSISTENT'
    else:
        verdict = 'AMBIGUOUS'
    log(f"  Verdict: {verdict}")

    # Figure
    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left: eigenvalue spectra for each group
        colors = plt.cm.viridis(np.linspace(0, 1, len(group_results)))
        for idx, (gname, gdata) in enumerate(sorted(group_results.items(),
                                                      key=lambda x: x[1]['rd_pct'])):
            ev = gdata['eigenvalues']
            ax1.plot(range(1, len(ev) + 1), ev, 'o-', color=colors[idx],
                     linewidth=1.5, markersize=8,
                     label=f"{gname} (R&D={gdata['rd_pct']}%)")
        ax1.set_xlabel('Eigenvalue rank', fontsize=10)
        ax1.set_ylabel('Eigenvalue', fontsize=10)
        ax1.set_title('Within-sector eigenvalue spectra', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=7, loc='upper right')
        ax1.grid(True, alpha=0.2)

        # Right: scatter R&D vs EV1 share
        ax2.scatter(rd_vals, ev1_shares, s=100, alpha=0.8, edgecolors='k',
                    linewidths=0.5, c='steelblue', zorder=3)
        for i, gname in enumerate(group_names):
            ax2.annotate(gname.split(' (')[0], (rd_vals[i], ev1_shares[i]),
                         fontsize=8, xytext=(5, 5), textcoords='offset points')
        ax2.set_xlabel('R&D intensity (%)', fontsize=10)
        ax2.set_ylabel('First eigenvalue share', fontsize=10)
        ax2.set_title(f'EV1 share vs R&D ($\\rho_s$={rho_ev1:.3f})',
                      fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.2)

        fig.suptitle('Test 16: Within-Sector Hierarchy Depth',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        for ext in ['.png', '.pdf']:
            plt.savefig(os.path.join(FIGDIR, f'test16_within_sector{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close()
        log("  Saved figures/framework_verification/test16_within_sector.png/.pdf")

    return {'verdict': verdict, 'rho_s_sig': rho_s, 'p_sig': p_val,
            'rho_s_ev1': rho_ev1, 'p_ev1': p_ev1,
            'group_results': {k: {kk: vv for kk, vv in v.items() if kk != 'eigenvalues'}
                             for k, v in group_results.items()}}


# ============================================================================
# TEST 17: Timescale Span vs R&D Intensity
# ============================================================================

def test17_timescale_span(series):
    """
    Prediction: Higher R&D sectors have wider timescale spans
    (ratio of slowest to fastest characteristic timescale).
    N_eff ≈ log(τ_max/τ_min) / log(r*).
    """
    log("\n" + "=" * 72)
    log("  TEST 17: Timescale Span vs R&D Intensity")
    log("=" * 72)
    log("  Prediction: higher R&D -> wider timescale span (tau_max/tau_min)")

    rd_vals = []
    span_vals = []
    labels = []

    for sid, (name, rd_pct, sigma) in SECTORS.items():
        g = get_growth_rate(series, sid)
        if len(g) < 120:
            continue

        # SSA to extract characteristic timescales
        x = g.values
        x = (x - np.mean(x)) / (np.std(x) + 1e-12)
        L = min(len(x) // 3, 60)

        if L < 10:
            continue

        H = hankel_matrix(x, L)
        try:
            U, s, Vt = np.linalg.svd(H, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        K = H.shape[1]
        eigenvalues = s ** 2 / K

        # Identify significant components
        noise_est = np.median(eigenvalues[-max(1, L // 4):])
        mp_upper = marchenko_pastur_threshold(K, L, noise_est)
        sig_ev = eigenvalues[eigenvalues > mp_upper]

        if len(sig_ev) < 2:
            continue

        # Timescale span = ratio of largest to smallest significant eigenvalue
        # (eigenvalue ~ amplitude^2 at that timescale)
        span = sig_ev[0] / sig_ev[-1]

        rd_vals.append(rd_pct)
        span_vals.append(span)
        labels.append(name)

    if len(rd_vals) < 5:
        log("  FAILED: insufficient data")
        return {'verdict': 'FAILED'}

    log(f"\n  {'Sector':<22} {'R&D%':>5} {'Span':>10}")
    log(f"  {'-'*40}")
    for i in range(len(labels)):
        log(f"  {labels[i]:<22} {rd_vals[i]:>5.1f} {span_vals[i]:>10.1f}")

    rho_s, p_val = spearmanr(rd_vals, span_vals)
    log(f"\n  Spearman rho(R&D%, timescale_span) = {rho_s:.4f}, p = {p_val:.4f}")

    if rho_s > 0 and p_val < 0.10:
        verdict = 'CONSISTENT'
    elif rho_s < 0 and p_val < 0.10:
        verdict = 'INCONSISTENT'
    else:
        verdict = 'AMBIGUOUS'
    log(f"  Verdict: {verdict}")

    # Figure
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(rd_vals, np.log10(span_vals), s=80, alpha=0.8, edgecolors='k',
                   linewidths=0.5, c='steelblue', zorder=3)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (rd_vals[i], np.log10(span_vals[i])), fontsize=7,
                        xytext=(5, 5), textcoords='offset points')
        ax.set_xlabel('R&D intensity (%)', fontsize=11)
        ax.set_ylabel('log$_{10}$(Timescale span)', fontsize=11)
        ax.set_title(f'Test 17: Timescale Span vs R&D ($\\rho_s$={rho_s:.3f}, p={p_val:.3f})',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        for ext in ['.png', '.pdf']:
            plt.savefig(os.path.join(FIGDIR, f'test17_timescale_span{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close()
        log("  Saved figures/framework_verification/test17_timescale_span.png/.pdf")

    return {'verdict': verdict, 'rho_s': rho_s, 'p': p_val}


# ============================================================================
# TEST 18: Temporal Emergence — Did Computer/Elec Gain Levels?
# ============================================================================

def test18_temporal_emergence(series):
    """
    Prediction: As technology complexifies, new hierarchy levels emerge.
    Computer/Electronics should show INCREASING effective dimensionality
    over time (especially post-2012 deep learning revolution), while
    Food manufacturing (control) should not.
    """
    log("\n" + "=" * 72)
    log("  TEST 18: Temporal Emergence of Hierarchy Levels")
    log("=" * 72)
    log("  Prediction: Computer/Elec gains levels over time; Food does not")

    test_sector = 'IPG334S'  # Computer/Elec
    ctrl_sector = 'IPG311S'  # Food

    results = {}

    for sid, label in [(test_sector, 'Computer/Elec'), (ctrl_sector, 'Food')]:
        g = get_growth_rate(series, sid)
        if len(g) < 240:
            log(f"  {label}: insufficient data ({len(g)} obs)")
            continue

        # Rolling SSA: compute effective dimensionality in 10-year windows
        window = 120  # 10 years of monthly data
        L_ssa = 36  # 3-year SSA window
        step = 12  # step by 1 year

        dims = []
        dates = []

        for start_idx in range(0, len(g) - window, step):
            chunk = g.values[start_idx:start_idx + window]
            n_sig, _, _ = ssa_effective_dim(chunk, L=L_ssa)
            center_date = g.index[start_idx + window // 2]
            dims.append(n_sig)
            dates.append(center_date)

        if len(dims) < 5:
            continue

        results[label] = {'dates': dates, 'dims': dims}

        # Trend test
        from scipy.stats import kendalltau
        tau, p = kendalltau(range(len(dims)), dims)
        log(f"\n  {label}:")
        log(f"    Windows: {len(dims)}, Dim range: {min(dims)}-{max(dims)}")
        log(f"    Kendall tau (trend) = {tau:.4f}, p = {p:.4f}")
        log(f"    Mean dim early (first third): {np.mean(dims[:len(dims)//3]):.2f}")
        log(f"    Mean dim late (last third): {np.mean(dims[-len(dims)//3:]):.2f}")
        results[label]['tau'] = tau
        results[label]['p'] = p
        results[label]['mean_early'] = np.mean(dims[:len(dims) // 3])
        results[label]['mean_late'] = np.mean(dims[-len(dims) // 3:])

    if 'Computer/Elec' not in results:
        log("  FAILED: Computer/Elec data missing")
        return {'verdict': 'FAILED'}

    tech = results['Computer/Elec']
    ctrl = results.get('Food', {})

    tech_increasing = tech['tau'] > 0 and tech['p'] < 0.10
    ctrl_flat = ctrl.get('tau', 0) <= 0 or ctrl.get('p', 1) >= 0.10

    if tech_increasing and ctrl_flat:
        verdict = 'CONSISTENT'
    elif tech_increasing:
        verdict = 'AMBIGUOUS'  # tech increases but so does control
    elif tech['tau'] <= 0:
        verdict = 'INCONSISTENT'
    else:
        verdict = 'AMBIGUOUS'
    log(f"\n  Tech increasing: {tech_increasing}, Control flat: {ctrl_flat}")
    log(f"  Verdict: {verdict}")

    # Figure
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(12, 6))

        for label, color, marker in [('Computer/Elec', 'blue', 'o'),
                                      ('Food', 'red', 's')]:
            if label in results:
                r = results[label]
                ax.plot(r['dates'], r['dims'], f'{color[0]}{marker}-', linewidth=1.5,
                        markersize=4, alpha=0.8,
                        label=f"{label} ($\\tau$={r['tau']:.3f}, p={r['p']:.3f})")

        # Mark key technology dates
        for date, event in [('2012-01-01', 'AlexNet'), ('2017-06-01', 'Transformer'),
                            ('2022-11-01', 'ChatGPT')]:
            ax.axvline(pd.Timestamp(date), color='gray', ls=':', alpha=0.5)
            ax.text(pd.Timestamp(date), ax.get_ylim()[1] * 0.95, f'  {event}',
                    fontsize=7, va='top', color='gray')

        ax.set_xlabel('Date (center of 10-year window)', fontsize=11)
        ax.set_ylabel('SSA effective dimensionality', fontsize=11)
        ax.set_title('Test 18: Temporal Emergence of Hierarchy Levels',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.annotate(f'Verdict: {verdict}', xy=(0.02, 0.02), xycoords='axes fraction',
                    fontsize=10, bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))

        plt.tight_layout()
        for ext in ['.png', '.pdf']:
            plt.savefig(os.path.join(FIGDIR, f'test18_temporal_emergence{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close()
        log("  Saved figures/framework_verification/test18_temporal_emergence.png/.pdf")

    return {'verdict': verdict,
            'tech_tau': tech['tau'], 'tech_p': tech['p'],
            'ctrl_tau': ctrl.get('tau', np.nan), 'ctrl_p': ctrl.get('p', np.nan),
            'tech_early': tech['mean_early'], 'tech_late': tech['mean_late']}


# ============================================================================
# SUMMARY
# ============================================================================

def make_summary_table(results):
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Endogenous Hierarchy Depth: Does Complexity Determine $N$?}')
    lines.append(r'\label{tab:framework_verification_3}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{clllc}')
    lines.append(r'\toprule')
    lines.append(r'Test & Prediction & Key statistic & Value & Verdict \\')
    lines.append(r'\midrule')

    def fmt(val, f='.3f'):
        return f'{val:{f}}' if isinstance(val, (int, float)) and not np.isnan(val) else 'N/A'

    test_rows = [
        ('13', 'SSA dimensionality $\\uparrow$ with R\\&D',
         r'Spearman $\rho$', fmt(results.get('test13', {}).get('rho_s', np.nan)),
         results.get('test13', {}).get('verdict', 'N/A')),
        ('14', 'ACF peaks $\\uparrow$ with R\\&D',
         r'Spearman $\rho$', fmt(results.get('test14', {}).get('rho_s', np.nan)),
         results.get('test14', {}).get('verdict', 'N/A')),
        ('15', 'Spectral peaks $\\uparrow$ with R\\&D',
         r'Spearman $\rho$', fmt(results.get('test15', {}).get('rho_s', np.nan)),
         results.get('test15', {}).get('verdict', 'N/A')),
        ('16', 'Within-sector eigenvalue depth $\\uparrow$',
         r'Spearman $\rho$ (EV1)', fmt(results.get('test16', {}).get('rho_s_ev1', np.nan)),
         results.get('test16', {}).get('verdict', 'N/A')),
        ('17', 'Timescale span $\\uparrow$ with R\\&D',
         r'Spearman $\rho$', fmt(results.get('test17', {}).get('rho_s', np.nan)),
         results.get('test17', {}).get('verdict', 'N/A')),
        ('18', 'Computer/Elec gains levels over time',
         r'Kendall $\tau$ (tech)', fmt(results.get('test18', {}).get('tech_tau', np.nan)),
         results.get('test18', {}).get('verdict', 'N/A')),
    ]

    for num, pred, stat, val, verdict in test_rows:
        v_fmt = r'\textbf{' + verdict + '}' if verdict == 'CONSISTENT' else verdict
        lines.append(f'  {num} & {pred} & {stat} & {val} & {v_fmt} \\\\')

    lines.append(r'\midrule')
    verdicts = [r.get('verdict', 'N/A') for r in results.values()]
    n_con = verdicts.count('CONSISTENT')
    n_amb = verdicts.count('AMBIGUOUS')
    n_inc = verdicts.count('INCONSISTENT')
    lines.append(f'  \\multicolumn{{4}}{{l}}{{Tests 13--18: {n_con} consistent, {n_amb} ambiguous, {n_inc} inconsistent}} & \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\vspace{4pt}')
    lines.append(r'\parbox{0.95\textwidth}{\footnotesize')
    lines.append(r'  R\&D intensity from NSF NCSES Business R\&D Survey 2019.')
    lines.append(r'  SSA window $L=48$ months (4 years); Marchenko--Pastur noise threshold.')
    lines.append(r'  4-digit NAICS subsectors from FRED for within-sector analysis (Test 16).')
    lines.append(r'  Temporal emergence (Test 18): 10-year rolling window, 1-year step.')
    lines.append(r'}')
    lines.append(r'\end{table}')

    table_str = '\n'.join(lines)
    path = os.path.join(RESULTS_DIR, 'framework_verification_3_summary.tex')
    with open(path, 'w') as f:
        f.write(table_str)
    log(f"\n  Saved {path}")
    return table_str


def main():
    print("=" * 72)
    print("  ENDOGENOUS HIERARCHY DEPTH: DOES COMPLEXITY DETERMINE N?")
    print("=" * 72)
    print()
    print("  Testing whether the number of effective dynamical levels")
    print("  is determined by technical complexity (R&D intensity)")
    print()

    log("Fetching FRED data (3-digit + 4-digit NAICS)...")
    series = fetch_all_series()

    n_loaded = sum(1 for s in series.values() if len(s) > 0)
    log(f"  Loaded {n_loaded}/{len(series)} series")

    results = {}
    results['test13'] = test13_ssa_vs_rd(series)
    results['test14'] = test14_acf_timescales(series)
    results['test15'] = test15_spectral_peaks(series)
    results['test16'] = test16_within_sector_depth(series)
    results['test17'] = test17_timescale_span(series)
    results['test18'] = test18_temporal_emergence(series)

    # Summary
    log("\n" + "=" * 72)
    log("  SUMMARY (Tests 13-18: Endogenous Hierarchy Depth)")
    log("=" * 72)

    for tname, r in results.items():
        v = r.get('verdict', 'N/A')
        log(f"  {tname}: {v}")

    verdicts = [r.get('verdict', 'N/A') for r in results.values()]
    n_con = verdicts.count('CONSISTENT')
    n_amb = verdicts.count('AMBIGUOUS')
    n_inc = verdicts.count('INCONSISTENT')
    log(f"\n  Round 3: {n_con} consistent, {n_amb} ambiguous, {n_inc} inconsistent")

    # Load all previous results
    total_verdicts = []
    for rnd in ['test_results.csv', 'test_results_2.csv']:
        path = os.path.join(OUTPUT, rnd)
        if os.path.exists(path):
            df = pd.read_csv(path)
            total_verdicts.extend(df['verdict'].tolist())
    total_verdicts.extend(verdicts)

    total_con = total_verdicts.count('CONSISTENT')
    total_amb = total_verdicts.count('AMBIGUOUS')
    total_inc = total_verdicts.count('INCONSISTENT')

    log(f"\n  All tests (1-18): {total_con} consistent, {total_amb} ambiguous, {total_inc} inconsistent out of {len(total_verdicts)}")

    if total_con >= 12:
        overall = "STRONG SUPPORT"
    elif total_con >= 8:
        overall = "MODERATE SUPPORT"
    elif total_inc >= 6:
        overall = "FRAMEWORK CHALLENGED"
    else:
        overall = "MIXED EVIDENCE"
    log(f"  Overall assessment: {overall}")

    make_summary_table(results)

    log_path = os.path.join(RESULTS_DIR, 'framework_verification_3_results.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(LOG_LINES))
    print(f"\n  Saved {log_path}")

    rows = []
    for tname, r in results.items():
        row = {'test': tname, 'verdict': r.get('verdict', 'N/A')}
        for k, v in r.items():
            if isinstance(v, (int, float, str)) and k != 'verdict':
                row[k] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT, 'test_results_3.csv'), index=False)

    print(f"\n{'='*72}")
    print(f"  DONE — {overall}")
    print(f"{'='*72}")


if __name__ == '__main__':
    main()
