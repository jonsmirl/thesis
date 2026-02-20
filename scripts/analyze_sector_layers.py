#!/usr/bin/env python3
"""
Sector Layer Analysis via Continuous Wavelet Transform
=======================================================

For each FRED industrial-production sector, this script:
  1. Computes the CWT power spectrum over the full available time range
  2. Averages across time to get a "characteristic frequency fingerprint"
  3. Identifies spectral peaks (distinct active layers)
  4. Computes effective dimensionality N_eff = count of peaks above threshold
  5. Tracks N_eff over rolling 20-year windows (5-year steps) to see how
     the number of active timescale layers evolves
  6. Computes cross-sector correlation of time-averaged spectra

Thesis predictions (CES hierarchy):
  - Low-rho sectors (Computer/Electronics) should have MORE active layers
    (innovation = new timescale gaps opening)
  - High-rho sectors (Primary Metals, Food) should have FEWER layers
    (maturation = timescale convergence, fewer distinct modes)
  - Kendall tau: Computer/Elec tau ~ +0.26 (increasing N_eff over time),
                 Steel tau ~ -0.27 (decreasing N_eff over time)

Data: Cached FRED monthly industrial production indices in
      thesis_data/framework_verification/
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    import pywt
except ImportError:
    print("ERROR: pip install PyWavelets")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
except ImportError:
    print("ERROR: pip install matplotlib")
    sys.exit(1)

from scipy.signal import find_peaks
from scipy.stats import kendalltau, spearmanr

# ============================================================================
# Configuration
# ============================================================================

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'thesis_data', 'framework_verification')
FIG_DIR = os.path.join(BASE, 'figures', 'sector_layers')
OUT_DIR = os.path.join(BASE, 'thesis_data', 'sector_layers')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

SECTORS = {
    'INDPRO':  ('Aggregate IP',              'aggregate'),
    'IPG334S': ('Computer & Electronics',     'low-rho (active tech wave)'),
    'IPG331S': ('Primary Metals',             'high-rho (mature)'),
    'IPG325S': ('Chemicals',                  'intermediate'),
    'IPG336S': ('Transport Equipment',        'EV wave'),
    'IPG311S': ('Food Manufacturing',         'control (stable)'),
    'IPG335S': ('Electrical Equipment',       'intermediate-low rho'),
}

# CWT parameters
MIN_PERIOD_MONTHS = 6       # shortest period to resolve (6 months)
MAX_PERIOD_MONTHS = 240     # longest period (20 years)
N_SCALES = 256              # number of wavelet scales (log-spaced)
WAVELET = 'morl'            # Morlet wavelet
MORLET_FACTOR = 1.03        # period ~= 1.03 * scale for Morlet

# Peak detection
PEAK_THRESHOLD_FRAC = 0.20  # 20% of max power = significance threshold
PEAK_PROMINENCE_FRAC = 0.05 # minimum prominence as fraction of max

# Rolling window
WINDOW_YEARS = 20
STEP_YEARS = 5


# ============================================================================
# Helper Functions
# ============================================================================

def load_series(series_id):
    """Load a cached FRED series from CSV."""
    path = os.path.join(DATA_DIR, f'{series_id}.csv')
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return None
    df = pd.read_csv(path, parse_dates=['date'], index_col='date')
    return df


def compute_growth(df):
    """Monthly log-difference growth rate, resampled to month-start."""
    s = df['value'].resample('MS').last().dropna()
    g = np.log(s).diff().dropna()
    return g


def get_scales():
    """Return log-spaced CWT scales and corresponding periods."""
    scales = np.geomspace(
        MIN_PERIOD_MONTHS / MORLET_FACTOR,
        MAX_PERIOD_MONTHS / MORLET_FACTOR,
        N_SCALES
    )
    periods_months = scales * MORLET_FACTOR
    periods_years = periods_months / 12.0
    return scales, periods_months, periods_years


def cwt_power(signal, scales):
    """Compute CWT power spectrum (|coefficients|^2), scale-normalized."""
    coeffs, freqs = pywt.cwt(signal, scales, WAVELET)
    power = np.abs(coeffs) ** 2
    # Normalize by scale so that large scales don't dominate visually
    for i in range(len(scales)):
        power[i] /= scales[i]
    return power


def time_averaged_spectrum(power):
    """Average CWT power across all time points -> 1D spectrum vs period."""
    return power.mean(axis=1)


def find_spectral_peaks(avg_spectrum, periods_years, threshold_frac=PEAK_THRESHOLD_FRAC,
                        prominence_frac=PEAK_PROMINENCE_FRAC):
    """
    Find peaks in the time-averaged power spectrum.
    Returns (peak_indices, peak_periods, peak_powers).
    """
    max_power = avg_spectrum.max()
    threshold = threshold_frac * max_power
    min_prominence = prominence_frac * max_power

    peak_idx, properties = find_peaks(
        avg_spectrum,
        height=threshold,
        prominence=min_prominence,
        distance=5  # minimum separation in index space
    )

    peak_periods = periods_years[peak_idx]
    peak_powers = avg_spectrum[peak_idx]

    # Sort by period (ascending)
    order = np.argsort(peak_periods)
    return peak_idx[order], peak_periods[order], peak_powers[order]


def characterize_sector(n_eff, dominant_period_yr):
    """Simple character label based on N_eff and dominant period."""
    if n_eff >= 4:
        char = "Multi-layer (innovative)"
    elif n_eff == 3:
        char = "Three-layer"
    elif n_eff == 2:
        char = "Two-layer"
    else:
        char = "Single-layer (mature)"

    if dominant_period_yr < 2.0:
        char += ", short-cycle dominant"
    elif dominant_period_yr < 5.0:
        char += ", business-cycle dominant"
    elif dominant_period_yr < 12.0:
        char += ", long-cycle dominant"
    else:
        char += ", secular dominant"

    return char


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_all_sectors():
    """Run CWT analysis on all sectors, produce summary table and figures."""

    scales, periods_months, periods_years = get_scales()

    # Storage for results
    results = {}
    avg_spectra = {}

    print("=" * 80)
    print("SECTOR LAYER ANALYSIS VIA CONTINUOUS WAVELET TRANSFORM")
    print("=" * 80)
    print()

    # ------------------------------------------------------------------
    # Step 1-4: Full-range CWT, time-averaged spectrum, peaks, N_eff
    # ------------------------------------------------------------------

    for sid, (name, desc) in SECTORS.items():
        print(f"\n--- {sid}: {name} ({desc}) ---")

        df = load_series(sid)
        if df is None:
            continue

        growth = compute_growth(df)
        print(f"  Data: {len(growth)} months, {growth.index.min().strftime('%Y-%m')} "
              f"to {growth.index.max().strftime('%Y-%m')}")

        # CWT
        power = cwt_power(growth.values, scales)

        # Time-averaged spectrum
        avg_spec = time_averaged_spectrum(power)
        avg_spectra[sid] = avg_spec

        # Find peaks
        peak_idx, peak_periods, peak_powers = find_spectral_peaks(avg_spec, periods_years)
        n_eff = len(peak_idx)

        # Dominant period = peak with highest power
        if n_eff > 0:
            dom_idx = np.argmax(peak_powers)
            dominant_period = peak_periods[dom_idx]
        else:
            dominant_period = np.nan

        # Characterize
        character = characterize_sector(n_eff, dominant_period) if n_eff > 0 else "No peaks detected"

        print(f"  N_eff (active layers): {n_eff}")
        print(f"  Peak periods (years):  {', '.join(f'{p:.1f}' for p in peak_periods)}")
        if not np.isnan(dominant_period):
            print(f"  Dominant period:       {dominant_period:.1f} yr")
        else:
            print(f"  Dominant period:       N/A")
        print(f"  Character:             {character}")

        results[sid] = {
            'name': name,
            'desc': desc,
            'n_eff': n_eff,
            'peak_periods': peak_periods,
            'peak_powers': peak_powers,
            'dominant_period': dominant_period,
            'character': character,
            'growth': growth,
            'power': power,
        }

    # ------------------------------------------------------------------
    # Step 5: Rolling analysis (20-year windows, 5-year steps)
    # ------------------------------------------------------------------

    print("\n\n" + "=" * 80)
    print("ROLLING WINDOW ANALYSIS (20-year windows, 5-year steps)")
    print("=" * 80)

    rolling_results = {}

    for sid in SECTORS:
        if sid not in results:
            continue

        growth = results[sid]['growth']

        # Determine window and step in months
        window_months = WINDOW_YEARS * 12
        step_months = STEP_YEARS * 12

        if len(growth) < window_months:
            print(f"  {sid}: insufficient data for 20-year window ({len(growth)} months)")
            rolling_results[sid] = ([], [], [])
            continue

        centers = []
        n_effs = []
        dominant_periods = []

        start = 0
        while start + window_months <= len(growth):
            window_data = growth.values[start:start + window_months]
            center_date = growth.index[start + window_months // 2]

            # CWT on window
            wp = cwt_power(window_data, scales)
            avg = time_averaged_spectrum(wp)

            # Peaks
            pidx, pperiods, ppowers = find_spectral_peaks(avg, periods_years)
            neff = len(pidx)

            # Dominant
            if neff > 0:
                dom = pperiods[np.argmax(ppowers)]
            else:
                dom = np.nan

            centers.append(center_date)
            n_effs.append(neff)
            dominant_periods.append(dom)

            start += step_months

        rolling_results[sid] = (centers, n_effs, dominant_periods)

        print(f"\n  {sid} ({results[sid]['name']}):")
        for c, ne, dp in zip(centers, n_effs, dominant_periods):
            dp_str = f"{dp:.1f}" if not np.isnan(dp) else "N/A"
            print(f"    {c.strftime('%Y')}: N_eff={ne}, dominant={dp_str} yr")

    # ------------------------------------------------------------------
    # Step 6: Kendall tau trend tests for N_eff over time
    # ------------------------------------------------------------------

    print("\n\n" + "=" * 80)
    print("KENDALL TAU TREND TESTS: N_eff vs Time")
    print("=" * 80)

    tau_results = {}
    for sid in SECTORS:
        if sid not in rolling_results:
            continue
        centers, n_effs, _ = rolling_results[sid]
        if len(centers) < 4:
            print(f"  {sid}: too few windows for Kendall test")
            continue

        time_idx = np.arange(len(n_effs))
        tau, p_value = kendalltau(time_idx, n_effs)
        tau_results[sid] = (tau, p_value)

        sig_str = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""
        print(f"  {sid:8s} ({results[sid]['name']:25s}): "
              f"tau = {tau:+.3f}, p = {p_value:.4f} {sig_str}")

    # ------------------------------------------------------------------
    # Step 7a: Summary Table
    # ------------------------------------------------------------------

    print("\n\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    header = f"{'Sector':<12s} {'Name':<25s} {'N_eff':>5s} {'Peak Periods (yr)':<35s} {'Dominant':>8s} {'Tau':>7s} {'Character'}"
    print(header)
    print("-" * len(header))

    summary_rows = []
    for sid in ['INDPRO', 'IPG334S', 'IPG335S', 'IPG336S', 'IPG325S', 'IPG331S', 'IPG311S']:
        if sid not in results:
            continue
        r = results[sid]
        pp_str = ', '.join(f'{p:.1f}' for p in r['peak_periods'])
        dom_str = f"{r['dominant_period']:.1f}" if not np.isnan(r['dominant_period']) else "N/A"
        tau_str = f"{tau_results[sid][0]:+.3f}" if sid in tau_results else "N/A"

        print(f"{sid:<12s} {r['name']:<25s} {r['n_eff']:>5d} {pp_str:<35s} {dom_str:>8s} {tau_str:>7s} {r['character']}")

        summary_rows.append({
            'sector_id': sid,
            'name': r['name'],
            'description': r['desc'],
            'n_eff': r['n_eff'],
            'peak_periods': pp_str,
            'dominant_period_yr': r['dominant_period'],
            'kendall_tau': tau_results.get(sid, (np.nan, np.nan))[0],
            'kendall_p': tau_results.get(sid, (np.nan, np.nan))[1],
            'character': r['character'],
        })

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, 'sector_layer_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary table saved to {summary_path}")

    # ------------------------------------------------------------------
    # Step 7b: Computer/Electronics vs Primary Metals divergence
    # ------------------------------------------------------------------

    print("\n\n" + "=" * 80)
    print("DIVERGENCE: Computer/Electronics vs Primary Metals")
    print("=" * 80)

    if 'IPG334S' in rolling_results and 'IPG331S' in rolling_results:
        c334, ne334, _ = rolling_results['IPG334S']
        c331, ne331, _ = rolling_results['IPG331S']

        print(f"\n  {'Year':>6s}  {'Comp/Elec N_eff':>15s}  {'Metals N_eff':>12s}  {'Gap':>5s}")
        print("  " + "-" * 45)

        divergence_rows = []
        for i in range(min(len(c334), len(c331))):
            gap = ne334[i] - ne331[i]
            print(f"  {c334[i].strftime('%Y'):>6s}  {ne334[i]:>15d}  {ne331[i]:>12d}  {gap:>+5d}")
            divergence_rows.append({
                'center_year': c334[i].year,
                'comp_elec_neff': ne334[i],
                'metals_neff': ne331[i],
                'gap': gap,
            })

        div_df = pd.DataFrame(divergence_rows)
        div_path = os.path.join(OUT_DIR, 'comp_metals_divergence.csv')
        div_df.to_csv(div_path, index=False)
        print(f"\n  Divergence data saved to {div_path}")

        # Trend in the gap
        if len(divergence_rows) >= 4:
            gaps = [row['gap'] for row in divergence_rows]
            tau_gap, p_gap = kendalltau(range(len(gaps)), gaps)
            print(f"\n  Kendall tau for gap trend: tau = {tau_gap:+.3f}, p = {p_gap:.4f}")
            if tau_gap > 0:
                print("  -> Gap is WIDENING over time (Comp/Elec gaining layers, Metals losing)")
            elif tau_gap < 0:
                print("  -> Gap is NARROWING over time")
            else:
                print("  -> No trend in gap")

    # ------------------------------------------------------------------
    # Step 7c: Cross-sector spectral correlation
    # ------------------------------------------------------------------

    print("\n\n" + "=" * 80)
    print("CROSS-SECTOR SPECTRAL CORRELATION (time-averaged spectra)")
    print("=" * 80)

    sector_ids_for_corr = [s for s in ['IPG334S', 'IPG335S', 'IPG336S', 'IPG325S', 'IPG331S', 'IPG311S']
                           if s in avg_spectra]

    n_corr = len(sector_ids_for_corr)
    corr_matrix = None
    if n_corr >= 2:
        corr_matrix = np.zeros((n_corr, n_corr))
        for i in range(n_corr):
            for j in range(n_corr):
                r, _ = spearmanr(avg_spectra[sector_ids_for_corr[i]],
                                 avg_spectra[sector_ids_for_corr[j]])
                corr_matrix[i, j] = r

        names_short = [SECTORS[s][0][:12] for s in sector_ids_for_corr]
        print(f"\n  {'':>14s}", end='')
        for nm in names_short:
            print(f"  {nm:>12s}", end='')
        print()

        for i in range(n_corr):
            print(f"  {names_short[i]:>14s}", end='')
            for j in range(n_corr):
                print(f"  {corr_matrix[i,j]:>12.3f}", end='')
            print()

        # Save
        corr_df = pd.DataFrame(corr_matrix,
                                index=[SECTORS[s][0] for s in sector_ids_for_corr],
                                columns=[SECTORS[s][0] for s in sector_ids_for_corr])
        corr_path = os.path.join(OUT_DIR, 'cross_sector_spectral_corr.csv')
        corr_df.to_csv(corr_path)
        print(f"\n  Correlation matrix saved to {corr_path}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    print("\n\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    # Figure 1: Time-averaged spectra for all sectors, overlaid
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['black', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    for i, sid in enumerate(['INDPRO', 'IPG334S', 'IPG331S', 'IPG325S', 'IPG336S', 'IPG311S', 'IPG335S']):
        if sid not in avg_spectra:
            continue
        spec = avg_spectra[sid]
        spec_norm = spec / spec.max()
        lw = 2.5 if sid in ('IPG334S', 'IPG331S') else 1.2
        ls = '-' if sid in ('IPG334S', 'IPG331S', 'INDPRO') else '--'
        ax.plot(periods_years, spec_norm, color=colors[i], linewidth=lw, linestyle=ls,
                label=f"{SECTORS[sid][0]}", alpha=0.85)

        # Mark peaks
        if sid in results:
            r = results[sid]
            for pp, pw in zip(r['peak_periods'], r['peak_powers']):
                pw_norm = pw / spec.max()
                ax.plot(pp, pw_norm, 'o', color=colors[i], markersize=6)

    ax.axhline(PEAK_THRESHOLD_FRAC, color='gray', linestyle=':', alpha=0.5,
               label=f'{int(PEAK_THRESHOLD_FRAC*100)}% threshold')
    ax.set_xscale('log')
    ax.set_xlabel('Period (years)', fontsize=12)
    ax.set_ylabel('Normalized wavelet power (time-averaged)', fontsize=12)
    ax.set_title('Time-Averaged CWT Power Spectra by Sector\n'
                 '(peaks = distinct active timescale layers)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(0.5, 20)
    ax.grid(True, alpha=0.3, which='both')

    # Add period annotations
    for period_yr, label in [(1.0, '1y'), (2.0, '2y'), (3.5, '3.5y\n(Kitchin)'),
                              (5.5, '5.5y'), (8.0, '8y\n(Juglar)'), (15, '15y\n(Kuznets)')]:
        if 0.5 <= period_yr <= 20:
            ax.axvline(period_yr, color='gray', alpha=0.15, linewidth=0.8)

    fig.tight_layout()
    fig1_path = os.path.join(FIG_DIR, 'time_averaged_spectra.pdf')
    fig.savefig(fig1_path, dpi=150, bbox_inches='tight')
    fig.savefig(fig1_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure 1: {fig1_path}")

    # Figure 2: Rolling N_eff for all sectors
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax = axes[0]
    for i, sid in enumerate(['INDPRO', 'IPG334S', 'IPG331S', 'IPG325S', 'IPG336S', 'IPG311S', 'IPG335S']):
        if sid not in rolling_results:
            continue
        centers, n_effs, _ = rolling_results[sid]
        if len(centers) == 0:
            continue
        years = [c.year for c in centers]
        lw = 2.5 if sid in ('IPG334S', 'IPG331S') else 1.0
        marker = 'o' if sid in ('IPG334S', 'IPG331S') else 's'
        ms = 6 if sid in ('IPG334S', 'IPG331S') else 3
        ax.plot(years, n_effs, color=colors[i], linewidth=lw, marker=marker,
                markersize=ms, label=f"{SECTORS[sid][0]}", alpha=0.85)

    ax.set_ylabel('N_eff (active layers)', fontsize=12)
    ax.set_title('Rolling Effective Dimensionality (20-year windows, 5-year steps)\n'
                 'Thesis: low-rho sectors gain layers, high-rho sectors lose layers',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    # Figure 2b: Gap between Computer/Elec and Metals
    ax = axes[1]
    if 'IPG334S' in rolling_results and 'IPG331S' in rolling_results:
        c334, ne334, _ = rolling_results['IPG334S']
        c331, ne331, _ = rolling_results['IPG331S']
        n_common = min(len(c334), len(c331))
        years = [c334[i].year for i in range(n_common)]
        gaps = [ne334[i] - ne331[i] for i in range(n_common)]

        ax.bar(years, gaps, width=4, color=['#e41a1c' if g > 0 else '#377eb8' for g in gaps],
               alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel('N_eff gap (Comp/Elec - Metals)', fontsize=12)
        ax.set_xlabel('Window center year', fontsize=12)
        ax.set_title('Layer Count Divergence: Computer/Electronics vs Primary Metals',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig2_path = os.path.join(FIG_DIR, 'rolling_neff.pdf')
    fig.savefig(fig2_path, dpi=150, bbox_inches='tight')
    fig.savefig(fig2_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure 2: {fig2_path}")

    # Figure 3: Cross-sector spectral correlation heatmap
    if corr_matrix is not None and n_corr >= 2:
        fig, ax = plt.subplots(figsize=(8, 7))
        names_full = [SECTORS[s][0] for s in sector_ids_for_corr]
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(n_corr))
        ax.set_yticks(range(n_corr))
        ax.set_xticklabels(names_full, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(names_full, fontsize=10)
        for i in range(n_corr):
            for j in range(n_corr):
                ax.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center',
                        fontsize=9, color='white' if abs(corr_matrix[i,j]) > 0.6 else 'black')
        plt.colorbar(im, ax=ax, label='Spearman correlation')
        ax.set_title('Cross-Sector Spectral Correlation\n(Spearman rank corr of time-averaged CWT spectra)',
                     fontsize=12, fontweight='bold')
        fig.tight_layout()
        fig3_path = os.path.join(FIG_DIR, 'spectral_correlation.pdf')
        fig.savefig(fig3_path, dpi=150, bbox_inches='tight')
        fig.savefig(fig3_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Figure 3: {fig3_path}")

    # Figure 4: Individual sector scalograms (2x3 panel)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    panel_sectors = ['IPG334S', 'IPG331S', 'IPG325S', 'IPG336S', 'IPG311S', 'IPG335S']

    for idx, sid in enumerate(panel_sectors):
        if sid not in results:
            continue
        ax = axes[idx // 3, idx % 3]
        r = results[sid]
        growth = r['growth']
        power = r['power']

        dates = growth.index
        X, Y = np.meshgrid(dates, periods_years)
        vmin = np.percentile(power, 5)
        vmax = np.percentile(power, 98)
        if vmin <= 0:
            vmin = 1e-10
        im = ax.pcolormesh(X, Y, power, cmap='magma', shading='gouraud',
                           norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.set_ylabel('Period (years)', fontsize=9)
        ax.set_title(f"{SECTORS[sid][0]} (N_eff={r['n_eff']})", fontsize=11, fontweight='bold')

        # Mark peaks as horizontal lines
        for pp in r['peak_periods']:
            ax.axhline(pp, color='cyan', linewidth=1.0, alpha=0.7, linestyle='--')

    fig.suptitle('CWT Scalograms -- IP Growth Rate by Sector (1972-present)\n'
                 'Horizontal cyan lines = detected spectral peaks (active layers)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig4_path = os.path.join(FIG_DIR, 'sector_scalograms.pdf')
    fig.savefig(fig4_path, dpi=150, bbox_inches='tight')
    fig.savefig(fig4_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure 4: {fig4_path}")

    # Figure 5: Comp/Elec vs Metals side-by-side averaged spectra
    if 'IPG334S' in avg_spectra and 'IPG331S' in avg_spectra:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        for ax, sid, color, title in [
            (axes[0], 'IPG334S', '#e41a1c', 'Computer & Electronics (low rho, innovative)'),
            (axes[1], 'IPG331S', '#377eb8', 'Primary Metals (high rho, mature)'),
        ]:
            spec = avg_spectra[sid]
            spec_norm = spec / spec.max()
            ax.fill_between(periods_years, spec_norm, alpha=0.3, color=color)
            ax.plot(periods_years, spec_norm, color=color, linewidth=2)
            ax.axhline(PEAK_THRESHOLD_FRAC, color='gray', linestyle=':', alpha=0.5)

            r = results[sid]
            for pp, pw in zip(r['peak_periods'], r['peak_powers']):
                pw_norm = pw / spec.max()
                ax.plot(pp, pw_norm, 'ko', markersize=8, zorder=5)
                ax.annotate(f'{pp:.1f}y', (pp, pw_norm),
                            textcoords="offset points", xytext=(8, 8),
                            fontsize=10, fontweight='bold')

            ax.set_xscale('log')
            ax.set_xlabel('Period (years)', fontsize=12)
            ax.set_title(f"{title}\nN_eff = {r['n_eff']}", fontsize=12, fontweight='bold')
            ax.set_xlim(0.5, 20)
            ax.grid(True, alpha=0.3, which='both')

        axes[0].set_ylabel('Normalized wavelet power', fontsize=12)

        tau_334 = tau_results.get('IPG334S', (np.nan, np.nan))
        tau_331 = tau_results.get('IPG331S', (np.nan, np.nan))
        fig.suptitle(f'CES Layer Structure: Innovation vs Maturation\n'
                     f'Kendall tau(N_eff trend): Comp/Elec = {tau_334[0]:+.3f} (p={tau_334[1]:.3f}), '
                     f'Metals = {tau_331[0]:+.3f} (p={tau_331[1]:.3f})',
                     fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig5_path = os.path.join(FIG_DIR, 'comp_vs_metals_spectra.pdf')
        fig.savefig(fig5_path, dpi=150, bbox_inches='tight')
        fig.savefig(fig5_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Figure 5: {fig5_path}")

    # ------------------------------------------------------------------
    # Save rolling results
    # ------------------------------------------------------------------

    rolling_rows = []
    for sid in SECTORS:
        if sid not in rolling_results:
            continue
        centers, n_effs, dom_periods = rolling_results[sid]
        for c, ne, dp in zip(centers, n_effs, dom_periods):
            rolling_rows.append({
                'sector_id': sid,
                'name': SECTORS[sid][0],
                'center_year': c.year,
                'n_eff': ne,
                'dominant_period_yr': dp,
            })
    rolling_df = pd.DataFrame(rolling_rows)
    rolling_path = os.path.join(OUT_DIR, 'rolling_neff.csv')
    rolling_df.to_csv(rolling_path, index=False)
    print(f"\n  Rolling results saved to {rolling_path}")

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------

    print("\n\n" + "=" * 80)
    print("THESIS PREDICTIONS vs RESULTS")
    print("=" * 80)

    print("\nPrediction 1: Low-rho sectors have MORE active layers than high-rho sectors")
    if 'IPG334S' in results and 'IPG331S' in results and 'IPG311S' in results:
        ne_comp = results['IPG334S']['n_eff']
        ne_metal = results['IPG331S']['n_eff']
        ne_food = results['IPG311S']['n_eff']
        print(f"  Computer/Electronics N_eff = {ne_comp}")
        print(f"  Primary Metals N_eff       = {ne_metal}")
        print(f"  Food Manufacturing N_eff   = {ne_food}")
        if ne_comp > ne_metal and ne_comp > ne_food:
            print("  RESULT: SUPPORTED -- Computer/Electronics has most layers")
        elif ne_comp >= ne_metal:
            print("  RESULT: PARTIALLY SUPPORTED -- Comp/Elec >= Metals")
        else:
            print("  RESULT: NOT SUPPORTED -- Metals has more layers than Comp/Elec")

    print("\nPrediction 2: Kendall tau for Comp/Elec should be positive (gaining layers)")
    if 'IPG334S' in tau_results:
        tau, p = tau_results['IPG334S']
        print(f"  Computer/Electronics: tau = {tau:+.3f}, p = {p:.4f}")
        print(f"  Paper claims: tau = +0.26")
        if tau > 0:
            print("  RESULT: SUPPORTED -- positive trend (increasing dimensionality)")
        else:
            print("  RESULT: NOT SUPPORTED -- trend is non-positive")

    print("\nPrediction 3: Kendall tau for Metals should be negative (losing layers)")
    if 'IPG331S' in tau_results:
        tau, p = tau_results['IPG331S']
        print(f"  Primary Metals: tau = {tau:+.3f}, p = {p:.4f}")
        print(f"  Paper claims: tau = -0.27")
        if tau < 0:
            print("  RESULT: SUPPORTED -- negative trend (decreasing dimensionality)")
        else:
            print("  RESULT: NOT SUPPORTED -- trend is non-negative")

    print("\nPrediction 4: Layer divergence between Comp/Elec and Metals widens over time")
    if 'IPG334S' in rolling_results and 'IPG331S' in rolling_results:
        c334, ne334, _ = rolling_results['IPG334S']
        c331, ne331, _ = rolling_results['IPG331S']
        n_common = min(len(ne334), len(ne331))
        if n_common >= 4:
            gaps = [ne334[i] - ne331[i] for i in range(n_common)]
            tau_gap, p_gap = kendalltau(range(len(gaps)), gaps)
            print(f"  Gap trend: Kendall tau = {tau_gap:+.3f}, p = {p_gap:.4f}")
            if tau_gap > 0:
                print("  RESULT: SUPPORTED -- divergence widening")
            elif tau_gap < 0:
                print("  RESULT: NOT SUPPORTED -- divergence narrowing")
            else:
                print("  RESULT: INCONCLUSIVE -- no trend")

    print("\n" + "=" * 80)
    print("DONE. Output files:")
    print(f"  {os.path.join(OUT_DIR, 'sector_layer_summary.csv')}")
    print(f"  {os.path.join(OUT_DIR, 'rolling_neff.csv')}")
    print(f"  {os.path.join(OUT_DIR, 'comp_metals_divergence.csv')}")
    print(f"  {os.path.join(OUT_DIR, 'cross_sector_spectral_corr.csv')}")
    print(f"  Figures in {FIG_DIR}/")
    print("=" * 80)


if __name__ == '__main__':
    analyze_all_sectors()
