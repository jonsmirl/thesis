#!/usr/bin/env python3
"""
Wavelet Analysis of Technology Waves in Industrial Production
==============================================================

Uses continuous wavelet transform (CWT) with Morlet wavelet to decompose
industrial production time series into time-frequency scalograms. Unlike
Fourier transforms, wavelets localize in BOTH time and frequency — so we
can see when each technology wave starts, peaks, and dies.

Outputs:
  1. Grand scalogram: INDPRO (1919-2025) — 106 years of aggregate IP
  2. Sector scalograms: 6 modern sectors (1972-2025) side by side
  3. Historical scalograms: steel (1899-1939) and auto (1913-1942)
  4. Dominant period evolution: how each sector's "active wavelength" changes
  5. Cross-wavelet coherence: identify GPT waves (coherent across sectors)
  6. Wave birth/death timeline: when each technology wave appears/disappears

Data: FRED API (cached from prior scripts)
"""

import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    import pywt
except ImportError:
    print("ERROR: pip install PyWavelets"); sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import LogNorm
except ImportError:
    print("ERROR: pip install matplotlib"); sys.exit(1)

from scipy.ndimage import uniform_filter1d

# ============================================================================
# Setup
# ============================================================================

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, 'thesis_data', 'framework_verification')
FIGDIR = os.path.join(BASE, 'figures', 'framework_verification')
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

FRED_API_KEY = os.environ.get('FRED_API_KEY', '')


def load_cached(series_id):
    """Load a cached FRED series."""
    cache = os.path.join(OUTPUT, f'{series_id}.csv')
    if os.path.exists(cache):
        return pd.read_csv(cache, parse_dates=['date'], index_col='date')
    return pd.DataFrame()


def get_growth(df):
    """Monthly log-difference growth rate."""
    s = df['value'].resample('MS').last().dropna()
    return np.log(s).diff().dropna()


def cwt_power(signal, scales, wavelet='morl'):
    """Compute CWT power spectrum (squared magnitude of coefficients)."""
    coeffs, freqs = pywt.cwt(signal, scales, wavelet)
    power = np.abs(coeffs) ** 2
    return power, freqs


# ============================================================================
# Figure 1: Grand Scalogram — INDPRO 1919-2025
# ============================================================================

def fig1_grand_scalogram():
    """
    CWT of total US industrial production, 1919-2025.
    Shows all technology waves as energy ridges in the time-frequency plane.
    """
    print("Figure 1: Grand scalogram — INDPRO (1919-2025)")

    df = load_cached('INDPRO')
    if len(df) == 0:
        print("  INDPRO not cached"); return
    g = get_growth(df)
    print(f"  {len(g)} months, {g.index.min().year}-{g.index.max().year}")

    # Scales: from 3 months to 30 years (360 months)
    # For Morlet wavelet, period ≈ 1.03 * scale
    min_period = 6    # 6 months
    max_period = 360   # 30 years
    n_scales = 200
    scales = np.geomspace(min_period / 1.03, max_period / 1.03, n_scales)
    periods_months = scales * 1.03  # approximate periods in months
    periods_years = periods_months / 12

    power, freqs = cwt_power(g.values, scales)

    # Normalize by scale for visual balance (larger scales have more energy)
    for i in range(len(scales)):
        power[i] /= scales[i]

    fig, axes = plt.subplots(2, 1, figsize=(18, 12), height_ratios=[3, 1],
                              sharex=True)

    # Top: scalogram
    ax = axes[0]
    dates = g.index
    X, Y = np.meshgrid(dates, periods_years)
    im = ax.pcolormesh(X, Y, power, cmap='magma', shading='gouraud',
                        norm=LogNorm(vmin=np.percentile(power, 10),
                                     vmax=np.percentile(power, 99)))
    ax.set_yscale('log')
    ax.set_ylabel('Period (years)', fontsize=12)
    ax.set_title('Continuous Wavelet Transform — US Industrial Production (1919-2025)',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    # Mark technology epochs
    epochs = [
        ('1920', 'Electrification'),
        ('1929', 'Crash'),
        ('1941', 'WWII'),
        ('1945', 'Postwar'),
        ('1970', 'Stagflation'),
        ('1981', 'Volcker'),
        ('1995', 'Internet'),
        ('2001', 'Dotcom'),
        ('2008', 'GFC'),
        ('2012', 'Deep Learning'),
        ('2020', 'COVID'),
    ]
    for date_str, label in epochs:
        d = pd.Timestamp(date_str)
        ax.axvline(d, color='white', ls='--', alpha=0.5, linewidth=0.8)
        ax.text(d, periods_years[2], f' {label}', color='white',
                fontsize=7, va='bottom', rotation=90, alpha=0.8)

    plt.colorbar(im, ax=ax, label='Wavelet power / scale', pad=0.01)

    # Bottom: growth rate for reference
    ax = axes[1]
    ax.plot(dates, g.values, 'k-', linewidth=0.3, alpha=0.5)
    smooth = uniform_filter1d(g.values, size=12)
    ax.plot(dates, smooth, 'b-', linewidth=1.5)
    ax.set_ylabel('IP growth rate', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.2)

    # NBER recessions shading
    recessions = [
        ('1920-01', '1921-07'), ('1923-05', '1924-07'), ('1926-10', '1927-11'),
        ('1929-08', '1933-03'), ('1937-05', '1938-06'), ('1945-02', '1945-10'),
        ('1948-11', '1949-10'), ('1953-07', '1954-05'), ('1957-08', '1958-04'),
        ('1960-04', '1961-02'), ('1969-12', '1970-11'), ('1973-11', '1975-03'),
        ('1980-01', '1980-07'), ('1981-07', '1982-11'), ('1990-07', '1991-03'),
        ('2001-03', '2001-11'), ('2007-12', '2009-06'), ('2020-02', '2020-04'),
    ]
    for start, end in recessions:
        try:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                      alpha=0.15, color='red')
        except:
            pass

    plt.tight_layout()
    for ext in ['.png', '.pdf']:
        plt.savefig(os.path.join(FIGDIR, f'wavelet_grand_scalogram{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved wavelet_grand_scalogram.png/.pdf")


# ============================================================================
# Figure 2: Sector Comparison — 6 Modern Sectors Side by Side
# ============================================================================

def fig2_sector_comparison():
    """
    CWT scalograms for 6 sectors, showing which frequency bands are active
    in each sector and when they appear/disappear.
    """
    print("\nFigure 2: Sector wavelet comparison (1972-2025)")

    sectors = [
        ('IPG334S', 'Computer/Elec\n(active wave)'),
        ('IPG331S', 'Primary Metals\n(mature)'),
        ('IPG325S', 'Chemicals\n(2nd wave)'),
        ('IPG336S', 'Transport Equip\n(EV wave)'),
        ('IPG311S', 'Food\n(control)'),
        ('IPG335S', 'Electrical Equip'),
    ]

    min_period = 6
    max_period = 180  # 15 years
    n_scales = 150
    scales = np.geomspace(min_period / 1.03, max_period / 1.03, n_scales)
    periods_years = scales * 1.03 / 12

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    for idx, (sid, label) in enumerate(sectors):
        ax = axes[idx // 3, idx % 3]
        df = load_cached(sid)
        if len(df) == 0:
            ax.set_title(f'{label}: NO DATA')
            continue

        g = get_growth(df)
        if len(g) < 120:
            ax.set_title(f'{label}: insufficient data')
            continue

        power, freqs = cwt_power(g.values, scales)
        for i in range(len(scales)):
            power[i] /= scales[i]

        dates = g.index
        X, Y = np.meshgrid(dates, periods_years)
        im = ax.pcolormesh(X, Y, power, cmap='magma', shading='gouraud',
                            norm=LogNorm(vmin=np.percentile(power, 10),
                                         vmax=np.percentile(power, 99)))
        ax.set_yscale('log')
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        if idx % 3 == 0:
            ax.set_ylabel('Period (years)', fontsize=10)
        if idx >= 3:
            ax.set_xlabel('Date', fontsize=10)

        # Mark key dates
        for d_str in ['1995', '2008', '2012', '2020']:
            d = pd.Timestamp(d_str)
            if dates.min() < d < dates.max():
                ax.axvline(d, color='white', ls=':', alpha=0.4, linewidth=0.5)

    fig.suptitle('Wavelet Scalograms by Manufacturing Sector (1972-2025)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    for ext in ['.png', '.pdf']:
        plt.savefig(os.path.join(FIGDIR, f'wavelet_sector_comparison{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved wavelet_sector_comparison.png/.pdf")


# ============================================================================
# Figure 3: Historical Scalograms — Steel and Auto
# ============================================================================

def fig3_historical():
    """
    CWT scalograms for historical steel and auto production.
    Shows the frequency structure of early technology waves.
    """
    print("\nFigure 3: Historical wavelet scalograms")

    historical = [
        ('M0135AUSM577NNBR', 'Steel Ingot Production (1899-1939)'),
        ('M0107AUSM543NNBR', 'Automobile Production (1913-1942)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for idx, (sid, title) in enumerate(historical):
        ax = axes[idx]
        df = load_cached(sid)
        if len(df) == 0:
            ax.set_title(f'{title}: NO DATA')
            continue

        g = get_growth(df)
        if len(g) < 60:
            ax.set_title(f'{title}: insufficient data')
            continue

        min_period = 4
        max_period = min(120, len(g) // 3)
        n_scales = 120
        scales = np.geomspace(min_period / 1.03, max_period / 1.03, n_scales)
        periods_years = scales * 1.03 / 12

        power, freqs = cwt_power(g.values, scales)
        for i in range(len(scales)):
            power[i] /= scales[i]

        dates = g.index
        X, Y = np.meshgrid(dates, periods_years)
        im = ax.pcolormesh(X, Y, power, cmap='magma', shading='gouraud',
                            norm=LogNorm(vmin=np.percentile(power, 10),
                                         vmax=np.percentile(power, 99)))
        ax.set_yscale('log')
        ax.set_ylabel('Period (years)', fontsize=10)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.invert_yaxis()

        # Mark events
        events = {
            'M0135AUSM577NNBR': [('1907', 'EAF'), ('1914', 'WWI'),
                                   ('1929', 'Crash'), ('1933', 'Recovery')],
            'M0107AUSM543NNBR': [('1913', 'Assembly\nline'), ('1929', 'Crash'),
                                   ('1941', 'WWII')],
        }
        for d_str, label in events.get(sid, []):
            d = pd.Timestamp(d_str)
            if dates.min() < d < dates.max():
                ax.axvline(d, color='white', ls='--', alpha=0.5, linewidth=0.8)
                ax.text(d, periods_years[2], f' {label}', color='white',
                        fontsize=7, va='bottom', rotation=90, alpha=0.8)

        plt.colorbar(im, ax=ax, label='Power/scale', pad=0.01)

    fig.suptitle('Historical Technology Waves — Wavelet Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    for ext in ['.png', '.pdf']:
        plt.savefig(os.path.join(FIGDIR, f'wavelet_historical{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved wavelet_historical.png/.pdf")


# ============================================================================
# Figure 4: Dominant Period Evolution — How Active Wavelength Changes
# ============================================================================

def fig4_dominant_period():
    """
    For each sector, extract the dominant wavelet period at each time point.
    The "dominant period" is the scale with maximum power at each time step.
    When a new technology wave arrives, the dominant period should shift.
    """
    print("\nFigure 4: Dominant period evolution")

    sectors = [
        ('INDPRO', 'Aggregate IP', 'black'),
        ('IPG334S', 'Computer/Elec', 'blue'),
        ('IPG331S', 'Primary Metals', 'red'),
        ('IPG325S', 'Chemicals', 'green'),
        ('IPG336S', 'Transport', 'purple'),
        ('IPG311S', 'Food', 'orange'),
    ]

    min_period = 12   # 1 year
    max_period = 180   # 15 years
    n_scales = 150
    scales = np.geomspace(min_period / 1.03, max_period / 1.03, n_scales)
    periods_years = scales * 1.03 / 12

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)

    # Top: INDPRO dominant period (1919-2025)
    ax = axes[0]
    df = load_cached('INDPRO')
    if len(df) > 0:
        g = get_growth(df)
        power, _ = cwt_power(g.values, scales)
        for i in range(len(scales)):
            power[i] /= scales[i]

        # Dominant period at each time step (smoothed)
        dom_idx = np.argmax(power, axis=0)
        dom_period = periods_years[dom_idx]
        # Smooth with 60-month window
        dom_smooth = uniform_filter1d(dom_period, size=60)

        ax.plot(g.index, dom_period, 'k-', alpha=0.15, linewidth=0.5)
        ax.plot(g.index, dom_smooth, 'b-', linewidth=2.5, label='60-month avg')
        ax.set_ylabel('Dominant period (years)', fontsize=11)
        ax.set_title('Aggregate IP — Dominant Wavelet Period (1919-2025)',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        # Mark eras
        for d, label in [('1945', 'Postwar'), ('1970', 'Stagflation'),
                          ('1995', 'Internet'), ('2008', 'GFC'), ('2012', 'DL')]:
            ax.axvline(pd.Timestamp(d), color='gray', ls=':', alpha=0.5)
            ax.text(pd.Timestamp(d), ax.get_ylim()[1] * 0.95, f' {label}',
                    fontsize=7, va='top', color='gray')

    # Bottom: sector dominant periods (1972-2025)
    ax = axes[1]
    for sid, name, color in sectors[1:]:  # skip INDPRO
        df = load_cached(sid)
        if len(df) == 0:
            continue
        g = get_growth(df)
        if len(g) < 120:
            continue

        power, _ = cwt_power(g.values, scales)
        for i in range(len(scales)):
            power[i] /= scales[i]

        dom_idx = np.argmax(power, axis=0)
        dom_period = periods_years[dom_idx]
        dom_smooth = uniform_filter1d(dom_period, size=60)
        ax.plot(g.index, dom_smooth, '-', color=color, linewidth=2, label=name)

    ax.set_ylabel('Dominant period (years)', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_title('Sector-Level Dominant Wavelet Periods (1972-2025)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    for ext in ['.png', '.pdf']:
        plt.savefig(os.path.join(FIGDIR, f'wavelet_dominant_period{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved wavelet_dominant_period.png/.pdf")


# ============================================================================
# Figure 5: Wavelet Energy by Band — Technology Wave Timeline
# ============================================================================

def fig5_energy_bands():
    """
    Decompose total wavelet energy into frequency bands:
      - Short cycle (6mo-2yr): business fluctuations
      - Medium cycle (2-7yr): business cycles
      - Long cycle (7-15yr): technology waves / Kuznets
      - Very long (15-30yr): Kondratieff / paradigm shifts

    Track how energy redistributes across bands over time.
    When a new technology wave arrives, energy should appear in the
    long-cycle band at a specific time.
    """
    print("\nFigure 5: Wavelet energy by frequency band")

    df = load_cached('INDPRO')
    if len(df) == 0:
        print("  INDPRO not cached"); return

    g = get_growth(df)

    # Full range of scales
    min_period = 6
    max_period = 360
    n_scales = 200
    scales = np.geomspace(min_period / 1.03, max_period / 1.03, n_scales)
    periods_months = scales * 1.03

    power, _ = cwt_power(g.values, scales)

    # Define frequency bands
    bands = {
        'Short (6mo-2yr)': (6, 24),
        'Business cycle (2-7yr)': (24, 84),
        'Technology wave (7-15yr)': (84, 180),
        'Paradigm shift (15-30yr)': (180, 360),
    }
    band_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(len(bands) + 1, 1, figsize=(18, 14), sharex=True)

    dates = g.index

    for idx, ((band_name, (low, high)), color) in enumerate(zip(bands.items(), band_colors)):
        ax = axes[idx]
        mask = (periods_months >= low) & (periods_months < high)
        if mask.sum() == 0:
            continue

        band_power = power[mask].mean(axis=0)
        # Smooth
        band_smooth = uniform_filter1d(band_power, size=60)

        ax.fill_between(dates, 0, band_power, alpha=0.2, color=color)
        ax.plot(dates, band_smooth, '-', color=color, linewidth=2)
        ax.set_ylabel(band_name, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.2)

        # Annotate peaks
        from scipy.signal import find_peaks
        if len(band_smooth) > 120:
            peaks, props = find_peaks(band_smooth,
                                       distance=120,
                                       prominence=np.std(band_smooth) * 0.5)
            for pk in peaks:
                if pk < len(dates):
                    ax.annotate(f'{dates[pk].year}', xy=(dates[pk], band_smooth[pk]),
                               fontsize=7, ha='center', va='bottom', color=color)

    # Bottom: growth rate for reference
    ax = axes[-1]
    ax.plot(dates, g.values, 'k-', linewidth=0.3, alpha=0.3)
    smooth = uniform_filter1d(g.values, size=12)
    ax.plot(dates, smooth, 'k-', linewidth=1.5)
    ax.set_ylabel('IP growth', fontsize=9)
    ax.set_xlabel('Date', fontsize=11)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.2)

    fig.suptitle('INDPRO Wavelet Energy by Frequency Band (1919-2025)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    for ext in ['.png', '.pdf']:
        plt.savefig(os.path.join(FIGDIR, f'wavelet_energy_bands{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved wavelet_energy_bands.png/.pdf")


# ============================================================================
# Figure 6: Cross-Sector Wavelet Coherence — GPT Detection
# ============================================================================

def fig6_cross_coherence():
    """
    Compute wavelet coherence between pairs of sectors.
    GPT waves should appear as periods of high coherence across ALL sectors
    at specific frequency bands.

    Method: for each time point, compute the average pairwise wavelet
    coherence at each scale. A GPT wave shows up as a ridge of high
    coherence at a specific scale/time.
    """
    print("\nFigure 6: Cross-sector wavelet coherence")

    sector_ids = ['IPG334S', 'IPG331S', 'IPG325S', 'IPG336S', 'IPG311S', 'IPG335S']
    sector_names = ['Comp/Elec', 'Metals', 'Chemicals', 'Transport', 'Food', 'Elec Equip']

    # Load all growth rates and align to common dates
    growth_rates = {}
    for sid in sector_ids:
        df = load_cached(sid)
        if len(df) > 0:
            growth_rates[sid] = get_growth(df)

    if len(growth_rates) < 4:
        print("  Insufficient sectors"); return

    # Align to common dates
    combined = pd.DataFrame({sid: g for sid, g in growth_rates.items()})
    combined = combined.dropna()
    dates = combined.index
    print(f"  {len(combined)} common months, {len(growth_rates)} sectors")

    min_period = 12
    max_period = 120
    n_scales = 100
    scales = np.geomspace(min_period / 1.03, max_period / 1.03, n_scales)
    periods_years = scales * 1.03 / 12

    # Compute CWT for each sector
    all_cwt = {}
    for sid in combined.columns:
        coeffs, _ = pywt.cwt(combined[sid].values, scales, 'morl')
        all_cwt[sid] = coeffs

    # Average pairwise coherence at each (scale, time) point
    sids = list(all_cwt.keys())
    n_pairs = 0
    coherence_sum = np.zeros_like(all_cwt[sids[0]], dtype=float)

    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            # Cross-wavelet power = |W_x * conj(W_y)|
            cross = all_cwt[sids[i]] * np.conj(all_cwt[sids[j]])
            # Smooth cross-wavelet and auto-spectra
            smooth_size = 24  # 2-year smoothing
            cross_smooth = uniform_filter1d(np.abs(cross), size=smooth_size, axis=1)
            auto_i = uniform_filter1d(np.abs(all_cwt[sids[i]]) ** 2, size=smooth_size, axis=1)
            auto_j = uniform_filter1d(np.abs(all_cwt[sids[j]]) ** 2, size=smooth_size, axis=1)
            coherence = cross_smooth ** 2 / (auto_i * auto_j + 1e-20)
            coherence_sum += coherence
            n_pairs += 1

    avg_coherence = coherence_sum / n_pairs

    fig, ax = plt.subplots(figsize=(16, 8))
    X, Y = np.meshgrid(dates, periods_years)
    im = ax.pcolormesh(X, Y, avg_coherence, cmap='hot', shading='gouraud',
                        vmin=0, vmax=np.percentile(avg_coherence, 95))
    ax.set_yscale('log')
    ax.set_ylabel('Period (years)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Cross-Sector Wavelet Coherence — GPT Detection (1972-2025)',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    for d, label in [('1995', 'Internet'), ('2008', 'GFC'),
                      ('2012', 'Deep Learning'), ('2020', 'COVID')]:
        ax.axvline(pd.Timestamp(d), color='cyan', ls='--', alpha=0.6, linewidth=0.8)
        ax.text(pd.Timestamp(d), periods_years[2], f' {label}', color='cyan',
                fontsize=8, va='bottom', rotation=90)

    plt.colorbar(im, ax=ax, label='Average pairwise coherence', pad=0.01)
    plt.tight_layout()
    for ext in ['.png', '.pdf']:
        plt.savefig(os.path.join(FIGDIR, f'wavelet_cross_coherence{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved wavelet_cross_coherence.png/.pdf")


# ============================================================================
# Figure 7: Sector-Specific Wavelet Energy — Individual Wave Fingerprints
# ============================================================================

def fig7_sector_energy_evolution():
    """
    For each sector, show how wavelet energy evolves in 3 bands:
    - Short (< 2yr): noise/seasonal
    - Medium (2-7yr): business cycle
    - Long (7-15yr): technology wave

    The ratio of long-band to total energy = "technology wave intensity".
    Sectors in active innovation should have increasing long-band share.
    """
    print("\nFigure 7: Sector wavelet energy evolution")

    sectors = [
        ('IPG334S', 'Computer/Elec', 'blue'),
        ('IPG331S', 'Primary Metals', 'red'),
        ('IPG325S', 'Chemicals', 'green'),
        ('IPG336S', 'Transport', 'purple'),
        ('IPG311S', 'Food', 'orange'),
        ('IPG335S', 'Electrical Equip', 'brown'),
    ]

    min_period = 6
    max_period = 180
    n_scales = 150
    scales = np.geomspace(min_period / 1.03, max_period / 1.03, n_scales)
    periods_months = scales * 1.03

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    for idx, (sid, name, color) in enumerate(sectors):
        ax = axes[idx // 3, idx % 3]
        df = load_cached(sid)
        if len(df) == 0:
            ax.set_title(f'{name}: NO DATA'); continue

        g = get_growth(df)
        if len(g) < 120:
            ax.set_title(f'{name}: insufficient'); continue

        power, _ = cwt_power(g.values, scales)

        # Bands
        short_mask = (periods_months >= 6) & (periods_months < 24)
        med_mask = (periods_months >= 24) & (periods_months < 84)
        long_mask = (periods_months >= 84) & (periods_months < 180)

        short_e = power[short_mask].sum(axis=0) if short_mask.sum() > 0 else np.zeros(len(g))
        med_e = power[med_mask].sum(axis=0) if med_mask.sum() > 0 else np.zeros(len(g))
        long_e = power[long_mask].sum(axis=0) if long_mask.sum() > 0 else np.zeros(len(g))
        total = short_e + med_e + long_e + 1e-20

        # Smoothed shares
        sm = 60
        short_share = uniform_filter1d(short_e / total, size=sm)
        med_share = uniform_filter1d(med_e / total, size=sm)
        long_share = uniform_filter1d(long_e / total, size=sm)

        dates = g.index
        ax.stackplot(dates,
                     short_share, med_share, long_share,
                     labels=['Short (<2yr)', 'Business (2-7yr)', 'Tech wave (7-15yr)'],
                     colors=['#a6cee3', '#1f78b4', '#b2df8a'],
                     alpha=0.8)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        if idx >= 3:
            ax.set_xlabel('Date', fontsize=10)
        if idx % 3 == 0:
            ax.set_ylabel('Energy share', fontsize=10)
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    fig.suptitle('Wavelet Energy Distribution by Frequency Band',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    for ext in ['.png', '.pdf']:
        plt.savefig(os.path.join(FIGDIR, f'wavelet_sector_energy{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved wavelet_sector_energy.png/.pdf")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  WAVELET ANALYSIS OF TECHNOLOGY WAVES")
    print("=" * 70)

    fig1_grand_scalogram()
    fig2_sector_comparison()
    fig3_historical()
    fig4_dominant_period()
    fig5_energy_bands()
    fig6_cross_coherence()
    fig7_sector_energy_evolution()

    print("\n" + "=" * 70)
    print("  COMPLETE — 7 figures saved to figures/framework_verification/")
    print("=" * 70)


if __name__ == '__main__':
    main()
