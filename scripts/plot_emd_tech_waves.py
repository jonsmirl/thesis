#!/usr/bin/env python3
"""
Discover technology waves from EEMD of detrended INDPRO.

Strategy:
  1. Remove smooth trend from log-INDPRO (HP filter, lambda=129600 for monthly)
  2. EEMD the cyclical component — no trend to swamp the slow modes
  3. The slow IMFs (period > 10yr) ARE the technology waves
  4. Their Hilbert amplitude envelopes show when each wave was active
  5. Compare discovered peaks/troughs to known technology eras

Also run on individual sectors to see sector-specific technology signatures.
"""

import os
import numpy as np
import pandas as pd
import warnings
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FW_DIR = os.path.join(BASE_DIR, 'thesis_data', 'framework_verification')
CACHE_DIR = os.path.join(BASE_DIR, 'thesis_data', 'fred_cache')
FIG_DIR = os.path.join(BASE_DIR, 'figures')

from PyEMD import EEMD


def load_series(series_id):
    path1 = os.path.join(FW_DIR, f'{series_id}.csv')
    if os.path.exists(path1):
        df = pd.read_csv(path1, parse_dates=['date'], index_col='date')
        return df['value'].dropna()
    path2 = os.path.join(CACHE_DIR, f'{series_id}.csv')
    if os.path.exists(path2):
        df = pd.read_csv(path2, index_col=0, parse_dates=True)
        return df.iloc[:, 0].dropna()
    return None


def hp_filter(y, lamb=129600):
    """Hodrick-Prescott filter. lambda=129600 for monthly (Ravn-Uhlig rule)."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    T = len(y)
    I = sparse.eye(T, format='csc')
    D2 = sparse.diags([1, -2, 1], [0, 1, 2], shape=(T-2, T), format='csc')
    trend = spsolve(I + lamb * D2.T @ D2, y)
    cycle = y - trend
    return trend, cycle


def run_eemd(signal, trials=200, noise_width=0.2, seed=42):
    np.random.seed(seed)
    eemd = EEMD(trials=trials, noise_width=noise_width)
    eemd.noise_seed(seed)
    return eemd.eemd(signal)


def characterize(imfs):
    total_var = sum(np.var(imf) for imf in imfs)
    info = []
    for k, imf in enumerate(imfs):
        efrac = np.var(imf) / total_var if total_var > 0 else 0
        analytic = hilbert(imf)
        amp = np.abs(analytic)
        inst_phase = np.unwrap(np.angle(analytic))
        inst_freq = np.abs(np.diff(inst_phase) / (2.0 * np.pi))
        inst_period_mo = np.where(inst_freq > 1e-6, 1.0 / inst_freq, np.nan)
        char_mo = np.nanmedian(inst_period_mo)
        char_yr = char_mo / 12.0
        info.append({
            'k': k, 'efrac': efrac, 'period_yr': char_yr,
            'amplitude': amp,
        })
    return info


# ═══════════════════════════════════════════════════════════════
# 1. AGGREGATE INDPRO — detrended
# ═══════════════════════════════════════════════════════════════

print("Loading INDPRO...")
raw = load_series('INDPRO')
vals = raw.values.astype(float)
log_ip = np.log(vals)
yr = raw.index.year + raw.index.month / 12.0

print("HP-filtering log-INDPRO...")
trend, cycle = hp_filter(log_ip)

print("Running EEMD on HP cycle...")
cycle_imfs = run_eemd(cycle)
cycle_info = characterize(cycle_imfs)

print(f"  {len(cycle_imfs)} IMFs from detrended INDPRO")
for info in cycle_info:
    pstr = f'{info["period_yr"]:.1f} yr' if info["period_yr"] >= 1 else f'{info["period_yr"]*12:.0f} mo'
    sig = '***' if info['efrac'] > 0.02 else ''
    print(f"  IMF {info['k']}: T={pstr}, energy={info['efrac']:.1%} {sig}")

# ═══════════════════════════════════════════════════════════════
# 2. SECTOR-LEVEL DECOMPOSITIONS
# ═══════════════════════════════════════════════════════════════

sectors = {
    'IPG334S': ('Computer & Electronics', '#984ea3'),
    'IPG331S': ('Primary Metals',         '#e41a1c'),
    'IPG336S': ('Transport Equipment',    '#377eb8'),
    'IPG325S': ('Chemicals',              '#4daf4a'),
    'IPG335S': ('Electrical Equipment',   '#ff7f00'),
}

sector_results = {}
for sid, (name, color) in sectors.items():
    s = load_series(sid)
    if s is None:
        continue
    svals = s.values.astype(float)
    slog = np.log(svals[svals > 0])
    s_yr = s.index.year + s.index.month / 12.0
    s_yr = s_yr[:len(slog)]

    s_trend, s_cycle = hp_filter(slog)
    s_imfs = run_eemd(s_cycle, trials=100)
    s_info = characterize(s_imfs)

    # Find the slowest significant IMF — this is the sector's technology wave
    slow_imfs = [i for i in s_info if i['period_yr'] > 5 and i['efrac'] > 0.005]
    sector_results[sid] = {
        'name': name, 'color': color,
        'yr': s_yr, 'cycle': s_cycle,
        'imfs': s_imfs, 'info': s_info,
        'slow': slow_imfs,
    }
    print(f"\n{name} ({sid}): {len(s_imfs)} IMFs")
    for i in s_info:
        if i['period_yr'] > 3:
            pstr = f'{i["period_yr"]:.1f} yr'
            print(f"  IMF {i['k']}: T={pstr}, energy={i['efrac']:.1%}")

# ═══════════════════════════════════════════════════════════════
# Known technology eras (for COMPARISON shading only)
# ═══════════════════════════════════════════════════════════════

known_waves = [
    (1919, 1929, 'Roaring 20s:\nElectrification &\nMass Production'),
    (1929, 1945, 'Depression\n& War'),
    (1945, 1973, 'Golden Age:\nPetrochemicals,\nSuburbs, Aviation'),
    (1973, 1995, 'Transition:\nMicroprocessors &\nSemiconductors'),
    (1995, 2008, 'Internet &\nMobile'),
    (2008, 2026, 'Cloud, AI &\nPlatforms'),
]
era_colors = ['#fee0d2', '#d9d9d9', '#deebf7', '#e5f5e0', '#f2e6ff', '#fff3e0']

# ═══════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 20))
gs = gridspec.GridSpec(8, 1, height_ratios=[1.5, 1.0, 1.0, 1.0, 0.3, 1.2, 1.2, 1.2],
                       hspace=0.15)

# ── Panel A: Detrended INDPRO cycle with technology-wave IMFs overlaid ──
ax_a = fig.add_subplot(gs[0])
for i, (start, end, label) in enumerate(known_waves):
    ax_a.axvspan(max(start, yr[0]), min(end, yr[-1]),
                 alpha=0.15, color=era_colors[i], zorder=0)
    mid = (max(start, yr[0]) + min(end, yr[-1])) / 2
    ax_a.text(mid, 0.97, label, transform=ax_a.get_xaxis_transform(),
              fontsize=7, ha='center', va='top', color='gray', fontstyle='italic')

ax_a.plot(yr, cycle, color='black', linewidth=0.3, alpha=0.4, label='HP cycle')

# Overlay slow IMFs (period > 8yr) — these are the technology waves
tech_colors = ['#e41a1c', '#377eb8', '#4daf4a']
tech_idx = 0
for info in cycle_info:
    if info['period_yr'] > 8 and info['efrac'] > 0.005:
        imf = cycle_imfs[info['k']][:len(yr)]
        pstr = f'{info["period_yr"]:.0f} yr'
        c = tech_colors[tech_idx % len(tech_colors)]
        ax_a.plot(yr, imf, color=c, linewidth=2, alpha=0.8,
                  label=f'IMF {info["k"]} (T≈{pstr}, {info["efrac"]:.1%})')
        tech_idx += 1

ax_a.set_ylabel('Deviation from trend', fontsize=10)
ax_a.legend(fontsize=8, loc='lower left', ncol=2)
ax_a.set_title('Technology Waves in Detrended INDPRO  (HP cycle → EEMD → slow IMFs)',
               fontsize=14, fontweight='bold')
ax_a.axhline(0, color='gray', linewidth=0.5)

# ── Panels B-D: Individual technology-wave IMFs with amplitude envelopes ──
tech_imfs_list = [i for i in cycle_info if i['period_yr'] > 8 and i['efrac'] > 0.005]
for panel_i, t_info in enumerate(tech_imfs_list[:3]):
    ax = fig.add_subplot(gs[1 + panel_i])
    k = t_info['k']
    imf = cycle_imfs[k][:len(yr)]
    amp = t_info['amplitude'][:len(yr)]
    amp_smooth = gaussian_filter1d(amp, sigma=24)
    period = t_info['period_yr']
    c = tech_colors[panel_i % len(tech_colors)]

    # Shade eras
    for i, (start, end, label) in enumerate(known_waves):
        ax.axvspan(max(start, yr[0]), min(end, yr[-1]),
                   alpha=0.1, color=era_colors[i], zorder=0)

    ax.plot(yr, imf, color=c, linewidth=0.6, alpha=0.7)
    ax.plot(yr, amp_smooth, color='black', linewidth=2, label='Amplitude envelope')
    ax.plot(yr, -amp_smooth, color='black', linewidth=2)
    ax.fill_between(yr, -amp_smooth, amp_smooth, alpha=0.06, color=c)
    ax.axhline(0, color='gray', linewidth=0.3)

    # Find envelope peaks = centers of technology bursts
    from scipy.signal import find_peaks
    min_dist = max(60, int(period * 4))
    env_peaks, _ = find_peaks(amp_smooth, distance=min_dist,
                               prominence=np.std(amp_smooth) * 0.2)
    for ep in env_peaks:
        if ep < len(yr):
            ax.plot(yr[ep], amp_smooth[ep], 'kv', markersize=8, zorder=5)
            ax.annotate(f'{yr[ep]:.0f}', xy=(yr[ep], amp_smooth[ep]),
                       xytext=(0, 10), textcoords='offset points',
                       fontsize=9, ha='center', fontweight='bold')

    pstr = f'{period:.0f} yr' if period >= 2 else f'{period:.1f} yr'
    ax.set_ylabel(f'IMF {k}\nT ≈ {pstr}', fontsize=10, fontweight='bold', color=c)
    ax.legend(fontsize=8, loc='upper right')

# ── Spacer ──
ax_sp = fig.add_subplot(gs[4])
ax_sp.axis('off')
ax_sp.text(0.5, 0.5, '── Sector-Level Technology Signatures (1972–2025) ──',
           transform=ax_sp.transAxes, fontsize=12, ha='center', va='center',
           fontweight='bold', color='gray')

# ── Panels E-G: Sector technology waves ──
sector_list = ['IPG334S', 'IPG336S', 'IPG331S']
for panel_i, sid in enumerate(sector_list):
    if sid not in sector_results:
        continue
    sr = sector_results[sid]
    ax = fig.add_subplot(gs[5 + panel_i])
    s_yr = sr['yr']
    s_cycle = sr['cycle']

    # Shade eras
    for i, (start, end, label) in enumerate(known_waves):
        if start < s_yr[-1] and end > s_yr[0]:
            ax.axvspan(max(start, s_yr[0]), min(end, s_yr[-1]),
                       alpha=0.1, color=era_colors[i], zorder=0)

    # Plot the cycle
    ax.plot(s_yr, s_cycle, color='gray', linewidth=0.3, alpha=0.3)

    # Overlay slow IMFs
    for s_info in sr['slow'][:2]:
        sk = s_info['k']
        s_imf = sr['imfs'][sk][:len(s_yr)]
        s_amp = gaussian_filter1d(s_info['amplitude'][:len(s_yr)], sigma=24)
        pstr = f'{s_info["period_yr"]:.0f} yr'

        ax.plot(s_yr, s_imf, color=sr['color'], linewidth=0.8, alpha=0.7)
        ax.plot(s_yr, s_amp, color='black', linewidth=2)
        ax.plot(s_yr, -s_amp, color='black', linewidth=2)
        ax.fill_between(s_yr, -s_amp, s_amp, alpha=0.08, color=sr['color'])

        # Envelope peaks
        min_dist = max(60, int(s_info['period_yr'] * 4))
        env_peaks, _ = find_peaks(s_amp, distance=min_dist,
                                   prominence=np.std(s_amp) * 0.15)
        for ep in env_peaks:
            if ep < len(s_yr):
                ax.plot(s_yr[ep], s_amp[ep], 'kv', markersize=7, zorder=5)
                ax.annotate(f'{s_yr[ep]:.0f}', xy=(s_yr[ep], s_amp[ep]),
                           xytext=(0, 8), textcoords='offset points',
                           fontsize=8, ha='center', fontweight='bold')

        ax.text(0.99, 0.95, f'Slow IMF: T≈{pstr}',
                transform=ax.transAxes, fontsize=9, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_ylabel(sr['name'], fontsize=10, fontweight='bold', color=sr['color'])
    ax.axhline(0, color='gray', linewidth=0.3)

axes_all = [fig.axes[i] for i in range(len(fig.axes)) if fig.axes[i] != ax_sp]
for ax in axes_all:
    ax.set_xlim(1919, 2026)

fig.axes[-1].set_xlabel('Year', fontsize=12)

plt.savefig(os.path.join(FIG_DIR, 'emd_tech_waves.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'emd_tech_waves.png'), dpi=200, bbox_inches='tight')
print("\nSaved figures/emd_tech_waves.pdf and .png")
plt.close()
