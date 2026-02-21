#!/usr/bin/env python3
"""
EMD Timescale Hierarchy: each IMF band plotted individually on its own row,
with time-varying amplitude visible. Rolling N_eff on top.
"""

import os
import numpy as np
import pandas as pd
import warnings
from scipy.signal import hilbert

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FW_DIR = os.path.join(BASE_DIR, 'thesis_data', 'framework_verification')
FIG_DIR = os.path.join(BASE_DIR, 'figures')

# ── Load and decompose ──
df = pd.read_csv(os.path.join(FW_DIR, 'INDPRO.csv'), parse_dates=['date'], index_col='date')
raw = df['value'].dropna()
vals = raw.values.astype(float)
growth = np.diff(np.log(vals))
dates = raw.index[1:len(growth)+1]
years = dates.year + dates.month / 12.0

from PyEMD import EEMD
np.random.seed(42)
eemd = EEMD(trials=200, noise_width=0.2)
eemd.noise_seed(42)
print("Running EEMD...")
imfs = eemd.eemd(growth)
n_imfs = len(imfs)
print(f"  {n_imfs} IMFs extracted")

# ── Characterize ──
total_var = sum(np.var(imf) for imf in imfs)
imf_info = []
for k in range(n_imfs):
    imf = imfs[k]
    efrac = np.var(imf) / total_var
    analytic = hilbert(imf)
    inst_freq = np.abs(np.diff(np.unwrap(np.angle(analytic))) / (2.0 * np.pi))
    inst_period_mo = np.where(inst_freq > 0, 1.0 / inst_freq, np.nan)
    char_mo = np.nanmedian(inst_period_mo)
    char_yr = char_mo / 12.0
    imf_info.append({'k': k, 'efrac': efrac, 'period_yr': char_yr, 'sig': efrac > 0.02})
    pstr = f'{char_yr:.2f} yr' if char_yr >= 1 else f'{char_mo:.0f} mo'
    sig = '***' if efrac > 0.02 else ''
    print(f"  IMF {k}: T={pstr}, energy={efrac:.1%} {sig}")

# Economic labels
econ_labels = {
    0: 'High-freq noise',
    1: 'Seasonal (7 mo)',
    2: 'Annual (1.1 yr)',
    3: 'Kitchin inventory (2.5 yr)',
    4: 'Business cycle (5.6 yr)',
    5: 'Juglar investment (12.4 yr)',
    6: 'Kondratiev wave (44 yr)',
    7: 'Secular trend (70 yr)',
    8: 'Residual',
}

# Colors: spectral from fast (violet) to slow (red)
band_colors = [
    '#7b3294',  # 0 - deep purple
    '#5e4fa2',  # 1 - indigo
    '#3288bd',  # 2 - blue
    '#66c2a5',  # 3 - teal
    '#abdda4',  # 4 - green
    '#e6f598',  # 5 - yellow-green
    '#f46d43',  # 6 - orange
    '#d53e4f',  # 7 - red
    '#9e0142',  # 8 - dark red
]

# ── Rolling N_eff (20-yr windows, 1-yr steps) ──
print("Computing rolling N_eff...")
window_mo = 240
step_mo = 12
rolling_years_list = []
rolling_neff_list = []
start = 0
while start + window_mo <= len(growth):
    center_idx = start + window_mo // 2
    center_year = years[min(center_idx, len(years)-1)]
    window = growth[start:start + window_mo]
    try:
        w_eemd = EEMD(trials=100, noise_width=0.2)
        w_eemd.noise_seed(42)
        w_imfs = w_eemd.eemd(window)
        w_total = sum(np.var(im) for im in w_imfs)
        neff = sum(1 for im in w_imfs if np.var(im) / w_total > 0.02)
    except:
        neff = np.nan
    rolling_years_list.append(center_year)
    rolling_neff_list.append(neff)
    start += step_mo

rolling_years = np.array(rolling_years_list)
rolling_neff = np.array(rolling_neff_list, dtype=float)
print(f"  {len(rolling_years)} windows, N_eff [{np.nanmin(rolling_neff):.0f}, {np.nanmax(rolling_neff):.0f}]")

# ── Show bands 0-7 (skip residual IMF 8) ──
show = list(range(min(n_imfs, 8)))
n_show = len(show)

# ── Figure: N_eff on top, then each band individually, original signal at bottom ──
n_panels = n_show + 2  # +1 for N_eff, +1 for original signal
height_ratios = [1.0] + [1.0] * n_show + [1.0]

fig, axes = plt.subplots(n_panels, 1, figsize=(16, 2.0 * n_panels),
                         sharex=True,
                         gridspec_kw={'hspace': 0.05, 'height_ratios': height_ratios})

# ── Panel 0: Rolling N_eff ──
ax = axes[0]
ax.fill_between(rolling_years, rolling_neff, alpha=0.25, color='steelblue')
ax.plot(rolling_years, rolling_neff, color='steelblue', linewidth=1.5)
ax.axhline(5, color='crimson', linestyle='--', linewidth=1, alpha=0.5)
ax.axhspan(4, 5, alpha=0.06, color='green')
ax.set_ylabel('$N_{eff}$', fontsize=10)
ax.set_ylim(2.5, 7.5)
ax.set_yticks([3, 4, 5, 6, 7])
ax.set_title('Timescale Bands Discovered by EEMD  —  INDPRO 1919–2025\n'
             'Each band plotted individually; $N_{eff}$ = number active at each time',
             fontsize=13, fontweight='bold', pad=8)
ax.text(0.99, 0.92, 'Rolling $N_{eff}$ (20-yr windows)',
        transform=ax.transAxes, fontsize=9, ha='right', va='top', color='steelblue')

# ── Panels 1..n_show: individual IMF bands (fastest at top, slowest at bottom) ──
for panel_i, k in enumerate(show):
    ax = axes[panel_i + 1]
    imf = imfs[k][:len(years)]
    efrac = imf_info[k]['efrac']
    period = imf_info[k]['period_yr']
    sig = imf_info[k]['sig']
    color = band_colors[k % len(band_colors)]
    label = econ_labels.get(k, f'IMF {k}')

    # Plot the waveform
    ax.plot(years, imf, color=color, linewidth=0.4 if k < 3 else 0.7, alpha=0.9)
    ax.fill_between(years, imf, 0, alpha=0.15, color=color)
    ax.axhline(0, color='gray', linewidth=0.3)

    # Symmetric y-limits
    ymax = np.percentile(np.abs(imf), 99.5) * 1.3
    ax.set_ylim(-ymax, ymax)

    # Left label: IMF number and period
    pstr = f'{period:.1f} yr' if period >= 1 else f'{period*12:.0f} mo'
    sig_str = '' if sig else '\n[sub-threshold]'
    ax.set_ylabel(f'IMF {k}\n{pstr}', fontsize=9, color=color, fontweight='bold')

    # Right label: economic identification + energy
    marker = '●' if sig else '○'
    ax.text(1.005, 0.5, f'{marker} {label}\n   {efrac:.1%} energy{sig_str}',
            transform=ax.transAxes, fontsize=8.5, va='center', ha='left',
            color=color, fontweight='bold' if sig else 'normal',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.08, edgecolor='none'))

    ax.tick_params(labelsize=7, labelleft=False)

    # Draw r* ratio annotation between this panel and next
    if panel_i < n_show - 1:
        next_k = show[panel_i + 1]
        next_period = imf_info[next_k]['period_yr']
        if period > 0:
            ratio = next_period / period
            # Small text at the boundary between panels
            ax.text(0.003, -0.02, f'r*≈{ratio:.1f}×',
                    transform=ax.transAxes, fontsize=7, color='gray',
                    va='top', ha='left', fontstyle='italic')

# ── Bottom panel: Original signal ──
ax = axes[-1]
ax.fill_between(years, growth, 0, where=growth > 0, alpha=0.2, color='green')
ax.fill_between(years, growth, 0, where=growth < 0, alpha=0.2, color='red')
ax.plot(years, growth, color='black', linewidth=0.25, alpha=0.6)
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_ylabel('Signal', fontsize=9)
ax.set_xlabel('Year', fontsize=11)
ax.text(1.005, 0.5, 'Original\nINDPRO growth',
        transform=ax.transAxes, fontsize=8.5, va='center', ha='left', color='black',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='gray', alpha=0.08, edgecolor='none'))
ax.tick_params(labelsize=7, labelleft=False)

axes[-1].set_xlim(years[0], years[-1])

plt.savefig(os.path.join(FIG_DIR, 'emd_levels.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'emd_levels.png'), dpi=200, bbox_inches='tight')
print("Saved figures/emd_levels.pdf and .png")
plt.close()
