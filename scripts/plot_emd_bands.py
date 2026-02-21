#!/usr/bin/env python3
"""Generate a clear visualization of the EMD timescale bands from INDPRO."""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.signal import hilbert

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FW_DIR = os.path.join(BASE_DIR, 'thesis_data', 'framework_verification')
FIG_DIR = os.path.join(BASE_DIR, 'figures')

# Load INDPRO
df = pd.read_csv(os.path.join(FW_DIR, 'INDPRO.csv'), parse_dates=['date'], index_col='date')
raw = df['value'].dropna()
vals = raw.values.astype(float)
growth = np.diff(np.log(vals))
dates = raw.index[1:len(growth)+1]
years = dates.year + dates.month / 12.0

# Run EEMD
from PyEMD import EEMD
np.random.seed(42)
eemd = EEMD(trials=200, noise_width=0.2)
eemd.noise_seed(42)
print("Running EEMD...")
imfs = eemd.eemd(growth)
print(f"  Got {len(imfs)} IMFs")

# Characterize
total_var = sum(np.var(imf) for imf in imfs)
info = []
for k, imf in enumerate(imfs):
    efrac = np.var(imf) / total_var
    analytic = hilbert(imf)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) / (2.0 * np.pi)
    inst_freq = inst_freq[inst_freq > 0]
    if len(inst_freq) > 10:
        period_mo = np.median(1.0 / inst_freq)
        period_yr = period_mo / 12.0
    else:
        period_yr = np.nan
    info.append({'k': k, 'efrac': efrac, 'period': period_yr, 'sig': efrac > 0.02})

# ── Figure: Timescale bands ──
# Show original signal + each significant IMF + residual trend
sig_indices = [i['k'] for i in info if i['sig']]
# Also include the first two sub-threshold IMFs (Juglar + Kondratiev)
extra = [i['k'] for i in info if not i['sig'] and i['efrac'] > 0.005][:2]
show_indices = sig_indices + extra

# Band labels
band_labels = {
    0: 'Sub-monthly noise',
    1: 'Seasonal (0.6 yr)',
    2: 'Annual (1.1 yr)',
    3: 'Kitchin inventory (2.5 yr)',
    4: 'Business cycle (5.6 yr)',
    5: 'Juglar investment (12.4 yr)',
    6: 'Kondratiev wave (43.5 yr)',
}

# Color scheme: cool to warm for fast to slow
colors = {
    0: '#4575b4',  # deep blue
    1: '#74add1',  # blue
    2: '#abd9e9',  # light blue
    3: '#fee090',  # light orange
    4: '#f46d43',  # orange
    5: '#d73027',  # red
    6: '#a50026',  # dark red
}

n_panels = len(show_indices) + 1  # +1 for original signal
fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.0 * n_panels),
                         sharex=True, gridspec_kw={'hspace': 0.08})

# Panel 0: Original growth signal
ax = axes[0]
ax.plot(years, growth, color='black', linewidth=0.3, alpha=0.8)
ax.fill_between(years, growth, 0, where=growth > 0, alpha=0.15, color='green')
ax.fill_between(years, growth, 0, where=growth < 0, alpha=0.15, color='red')
ax.set_ylabel('Growth', fontsize=9)
ax.set_title('INDPRO Monthly Log-Growth (1919–2025)', fontsize=11, fontweight='bold', loc='left')
ax.axhline(0, color='gray', linewidth=0.3)
ax.tick_params(labelsize=8)

# Each IMF band
for panel_idx, k in enumerate(show_indices):
    ax = axes[panel_idx + 1]
    imf = imfs[k][:len(years)]
    efrac = info[k]['efrac']
    period = info[k]['period']
    sig = info[k]['sig']

    c = colors.get(k, '#888888')
    ax.plot(years, imf, color=c, linewidth=0.5 if k < 3 else 0.8)
    ax.fill_between(years, imf, 0, alpha=0.2, color=c)
    ax.axhline(0, color='gray', linewidth=0.3)

    label = band_labels.get(k, f'IMF {k}')
    energy_pct = f'{efrac:.1%}'
    sig_marker = '' if sig else ' [sub-threshold]'
    ax.set_ylabel(f'IMF {k}', fontsize=9)

    # Right-side annotation
    ax.text(1.01, 0.5, f'{label}\n{energy_pct} energy{sig_marker}',
            transform=ax.transAxes, fontsize=8, va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=c, alpha=0.15))
    ax.tick_params(labelsize=8)

axes[-1].set_xlabel('Year', fontsize=10)
axes[-1].set_xlim(years[0], years[-1])

# Add r* annotations between adjacent panels
for i in range(len(show_indices) - 1):
    k1 = show_indices[i]
    k2 = show_indices[i + 1]
    p1 = info[k1]['period']
    p2 = info[k2]['period']
    if p1 > 0 and p2 > 0:
        ratio = p2 / p1
        # Place text at left margin between panels
        y_mid = (axes[i + 1].get_position().y0 + axes[i + 2].get_position().y0) / 2
        fig.text(0.005, y_mid, f'×{ratio:.1f}', fontsize=7, color='gray',
                 va='center', ha='left', fontstyle='italic')

plt.savefig(os.path.join(FIG_DIR, 'emd_bands.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'emd_bands.png'), dpi=200, bbox_inches='tight')
print(f"Saved figures/emd_bands.pdf and .png")
plt.close()
