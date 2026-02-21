#!/usr/bin/env python3
"""
FDT Technology Wave Detector
============================

Uses the fluctuation-dissipation theorem from Paper 12 to locate technology
waves from financial + production data. No patent data, no technology labels.

The FDT says:  chi = sigma^2 / T

where:
  chi   = responsiveness of sector output to shocks (impulse response)
  sigma^2 = productivity variance (rolling variance of sector growth)
  T     = information temperature (measured from VIX / yield spread)

Two signatures of an active technology wave in sector s:
  1. HIGH sigma^2/T  =>  flat free energy landscape (phase transition)
  2. HIGH autocorrelation time  =>  critical slowing down

Both should peak when a technology wave is active in that sector.

We compute these for 5 manufacturing sectors (1972-2025) and compare
to the EEMD-discovered technology waves.

Data: Sector IP from FRED (cached), VIX (2000+), DGS10-DGS2 yield spread
"""

import os
import numpy as np
import pandas as pd
import warnings
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FW_DIR = os.path.join(BASE_DIR, 'thesis_data', 'framework_verification')
CACHE_DIR = os.path.join(BASE_DIR, 'thesis_data', 'fred_cache')
FIG_DIR = os.path.join(BASE_DIR, 'figures')


def load_series(series_id):
    for path in [os.path.join(FW_DIR, f'{series_id}.csv'),
                 os.path.join(CACHE_DIR, f'{series_id}.csv')]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, parse_dates=['date'], index_col='date')
                return df['value'].dropna()
            except:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                return df.iloc[:, 0].dropna()
    return None


def rolling_acf_time(series, window=60, max_lag=24):
    """Rolling autocorrelation decay time (months to reach 1/e)."""
    n = len(series)
    tau = np.full(n, np.nan)
    for t in range(window, n):
        chunk = series[t-window:t]
        if np.std(chunk) < 1e-10:
            continue
        chunk_centered = chunk - np.mean(chunk)
        acf_vals = []
        for lag in range(1, min(max_lag + 1, window // 2)):
            c = np.corrcoef(chunk_centered[:-lag], chunk_centered[lag:])[0, 1]
            acf_vals.append(c)
        acf_vals = np.array(acf_vals)
        # Find first crossing below 1/e
        threshold = 1.0 / np.e
        crossings = np.where(acf_vals < threshold)[0]
        if len(crossings) > 0:
            tau[t] = crossings[0] + 1  # months
        else:
            tau[t] = max_lag  # still correlated at max lag
    return tau


# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════

print("Loading data...")

# Aggregate INDPRO
indpro = load_series('INDPRO')
indpro_growth = np.diff(np.log(indpro.values.astype(float)))
indpro_dates = indpro.index[1:]

# VIX — daily, resample to monthly
vix_raw = load_series('VIXCLS')
if vix_raw is not None:
    vix_monthly = vix_raw.resample('MS').mean().dropna()
    print(f"  VIX: {len(vix_monthly)} monthly obs, {vix_monthly.index[0].strftime('%Y-%m')} to {vix_monthly.index[-1].strftime('%Y-%m')}")
else:
    print("  WARNING: VIX not found")
    vix_monthly = None

# Yield spread as T proxy (goes back further if available)
dgs10 = load_series('DGS10')
dgs2 = load_series('DGS2')
if dgs10 is not None and dgs2 is not None:
    spread_daily = dgs10 - dgs2.reindex(dgs10.index)
    spread_monthly = spread_daily.resample('MS').mean().dropna()
    print(f"  Yield spread: {len(spread_monthly)} monthly obs")
else:
    spread_monthly = None

# Sectors
sectors = {
    'IPG334S': ('Computer & Electronics', '#984ea3'),
    'IPG331S': ('Primary Metals',         '#e41a1c'),
    'IPG336S': ('Transport Equipment',    '#377eb8'),
    'IPG325S': ('Chemicals',              '#4daf4a'),
    'IPG335S': ('Electrical Equipment',   '#ff7f00'),
}

sector_data = {}
for sid, (name, color) in sectors.items():
    s = load_series(sid)
    if s is not None:
        svals = s.values.astype(float)
        sgrowth = np.diff(np.log(svals[svals > 0]))
        sdates = s.index[1:len(sgrowth)+1]
        sector_data[sid] = {
            'name': name, 'color': color,
            'growth': sgrowth, 'dates': sdates,
            'raw': s,
        }
        print(f"  {name}: {len(sgrowth)} obs")


# ═══════════════════════════════════════════════════════════════
# 2. COMPUTE FDT INDICATORS FOR EACH SECTOR
# ═══════════════════════════════════════════════════════════════

print("\nComputing FDT indicators...")

ROLL_WINDOW = 60  # 5 years rolling
SMOOTH_SIGMA = 12  # 1 year smoothing

for sid, sd in sector_data.items():
    growth = sd['growth']
    dates = sd['dates']
    n = len(growth)

    # (A) Rolling variance  sigma^2_s(t)
    sigma2 = pd.Series(growth, index=dates).rolling(ROLL_WINDOW).var().values

    # (B) Rolling responsiveness chi_s(t):
    #     correlation of sector growth with aggregate IP growth (as shock proxy)
    #     in rolling windows
    agg_aligned = pd.Series(indpro_growth, index=indpro_dates).reindex(dates).values
    chi = np.full(n, np.nan)
    for t in range(ROLL_WINDOW, n):
        sg = growth[t-ROLL_WINDOW:t]
        ag = agg_aligned[t-ROLL_WINDOW:t]
        mask = ~(np.isnan(sg) | np.isnan(ag))
        if mask.sum() > 20:
            slope, _, _, _, _ = stats.linregress(ag[mask], sg[mask])
            chi[t] = abs(slope)

    # (C) Autocorrelation decay time  tau_s(t)
    tau = rolling_acf_time(growth, window=ROLL_WINDOW)

    # (D) Information temperature proxy: use VIX aligned to sector dates
    T_proxy = np.full(n, np.nan)
    if vix_monthly is not None:
        vix_aligned = vix_monthly.reindex(dates, method='ffill')
        T_proxy = vix_aligned.values / 100.0  # scale to reasonable units

    # (E) FDT indicator:  sigma^2 / T
    #     High when landscape is flat (technology wave active)
    fdt_indicator = sigma2 / (T_proxy + 1e-6)

    # (F) Critical slowing down indicator: just tau itself

    # Smooth everything
    sd['sigma2'] = gaussian_filter1d(np.nan_to_num(sigma2), sigma=SMOOTH_SIGMA)
    sd['chi'] = gaussian_filter1d(np.nan_to_num(chi), sigma=SMOOTH_SIGMA)
    sd['tau_acf'] = gaussian_filter1d(np.nan_to_num(tau), sigma=SMOOTH_SIGMA)
    sd['T_proxy'] = gaussian_filter1d(np.nan_to_num(T_proxy), sigma=SMOOTH_SIGMA)
    sd['fdt_indicator'] = gaussian_filter1d(np.nan_to_num(fdt_indicator), sigma=SMOOTH_SIGMA)

    # Also compute the implied landscape curvature:  lambda = chi * T / sigma^2
    # (inverse of FDT indicator times chi)
    lambda_implied = (np.nan_to_num(chi) * np.nan_to_num(T_proxy)) / (np.nan_to_num(sigma2) + 1e-10)
    sd['lambda_implied'] = gaussian_filter1d(lambda_implied, sigma=SMOOTH_SIGMA)

    print(f"  {sd['name']}: FDT indicators computed")


# ═══════════════════════════════════════════════════════════════
# 3. ALSO RUN EEMD ON SECTORS FOR COMPARISON
# ═══════════════════════════════════════════════════════════════

print("\nRunning EEMD for comparison envelopes...")
from PyEMD import EEMD
from scipy.signal import find_peaks


def hp_filter(y, lamb=129600):
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    T = len(y)
    I = sparse.eye(T, format='csc')
    D2 = sparse.diags([1, -2, 1], [0, 1, 2], shape=(T-2, T), format='csc')
    trend = spsolve(I + lamb * D2.T @ D2, y)
    return y - trend


for sid, sd in sector_data.items():
    raw = sd['raw']
    slog = np.log(raw.values.astype(float))
    cycle = hp_filter(slog)
    np.random.seed(42)
    eemd = EEMD(trials=100, noise_width=0.2)
    eemd.noise_seed(42)
    imfs = eemd.eemd(cycle)

    # Find slowest significant IMF
    total_var = sum(np.var(im) for im in imfs)
    slow_amp = np.zeros(len(cycle))
    for im in imfs:
        efrac = np.var(im) / total_var
        analytic = hilbert(im)
        inst_freq = np.abs(np.diff(np.unwrap(np.angle(analytic))) / (2.0 * np.pi))
        inst_period = np.where(inst_freq > 1e-6, 1.0 / inst_freq, np.nan)
        char_yr = np.nanmedian(inst_period) / 12.0
        if char_yr > 5 and efrac > 0.003:
            slow_amp[:len(im)] += np.abs(analytic)

    # Smooth the EEMD envelope
    sd['eemd_envelope'] = gaussian_filter1d(slow_amp, sigma=24)
    sd['eemd_dates'] = raw.index[:len(slow_amp)]
    print(f"  {sd['name']}: EEMD envelope computed")


# ═══════════════════════════════════════════════════════════════
# 4. FIGURE
# ═══════════════════════════════════════════════════════════════

print("\nGenerating figure...")

n_sectors = len(sector_data)
fig = plt.figure(figsize=(18, 4.5 * n_sectors + 2))
gs = gridspec.GridSpec(n_sectors, 1, hspace=0.25)

known_waves = [
    (1973, 1995, 'Microprocessor\nRevolution', '#e5f5e0'),
    (1995, 2008, 'Internet &\nMobile', '#f2e6ff'),
    (2008, 2026, 'Cloud &\nAI', '#fff3e0'),
]

for panel_i, (sid, sd) in enumerate(sector_data.items()):
    ax = fig.add_subplot(gs[panel_i])
    dates = sd['dates']
    yr = dates.year + dates.month / 12.0
    color = sd['color']

    # Shade known eras
    for start, end, label, fc in known_waves:
        if start < yr[-1] and end > yr[0]:
            ax.axvspan(max(start, yr[0]), min(end, yr[-1]),
                       alpha=0.15, color=fc, zorder=0)
            mid = (max(start, yr[0]) + min(end, yr[-1])) / 2
            ax.text(mid, 0.98, label, transform=ax.get_xaxis_transform(),
                    fontsize=7, ha='center', va='top', color='gray', fontstyle='italic')

    # Create twin axes for the three indicators
    # Main axis: FDT indicator (sigma^2/T)
    fdt = sd['fdt_indicator']
    # Normalize for plotting
    fdt_norm = fdt / (np.nanmax(fdt[ROLL_WINDOW:]) + 1e-10)
    ax.fill_between(yr, fdt_norm, alpha=0.2, color=color)
    ax.plot(yr, fdt_norm, color=color, linewidth=2,
            label=r'FDT indicator $\sigma^2/T$')

    # Autocorrelation time (critical slowing down) — on same scale
    tau = sd['tau_acf']
    tau_norm = tau / (np.nanmax(tau[ROLL_WINDOW:]) + 1e-10)
    ax.plot(yr, tau_norm, color='black', linewidth=1.5, linestyle='--',
            label=r'ACF decay time $\tau$ (critical slowing)')

    # EEMD envelope for comparison (dashed gray)
    eemd_yr = sd['eemd_dates'].year + sd['eemd_dates'].month / 12.0
    eemd_env = sd['eemd_envelope']
    eemd_norm = eemd_env / (np.nanmax(eemd_env) + 1e-10)
    # Align to sector date range
    ax.plot(eemd_yr, eemd_norm, color='gray', linewidth=1.5, linestyle=':',
            alpha=0.7, label='EEMD envelope (comparison)')

    # Mark peaks of FDT indicator
    fdt_valid = fdt_norm.copy()
    fdt_valid[:ROLL_WINDOW + 12] = 0  # mask warmup
    peaks, _ = find_peaks(fdt_valid, distance=60, prominence=0.1)
    for p in peaks:
        if p < len(yr):
            ax.plot(yr[p], fdt_norm[p], 'v', color=color, markersize=10, zorder=5)
            ax.annotate(f'{yr[p]:.0f}', xy=(yr[p], fdt_norm[p]),
                       xytext=(0, 10), textcoords='offset points',
                       fontsize=9, ha='center', fontweight='bold', color=color)

    ax.set_ylabel(sd['name'], fontsize=11, fontweight='bold', color=color)
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['Low', 'Med', 'High'], fontsize=8)
    ax.legend(fontsize=8, loc='upper left', ncol=3)
    ax.set_xlim(1977, 2025)

    # Add implied curvature annotation
    ax2 = ax.twinx()
    lam = sd['lambda_implied']
    lam_norm = lam / (np.nanmax(lam[ROLL_WINDOW:]) + 1e-10)
    ax2.plot(yr, 1.0 - lam_norm, color='red', linewidth=1, alpha=0.4,
             label=r'$1/\lambda$ (landscape flatness)')
    ax2.set_ylabel(r'$1/\lambda$', fontsize=9, color='red', alpha=0.5)
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_yticks([])

fig.suptitle('FDT Technology Wave Detector\n'
             r'$\chi = \sigma^2/T$: high when free energy landscape is flat (technology transition active)'
             '\nCompared with EEMD amplitude envelope — no patent or technology data used',
             fontsize=14, fontweight='bold', y=1.01)

fig.axes[-2].set_xlabel('Year', fontsize=12)  # last main axis

plt.savefig(os.path.join(FIG_DIR, 'fdt_tech_waves.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fdt_tech_waves.png'), dpi=200, bbox_inches='tight')
print("Saved figures/fdt_tech_waves.pdf and .png")
plt.close()

# ═══════════════════════════════════════════════════════════════
# 5. CORRELATION: FDT vs EEMD
# ═══════════════════════════════════════════════════════════════

print("\nFDT vs EEMD correlation:")
for sid, sd in sector_data.items():
    dates = sd['dates']
    yr = dates.year + dates.month / 12.0
    fdt = sd['fdt_indicator']
    eemd_yr = sd['eemd_dates'].year + sd['eemd_dates'].month / 12.0
    eemd_env = sd['eemd_envelope']

    # Align: find common date range
    eemd_series = pd.Series(eemd_env, index=sd['eemd_dates'])
    fdt_series = pd.Series(fdt, index=dates)
    common = pd.DataFrame({'fdt': fdt_series, 'eemd': eemd_series}).dropna()
    if len(common) > 60:
        r, p = stats.spearmanr(common['fdt'], common['eemd'])
        print(f"  {sd['name']:25s}: Spearman r={r:.3f} (p={p:.4f}), n={len(common)}")
    else:
        print(f"  {sd['name']:25s}: insufficient overlap")
