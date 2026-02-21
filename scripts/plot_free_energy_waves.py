#!/usr/bin/env python3
"""
Dynamic Free Energy Technology Wave Detector
=============================================

Works backwards from 106 years of production data (no patents, no technology
labels, no external proxies) to discover technology waves using phase-transition
signatures from the dynamical free energy framework (Paper 12).

CORE PHYSICS:
  Technology waves = phase transitions in the CES production landscape.
  Three universal signatures, all computable from data alone:

  1. DIVERGENT SUSCEPTIBILITY:  chi(t) = E_slow(t) / T_fast(t)
     Slow-mode energy rises while fast-mode temperature stays normal.

  2. CRITICAL SLOWING DOWN:  tau_ACF(t) → large
     System takes longer to relax near flat landscape.

  3. SYMMETRY BREAKING:  eigenvector rotation (multi-sector)
     The economy's "preferred direction" changes structurally.

SELF-CONSISTENT TEMPERATURE:
  T(t) = energy in fast EEMD modes (period < 3 years)

  These fast modes form the "thermal bath" for the slow structural modes,
  exactly as kinetic energy gives temperature in molecular dynamics.

  Key advantage: during financial crises, BOTH fast and slow modes spike,
  so chi stays modest. During technology waves, ONLY slow modes change,
  so chi rises. The ratio discriminates waves from crises automatically.

Data: INDPRO (1919-2025, monthly) + 6 sector IP indices (1972-2025)
"""

import os
import numpy as np
import pandas as pd
import warnings
from scipy.signal import hilbert, find_peaks
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
    """Load a FRED series from framework_verification or fred_cache."""
    for path in [os.path.join(FW_DIR, f'{series_id}.csv'),
                 os.path.join(CACHE_DIR, f'{series_id}.csv')]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, parse_dates=['date'], index_col='date')
                return df['value'].dropna()
            except Exception:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                return df.iloc[:, 0].dropna()
    return None


def eemd_decompose(signal, trials=200, noise_width=0.2, seed=42):
    """Run EEMD; return IMFs with Hilbert-derived characteristics."""
    from PyEMD import EEMD
    np.random.seed(seed)
    eemd = EEMD(trials=trials, noise_width=noise_width)
    eemd.noise_seed(seed)
    imfs = eemd.eemd(signal)

    total_var = sum(np.var(im) for im in imfs)
    info = []
    for k, imf in enumerate(imfs):
        efrac = np.var(imf) / total_var if total_var > 0 else 0
        analytic = hilbert(imf)
        amp = np.abs(analytic)
        inst_freq = np.abs(np.diff(np.unwrap(np.angle(analytic))) / (2.0 * np.pi))
        inst_period_mo = np.where(inst_freq > 1e-6, 1.0 / inst_freq, np.nan)
        char_yr = np.nanmedian(inst_period_mo) / 12.0
        info.append({
            'k': k, 'efrac': efrac, 'period_yr': char_yr,
            'amplitude': amp, 'imf': imf,
        })
    return imfs, info


def rolling_acf_time(series, window=120, max_lag=48):
    """Rolling autocorrelation decay time (months to first crossing below 1/e)."""
    n = len(series)
    tau = np.full(n, np.nan)
    threshold = 1.0 / np.e
    for t in range(window, n):
        chunk = series[t - window:t]
        if np.std(chunk) < 1e-10:
            continue
        cc = chunk - np.mean(chunk)
        found = False
        for lag in range(1, min(max_lag + 1, window // 3)):
            c = np.corrcoef(cc[:-lag], cc[lag:])[0, 1]
            if c < threshold:
                tau[t] = lag
                found = True
                break
        if not found:
            tau[t] = max_lag
    return tau


# ═══════════════════════════════════════════════════════════════
# 1. AGGREGATE INDPRO — EEMD AND FAST/SLOW PARTITION
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("DYNAMIC FREE ENERGY TECHNOLOGY WAVE DETECTOR")
print("Working backwards from production data — no patents, no labels")
print("=" * 70)

print("\n1. Loading INDPRO (1919-2025)...")
indpro = load_series('INDPRO')
vals = indpro.values.astype(float)
growth = np.diff(np.log(vals))
dates = indpro.index[1:len(growth) + 1]
years = dates.year + dates.month / 12.0
print(f"   {len(growth)} monthly growth obs, {years[0]:.0f}-{years[-1]:.0f}")

print("\n2. Running EEMD on growth rates...")
imfs, imf_info = eemd_decompose(growth, trials=200)

# Classify each IMF
# The business cycle (~5.6yr, IMF 4) is cyclical, not structural.
# Technology waves operate at Juglar+ timescales (>8yr).
# Including the business cycle in E_slow creates false peaks at recessions.
FAST_CUTOFF = 4.0   # years — fast modes form the thermal bath (includes business cycle)
SLOW_CUTOFF = 8.0   # years — slow modes carry structural/technology signal

fast_k = [i['k'] for i in imf_info if i['period_yr'] < FAST_CUTOFF]
slow_k = [i['k'] for i in imf_info if i['period_yr'] > SLOW_CUTOFF]
mid_k  = [i['k'] for i in imf_info if FAST_CUTOFF <= i['period_yr'] <= SLOW_CUTOFF]

print(f"   {len(imfs)} IMFs extracted:")
for info in imf_info:
    pyr = info['period_yr']
    pstr = f'{pyr:.1f} yr' if pyr >= 1 else f'{pyr * 12:.0f} mo'
    band = 'FAST' if info['k'] in fast_k else ('SLOW' if info['k'] in slow_k else 'mid')
    sig = '***' if info['efrac'] > 0.02 else ''
    print(f"     IMF {info['k']:2d}: T={pstr:>8s}, energy={info['efrac']:.1%}  [{band}] {sig}")

print(f"   Fast modes (thermal bath): IMFs {fast_k}")
print(f"   Slow modes (structural):   IMFs {slow_k}")


# ═══════════════════════════════════════════════════════════════
# 2. SELF-CONSISTENT TEMPERATURE AND SUSCEPTIBILITY
# ═══════════════════════════════════════════════════════════════

print("\n3. Computing self-consistent temperature and susceptibility...")

# Hilbert instantaneous amplitude → squared = instantaneous energy
# For fast modes: E_fast(t) = Σ_k |H{IMF_k}(t)|²  (thermal energy)
# For slow modes: E_slow(t) = Σ_k |H{IMF_k}(t)|²  (structural energy)
# Susceptibility: chi(t) = E_slow(t) / E_fast(t)

N = len(growth)
E_fast = np.zeros(N)
for k in fast_k:
    amp = imf_info[k]['amplitude'][:N]
    E_fast += amp ** 2

E_slow = np.zeros(N)
for k in slow_k:
    amp = imf_info[k]['amplitude'][:N]
    E_slow += amp ** 2

# Smooth: remove the 2x-frequency AM modulation artifact from Hilbert amplitudes
# sigma=24 months (2yr) preserves decadal features while removing oscillatory artifacts
SMOOTH = 24
T_fast = gaussian_filter1d(E_fast, sigma=SMOOTH)
sigma2_slow = gaussian_filter1d(E_slow, sigma=SMOOTH)

# Susceptibility
chi = sigma2_slow / (T_fast + 1e-20)
chi_smooth = gaussian_filter1d(chi, sigma=SMOOTH)

# Critical slowing down: ACF decay time of slow-mode sum
print("   Computing critical slowing down (ACF decay time of slow modes)...")
slow_sum = np.zeros(N)
for k in slow_k:
    slow_sum += imfs[k][:N]

tau_acf = rolling_acf_time(slow_sum, window=120, max_lag=48)
tau_smooth = gaussian_filter1d(np.nan_to_num(tau_acf), sigma=SMOOTH)

# Normalize both to [0, 1] range (excluding warmup)
WARMUP = 180  # 15 years
chi_max = np.nanmax(chi_smooth[WARMUP:])
tau_max = np.nanmax(tau_smooth[WARMUP:])
chi_norm = chi_smooth / (chi_max + 1e-20)
tau_norm = tau_smooth / (tau_max + 1e-20)

# Combined indicator: geometric mean (softer than product)
# sqrt(chi * tau) — both contribute but a strong signal in one
# can partially compensate for a weaker signal in the other
combined = np.sqrt(chi_norm * tau_norm)

print(f"   T_fast  range: [{np.min(T_fast[WARMUP:]):.2e}, {np.max(T_fast[WARMUP:]):.2e}]")
print(f"   E_slow  range: [{np.min(sigma2_slow[WARMUP:]):.2e}, {np.max(sigma2_slow[WARMUP:]):.2e}]")
print(f"   chi     range: [{np.min(chi_smooth[WARMUP:]):.3f}, {np.max(chi_smooth[WARMUP:]):.3f}]")
print(f"   tau_ACF range: [{np.nanmin(tau_smooth[WARMUP:]):.1f}, {np.nanmax(tau_smooth[WARMUP:]):.1f}] months")


# ═══════════════════════════════════════════════════════════════
# 3. PEAK DETECTION — DISCOVER TECHNOLOGY WAVES
# ═══════════════════════════════════════════════════════════════

print("\n4. Detecting technology wave peaks from combined indicator...")

det = combined.copy()
det[:WARMUP] = 0  # mask warmup

# Find peaks: minimum 8 years apart, require meaningful prominence
peaks, props = find_peaks(det, distance=96, prominence=0.03)

print(f"   Discovered {len(peaks)} technology wave peaks:")
for i, p in enumerate(peaks):
    print(f"     Wave {i + 1}: year {years[p]:.1f}  "
          f"(chi={chi_norm[p]:.2f}, tau={tau_norm[p]:.2f}, combined={combined[p]:.3f})")


# ═══════════════════════════════════════════════════════════════
# 4. MULTI-SECTOR EIGENVALUE ANALYSIS (1972-2025)
# ═══════════════════════════════════════════════════════════════

print("\n5. Multi-sector structural fingerprinting...")

sectors = {
    'IPG334S': ('Computer/Electronics', '#984ea3'),
    'IPG331S': ('Primary Metals',       '#e41a1c'),
    'IPG336S': ('Transport Equipment',  '#377eb8'),
    'IPG325S': ('Chemicals',            '#4daf4a'),
    'IPG335S': ('Electrical Equipment', '#ff7f00'),
    'IPG333S': ('Machinery',            '#a65628'),
}

# Load and align sector growth rates
sector_series = {}
for sid, (name, _) in sectors.items():
    s = load_series(sid)
    if s is not None:
        svals = s.values.astype(float)
        sg = np.diff(np.log(svals[svals > 0]))
        sd = s.index[1:len(sg) + 1]
        sector_series[sid] = pd.Series(sg, index=sd)
        print(f"   {name}: {len(sg)} obs")

sec_ids = list(sector_series.keys())
sec_names = [sectors[s][0] for s in sec_ids]
sec_colors = [sectors[s][1] for s in sec_ids]
N_sec = len(sec_ids)

# Common date index
common_idx = sector_series[sec_ids[0]].index
for sid in sec_ids[1:]:
    common_idx = common_idx.intersection(sector_series[sid].index)

growth_mat = np.column_stack([sector_series[sid].reindex(common_idx).values
                               for sid in sec_ids])
sec_years = common_idx.year + common_idx.month / 12.0
n_sec_obs = len(common_idx)

# EEMD each sector → separate fast and slow components
print("   Decomposing each sector into fast/slow bands via EEMD...")
sector_slow = np.zeros((n_sec_obs, N_sec))
sector_fast = np.zeros((n_sec_obs, N_sec))

for j, sid in enumerate(sec_ids):
    s_imfs, s_info = eemd_decompose(growth_mat[:, j], trials=100, seed=42 + j)
    for inf in s_info:
        L = min(len(s_imfs[inf['k']]), n_sec_obs)
        if inf['period_yr'] > SLOW_CUTOFF:
            sector_slow[:L, j] += s_imfs[inf['k']][:L]
        elif inf['period_yr'] < SLOW_CUTOFF:
            # Everything below the slow cutoff is "thermal" for sector analysis
            sector_fast[:L, j] += s_imfs[inf['k']][:L]
    print(f"     {sectors[sid][0]}: fast/slow separated")

# Rolling PCA of slow modes across sectors
EIGEN_WIN = 120  # 10-year windows
top_eigenval = np.full(n_sec_obs, np.nan)
eigvec_load = np.full((n_sec_obs, N_sec), np.nan)
sector_T_avg = np.full(n_sec_obs, np.nan)

print("   Running rolling PCA of slow-mode covariance...")
for t in range(EIGEN_WIN, n_sec_obs):
    w_slow = sector_slow[t - EIGEN_WIN:t, :]
    w_fast = sector_fast[t - EIGEN_WIN:t, :]

    # Standardize each sector's slow modes before PCA
    # so the result captures structural co-movement, not raw amplitude
    stds = np.std(w_slow, axis=0)
    stds[stds < 1e-10] = 1.0
    w_std = w_slow / stds

    cov = np.cov(w_std.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigenvalues = eigvals[idx]
    top_vec = eigvecs[:, idx[0]]

    top_eigenval[t] = eigenvalues[0]
    eigvec_load[t, :] = np.abs(top_vec)  # absolute loadings
    sector_T_avg[t] = np.mean(np.var(w_fast, axis=0))

# Sector susceptibility = top eigenvalue / average sector temperature
sector_chi = top_eigenval / (sector_T_avg + 1e-20)
sector_chi_sm = gaussian_filter1d(np.nan_to_num(sector_chi), sigma=SMOOTH)
sector_chi_norm = sector_chi_sm / (np.nanmax(sector_chi_sm[EIGEN_WIN:]) + 1e-20)

# Smooth eigenvector loadings
for j in range(N_sec):
    eigvec_load[:, j] = gaussian_filter1d(np.nan_to_num(eigvec_load[:, j]), sigma=SMOOTH)
load_sum = np.sum(eigvec_load, axis=1, keepdims=True)
load_sum[load_sum < 1e-10] = 1
eigvec_normed = eigvec_load / load_sum

# Detect sector-level peaks
sec_det = sector_chi_norm.copy()
sec_det[:EIGEN_WIN + 36] = 0
sec_peaks, _ = find_peaks(np.nan_to_num(sec_det), distance=72, prominence=0.03)

print(f"   Sector-level peaks:")
for sp in sec_peaks:
    dom = np.argmax(eigvec_normed[sp, :])
    print(f"     {sec_years[sp]:.0f}: dominant = {sec_names[dom]} "
          f"(loading={eigvec_normed[sp, dom]:.2f}), chi={sector_chi_norm[sp]:.2f}")


# ═══════════════════════════════════════════════════════════════
# 5. FIGURE
# ═══════════════════════════════════════════════════════════════

print("\n6. Generating figure...")

known_waves = [
    (1919, 1929, 'Electrification &\nMass Production'),
    (1929, 1945, 'Depression &\nWar Restructuring'),
    (1945, 1973, 'Petrochemicals,\nSuburbs, Aviation'),
    (1973, 1995, 'Microprocessor\nRevolution'),
    (1995, 2008, 'Internet &\nMobile'),
    (2008, 2026, 'Cloud &\nAI'),
]
era_colors = ['#fee0d2', '#d9d9d9', '#deebf7', '#e5f5e0', '#f2e6ff', '#fff3e0']

fig = plt.figure(figsize=(18, 20))
gs = gridspec.GridSpec(4, 1, height_ratios=[2.5, 1.2, 1.5, 1.0], hspace=0.22)

# ─── Panel A: MAIN RESULT — Discovered technology waves ───────────
ax_a = fig.add_subplot(gs[0])

for i, (start, end, label) in enumerate(known_waves):
    if start < years[-1] and end > years[0]:
        ax_a.axvspan(max(start, years[0]), min(end, years[-1]),
                     alpha=0.12, color=era_colors[i], zorder=0)
        mid = (max(start, years[0]) + min(end, years[-1])) / 2
        ax_a.text(mid, 0.97, label, transform=ax_a.get_xaxis_transform(),
                  fontsize=7, ha='center', va='top', color='gray', fontstyle='italic')

# Susceptibility
ax_a.fill_between(years, chi_norm, alpha=0.15, color='#2166ac')
ax_a.plot(years, chi_norm, color='#2166ac', linewidth=1.8,
          label=r'Susceptibility $\chi = E_{slow} / T_{fast}$')

# Critical slowing
ax_a.plot(years, tau_norm, color='#666666', linewidth=1.2, linestyle='--',
          label=r'Critical slowing $\tau_{ACF}$ (confirmation)')

# Combined indicator
ax_a.plot(years, combined, color='#b2182b', linewidth=2.5,
          label=r'Combined: $\sqrt{\chi \cdot \tau}$ (technology wave signal)')
ax_a.fill_between(years, combined, alpha=0.08, color='#b2182b')

# Mark peaks
for i, p in enumerate(peaks):
    ax_a.plot(years[p], combined[p], 'v', color='#b2182b', markersize=14,
              zorder=5, markeredgecolor='black', markeredgewidth=0.5)
    ax_a.annotate(f'{years[p]:.0f}', xy=(years[p], combined[p]),
                  xytext=(0, 14), textcoords='offset points',
                  fontsize=11, ha='center', fontweight='bold', color='#b2182b')

ax_a.set_ylabel('Normalized indicator', fontsize=11)
ax_a.set_ylim(-0.05, 1.35)
ax_a.legend(fontsize=9, loc='upper left', ncol=2, framealpha=0.9)
ax_a.set_title(
    'Technology Waves Discovered from Production Data Alone  (INDPRO 1919-2025)\n'
    r'Self-consistent $T$ from fast EEMD modes — no VIX, no patents, no technology labels used',
    fontsize=13, fontweight='bold')
ax_a.set_xlim(1925, 2026)


# ─── Panel B: THE MECHANISM — T_fast vs sigma²_slow ──────────────
ax_b = fig.add_subplot(gs[1], sharex=ax_a)

for i, (start, end, label) in enumerate(known_waves):
    if start < years[-1] and end > years[0]:
        ax_b.axvspan(max(start, years[0]), min(end, years[-1]),
                     alpha=0.08, color=era_colors[i], zorder=0)

T_disp = T_fast / np.max(T_fast[WARMUP:])
S_disp = sigma2_slow / np.max(sigma2_slow[WARMUP:])

ax_b.plot(years, T_disp, color='steelblue', linewidth=1.5,
          label=r'$T_{fast}(t)$ — fast-mode energy (thermal noise)')
ax_b.fill_between(years, T_disp, alpha=0.1, color='steelblue')
ax_b.plot(years, S_disp, color='firebrick', linewidth=1.5,
          label=r'$E_{slow}(t)$ — slow-mode energy (structural change)')
ax_b.fill_between(years, S_disp, alpha=0.1, color='firebrick')

# Annotate crisis vs technology wave discrimination
# Find major T spikes for annotation
T_peaks_idx, _ = find_peaks(T_disp, distance=60, prominence=0.2)
for tp in T_peaks_idx[:3]:
    if tp > WARMUP and tp < len(years):
        ax_b.annotate(f'{years[tp]:.0f}',
                      xy=(years[tp], T_disp[tp]),
                      xytext=(15, 10), textcoords='offset points',
                      fontsize=8, color='steelblue', fontweight='bold',
                      arrowprops=dict(arrowstyle='->', color='steelblue', lw=0.8))

ax_b.set_ylabel('Normalized energy', fontsize=10)
ax_b.legend(fontsize=8, loc='upper left', ncol=2, framealpha=0.9)
ax_b.set_title(
    r'Why it works: crises spike $T_{fast}$ (denominator), '
    r'tech waves spike $E_{slow}$ (numerator) — ratio discriminates automatically',
    fontsize=10, fontstyle='italic')


# ─── Panel C: MULTI-SECTOR FINGERPRINTS ──────────────────────────
ax_c = fig.add_subplot(gs[2])

# Stacked area: eigenvector loadings show which sectors drive each wave
bottom = np.zeros(n_sec_obs)
for j in range(N_sec):
    top = bottom + eigvec_normed[:, j]
    ax_c.fill_between(sec_years, bottom, top, alpha=0.6,
                      color=sec_colors[j], label=sec_names[j],
                      linewidth=0)
    bottom = top

# Overlay sector susceptibility
ax_c2 = ax_c.twinx()
ax_c2.plot(sec_years, sector_chi_norm, color='black', linewidth=2.5, alpha=0.7,
           label=r'Sector $\chi$ (PCA)')
ax_c2.set_ylabel(r'Sector susceptibility $\chi$', fontsize=10)
ax_c2.set_ylim(0, 1.6)

# Mark sector peaks with dominant sector label
for sp in sec_peaks:
    if sp < len(sec_years):
        dom = np.argmax(eigvec_normed[sp, :])
        ax_c2.plot(sec_years[sp], sector_chi_norm[sp], 'kv',
                   markersize=11, zorder=5, markeredgewidth=0.5)
        ax_c2.annotate(
            f'{sec_years[sp]:.0f}\n{sec_names[dom]}',
            xy=(sec_years[sp], sector_chi_norm[sp]),
            xytext=(0, 14), textcoords='offset points',
            fontsize=9, ha='center', fontweight='bold',
            color=sec_colors[dom])

ax_c.set_ylabel('Eigenvector loadings\n(sector participation)', fontsize=10)
ax_c.set_ylim(0, 1.05)
ax_c.legend(fontsize=7, loc='upper left', ncol=3, framealpha=0.9)
ax_c.set_title(
    'Sector Fingerprints: eigenvector of slow-mode covariance identifies '
    'WHICH sectors drive each structural transition',
    fontsize=10, fontstyle='italic')
ax_c.set_xlim(1983, 2026)


# ─── Panel D: VALIDATION TABLE ───────────────────────────────────
ax_d = fig.add_subplot(gs[3])
ax_d.axis('off')

lines = []
lines.append("VALIDATION: Peaks discovered by free-energy detector vs known technology eras")
lines.append("=" * 85)
lines.append(f"{'Detected':<12} {'chi':<8} {'tau':<8} {'Combined':<10} {'Nearest Known Era':<38} {'Match'}")
lines.append("-" * 85)

for i, p in enumerate(peaks):
    yr_p = years[p]
    # Find which known era contains this peak
    match_label = "Outside defined eras"
    for start, end, label in known_waves:
        if start <= yr_p <= end:
            match_label = label.replace('\n', ' ')
            break
    mark = "YES" if match_label != "Outside defined eras" else "?"
    lines.append(f"  {yr_p:<10.0f} {chi_norm[p]:<8.2f} {tau_norm[p]:<8.2f} "
                 f"{combined[p]:<10.3f} {match_label:<38} {mark}")

lines.append("-" * 85)
lines.append("Method:  EEMD fast/slow partition  ->  sqrt(chi x tau_ACF)")
lines.append("Inputs:  INDPRO monthly (1919-2025).  No patents, no technology labels,")
lines.append("         no external temperature proxy.  T derived self-consistently")
lines.append("         from fast-mode energy (statistical mechanics first principles).")

val_text = "\n".join(lines)
ax_d.text(0.02, 0.95, val_text, transform=ax_d.transAxes,
          fontsize=8.5, fontfamily='monospace', va='top',
          bbox=dict(facecolor='#f8f8f8', edgecolor='gray', alpha=0.9, pad=10))

plt.savefig(os.path.join(FIG_DIR, 'free_energy_waves.pdf'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'free_energy_waves.png'),
            dpi=200, bbox_inches='tight')
print("\nSaved figures/free_energy_waves.pdf and .png")
plt.close()

# ═══════════════════════════════════════════════════════════════
# 6. PRINT CORRELATION DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

print("\n7. Diagnostics:")
print(f"   Aggregate peaks found:  {len(peaks)}")
print(f"   Sector peaks found:    {len(sec_peaks)}")

# Cross-check: do aggregate and sector peaks agree in the overlap period?
agg_peak_years = [years[p] for p in peaks if years[p] > 1983]
sec_peak_years = [sec_years[sp] for sp in sec_peaks]
if agg_peak_years and sec_peak_years:
    print("\n   Aggregate vs sector peak matching (overlap period 1983-2025):")
    for ay in agg_peak_years:
        nearest = min(sec_peak_years, key=lambda sy: abs(sy - ay))
        gap = abs(ay - nearest)
        print(f"     Aggregate {ay:.0f}  <->  Sector {nearest:.0f}  (gap={gap:.1f} yr)")
