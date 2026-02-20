#!/usr/bin/env python3
"""
Wavelet layer analysis of US Industrial Production (INDPRO).

Computes the continuous wavelet transform of log-differenced INDPRO,
then for each decade identifies distinct spectral peaks, counts active
frequency bands, and computes timescale separation ratios.

Key question: does the data show a FIXED number of layers with fixed
timescale ratios, or do the number of active layers and their
separations change over time?

Author: Connor Doll / Claude Code analysis
"""

import numpy as np
import pandas as pd
import pywt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. Load and preprocess data
# ============================================================

DATA_PATH = "/home/jonsmirl/thesis/thesis_data/framework_verification/INDPRO.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Log-difference to get growth rates (stationary series)
log_val = np.log(df["value"].values)
growth = np.diff(log_val)  # monthly log-growth
dates = df["date"].values[1:]  # one shorter after differencing

print(f"Data: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
print(f"After log-differencing: {len(growth)} observations")
print()

# ============================================================
# 2. Continuous Wavelet Transform
# ============================================================

# We want to resolve periods from ~3 months to ~30 years.
# With monthly data, period = scale * dt where dt = 1/12 year.
# For Morlet wavelet (cmor1.5-1.0), the relationship is:
#   period_months = scale * central_period_of_wavelet
# For cmor1.5-1.0: central frequency ~ 1.0 Hz, so period_months ~ scale
#
# We'll use a log-spaced set of scales covering 3 months to 40 years (480 months).

# Use complex Morlet wavelet
wavelet = "cmor1.5-1.0"

# Scales: from 3 months to 480 months (40 years), log-spaced
min_period_months = 3
max_period_months = 480
n_scales = 300

# For cmor wavelet, period = scale / center_frequency
# center_frequency for cmor1.5-1.0 is 1.0, so period_months ~ scale
scales = np.logspace(np.log10(min_period_months), np.log10(max_period_months), n_scales)

# Compute CWT
coefficients, frequencies = pywt.cwt(growth, scales, wavelet, sampling_period=1.0)
# frequencies are in cycles/month; convert to periods in months
periods_months = 1.0 / frequencies
periods_years = periods_months / 12.0

# Wavelet power
power = np.abs(coefficients) ** 2

print(f"CWT computed: {power.shape[0]} scales x {power.shape[1]} time points")
print(f"Period range: {periods_years.min():.2f} to {periods_years.max():.1f} years")
print()

# ============================================================
# 3. Decade-averaged power spectra
# ============================================================

# Build decade labels
date_years = pd.to_datetime(dates).year
decade_labels = (date_years // 10) * 10

# Define decades to analyze
decades = sorted(set(decade_labels))
# Filter to full or near-full decades
decades = [d for d in decades if d >= 1920 and d <= 2020]

print("=" * 100)
print(f"{'Decade':<10} {'N_peaks':>8} {'Peak periods (years)':>45} {'Separation ratios':>35}")
print("=" * 100)

results = []

for decade in decades:
    mask = decade_labels == decade
    n_months = mask.sum()
    
    if n_months < 24:  # need at least 2 years
        continue
    
    # Average power spectrum for this decade
    avg_power = power[:, mask].mean(axis=1)
    
    # Normalize to unit max for peak detection
    avg_power_norm = avg_power / avg_power.max()
    
    # Smooth slightly to suppress noise (sigma in log-scale index units)
    avg_power_smooth = gaussian_filter1d(avg_power_norm, sigma=3)
    
    # Find peaks in the smoothed spectrum
    # Use prominence-based peak detection to find DISTINCT spectral peaks
    # Require prominence > 5% of max to filter noise
    peaks, properties = find_peaks(
        avg_power_smooth,
        prominence=0.05,
        distance=10,  # minimum separation of ~10 scale indices
        height=0.05   # minimum height
    )
    
    # If no peaks found, relax criteria
    if len(peaks) == 0:
        peaks, properties = find_peaks(
            avg_power_smooth,
            prominence=0.02,
            distance=5,
            height=0.02
        )
    
    # Get periods of peaks, sorted from shortest to longest
    peak_periods = np.sort(periods_years[peaks])
    
    # Also check if there's power at the longest scales (secular trend)
    # by looking at the last 20% of scales
    long_scale_power = avg_power_smooth[-int(0.2 * len(avg_power_smooth)):].mean()
    
    # Compute separation ratios between adjacent peaks
    if len(peak_periods) >= 2:
        ratios = peak_periods[1:] / peak_periods[:-1]
        ratio_str = ", ".join([f"{r:.1f}" for r in ratios])
    else:
        ratios = []
        ratio_str = "--"
    
    period_str = ", ".join([f"{p:.1f}" for p in peak_periods])
    
    print(f"{decade}s    {len(peak_periods):>5}     {period_str:>45}   {ratio_str:>35}")
    
    results.append({
        "decade": decade,
        "n_peaks": len(peak_periods),
        "peak_periods": peak_periods.tolist(),
        "separation_ratios": ratios.tolist() if len(ratios) > 0 else [],
        "n_months": n_months,
    })

print("=" * 100)
print()

# ============================================================
# 4. Detailed analysis: stability of structure
# ============================================================

print("\n" + "=" * 80)
print("DETAILED ANALYSIS: LAYER STABILITY")
print("=" * 80)

# Count how many peaks fall in each frequency band across decades
bands = {
    "Sub-annual (< 1 yr)": (0, 1),
    "Short cycle (1-3 yr)": (1, 3),
    "Business cycle (3-8 yr)": (3, 8),
    "Juglar/Long (8-20 yr)": (8, 20),
    "Kuznets/Secular (20+ yr)": (20, 100),
}

print(f"\n{'Band':<30}", end="")
for r in results:
    print(f" {r['decade']}s", end="")
print()
print("-" * 30 + "-" * 8 * len(results))

for band_name, (lo, hi) in bands.items():
    print(f"{band_name:<30}", end="")
    for r in results:
        count = sum(1 for p in r["peak_periods"] if lo < p <= hi)
        marker = "  *" * count if count > 0 else "  ."
        print(f"{marker:>8}", end="")
    print()

print()

# ============================================================
# 5. Summary statistics
# ============================================================

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

n_peaks_list = [r["n_peaks"] for r in results]
print(f"\nNumber of distinct peaks per decade:")
print(f"  Range: {min(n_peaks_list)} to {max(n_peaks_list)}")
print(f"  Mean:  {np.mean(n_peaks_list):.1f}")
print(f"  Std:   {np.std(n_peaks_list):.1f}")

all_ratios = []
for r in results:
    all_ratios.extend(r["separation_ratios"])

if all_ratios:
    print(f"\nTimescale separation ratios (all decades pooled):")
    print(f"  N ratios: {len(all_ratios)}")
    print(f"  Mean:     {np.mean(all_ratios):.2f}")
    print(f"  Median:   {np.median(all_ratios):.2f}")
    print(f"  Std:      {np.std(all_ratios):.2f}")
    print(f"  Range:    {min(all_ratios):.2f} to {max(all_ratios):.2f}")

# ============================================================
# 6. Test: fixed vs changing structure
# ============================================================

print("\n" + "=" * 80)
print("KEY QUESTION: FIXED vs CHANGING LAYER STRUCTURE")
print("=" * 80)

# Test 1: Is the number of peaks constant?
peak_counts = [r["n_peaks"] for r in results]
cv_peaks = np.std(peak_counts) / np.mean(peak_counts) if np.mean(peak_counts) > 0 else 0

print(f"\n1. Peak count variation (CV = std/mean): {cv_peaks:.3f}")
if cv_peaks < 0.15:
    print("   -> LOW variation: number of active layers is approximately constant")
elif cv_peaks < 0.30:
    print("   -> MODERATE variation: some change in number of active layers")
else:
    print("   -> HIGH variation: number of active layers changes substantially")

# Test 2: Are peak locations stable?
# For each band, compute the fraction of decades with a peak in that band
print(f"\n2. Band activation frequency:")
for band_name, (lo, hi) in bands.items():
    freq = sum(1 for r in results if any(lo < p <= hi for p in r["peak_periods"])) / len(results)
    stability = "STABLE" if freq > 0.7 else ("INTERMITTENT" if freq > 0.3 else "RARE")
    print(f"   {band_name:<30}: {freq:.0%} of decades ({stability})")

# Test 3: Are separation ratios stable?
print(f"\n3. Separation ratio stability:")
# Group ratios by decade and look at the distribution per decade
for r in results:
    if r["separation_ratios"]:
        ratio_str = ", ".join([f"{x:.1f}" for x in r["separation_ratios"]])
        print(f"   {r['decade']}s: [{ratio_str}]")
    else:
        print(f"   {r['decade']}s: [single peak]")

# ============================================================
# 7. Finer time resolution: 20-year sliding windows
# ============================================================

print("\n\n" + "=" * 80)
print("SLIDING WINDOW ANALYSIS (20-year windows, 5-year steps)")
print("=" * 80)

window_months = 240  # 20 years
step_months = 60     # 5 years

print(f"\n{'Window':<18} {'N_peaks':>8} {'Peak periods (years)':>50} {'Ratios':>30}")
print("-" * 110)

window_results = []
start_idx = 0

while start_idx + window_months <= len(growth):
    end_idx = start_idx + window_months
    
    start_date = pd.Timestamp(dates[start_idx])
    end_date = pd.Timestamp(dates[end_idx - 1])
    
    # Average power over this window
    avg_power = power[:, start_idx:end_idx].mean(axis=1)
    avg_power_norm = avg_power / avg_power.max()
    avg_power_smooth = gaussian_filter1d(avg_power_norm, sigma=3)
    
    peaks, properties = find_peaks(
        avg_power_smooth,
        prominence=0.05,
        distance=10,
        height=0.05
    )
    
    if len(peaks) == 0:
        peaks, properties = find_peaks(
            avg_power_smooth,
            prominence=0.02,
            distance=5,
            height=0.02
        )
    
    peak_periods = np.sort(periods_years[peaks])
    
    if len(peak_periods) >= 2:
        ratios = peak_periods[1:] / peak_periods[:-1]
        ratio_str = ", ".join([f"{r:.1f}" for r in ratios])
    else:
        ratios = []
        ratio_str = "--"
    
    period_str = ", ".join([f"{p:.1f}" for p in peak_periods])
    label = f"{start_date.year}-{end_date.year}"
    
    print(f"{label:<18} {len(peak_periods):>5}     {period_str:>50}   {ratio_str:>30}")
    
    window_results.append({
        "window": label,
        "n_peaks": len(peak_periods),
        "peak_periods": peak_periods.tolist(),
        "separation_ratios": ratios.tolist(),
    })
    
    start_idx += step_months

# ============================================================
# 8. Overall verdict
# ============================================================

print("\n\n" + "=" * 80)
print("OVERALL VERDICT")
print("=" * 80)

# Compute statistics on the sliding window results
sw_peaks = [r["n_peaks"] for r in window_results]
sw_ratios = []
for r in window_results:
    sw_ratios.extend(r["separation_ratios"])

if sw_ratios:
    print(f"""
Sliding window statistics:
  Peak count:  mean = {np.mean(sw_peaks):.1f}, std = {np.std(sw_peaks):.1f}, range = [{min(sw_peaks)}, {max(sw_peaks)}]
  Separation ratios (N={len(sw_ratios)}):
    mean = {np.mean(sw_ratios):.2f}, median = {np.median(sw_ratios):.2f}, std = {np.std(sw_ratios):.2f}
    IQR = [{np.percentile(sw_ratios, 25):.2f}, {np.percentile(sw_ratios, 75):.2f}]
""")
else:
    print(f"""
Sliding window statistics:
  Peak count: mean = {np.mean(sw_peaks):.1f}, std = {np.std(sw_peaks):.1f}
""")

# Characterize the typical separation
if sw_ratios:
    med_ratio = np.median(sw_ratios)
    # Does the typical ratio match epsilon-separation?
    # Port-Hamiltonian timescale separation needs ratio >> 1 (typically 3-10)
    if med_ratio > 3:
        separation_verdict = "CLEAR timescale separation (ratio > 3x)"
    elif med_ratio > 2:
        separation_verdict = "MODERATE timescale separation (ratio 2-3x)"
    else:
        separation_verdict = "WEAK timescale separation (ratio < 2x)"
    
    print(f"Typical timescale separation: median ratio = {med_ratio:.1f}x -> {separation_verdict}")
    
    # Is the number of layers approximately constant?
    if np.std(sw_peaks) / np.mean(sw_peaks) < 0.2:
        layer_verdict = "approximately FIXED number of active layers"
    else:
        layer_verdict = "VARIABLE number of active layers"
    
    print(f"Layer count: {layer_verdict} ({np.mean(sw_peaks):.1f} +/- {np.std(sw_peaks):.1f})")
    
    # Are the separation ratios stable or drifting?
    # Compare first half vs second half of windows
    mid = len(window_results) // 2
    early_ratios = []
    late_ratios = []
    for r in window_results[:mid]:
        early_ratios.extend(r["separation_ratios"])
    for r in window_results[mid:]:
        late_ratios.extend(r["separation_ratios"])
    
    if early_ratios and late_ratios:
        print(f"\nSeparation ratio drift:")
        print(f"  Early windows (pre-{window_results[mid]['window']}): "
              f"median = {np.median(early_ratios):.2f}")
        print(f"  Late windows  (post-{window_results[mid]['window']}): "
              f"median = {np.median(late_ratios):.2f}")
        
        ratio_change = abs(np.median(late_ratios) - np.median(early_ratios)) / np.median(early_ratios)
        if ratio_change < 0.2:
            print(f"  -> Separation ratios are STABLE over time ({ratio_change:.0%} change)")
        else:
            print(f"  -> Separation ratios DRIFT over time ({ratio_change:.0%} change)")

print("\n" + "=" * 80)
print("INTERPRETATION FOR THESIS FRAMEWORK")
print("=" * 80)
print("""
The thesis posits a 4-level hierarchy with strict timescale separation:
  Level 1 (slowest): Decades (hardware learning curves)
  Level 2: Years (agent adoption dynamics)
  Level 3: Months (training feedback)
  Level 4 (fastest): Days-weeks (settlement)

INDPRO (monthly, 1919-2025) can resolve Levels 1-3 but NOT Level 4.
The CWT analysis above tests whether industrial production shows:
  (a) A stable number of frequency layers across eras
  (b) Consistent timescale separation between layers
  (c) Fixed or drifting characteristic periods

If the layer structure is stable with consistent ratios ~3-10x,
this supports the port-Hamiltonian timescale separation assumption.
If layers appear and disappear, or ratios are highly variable,
the fixed-hierarchy model may need modification.
""")
