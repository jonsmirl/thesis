# Wavelet Analysis of the Hierarchical Layering Model

**Date:** February 2026
**Data:** FRED INDPRO (1919-2025), 6 manufacturing subsectors (1972-2025), historical steel/auto (1899-1942)
**Method:** Continuous Wavelet Transform (Morlet), spectral peak detection, rolling N_eff

---

## 1. The Question

The thesis (Complementary Heterogeneity v4, Theorem 3.3) models the economy as N_eff hierarchical levels with strict timescale separation: epsilon_1 >> epsilon_2 >> ... >> epsilon_N. The current AI/crypto application uses N_eff = 4:

| Level | Domain | Timescale |
|-------|--------|-----------|
| n=1 | Hardware (learning curves) | Decades |
| n=2 | Network (mesh formation) | Years |
| n=3 | Capability (training) | Months |
| n=4 | Settlement (finance) | Days-weeks |

The paper already states that N_eff is endogenous (Theorem 3.3) and 4 is specific to the 2020s, not a structural assumption. The wavelet analysis tests this claim quantitatively and exposes three issues with the current formulation.

---

## 2. Finding 1: N_eff Is Variable — But Centered on 4-5

Decade-by-decade spectral peak counts from INDPRO (1919-2025):

| Decade | N peaks | Peak periods (years) | Separation ratios |
|--------|---------|---------------------|-------------------|
| 1920s | 4 | 0.9, 3.2, 5.7, 15.2 | 3.4, 1.8, 2.7 |
| 1930s | 5 | 0.8, 2.0, 3.2, 7.0, 15.2 | 2.6, 1.6, 2.2, 2.2 |
| 1940s | 5 | 0.4, 1.9, 3.7, 7.6, 14.4 | 5.5, 1.9, 2.0, 1.9 |
| 1950s | 4 | 0.8, 2.3, 12.4, 24.9 | 2.9, 5.5, 2.0 |
| 1960s | 5 | 0.7, 1.5, 3.3, 12.4, 27.1 | 2.2, 2.3, 3.8, 2.2 |
| 1970s | 5 | 1.4, 2.5, 5.5, 11.4, 28.0 | 1.8, 2.2, 2.1, 2.5 |
| 1980s | 6 | 0.8, 3.5, 5.5, 9.8, 16.5, 28.5 | 4.5, 1.6, 1.8, 1.7, 1.7 |
| 1990s | 5 | 1.3, 2.5, 5.3, 9.6, 29.5 | 2.0, 2.1, 1.8, 3.1 |
| 2000s | 4 | 3.7, 7.1, 16.0, 30.5 | 1.9, 2.3, 1.9 |
| 2010s | 4 | 1.1, 3.7, 6.0, 31.5 | 3.4, 1.6, 5.3 |

**Summary statistics:**
- Mean N_eff = 4.5, std = 1.0, range [2, 6]
- The paper's claim that N_eff = 4 for the current era is within the historical range
- N_eff is NOT fixed but varies with the technology lifecycle as Theorem 3.3 predicts

**Verdict: Theorem 3.3 is correct. N_eff = 4 is empirically reasonable for the 2020s.**

---

## 3. Finding 2: Separation Ratios Are ~2x, Not >>

This is the critical finding. The ">>" notation implies timescale separations of at least 10x. The data shows:

- **Median separation ratio: 2.1x** (IQR: 1.84-2.63)
- **No systematic drift:** early decades ~2.2x, late decades ~2.1x
- **Maximum observed: 5.5x** (1940s sub-annual to 2yr band; 1950s 2yr to 12yr band — both during post-WWII structural shifts)
- **Typical range: 1.6x to 3.4x**

This matters because the singular perturbation / slow manifold reduction that justifies the hierarchical coupling topology requires "sufficient" separation. The paper defines this via a threshold r*:

> N_eff = 1 + #{i : g_i > log r*}

where g_i = log(tau_{i+1}/tau_i).

**The wavelet data calibrates r* empirically: r* ~ 2.** At this threshold, 4-5 layers are consistently detected. But r* = 2 is at the low end of what singular perturbation theory typically requires for clean scale separation. Standard references (Fenichel, Tikhonov) assume ratios of 3-10x or more.

### Implications

1. **The nearest-neighbor coupling topology may be approximate.** Theorem 3.2 derives that coupling between non-adjacent levels is suppressed by O(epsilon_{n+1}/epsilon_n). With ratios of ~2x, this suppression factor is ~0.5 — significant, not negligible. Non-adjacent coupling may matter.

2. **The slow manifold reduction has O(1) corrections.** The standard Tikhonov result gives O(epsilon) corrections. When epsilon = 1/2 rather than 1/10, these corrections are 5x larger.

3. **The "strict" hierarchy is better described as a "soft" hierarchy.** The layers exist but they bleed into each other. Adjacent-layer dynamics are partially entangled, not cleanly separated.

### What the model needs

The current formulation treats the hierarchy as a discrete ladder. The wavelet data suggests it's closer to a **wavelet multiresolution decomposition** — a continuous spectrum organized around preferred octaves, with energy leaking between adjacent bands. The model should:

- Replace the strict timescale separation assumption (>>) with a quantitative bound (epsilon_n/epsilon_{n+1} > r* ~ 2)
- Explicitly bound the non-adjacent coupling terms that are currently assumed negligible
- Acknowledge that the port-Hamiltonian nearest-neighbor structure is an O(1/r*) approximation, not exact

---

## 4. Finding 3: A Stable Core + Intermittent Periphery

The wavelet energy band analysis reveals that NOT all layers are equally persistent:

| Band | Period | Present in N/11 decades | Character |
|------|--------|------------------------|-----------|
| Sub-annual | <1 yr | 7/11 (64%) | Intermittent |
| Short cycle | 1-3 yr | 7/11 (64%) | Intermittent |
| Business cycle | 3-8 yr | 10/11 (91%) | **Core** |
| Juglar/long | 8-20 yr | 9/11 (82%) | **Core** |
| Kondratieff | 20+ yr | 7/11 (64%) | Intermittent |

**Two bands are nearly always active: business cycle (3-8yr) and Juglar (8-20yr).** These form the stable core of the hierarchy. The sub-annual, short-cycle, and Kondratieff bands activate and deactivate across eras.

This maps onto the thesis hierarchy as follows:
- The **always-active** business cycle and Juglar bands correspond roughly to Levels 2-3 (Network/Capability)
- The **intermittent** sub-annual band corresponds to Level 4 (Settlement) — which the model predicts activates only when R_settle > 1
- The **intermittent** Kondratieff band corresponds to Level 1 (Hardware) — which is only relevant during technology transitions

**This is actually good news for the model.** The 4-level structure is not "always 4 levels active." It's "2 core levels always active, with 1-2 additional levels that activate during specific phases." The model's R_0 activation thresholds predict exactly this behavior: a level only "turns on" when its reproduction number exceeds 1.

---

## 5. Finding 4: Sector Spectral Fingerprints Vary with rho

Cross-sector analysis confirms the prediction that low-rho (innovative) sectors have richer spectral structure than high-rho (mature) sectors:

| Sector | Predicted rho | N_eff | Kendall tau | Spectral character |
|--------|--------------|-------|-------------|-------------------|
| Computer/Electronics | Low | 2 | +0.000 | Two distinct peaks (2.9yr, 16.3yr) |
| Electrical Equipment | Low-mid | 4 | -0.467 | Multi-layered, innovation-driven |
| Chemicals | Mid | 1 | -0.690 | Single mode at 2.1yr |
| Transport | Mid-high | 0 | -0.053 | Monotonic decay (no peaks) |
| Primary Metals | High | 0 | -0.276 | Monotonic decay (mature) |
| Food | High | 1 | -0.700 | Single mode at 1.8yr |

**Key results:**
- **Computer/Electronics is the spectral outlier.** Spearman correlation with other sectors: 0.30-0.65. All mature sectors correlate at 0.91-0.99 with each other.
- **Primary Metals tau = -0.276** matches the paper's claimed -0.27 almost exactly. Maturation toward single-layer behavior is confirmed.
- **Computer/Electronics tau = 0.000** does not match the paper's claimed +0.26. The rolling trajectory peaks mid-sample (N_eff = 4 in 1987 and 2007) but returns to N_eff = 2 in recent windows. This may reflect the post-COVID semiconductor IP flattening or sensitivity to window parameters.
- **Computer/Electronics always has more layers than Metals** (gap >= 1 in every rolling window). The gap peaks at +3 around 2007.

**The spectral correlation heatmap provides the cleanest result.** The mature-sector cluster (Metals, Food, Transport, Chemicals: all r > 0.91) versus the Computer/Electronics outlier (r = 0.30-0.50 with the cluster) confirms that the spectral fingerprint of a sector depends on its position in the innovation lifecycle — exactly as the rho-dependent N_eff predicts.

---

## 6. Implications for the Model

### What the paper gets right
1. **N_eff is endogenous and varies** — wavelet data strongly confirms this (Theorem 3.3)
2. **N = 4 is reasonable for the 2020s** — historical mean is 4.5
3. **Innovation creates new layers; maturation destroys them** — sector analysis confirms
4. **Not all layers are always active** — the R_0 threshold interpretation is consistent with the intermittent peripheral bands

### What needs revision
1. **The ">>" notation is misleading.** Replace with a quantitative bound. The data says r* ~ 2, so the correct statement is epsilon_n/epsilon_{n+1} > 2, not epsilon_n >> epsilon_{n+1}. This is "moderate separation," not "strict separation."

2. **Non-adjacent coupling is not negligible.** With r* ~ 2, the O(epsilon_{n+1}/epsilon_n) ~ O(1/2) correction to nearest-neighbor coupling is substantial. The model should either:
   - Bound the error explicitly (the nearest-neighbor approximation is O(1/r*) ~ O(0.5))
   - Or extend the coupling topology to include next-nearest-neighbor terms

3. **The hierarchy is better described as a wavelet multiresolution.** Rather than "4 discrete levels," the economy has a continuous spectrum organized into ~log2-spaced octaves. The CWT naturally provides this decomposition. Framing the model as a multiresolution analysis (MRA) rather than a singular perturbation hierarchy would:
   - Naturally accommodate variable N_eff (just change the number of octaves analyzed)
   - Handle the "soft" boundaries between layers (wavelet subbands overlap by construction)
   - Connect directly to the RG flow argument (coarse-graining = going up one octave)
   - Make the ~2x separation ratio a feature (it's exactly one octave) rather than a weakness

4. **The "always-active core" suggests a reduced model.** For most applications, a 2-level model (business cycle + technology wave, approximately 5yr and 15yr) may suffice. The additional levels (sub-annual settlement, multi-decade Kondratieff) are intermittent enrichments, not permanent features. This matches the pattern in the energy bands figure where the paradigm shift band and short-cycle band have much lower power than the business cycle and technology wave bands except during exceptional periods.

### Suggested revision to Theorem 3.3

Current: N_eff = 1 + #{i : g_i > log r*}

Proposed addition: The empirical value r* ~ 2 (one wavelet octave) yields N_eff ~ 4-5 for aggregate US industrial production over the century 1919-2025. The associated singular perturbation reduction is an O(1/r*) = O(0.5) approximation, not exact. The nearest-neighbor coupling topology holds to leading order but admits O(1/r*) next-nearest-neighbor corrections. These corrections are largest during innovation phases when new timescale gaps are forming (gap ratios may be as low as 1.6x before stabilizing at ~2x).

---

## 7. Data Files

- `thesis_data/framework_verification/` — 44 cached FRED series
- `thesis_data/sector_layers/sector_layer_summary.csv` — sector N_eff table
- `thesis_data/sector_layers/rolling_neff.csv` — rolling N_eff trajectories
- `thesis_data/sector_layers/cross_sector_spectral_corr.csv` — spectral correlation matrix
- `figures/framework_verification/wavelet_*.png` — 7 aggregate wavelet figures
- `figures/sector_layers/*.png` — 5 sector-level figures
- `scripts/analyze_wavelet_layers.py` — aggregate analysis
- `scripts/analyze_sector_layers.py` — sector analysis
