# The α Problem: Diagnosis and Solutions

## Working Notes for Connor's "Endogenous Decentralization" Paper

---

## 1. The Problem Restated

Connor's panel OLS regression of semiconductor price on cumulative production yields α ≈ 0.77 (implying a ~59% cost reduction per doubling of cumulative output). This is far too high. The literature consensus:

| Study | Method | Data | Learning Rate (α) | Cost Reduction per Doubling |
|-------|--------|------|-------|------|
| **Irwin & Klenow (1994)** | IV, firm-level panel | 7 DRAM generations, 1974–92 | ~0.32 | ~20% |
| **Goldberg et al. (2024)** | Structural model, proprietary | Microprocessors/SoCs, 2004–15 | ~0.05 (firm-node) | ~3.4% |
| **Goldberg et al. (2024)** | w/ within-firm spillovers | Same | ~0.07 | ~4.7% |
| **Goldberg et al. (2024)** | w/ cross-border spillovers | Same | ~0.12 | ~8% |
| **Carlino et al. (2025)** | Piecewise regression | 87 technologies incl. DRAM | Varies by segment | Varies |
| **Industry "lore"** | Various | Mixed | ~0.32–0.51 | ~20–30% |
| **Connor's OLS** | Pooled OLS | 1984–2024 | ~0.77 | ~59% |

Connor's estimate is 2–4x higher than the canonical Irwin & Klenow estimate and an order of magnitude above Goldberg et al.'s more recent structural estimates. A referee will notice this immediately.

## 2. Root Cause Diagnosis

### 2.1 Measurement Error in Early Cumulative Production

The script uses 1984 cumulative production at 0.00001 exabytes and 2024 at 3,200 EB — a 320-million-fold range. The 1984–1990 figures are almost certainly too low because:

- WSTS began collecting **revenue** data in 1976, but did not track **bits shipped** as a separate series. Their Blue Book (freely downloadable) covers monthly semiconductor billings by region, not physical production volumes.
- Converting revenue to bits requires price-per-bit data, which itself is what we're trying to estimate — creating circularity.
- Before the mid-1990s, DRAM was only one component of total semiconductor output. Total semiconductor bits shipped is dominated by memory since ~2000, but in the 1980s the mix was different.
- The Irwin & Klenow data came directly from individual firms' quarterly reports — not from aggregate industry estimates.

**If the 1984 cumulative figure is even 10× too low** (e.g., should be 0.0001 EB instead of 0.00001), it compresses the log-log x-axis and inflates α. The Chow tests confirm the structural break: the curve isn't a single power law.

### 2.2 The Structural Break Pattern

The subsample estimates tell a coherent story:

| Period | α estimate | Interpretation |
|--------|-----------|----------------|
| 1984–1995 | ~0.38 | Plausible — close to Irwin & Klenow's 0.32 |
| 1996–2010 | ~0.75–0.92 | Implausibly high — likely artifact of measurement error or regime change |
| 2011–2024 | ~0.31 | Plausible — consistent with mature industry slowing |

The middle period coincides with the DRAM consolidation era (many firms exited), the transition from SDRAM→DDR→DDR2→DDR3, and massive capacity buildouts in Korea and Taiwan. It also coincides with the period where AI Impacts documents a slowdown in the DRAM price decline rate (from ~36% annual decline pre-2010 to ~12–15% after 2010).

### 2.3 The Endogeneity Problem

Even without the measurement error, OLS on `log(price) = a + α·log(cumulative_production)` has a classic simultaneous equations problem: **lower prices stimulate demand, which increases cumulative production**. This is exactly why Irwin & Klenow used instrumental variables — and it's why their IV estimates (α ≈ 0.32, learning rate ≈ 20%) are lower than naïve OLS would suggest. Upward bias in α from OLS is expected.

## 3. Recommended Solutions (In Order of Preference)

### Option A: Use Irwin & Klenow as the Baseline, with Honest Discussion (RECOMMENDED)

This is the cleanest approach for a thesis that isn't primarily about estimating learning curves but rather about using them as a parameter in a decentralization argument.

**Implementation:**
1. Cite Irwin & Klenow (1994) α ≈ 0.32 (learning rate ≈ 20%) as the canonical estimate
2. Cite Goldberg et al. (2024) as showing even lower rates (3.4–8%) in modern data
3. Cite Carlino et al. (2025) for the evidence that learning rates change over time (structural breaks are the norm, not the exception — they find breakpoints in 66% of 87 technology datasets)
4. Show Connor's own OLS results as a robustness check, flagging the structural breaks
5. Run the decentralization model with α ∈ {0.15, 0.20, 0.32} as a sensitivity analysis

**Draft language:**
> "Following Irwin and Klenow (1994), who estimate learning rates averaging 20% using instrumental variables on firm-level DRAM data (1974–1992), we adopt a baseline learning exponent of α = 0.32. More recent structural estimates by Goldberg et al. (2024) suggest even lower rates at the firm-technology node level (3.4–8%), while Carlino et al. (2025) document that learning rates frequently change over time — structural breaks are present in two-thirds of technology datasets examined. Our own panel OLS using aggregate semiconductor data (1984–2024) yields a substantially higher α = 0.77, but Chow tests reject the null of parameter stability at all conventional significance levels (F-statistics ranging from 4.2 to 41.1 against critical values of ~3.2), consistent with well-documented measurement challenges in pre-1995 cumulative production estimates and the simultaneous equations bias that motivates IV estimation. Table X reports sensitivity analysis across the plausible range of learning exponents."

### Option B: Estimate a Piecewise Model (More Work, Higher Payoff)

If Connor wants to contribute an original empirical result, estimate a piecewise log-linear model following Carlino et al. (2025):

**Implementation:**
1. Use the Bai-Perron (1998, 2003) method for endogenous structural break detection in the log-log regression
2. This will formally identify breakpoint years and estimate separate α values for each regime
3. Report the regime-specific estimates alongside the pooled OLS
4. Use R package `strucchange` or Python `ruptures` for implementation

**Advantages:** Directly addresses the referee concern. Shows Connor engaged seriously with the data rather than papering over problems.

**Key reference:** Carlino, A., Wongel, A., Duan, L., Virgüez, E., Davis, S.J., Edwards, M.R., & Caldeira, K. (2025). "Variability of technology learning rates." *Joule* (available as working paper, published in revised form).

### Option C: Reconstruct Better Cumulative Production Data (Most Work)

If Connor wants to actually fix the data:

1. **WSTS Blue Book** — freely downloadable from wsts.org, has monthly semiconductor revenue by region back to the 1970s. Convert to bits using price-per-bit estimates from VLSI Research / ITRS (available in the Kurzweil Singularity Is Near database and the McCallum historical memory price dataset at jcmit.net/memoryprice.htm).

2. **IC Insights / VLSI Research** — commercial databases that may be accessible through Tufts library. These track unit shipments and bit shipments for DRAM specifically.

3. **Performance Curve Database** (pcdb.santafe.edu) — the Santa Fe Institute's database used by Carlino et al. Contains cost-vs-cumulative-production data for DRAM and other technologies. **Connor should check this first** — it may already have exactly the series he needs.

4. **John McCallum's dataset** (jcmit.net/memoryprice.htm) — historical DRAM prices per GB from 1957 to ~2018, used by AI Impacts. Combines VLSI Research data (1971–2000) with ITRS projections (2001–2018).

## 4. What NOT to Do

- **Don't just report α = 0.77 without qualification.** A referee who knows Irwin & Klenow will reject immediately.
- **Don't claim the structural breaks are a "novel finding."** Carlino et al. (2025) just published exactly this point, and it's been discussed in the learning curve literature for decades.
- **Don't try to instrument without good instruments.** Bad IV is worse than OLS with honest caveats.
- **Don't drop early observations to "fix" the break.** Truncating the sample to get a nicer α is exactly what a referee will suspect.

## 5. Literature to Cite

### Essential
- **Irwin, D.A. & Klenow, P.J. (1994).** "Learning-by-Doing Spillovers in the Semiconductor Industry." *Journal of Political Economy*, 102(6), 1200–1227. [The canonical cite — IV estimation, α ≈ 0.32, learning rate ≈ 20%]
- **Goldberg, P., Juhász, R., Lane, N., Lo Forte, G., & Thurk, J. (2024).** "Industrial Policy in the Global Semiconductor Sector." NBER Working Paper 32651. [Most recent estimates — learning rate 3.4–8%, much lower than "industry lore"]
- **Carlino, A. et al. (2025).** "Variability of technology learning rates." *Joule*. [Documents that structural breaks in learning curves are the norm, not the exception]

### Recommended
- **Wright, T.P. (1936).** "Factors Affecting the Cost of Airplanes." *Journal of the Aeronautical Sciences*, 3(4), 122–128. [The original]
- **Nagy, B., Farmer, J.D., Bui, Q.M., & Trancik, J.E. (2013).** "Statistical Basis for Predicting Technological Progress." *PLoS ONE*, 8(2), e52669. [Wright's Law vs. Moore's Law comparison — Wright slightly better for hardware technologies]
- **Thompson, P. (2010).** "Learning by Doing." In *Handbook of the Economics of Innovation*, Vol. 1, 429–476. Elsevier. [Comprehensive survey]

## 6. Sensitivity Analysis Framework

For the decentralization model, run with these parameter values:

| Scenario | α | Learning Rate | Source/Justification |
|----------|---|--------------|---------------------|
| Low | 0.12 | ~8% | Goldberg et al. (2024) w/ cross-border spillovers |
| Baseline | 0.32 | ~20% | Irwin & Klenow (1994) canonical estimate |
| High | 0.51 | ~30% | Upper bound of industry estimates / BCG experience curve |
| Connor's OLS | 0.77 | ~59% | Reported for completeness, with structural break caveats |

**Key question for the thesis:** Does the endogenous decentralization argument hold across the plausible range? If it requires α > 0.5 to work, the argument is fragile. If it works at α = 0.20, it's robust.

## 7. Quick Data Sources Connor Can Access This Week

1. **Santa Fe Institute Performance Curve Database:** pcdb.santafe.edu — may already have DRAM cost/production series
2. **WSTS Historical Billings Report:** Free Excel download from wsts.org/67/Historical-Billings-Report — semiconductor revenue by region, monthly, since ~1976
3. **McCallum Memory Price Dataset:** jcmit.net/memoryprice.htm — DRAM $/GB from 1957–2018
4. **AI Impacts DRAM analysis:** aiimpacts.org/trends-in-dram-price-per-gigabyte/ — includes Google Sheets with extracted data
5. **Kurzweil/Singularity.com charts:** singularity.com/charts/page58.html — DRAM bits/dollar using VLSI Research and ITRS data

---

*Working document — Jon & Connor — February 2026*
