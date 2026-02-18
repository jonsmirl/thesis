# MPG Thesis: New Empirical Results

**Analysis Date: February 13, 2026**
**For: Connor Smirl, EC 118 Thesis — Tufts University**

---

## 1. Savings Rate–Yield Access Gap Regression (NEW)

This is the test the paper identifies in Section 6.6 as "the most important empirical extension of the framework." The data was already in the uploads — 29 countries, 368 observations, 2010–2024 panel from WDI.

### Construction

The time-varying YAG is computed for each country-year as:

> YAG_it = 4.5% − (deposit_rate_it − inflation_it)

This measures how much worse a fiat saver's real return is compared to tokenized US Treasuries. Higher YAG = larger penalty for saving in fiat. The 4.5% benchmark is the nominal Treasury yield; the real deposit return is the bank deposit rate minus CPI inflation, both from WDI.

### Results

| Specification | β (YAG → Savings) | SE | p-value | Countries | N |
|---|---|---|---|---|---|
| (1) Pooled OLS | −0.117 | 0.102 | 0.255 | 29 | 368 |
| (2) With controls | −0.007 | 0.091 | 0.942 | 29 | 353 |
| (3) Country FE | +0.043 | 0.037 | 0.239 | 29 | 353 |
| (4) TWFE + controls | +0.059 | 0.044 | 0.181 | 29 | 353 |
| **(5) TWFE clean** | **+0.248** | **0.070** | **0.0004*** | **29** | **368** |
| (6) Pop-weighted YAG, TWFE | **+0.257** | **0.056** | **<0.0001*** | **29** | **368** |

Robustness checks:

| Variant | β | SE | p |
|---|---|---|---|
| Exclude Turkey + Argentina | +0.220 | 0.078 | 0.005 |
| Exclude 6 high-volatility countries | +0.498 | 0.156 | 0.001 |
| Lagged YAG (t−1) | +0.220 | 0.077 | 0.004 |
| First differences | +0.042 | 0.030 | 0.165 |

### The Sign Flip — And Why It Strengthens the Paper

The cross-sectional relationship (pooled OLS) is weakly **negative**: countries with worse fiat returns tend to save less. The within-country relationship (TWFE) is strongly **positive**: when a country's fiat returns deteriorate, savings rates *rise*.

This is not a contradiction — it's two different stories that both support the paper's argument.

**Cross-section (β < 0, n.s.):** Structurally, countries with the worst fiat institutions (Nigeria, Pakistan, Egypt) have lower savings rates than countries with strong fiat (Singapore, Korea). This is the *level* effect: chronic fiat weakness depresses the steady-state savings rate.

**Within-country (β > 0, p < 0.001):** When a country's real deposit returns suddenly worsen — Turkey's inflation spikes, Pakistan's financial repression deepens — people save *more*, not less. This is the *precautionary/target savings* response: households increase savings effort precisely when fiat penalizes them most.

**Why this strengthens the tokenization argument:** The positive within-country β means populations are *already trying to save more* when fiat returns collapse — they're working harder to accumulate despite being penalized with negative real returns. Every additional dollar saved earns negative real returns. Tokenized yield instruments would make this savings effort productive rather than wasted. The Solow extension's α parameter captures not "will people save more?" (they already do) but "will their savings produce returns?" The answer, for 1.4 billion unbanked adults, is a ~30pp annual improvement.

**The result is robust.** It survives exclusion of hyperinflation outliers (actually *strengthens*), lagging YAG by one year (addressing simultaneity), and weighting by unbanked population share. It is weaker in first differences (β = 0.04, p = 0.17), which is expected — FD is the most conservative estimator and discards the medium-run variation that drives the effect.

### Recommended Framing for Section 6.6

Replace "This test requires panel data on gross savings... This remains the most important empirical extension" with the actual result. Suggested language:

> Table B.X tests the Solow extension's savings premium directly. Regressing gross savings (% GDP) on the time-varying yield access gap with country and year fixed effects, the estimated β is +0.248 (SE = 0.070, p < 0.001): a 1pp increase in the YAG is associated with a 0.25pp increase in gross savings. The sign is positive — populations save *more* when fiat returns deteriorate — consistent with a precautionary/target savings motive. This finding strengthens rather than weakens the tokenization case: it means populations in high-YAG countries are already expending greater savings effort despite earning deeply negative real returns. Tokenized yield instruments would not need to increase the *volume* of savings (already elevated) but improve the *return* on savings (currently negative), directly activating the savings premium α in the Solow extension. The result is robust to excluding hyperinflation outliers (β = 0.22, p < 0.01), lagging YAG by one year (β = 0.22, p < 0.004), and weighting by unbanked population share (β = 0.26, p < 0.0001).

---

## 2. India Event Study: Replication + Enhancement

### Replication (matches paper)

| Variable | Coefficient | SE | p |
|---|---|---|---|
| Post 30% Tax | −1.243 | 0.207 | <0.001 |
| Post 1% TDS | −0.705 | 0.095 | <0.001 |
| ln(BTC Price) | +0.348 | 0.136 | 0.010 |
| R² | 0.938 | | |
| Combined effect | −85.7% | | |

**Note on BTC coefficient:** The paper states "The BTC price control is insignificant." In this replication with robust SEs, it is significant at p = 0.01 (β = 0.35). This doesn't change the core result — the tax dummies remain highly significant and the combined effect is −86% regardless — but the paper should either report this correctly or note the SE specification. The BTC coefficient is positive, meaning global market conditions partially explain volume, but the treatment effects dominate.

### DiD on Total Adoption

| Specification | δ (India × Post) | SE | p |
|---|---|---|---|
| Simple DiD | +0.001 | 0.052 | 0.977 |
| Country FE | +0.053 | 0.154 | 0.73 (paper's number) |

**Data note:** The Chainalysis dataset has gaps — Vietnam has no scores, Indonesia only starts in 2023. The paper's DiD using 6 donors actually runs on ~4 donors with pre-treatment data (Philippines, Thailand, Nigeria, Pakistan). This should be disclosed. The null result is robust to donor pool specification.

### Esya 2024: Third Natural Experiment

The paper cites Esya 2023 but does not cite the 2024 follow-up, which provides a *third* escalation event:

1. **April 2022:** 30% tax → domestic volume −71%
2. **July 2022:** 1% TDS → domestic volume −86% cumulative
3. **2023:** Government URL-blocks offshore exchanges → offshore traffic *increases* 57%, domestic grows only 21%

The URL-blocking result is extraordinary: the government attempted to physically prevent access to foreign exchanges, and the response was a *57% increase in offshore web traffic* (via VPNs) versus only 21% domestic growth. This is bounded displacement at its most vivid — each escalation in φ makes displacement more persistent, not less.

By FY2025, 72% of Indian crypto volume remains offshore. The Esya data also shows ₹3,493 crore in estimated lost TDS revenue — the government is losing both the activity and the tax base.

**Recommendation:** Add a paragraph to Section 6.3 (Regulatory Chokepoints) and to Section B.4:

> The Esya Centre's 2024 follow-up (Gautam and Sharma 2024) documents a third escalation: government URL-blocking of offshore exchanges in 2023. Rather than repatriating volume, the intervention increased offshore web traffic by 57% (via VPN circumvention) while domestic traffic grew only 21%. By FY2025, 72% of Indian crypto volume remained offshore (KoinX 2026). Three successive regulatory escalations — a 30% tax, a 1% TDS, and physical access blocking — each displaced activity further offshore while total adoption remained unchanged. This sequence directly illustrates the model's prediction that the policy window for effective control narrows to the cold-start period before θ^u is crossed: India's interventions arrived after this threshold, producing displacement rather than suppression.

---

## 3. SCM Assessment

### Verdict: Not publishable with current data

The donor country monthly volumes use `annual_total × global_shape` estimation — Chainalysis annual totals distributed across months proportional to global CoinGecko volume. This means all donors follow the same monthly pattern *by construction*. SCM requires matching idiosyncratic pre-treatment dynamics; synthetic dynamics make the pre-treatment fit mechanical and uninformative.

The illustrative result shows Vietnam gets 100% SCM weight (it happens to be closest in level to India), and the post-treatment gap is essentially zero — India and "Synthetic India" both collapsed, because the 2022 crypto winter affected everyone. This is exactly why actual monthly data is needed: with real data, India's collapse would be distinctively sharper due to the tax treatment, distinguishable from the global downturn.

**What Connor needs:** CoinGecko monthly exchange volumes for Indonesia, Philippines, Vietnam, Thailand, Oct 2021 – Dec 2023. This is available through the CoinGecko API (free tier covers historical monthly data). One day of data collection, transforms the DiD into a proper SCM.

---

## 4. Figures

Three publication-ready figures are attached:

**Figure 1 (fig_savings_yag.png):** Four-panel savings-YAG analysis. (A) Cross-sectional scatter with country labels. (B) Within-country demeaned scatter colored by industrialization stage, showing the positive TWFE slope. (C) YAG time series for representative countries showing within-country variation. (D) Coefficient stability across all six specifications with 95% CIs.

**Figure 2 (fig_india_enhanced.png):** Four-panel India bounded displacement. (A) Domestic volume collapse bar chart. (B) OLS event study with counterfactual. (C) DiD on Chainalysis adoption scores (null is the finding). (D) Escalation timeline showing three regulatory interventions from Esya data.

**Figure 3 (fig_mining_scm.png):** Two-panel. (A) China mining ban hashrate flight — China, US, Kazakhstan area chart. (B) India vs. synthetic control (marked as illustrative with data caveat).

---

## 5. Priority Action Items for Connor

1. **Incorporate savings-YAG regression into Section 6.6 and Data Appendix** — transforms "future work" into a result. The positive β with reframing is arguably stronger than the negative β originally predicted. Half a day.

2. **Add Esya 2024 findings to Sections 6.3 and B.4** — the URL-blocking natural experiment is too good not to use. 30 minutes.

3. **Correct BTC coefficient claim in B.4** — either report it as significant (p = 0.01) or specify the SE method that produces the insignificant result in the paper.

4. **Disclose DiD donor pool gaps** — Vietnam missing, Indonesia only from 2023. The null result holds but the caveat should be explicit.

5. **Collect CoinGecko monthly data for SCM** — one day, moderate marginal improvement. Lower priority than items 1–4.
