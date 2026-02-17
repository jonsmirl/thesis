## 5.x  Bounding R₀ from Open-Weight Adoption Dynamics

The R₀ framework (Section 3.9) predicts that hardware cost parity precedes self-sustaining distributed adoption by 2–3 years, during which coordination friction κ declines as deployment infrastructure matures. This section bounds the R₀ parameters from observed open-weight adoption dynamics, providing independent empirical discipline for the framework rather than post-hoc calibration.

### 5.x.1  Methodology

Model open-weight token share s(t) as following logistic-SIR dynamics:

> ds/dt = r(t) · s(t) · (1 − s(t)),    where  r(t) = (R₀(t) − 1) · δ

From discrete observations of the OpenRouter token-volume series, back out the implied growth rate and composite R₀:

> R₀(t) ≈ 1 + [Δs/Δt] / [δ · s(t) · (1 − s(t))]

with δ normalized to monthly frequency. This yields an implied R₀ trajectory that can be decomposed into cost-driven and coordination-driven components.

**Critical scope distinction.** The OpenRouter series measures open-weight model share through a *centralized* aggregator. An enterprise running Qwen-2.5 via OpenRouter uses open-weight models but centralized infrastructure. The paper's R₀ > 1 crossing condition (Prediction 2*) refers to self-sustaining *distributed* inference adoption, which faces additional coordination friction from hardware heterogeneity and edge deployment complexity. The OpenRouter-implied R₀ therefore bounds the *upper envelope* of the broader open-weight ecosystem's reproduction number; the distributed-specific R₀ is strictly lower.

### 5.x.2  Implied R₀ Trajectory

| Period | s(t) | Δs/Δt | R₀(t) |
|---|---|---|---|
| Jan-24 → Mar-24 | 0.025 | 0.008 | 1.44 |
| Mar-24 → Jun-24 | 0.050 | 0.008 | 1.23 |
| Jun-24 → Sep-24 | 0.080 | 0.010 | 1.16 |
| Sep-24 → Nov-24 | 0.120 | 0.020 | 1.22 |
| Nov-24 → Dec-24 | 0.150 | 0.030 | 1.26 |
| Dec-24 → Jan-25 | 0.250 | 0.098 | 1.61 |
| Jan-25 → Feb-25 | 0.180 | −0.069 | 0.59 |

**Table X.** Implied R₀ from OpenRouter open-weight token share dynamics. The January 2025 spike reflects the DeepSeek R1 release; the February reversion to 18% represents the post-novelty plateau. Excluding the R1 spike-reversion, mean implied R₀ = 1.15.

Three features of this trajectory are notable:

First, implied R₀ for centralized open-weight adoption is *above* unity for most of the observation period (mean ≈ 1.2, excluding the spike-reversion). This is consistent with open-weight models gaining share through centralized providers—a stage that precedes and enables distributed deployment. The fact that even this easier adoption path yields R₀ only modestly above 1 (not 2 or 3) indicates the ecosystem remains in early-stage growth, not yet in the rapid-expansion phase characteristic of mature network effects.

Second, the trajectory is approximately flat at R₀ ≈ 1.2 from March through November 2024, then exhibits a sharp perturbation (DeepSeek R1 release) followed by reversion. This pattern—steady growth punctuated by model-release shocks that partially revert—is characteristic of an ecosystem where adoption is driven by capability events (model releases) rather than self-sustaining network dynamics. Under mature network effects, we would expect accelerating R₀, not flat-with-shocks.

Third, the February 2025 reversion (R₀ = 0.59) demonstrates that the open-weight ecosystem can still enter sub-critical regimes when novelty effects dissipate. This is the strongest evidence that even centralized open-weight adoption has not yet achieved robust R₀ > 1 driven by structural advantages rather than event-driven surges.

### 5.x.3  Parameter Bounds

**Latency advantage λ.** Structural and directly measurable: edge inference latency < 10ms versus cloud round-trip 50–200ms, a 5–20× advantage. This is hardware-determined and independent of the adoption dynamics; it enters R₀ as a quality dimension that can push adoption even before cost parity.

**Churn rate μ.** Bounded from model lifecycle data on Hugging Face. The rapid succession of model families—Llama 2 to Llama 3 (9 months), Qwen 2 to Qwen 2.5 (3 months)—implies deployment-weighted model lifetimes of approximately 6–12 months, or μ ≈ 0.08–0.17/month.

**Coordination friction κ.** From R₀ = β·γ/(κ + μ), with β·γ calibrated to the observed adoption dynamics: κ ranges from approximately 0.05 (January 2025, peak adoption) to 0.11 (mid-2024, pre-Qwen-2.5 coordination infrastructure). The trajectory of κ decline is corroborated by observable coordination indicators: in June 2024, major model releases required weeks of community effort to produce optimized edge runtimes; by January 2025, DeepSeek R1 shipped with day-zero GGUF quantizations, ONNX exports, and multi-hardware deployment scripts—a compression of coordination latency from weeks to hours.

**Composite β·γ.** The adoption rate × network effect product is calibrated at approximately 0.24 (monthly). This is consistent with the observation that open-weight share growth is primarily linear rather than exponential over the observation period—the logistic dynamics are in the early, approximately linear regime where s(t) ≪ 1 and network effects have not yet produced the characteristic acceleration.

### 5.x.4  Implications for the Distributed R₀ Crossing

The distinction between centralized open-weight R₀ (≈ 1.2) and distributed inference R₀ disciplines Prediction 2*. If the upper-envelope ecosystem achieves R₀ ≈ 1.2 with the coordination advantages of centralized deployment (single-provider APIs, managed infrastructure, no hardware heterogeneity), then the distributed-specific R₀ is reduced by the additional friction of edge deployment:

> R₀_distributed ≈ R₀_centralized · (κ_central / κ_distributed)⁻¹

With κ_distributed plausibly 2–5× higher than κ_central (reflecting hardware heterogeneity, absence of managed infrastructure, consumer deployment friction), R₀_distributed ≈ 0.4–0.8 in the current period—firmly in the sub-critical regime. The prediction that R₀_distributed > 1 by 2030–2032 then requires:

(a) Continued hardware cost decline along the learning curve (reducing the cost threshold)
(b) Coordination friction κ_distributed declining as edge runtimes mature (the mechanism documented in Section 3.9.3)
(c) The structural latency advantage λ becoming salient as real-time applications grow

The implied rate of κ decline from the centralized data (approximately 30–50% per year during 2024) provides a lower bound on the coordination maturation rate. If distributed coordination friction follows a similar trajectory with a lag, R₀_distributed > 1 by 2030–2032 requires κ_distributed to decline at ≥ 25% annually—a rate consistent with the observed centralized trajectory but not yet independently confirmed for edge deployment.

### 5.x.5  Limitations

This exercise bounds rather than structurally estimates the R₀ parameters. Three limitations merit acknowledgment. First, the OpenRouter series covers only 14 months with 8 observations—sufficient for bounding but not for formal time-series inference. Second, the logistic-SIR specification imposes functional form assumptions; alternative adoption models (Bass diffusion, threshold models) would yield different implied parameters, though the qualitative trajectory (R₀ rising, κ declining) is robust to specification. Third, the decomposition of R₀ into cost and coordination components requires auxiliary assumptions about the cost-advantage trajectory that introduce additional uncertainty.

A more rigorous test awaits longer time series and, crucially, direct measurement of distributed (edge) inference volumes—data that does not yet exist at the granularity required but that the model's predictions (Prediction 2*, Prediction 7) are designed to be tested against.

---

**[Figure X: Four-panel figure]**
(a) Open-weight token share trajectory (OpenRouter)
(b) Implied R₀ with R₀ = 1 threshold
(c) R₀ decomposition: cost advantage vs coordination effect
(d) Coordination friction κ decline with ecosystem breadth overlay
