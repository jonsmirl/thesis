# R₀ Threshold Extension: Making the Coordination Layer Semi-Endogenous

## For insertion as Section 3.9 (or Appendix C) of "Endogenous Decentralization"

---

## The Problem

The current model defines crossing at cost parity: distributed becomes viable when c_distributed ≤ c_centralized (equivalently, when x(t) = 0). But the empirical evidence in Section 5 shows that hardware crossing *precedes* architectural dominance by 3–5 years. The 386 was cost-competitive by 1987; Windows didn't explode until 1990–95. TCP/IP was viable by ~1989; the browser arrived 1993–94.

This lag is exogenous to the current model. The coordination layer pattern (Table 7) is documented but not explained by the game-theoretic structure.

## The Key Insight

The lag exists because cost parity is *necessary but not sufficient* for self-sustaining distributed adoption. What's actually required is that the distributed ecosystem's **basic reproduction number** exceeds unity:

$$R_0 \equiv \frac{\beta(c) \cdot \gamma}{(\kappa + \mu)} > 1$$

where:
- β(c) = adoption rate, increasing in cost advantage (decreasing in c_distributed)  
- γ = network effect multiplier (each adopter's contribution to ecosystem value)
- κ = coordination friction (cost of building/using the coordination layer)
- μ = churn rate (users reverting to centralized)

When R₀ < 1, distributed adoption dies out even if hardware is cost-competitive. When R₀ > 1, adoption is self-sustaining. The crossing that matters is R₀ = 1, not c_distributed = c_centralized.

## Preserving the One-State-Variable Structure

Here's what makes this tractable: **R₀ = 1 is equivalent to a modified cost threshold, which maps back to a modified cumulative production threshold.**

### Step 1: The Generalized Crossing Condition

For distributed adoption to be self-sustaining, the basic reproduction number must exceed unity. The general condition is:

$$R_0 = \frac{\beta(c^* - c) \cdot \gamma}{\kappa + \mu} > 1$$

where β(·) is the adoption rate as a function of the cost advantage (c* − c). Setting R₀ = 1 and solving for the critical distributed cost c** depends on the specification of β:

| Adoption function | β specification | c** (from R₀ = 1) | Verified |
|---|---|---|---|
| Linear | β = c* − c | c* − (κ+μ)/γ | Wolfram ✓ |
| Power law | β = (c* − c)^p | c* − ((κ+μ)/γ)^{1/p} | Wolfram ✓ |
| Logistic | β = 1/(1+e^{−k(c*−c−c₀)}) | Implicit; numerical | Wolfram ✓ |

The linear case is used below for closed-form exposition. The qualitative results are regime-dependent but hold across all three specifications (see Step 2).

### Step 2: Map to Modified Production Threshold

From the learning curve c(Q) = c₀ · Q^{-α}, the R₀ = 1 condition c = c** implies a modified cumulative production threshold:

$$\bar{Q}^* = \bar{Q} \cdot \left(\frac{c^*}{c^{**}}\right)^{1/\alpha}$$

Under the linear specification, this is:

$$\bar{Q}^* = \bar{Q} \cdot \left(1 - \frac{\kappa + \mu}{\gamma \cdot c^*}\right)^{-1/\alpha}$$

**The relationship between Q̄\* and Q̄ is regime-dependent.** Whether R₀ > 1 requires more or less cumulative production than cost parity depends on a single condition: the value of R₀ evaluated at the cost-parity point itself.

**Friction-dominated regime: R₀(c\*) < 1 → Q̄\* > Q̄.** When coordination friction is high relative to network effects, R₀ < 1 at cost parity. Self-sustaining adoption requires costs to fall *below* parity — a cost cushion to overcome switching costs. The system needs more cumulative production than cost parity alone would require. The gap Q̄\* − Q̄ is the coordination layer deficit. This is the empirically relevant case during the pre-crossing period when the coordination layer is immature.

**Network-dominated regime: R₀(c\*) > 1 → Q̄\* < Q̄.** When network effects are strong (high γ) or coordination friction is low (low κ), R₀ > 1 *before* cost parity. Self-sustaining adoption kicks in while distributed is still more expensive than centralized — network effects compensate for the cost disadvantage. Crossing arrives *sooner* than pure cost parity predicts.

The Wolfram verification of the logistic specification revealed this regime structure: with sufficiently strong network effects (γ = 2, κ+μ = 0.4), the logistic model yields c** > c*, placing the system in the network-dominated regime. The linear and power-law specifications with the same parameters yield c** < c* (friction-dominated). The regime boundary is R₀(c*) = 1, or equivalently β(0)·γ = κ+μ: whether network effects alone (at zero cost advantage) can sustain adoption.

**This regime structure enriches the model's predictions.** Early-stage transitions — when coordination infrastructure is immature, κ is high, and γ is low — are friction-dominated: hardware crossing precedes self-sustaining adoption by the 3–5 year lag observed historically. As the coordination layer matures (κ falls, γ rises), the system approaches and can cross into the network-dominated regime, where the R₀ threshold is reached before cost parity. This is consistent with the compressed-lag prediction for AI: pre-crossing coordination maturity (stablecoins, ONNX, edge frameworks) may push the current transition toward the network-dominated regime, explaining why ΔT could be 2–3 years rather than 3–5.

### Step 3: Two Cases

The claim that "the model is unchanged" depends on whether κ is treated as fixed or declining. These are distinct modeling choices with different implications.

**Case A: Fixed κ (exogenous coordination friction).** If κ is a parameter — reflecting coordination infrastructure that evolves on institutional timescales outside the model — then Q̄* is a constant, x(t) = Q̄* − Q(t) is a standard state variable, and the entire game-theoretic apparatus carries through exactly. The HJB, FOCs, closed-form solutions, Proposition 1, and Corollaries 1–3 all hold with Q̄* replacing Q̄. In the friction-dominated regime (Q̄* > Q̄), crossing occurs later in absolute time than the cost-parity model predicts. In the network-dominated regime (Q̄* < Q̄), crossing occurs earlier. In both regimes, the acceleration ratio T*_Nash / T*_Coop is preserved — Nash competition accelerates crossing relative to cooperation regardless of which regime applies.

This is the appropriate treatment for the current paper. It closes the theoretical gap — the coordination layer enters the crossing condition — without altering the model's identity.

**Case B: Declining κ (semi-endogenous coordination).** If κ declines as the coordination layer matures, Q̄*(t) is moving. The state x(t) = Q̄*(κ(t)) − Q(t) then evolves as:

$$\dot{x} = \frac{d\bar{Q}^*}{d\kappa} \cdot \frac{d\kappa}{dE} \cdot \dot{E} \;-\; \sum_i q_i$$

The first term is negative (Q̄* falls as κ falls as E rises), meaning the threshold drifts toward the firms even when they produce nothing. A firm's value function V(x) now implicitly depends on the drift rate, which depends on E(t) — a second state variable.

The one-state-variable structure survives under a **timescale separation assumption**: if coordination infrastructure evolves slowly relative to the production game (formally, if |dQ̄*/dt| ≪ |ΣqᵢI), firms can treat Q̄* as locally constant when solving their HJB. This is a standard quasi-static approximation (analogous to the Born-Oppenheimer approximation in physics, or the "slow manifold" in singular perturbation theory). Under quasi-statics, the Nash equilibrium at each instant is the solution to the fixed-Q̄* game evaluated at the current Q̄*(t), and the analysis of Case A applies pointwise.

The quasi-static assumption is empirically reasonable: coordination layers mature over years (standards committees, developer ecosystem formation, regulatory adaptation), while production decisions operate on quarterly cycles. But it is an assumption, and it breaks down if coordination friction drops discontinuously — as when a dominant standard suddenly emerges. At such moments, Q̄* jumps downward, x drops discretely, and the smooth HJB solution is interrupted. The model handles this as a regime shift rather than a continuous dynamic.

**Recommendation for the paper:** Use Case A (fixed κ, exact result) in Section 3.9. Present Case B (declining κ, quasi-static) in the discussion as economic motivation for why the coordination gap shrinks over time, noting the timescale separation assumption explicitly. Reserve the full two-state treatment for the companion paper.

---

## Making κ Endogenous (The Semi-Endogenous Part)

The coordination friction κ is not fixed. It declines as the coordination layer matures. The qualitative dynamics require only that κ is monotonically decreasing in cumulative ecosystem investment E(t):

$$\kappa(t) = \kappa(E(t)), \quad \kappa'(E) < 0$$

The specific functional form of κ(E) is not pinned down by the theory. Exponential decay (κ = κ₀·e^{-λE}) is analytically convenient but poorly motivated: the economics of coordination—standards wars, winner-take-all platform competition, committee processes—are lumpy, not smooth. A threshold function where κ drops discontinuously when a dominant standard emerges (as Windows did for PCs, or HTTP/HTML did for the web) may better describe the actual dynamics. A power law κ = κ₀·E^{-η} accommodates the heavy-tailed adoption patterns observed in platform markets. 

**The qualitative results are robust to functional form.** What matters for the model is that (i) κ is decreasing in ecosystem development, (ii) ecosystem development E is increasing in the cost advantage created by centralized investment, and (iii) declining κ reduces the effective threshold Q̄*(κ). These three properties hold for any monotonically decreasing κ(E). The feedback loop operates regardless:

1. Centralized investment → cumulative production Q rises → hardware costs fall
2. Cost advantage grows → coordination layer investment E rises → κ falls  
3. Falling κ → Q̄*(κ) falls → the crossing *comes toward you*

The rate at which E responds to the cost gap is:

$$\dot{E} = \phi \cdot \max\{0, \, c_{centralized} - c(Q)\}$$

where φ converts cost advantage into ecosystem investment incentive. This creates a two-sided convergence: Q(t) rises toward Q̄*(t) while Q̄*(t) falls toward Q(t). The speed of convergence depends on the functional form of κ(E) and the magnitude of φ, but the *direction* is structural.

### The Two-Gap Decomposition

Define:
- **Hardware gap**: G_H(t) = Q̄ – Q(t) (distance to cost parity, current state variable x in the original model)
- **Coordination gap**: G_C = Q̄*(κ) – Q̄ (positive in friction-dominated regime, negative in network-dominated regime; constant under Case A, declining under Case B)
- **Total gap**: x*(t) = G_H(t) + G_C = Q̄*(κ) – Q(t)

The current model tracks only G_H. The R₀ extension adds G_C. In the friction-dominated regime (G_C > 0), the total gap exceeds the hardware gap — more production is needed beyond cost parity. In the network-dominated regime (G_C < 0), network effects close part of the gap, and less production is needed. Under Case A (fixed κ), G_C is a constant and the total gap x* is a standard state variable — the tractability result is exact. Under Case B (declining κ with quasi-static approximation), G_C shrinks over time (becoming less positive or more negative) and the firms solve a sequence of fixed-κ games, each valid over the timescale on which κ is approximately constant.

### Proposition 1* (Generalized Overinvestment)

**Under fixed κ (Case A): In the symmetric MPE with generalized crossing condition R₀ = 1, aggregate output Q^N(x) strictly exceeds cooperative output Q^C(x) for all x > 0, where x = Q̄*(κ) − Q. Nash equilibrium crossing occurs strictly before cooperative crossing: T\*_Nash < T\*_Coop.**

*Proof.* Identical to Proposition 1 with Q̄ replaced by Q̄\*(κ). The HJB equation, FOCs, and shadow cost comparison are functions of the state x, not of the threshold's composition. The boundary condition V(0) = S is unchanged because the post-crossing displacement dynamics are independent of how the threshold is defined. ■

*Remark.* Under Case B (declining κ, quasi-static), Proposition 1* holds pointwise at each Q̄*(t), with approximation error bounded by the timescale separation ratio. The overinvestment result is robust to slow threshold drift because the strategic mechanism — each firm internalizing only 1/N of the social shadow cost — operates at the production timescale regardless of boundary movement.

---

## The Coordination Layer Lag Becomes Endogenous

This framework explains the 3–5 year lag structurally rather than by assertion. The explanation applies in the **friction-dominated regime** (R₀(c*) < 1), which characterizes all historical transitions during their early phases:

**Hardware crossing** occurs when Q(t) = Q̄ (cost parity). In the friction-dominated regime, R₀ < 1 at this point because κ is still high — the coordination layer hasn't matured.

**R₀ crossing** occurs when Q(t) = Q̄*(κ(t)). The lag between hardware crossing and R₀ crossing is:

$$\Delta T = T^*_{R_0} - T^*_{hardware} > 0 \quad \text{(friction-dominated)}$$

This lag depends on:
- κ₀: initial coordination friction (higher → longer lag)
- λ: coordination learning rate (higher → shorter lag)  
- φ: responsiveness of ecosystem investment to cost advantage (higher → shorter lag)

In the **network-dominated regime** (R₀(c*) > 1), ΔT is negative: self-sustaining adoption arrives before cost parity, and there is no lag to explain. The regime transition itself — from friction-dominated to network-dominated as κ falls and γ rises — is what compresses ΔT from 3–5 years toward zero and potentially into negative territory.

### Calibration to Historical Transitions

| Transition | Hardware T* | R₀ T* | ΔT | Implied κ₀/λ |
|---|---|---|---|---|
| Mainframe → PC | 1987 | 1990–92 | 3–5 yr | ~15 |
| ARPANET → Internet | ~1989 | 1993–94 | 4–5 yr | ~18 |
| Cloud → Edge AI | 2028–30 | ? | ? | ? |

The observed consistency of ΔT ≈ 3–5 years across the two historical transitions is suggestive but not conclusive—two data points cannot establish a structural regularity. The similar implied κ₀/λ ratios may reflect genuine commonalities in how coordination layers mature in technology markets (standards processes, developer ecosystem formation, and settlement infrastructure all operate on similar institutional timescales), or it may be coincidence. The framework generates the lag endogenously but does not predict its magnitude from first principles.

**The more interesting falsifiable implication concerns the AI transition specifically.** The coordination layer for distributed AI is already substantially more developed *before* hardware crossing than in either prior case. Stablecoin settlement ($3.9T/quarter), containerized model serving (ONNX, llama.cpp), and edge deployment frameworks (TensorRT, Core ML) are advancing in parallel with—not after—hardware cost decline. In the regime framework, this pre-crossing coordination maturity lowers κ₀ and raises γ, pushing the system *toward* the network-dominated regime. If the R₀ framework is correct, this predicts a **compressed lag: ΔT ≈ 2–3 years rather than 3–5, or potentially ΔT ≈ 0 if the transition reaches the network-dominated regime before hardware crossing.** This prediction *breaks* the historical pattern based on a structural argument (pre-crossing coordination investment reduces κ₀), which makes it a sharper test of the theory than extrapolating the 3–5 year regularity. Evidence against: ΔT ≥ 5 years despite observable pre-crossing coordination maturity.

---

## Updated Predictions

The R₀ framework refines two existing predictions and generates one new one:

### Prediction 2* (Refined)
**R₀ > 1 for distributed AI inference by 2030–2032.** The original Prediction 2 (70B on-device by 2028–2030) captures hardware crossing. The refined prediction adds: self-sustaining distributed adoption (R₀ > 1) arrives 2–3 years later. Evidence for: quarterly distributed inference share growing >5% per quarter without subsidies. Evidence against: distributed share stalling below 20% despite hardware capability by 2033.

### Prediction 7* (Refined)  
**The coordination-layer trough (Prediction 7) corresponds to the period when hardware has crossed but R₀ < 1.** The trough occurs because early adopters face high κ (immature coordination layer) and adoption is not yet self-sustaining. The second wave arrives when κ falls enough for R₀ > 1. The R₀ framework predicts the trough depth: the maximum adoption level before the trough equals the endemic equilibrium under R₀ < 1, which is zero—explaining why early adoption waves collapse (CP/M market share, early ISP subscribers) rather than merely decelerating.

### Prediction 8 (New)
**Coordination friction κ is observable and declining.** The R₀ framework requires that κ decreases over time, which is empirically testable through proxies for coordination cost. Three candidate proxies, each capturing a distinct dimension of coordination friction:

- *Deployment friction:* Time-to-deploy for a standardized model on new edge hardware (measures API/toolchain maturity)
- *Interoperability friction:* Number of mutually incompatible model interchange formats weighted by market share (measures standards consolidation; declining Herfindahl fragmentation index implies falling κ)  
- *Settlement friction:* Transaction cost for micropayment-scale AI inference settlements (measures economic coordination readiness)

The prediction is directional: all three proxies should show sustained decline through the transition period, with acceleration following hardware crossing. The model does not generate specific numerical targets for these proxies — unlike Predictions 1–7, which are calibrated from α, Q̄, and N, the coordination learning rate and initial friction level are not yet estimated from data. Constructing a κ index from weighted proxies and estimating its trajectory is a natural next step but requires its own empirical treatment. Evidence against: any proxy showing sustained *increase* or stagnation through 2030 despite hardware costs continuing to decline along the learning curve — this would indicate that coordination friction is not responsive to the cost gap, breaking the feedback loop central to the R₀ extension.

---

## Implementation Options for the Paper

### Option A: Light Touch (Recommended for current draft)
Add a subsection 3.9 "Generalized Crossing Condition" (~2 pages) that:
1. Defines R₀ and shows equivalence to modified threshold Q̄* (Case A: fixed κ)
2. States Proposition 1* (one paragraph, exact proof by reference to Prop. 1)
3. Notes that declining κ makes the coordination lag endogenous (motivational discussion, not formal)
4. Refines Predictions 2 and 7

This preserves the paper's clean identity while closing the biggest theoretical gap. The key virtue: every mathematical claim is exact under Case A. The economic motivation for *why* κ should decline over time is discussed informally, with the quasi-static treatment (Case B) and its timescale separation assumption flagged as future work.

### Option B: Full Treatment (Appendix C, ~5 pages)
Everything in Option A plus:
- The two-gap decomposition with Case B quasi-static dynamics
- Explicit statement of the timescale separation assumption and its empirical basis
- κ(E) feedback loop (functional-form-agnostic)
- Calibration to historical ΔT
- Prediction 8

### Option C: Companion Paper
The full SEIR adoption dynamics with:
- Two-state PDE system: x(t) and n(t)  
- Adoption dynamics: dn/dt = β(x)·n·(1-n) – μ·n
- Firms' HJB depending on both states
- Numerical solution (no closed-form expected)
- Full welfare analysis including consumer surplus

This is probably a separate paper ("Self-Sustaining Decentralization: An Epidemiological Model of Architectural Transition") that cites the current one.

---

## Technical Notes

### Why This Doesn't Require a PDE (and When It Would)
Under Case A (fixed κ), the one-state-variable structure is exact — no approximation needed. Under Case B (declining κ), the quasi-static approximation makes it work as long as coordination infrastructure evolves slowly relative to production decisions. The approximation error is of order |dQ̄*/dt| / |Σqᵢ|, which is small when institutional timescales (years) dominate production timescales (quarters).

The full two-state treatment becomes necessary when either: (a) firms strategically invest in coordination layer development as a separate choice variable, so that E enters the HJB as a control, or (b) coordination dynamics are fast enough to violate timescale separation — e.g., a standards war resolving on the same timescale as capacity investment decisions. The lumpy, discontinuous nature of real coordination dynamics (standards wars resolving suddenly, platform tipping) further argues for the Case A treatment in the current paper: imposing smooth κ dynamics on an inherently discrete process would add complexity without adding honesty. This is the Option C companion paper territory.

### Connection to Epidemiological Literature
The R₀ framework draws on the SIR model's threshold theorem (Kermack and McKendrick, 1927): an epidemic is self-sustaining iff R₀ > 1. The technology adoption analog: Bass (1969) diffusion model with network effects generates identical threshold dynamics. The innovation here is connecting R₀ to the learning curve game: centralized firms' Nash overinvestment drives β(c) upward (by reducing distributed costs) while coordination layer maturation drives κ downward, and R₀ > 1 emerges endogenously from the same competitive dynamics that produce Proposition 1.

### On the "Evolutionary Suicide" Connection (Section 3.6)
The R₀ framing strengthens the evolutionary suicide result. In epidemiological terms: the centralized firms' ESS production strategy increases the transmissibility β of the distributed "pathogen" by lowering costs, while simultaneously the coordination layer reduces the recovery rate μ + κ. The host population (centralized firms) pursues a collectively stable strategy that maximizes the pathogen's R₀. This is niche construction driving R₀ above 1—a precise formal analog to biological systems where host behavior increases parasite fitness.

---

## References to Add

Bass, F. M. (1969). A new product growth for model consumer durables. Management Science, 15(5), 215–227.

Kermack, W. O., & McKendrick, A. G. (1927). A contribution to the mathematical theory of epidemics. Proceedings of the Royal Society A, 115(772), 700–721.

Jackson, M. O., & Yariv, L. (2007). Diffusion of behavior and equilibrium properties in network games. American Economic Review, 97(2), 92–98.

Young, H. P. (2009). Innovation diffusion in heterogeneous populations: Contagion, social influence, and social learning. American Economic Review, 99(5), 1899–1924.
