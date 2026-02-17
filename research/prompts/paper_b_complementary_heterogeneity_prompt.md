# Writing Prompt: Paper B — "Complementary Heterogeneity in Hierarchical Economies"

## Target

A publication-ready economics paper (~40 pages + appendix) that develops a dynamic theory of hierarchical economies with CES-aggregated agents at each level. The paper takes the CES Triple Role (Paper A) as a given static result and builds everything dynamic on top of it: architecture derivation, cross-level activation, stability, welfare decomposition, and policy implications.

The paper must be readable by economists. Every piece of non-standard mathematics gets an economic translation. The language strategy is: **prove in the general framework, then immediately translate the result into economics before moving on.**

**Target journals:** Review of Economic Studies, Econometrica, or Quarterly Journal of Economics.

**Key insight to convey:** A hierarchical economy with CES production at each level has a *derived* architecture — aggregate coupling, directed feed-forward, nearest-neighbor chain — that is forced by the geometry of the isoquant, not assumed by the modeler. This architecture produces a spectral activation threshold, a hierarchical ceiling cascade, and a welfare decomposition that identifies the binding constraint as the most institutionally rigid layer, not the most visible disequilibrium.

---

## Economic Language Translation Strategy

This is the most important meta-instruction. The underlying mathematics uses dynamical systems theory (slow manifolds, bifurcation, port-Hamiltonian systems, next-generation matrices). Economists will not read a paper that presents these tools raw. The translation strategy is:

| Mathematical concept | Economic translation |
|---------------------|---------------------|
| Slow manifold | Long-run equilibrium conditional on the slower-moving level |
| Fenichel's theorem | The fast sector equilibrates before the slow sector moves appreciably |
| Timescale separation ε_n ≪ ε_{n-1} | Level n adjusts much faster than level n−1 (e.g., financial markets adjust faster than training pipelines, which adjust faster than semiconductor fabs) |
| Within-level eigenstructure | Diversity effects wash out faster than aggregate effects — the CES aggregate is a sufficient statistic for what other levels see |
| (2−ρ) low-pass filter | Compositional changes within a sector are invisible to other sectors; only the aggregate matters |
| Port-Hamiltonian structure | The economy has an internal energy function (the production technology Φ) and loses energy through institutional friction (damping σ_n) |
| Non-gradient flow / topological obstruction | There is no social planner's problem that generates these dynamics — the directed feed-forward structure cannot be produced by optimizing any potential function. This is a structural feature, not a modeling choice |
| Next-generation matrix K | Cross-level amplification matrix: entry K_{n,n-1} measures how much growth at level n−1 amplifies level n per unit of depreciation |
| Spectral radius ρ(K) > 1 | The hierarchy is "activated" — cross-level feedbacks are strong enough to sustain non-trivial activity |
| Cycle product P_cycle | The geometric mean of all cross-level amplification rates around the loop |
| Sub-threshold → super-threshold | Each sector alone is too weak to sustain itself, but the cross-sector feedbacks are strong enough that the system as a whole sustains activity |
| Transcritical bifurcation | A structural transition: below threshold, only the trivial equilibrium (no activity) is stable; above threshold, a non-trivial equilibrium (positive activity at all levels) becomes stable |
| Canard / delayed loss of stability | The transition takes time even after conditions are favorable — the system lingers near the old equilibrium before snapping to the new one |
| Bridge equation ∇²Φ = W⁻¹ · ∇²V | The production technology (Φ, what the economy can do) is connected to welfare (V, how far from efficiency) through institutional quality (W, how efficiently the economy adjusts) |
| Damping cancellation | Tightening regulation at level n has zero net welfare effect because faster convergence exactly offsets lower equilibrium output — reform must target the upstream level |
| Lyapunov function V | Welfare distance from equilibrium — a computable measure of how inefficient the economy currently is, decomposed by level |

**Rule:** After every theorem or proposition, write a paragraph labeled "Economic interpretation" that restates the result using only the right column of this table. A reader who skips all proofs and reads only the interpretations should understand the paper's contribution.

---

## Structure

### 1. Introduction (~3 pages)

Open with the economic question: what happens to a multi-sector economy when AI agents enter as market participants at machine speed?

The paper studies a hierarchical economy with four sectors operating at different timescales:
1. **Hardware** (decades): semiconductor learning curves, capital investment, the pace set by Wright's Law
2. **Network** (years): formation of AI agent ecosystems, adoption dynamics, platform competition
3. **Capability** (months): training efficiency, model improvement, autocatalytic skill accumulation
4. **Settlement** (days): financial settlement, stablecoin flows, monetary policy transmission

Each sector aggregates heterogeneous inputs using CES production technology. The CES Triple Role (Paper A) guarantees that within each sector, diversity is productive, informationally robust, and resistant to strategic manipulation.

The paper asks: what happens *between* sectors? The answer is surprising — the architecture of cross-sector interaction is not a free modeling choice but is *derived* from the CES geometry. Five results follow:

1. **Derived architecture** (Port Topology Theorem): Each sector communicates with others only through its aggregate output. The coupling must be directed (feed-forward). Long-range cross-sector links have no effect on dynamics. The architecture is a nearest-neighbor chain.

2. **Activation threshold** (Master R₀): The hierarchy activates when the spectral radius of a cross-sector amplification matrix exceeds 1. Individual sectors can each be too weak to sustain themselves, yet the system as a whole can sustain activity through cross-sector feedback.

3. **Hierarchical ceiling** (Baumol Bottleneck): Each sector's output is bounded by the sector below it in the timescale hierarchy. The long-run growth rate equals the growth rate of the slowest sector (hardware). The Baumol cost disease and the Triffin dilemma are the same mathematical object at adjacent layers.

4. **Welfare decomposition** (Eigenstructure Bridge): A computable welfare-distance function attributes inefficiency to each sector. The binding welfare constraint is the most institutionally *rigid* sector, not the most *visible* disequilibrium.

5. **Damping cancellation** (Upstream Reform Principle): Tightening regulation at sector n has zero net welfare effect — faster convergence exactly offsets lower equilibrium output. To improve welfare at sector n, reform sector n−1 (upstream) or increase the responsiveness of sector n's cross-sector coupling.

Preview the transition dynamics: the low-activity → high-activity transition takes O(1/√ε_drift) time, where ε_drift is the rate of secular improvement (e.g., Wright's Law). At current rates, approximately 8 years.

### 2. Setup (~3 pages)

#### 2.1. The CES Aggregate (Brief — cite Paper A)

State the CES aggregate F_n(x_n) with curvature parameter K = (1−ρ)(J−1)/J. Cite the Triple Role from Paper A: within each level, K simultaneously controls superadditivity (gap ∝ K), correlation robustness (bonus ∝ K²), and strategic independence (penalty ∝ K). These are static, within-level properties assumed henceforth.

#### 2.2. The Hierarchical Economy

Present the dynamics:

$$\varepsilon_n \dot{x}_{nj} = T_n(\mathbf{x}_{n-1}) \cdot \frac{\partial F_n}{\partial x_{nj}} - \sigma_n x_{nj}$$

**Economic translation of each term:**
- ε_n: characteristic adjustment speed of sector n (hardware adjusts over decades, finance over days)
- T_n(x_{n-1}) = φ_n(F_{n-1}(x_{n-1})): demand from the downstream sector — the amount of resources flowing into sector n, determined by the aggregate performance of sector n−1
- ∂F_n/∂x_{nj}: the marginal product of component j within sector n (how much adding one more unit of component j increases the sector's output)
- σ_n: depreciation/friction/institutional drag at sector n (how fast gains erode without continued input)

**Standing assumptions** (state each with economic motivation):
1. ρ < 1: inputs within each sector are not perfect substitutes (they have heterogeneous capabilities)
2. J ≥ 2 components per level, N ≥ 2 levels: nontrivial diversity and nontrivial hierarchy
3. Timescale separation ε_1 ≫ ε_2 ≫ ··· ≫ ε_N: hardware adjusts much slower than finance (empirically uncontroversial)
4. Positive depreciation σ_n > 0: without continued investment, capability erodes
5. Monotone coupling φ_n with φ_n(0) = 0: more upstream output means more downstream demand; no output means no demand

#### 2.3. The Free Energy and the Hamiltonian Interpretation

Define Φ = −Σ_n log F_n and explain its economic meaning: Φ is the "production difficulty" of the economy — the log of the reciprocal of aggregate output at each level. It is the natural internal energy function of the production technology. Its convexity (all eigenvalues of ∇²Φ positive) means the economy is always locally "trying" to increase output — but institutional friction (σ_n) and the directed structure (feed-forward coupling) prevent it from reaching a global optimum.

### 3. The Port Topology Theorem (~5 pages)

This is the architecture-derivation section. It should be the most carefully written section of the paper because it contains the most surprising result: the network topology is derived, not assumed.

**Theorem (CES-Forced Topology).** Under the standing assumptions:

**(i) Aggregate coupling.** Each sector communicates with other sectors only through its aggregate output F_n. Individual component states within a sector are invisible to other sectors.

*Proof sketch:* Three steps — equilibrium uniqueness (the CES first-order conditions force all components to equalize), normal hyperbolicity (the diversity modes decay (2−ρ) times faster than the aggregate mode, creating a spectral gap), and Fenichel persistence (the fast-decaying diversity modes create an invariant manifold parameterized by F_n alone, which persists under perturbation).

*Economic interpretation:* The aggregate output of a sector is a sufficient statistic for cross-sector planning. A policymaker or downstream sector does not need to know the composition of upstream output — only its level. This is a consequence of CES complementarity, not an assumption of the model. The (2−ρ) factor quantifies the "macro is blind to micro" result: compositional changes within a sector decay (2−ρ) times faster than aggregate changes, so by the time the downstream sector responds, the composition has already equilibrated.

**(ii) Directed coupling.** The cross-sector coupling must be directed (non-reciprocal). Any bidirectional coupling yields an unconditionally stable system incapable of structural transition.

*Proof:* Show that a bidirectional two-sector system has eigenvalues with strictly negative real parts for all parameter values — it cannot bifurcate. Then show that passive bidirectional coupling (any coupling that dissipates energy) only strengthens stability. The bifurcation at ρ(K) = 1 requires net energy injection through the hierarchy, which requires directed coupling.

*Economic interpretation:* Feedback between sectors must be asymmetric — semiconductor improvements enable AI training, but AI training does not directly improve semiconductor fabrication (at least not on the same timescale). If the coupling were symmetric, the economy would be unconditionally stable — which means it could never undergo a structural transition. The possibility of structural change (the "crossing point" where the AI economy activates) requires directed, asymmetric flows.

**(iii) Port alignment.** The coupling enters through the gradient of the CES aggregate (proportional to 1 at equal weights). The coupling *functions* φ_n (how much downstream demand responds to upstream output) are free parameters not determined by ρ.

*Economic interpretation:* The geometry determines *which direction* cross-sector coupling takes (through the aggregate, not through individual components). The *strength* of coupling — the gain functions φ_n — is where the discipline-specific economics lives. Semiconductor economics determines φ_1 (learning curves). Platform economics determines φ_2 (network effects). AI research determines φ_3 (training efficiency). Monetary economics determines φ_4 (settlement demand).

**(iv) Nearest-neighbor topology.** Under timescale separation, long-range coupling (e.g., from hardware directly to settlement, skipping network and capability) has no dynamical effect — the fast intermediate sectors equilibrate before the slow sectors notice.

*Proof:* Explicit computation for a 3-level system: the fastest level equilibrates to a constant, absorbing the long-range coupling into a modified constant input. The reduced Jacobian is lower-triangular regardless of long-range coupling strength. Induction extends to N levels.

*Economic interpretation:* You cannot skip levels in the hierarchy. A semiconductor subsidy does not directly improve financial settlement — it first improves hardware, which enables more AI agents, which develop capabilities, which create settlement demand. Each link in the chain must be traversed. Interventions that try to "jump" levels (e.g., mandating stablecoin adoption without the underlying AI ecosystem) have the same equilibrium effect as doing nothing.

#### The Moduli Space Theorem

**Theorem (Structural Determination).** Fix ρ < 1 and the integers (J, N). Then:

*Qualitative invariants* (determined by ρ): within-level eigenstructure, coupling topology (aggregate, directed, nearest-neighbor), existence of a bifurcation threshold, the Triple Role (from Paper A), and the Eigenstructure Bridge.

*Free parameters* (quantitative degrees of freedom): timescales (ε_1, ..., ε_N), damping rates (σ_1, ..., σ_N), and gain functions (φ_1, ..., φ_N).

*Economic interpretation:* The CES geometry determines the *architecture* of the economy — which variables couple to which, through what channels, in what direction. What the CES geometry leaves free is where the economics lives: how fast each sector adjusts, how much friction each faces, and how responsive cross-sector coupling is. This is a model selection result: the space of possible models collapses from an arbitrary N-sector interaction graph to a specific nearest-neighbor chain with scalar coupling. The analogy to electrical circuits is exact: Kirchhoff's laws constrain the circuit topology; component values (resistors, capacitors) are free.

### 4. The Reduced System and Activation Threshold (~4 pages)

#### 4.1. Reduction to N Dimensions

On the slow manifold, the NJ-dimensional system reduces to N scalar equations:

$$\varepsilon_n \dot{F}_n = \phi_n(F_{n-1})/J − \sigma_n F_n$$

Prove this via projection onto the aggregate (using Euler's theorem for degree-1 homogeneous functions). This is the system that all subsequent analysis uses.

#### 4.2. The Next-Generation Matrix

At the nontrivial equilibrium, decompose the Jacobian as J = T + Σ (transmission + damping). Define the next-generation matrix K = −TΣ⁻¹.

*Economic translation:* Entry K_{n,n-1} = φ_n'(F̄_{n-1})·F̄_{n-1}/|σ_{n-1}| measures: if sector n−1 generates one additional unit of output, how many "next-generation" units of sector n activity does this produce (accounting for depreciation)? This is the cross-sector multiplier.

#### 4.3. Characteristic Polynomial

**Theorem.** The characteristic polynomial of K for a cyclic N-level system is:

$$p(λ) = \prod_i(d_i − λ) − P_{\text{cycle}}$$

where d_i are diagonal entries (within-level reproduction) and P_cycle = Π k_{n+1,n} is the cycle product.

*Proof:* Leibniz expansion of det(K − λI). Only two permutations contribute nonzero products: the identity (giving Π(d_i − λ)) and the full N-cycle (giving (−1)^{N−1} P_cycle). All other permutations select at least one zero entry. One-paragraph proof.

*Economic corollary:* The sensitivity ∂ρ/∂k_{ij} ∝ P_cycle/k_{ij} is largest for the smallest cross-sector multiplier. The system's activation is bottlenecked by its weakest cross-sector link. **Investment thesis: invest in the weakest link, not the strongest.**

#### 4.4. The Spectral Threshold

**Theorem.** The nontrivial equilibrium exists and is stable iff ρ(K) > 1. The transition at ρ(K) = 1 is a structural transition (transcritical bifurcation).

*Key result:* The system can be globally activated (ρ(K) > 1) from individually sub-threshold sectors (d_n < 1 for all n) when P_cycle^{1/N} > 1 − max_i d_i. The cross-sector amplification compensates for sub-threshold individual sectors.

*Economic interpretation:* No single sector — not AI hardware, not the agent ecosystem, not training capability, not financial settlement — is individually strong enough to sustain itself. But the cross-sector feedbacks (cheaper hardware → more agents → better training → more settlement demand → more investment → cheaper hardware) create a self-sustaining cycle. The threshold ρ(K) = 1 is the "crossing point" where this cycle becomes self-sustaining.

### 5. The Four Economic Levels (~6 pages)

Apply the framework to each of the four levels. For each level, present:
- The economic content (what the level represents)
- The gain function φ_n (what determines cross-sector coupling)
- The relevant CES property from the Triple Role
- The ceiling from the slow manifold cascade

#### 5.1. Hardware (Level 1, Slowest — Decades)

Wright's Law learning curves. The gain function φ_1 is a power law with exponent determined by the learning rate (empirically α ≈ 0.23 for semiconductors). This is the pace car — it sets the long-run growth rate of the entire economy.

#### 5.2. Network (Level 2 — Years)

Self-organizing mesh of heterogeneous AI agents. Logistic adoption dynamics: Ḟ_2 = β(F_1)·F_2·(1 − F_2/N*(F_1)) − μF_2. The CES superadditivity (from Paper A) quantifies the diversity premium: combining agents with different capability profiles produces more than the sum.

#### 5.3. Capability (Level 3 — Months)

Autocatalytic training improvement. The effective production multiplier φ_eff = φ_0/(1 − β_auto·φ_0) includes autocatalytic feedback. Three growth regimes with sharp boundaries. The CES correlation robustness (from Paper A) provides collapse protection: the diversity of the training data prevents model collapse (Shumailov et al. 2024).

The Baumol bottleneck: as the mesh automates inference, the remaining non-automated task (frontier training) becomes the binding constraint. Mesh growth converges to the frontier training rate.

#### 5.4. Settlement (Level 4, Fastest — Days)

Stablecoin settlement infrastructure. The CES strategic independence (from Paper A) makes manipulation unprofitable: diversity modes decay (2−ρ) times faster than aggregate modes, suppressing manipulation signals faster than legitimate price signals.

Monetary policy degradation: forward guidance → QE → financial repression degrade in sequence. The Triffin squeeze: stablecoin demand pushes Treasury supply up while mesh participation makes the safety boundary lower.

### 6. The Eigenstructure Bridge and Welfare Decomposition (~5 pages)

#### 6.1. The Non-Gradient Obstruction

**Proposition.** The hierarchical CES economy is *not* a gradient flow. The lower-triangular Jacobian (from directed coupling) is a topological obstruction — no coordinate transformation can symmetrize it.

*Economic interpretation:* There is no social planner's problem whose first-order conditions generate these dynamics. The economy's directed, hierarchical structure is fundamentally incompatible with welfare optimization. Standard welfare theorems do not apply. This is not a market failure — it is a structural feature of hierarchical economies with directed cross-sector flows.

#### 6.2. The Storage Function (Lyapunov)

**Theorem.** The function V(x) = Σ_n c_n D_KL(x_n ‖ x_n*), with tree coefficients c_n = P_cycle/k_{n,n-1}, is a Lyapunov function (welfare distance from equilibrium) for the nontrivial equilibrium.

Construction via Li-Shuai-van den Driessche (2010). The cross-level contributions cancel by the tree condition — this is not obvious and requires the specific coefficients c_n.

*Economic interpretation:* V measures the total welfare loss from being out of equilibrium, decomposed by sector. Each sector's contribution is c_n · D_KL(x_n ‖ x_n*), where c_n is determined by the system's structure. V always decreases along the economy's trajectory (the economy always moves toward equilibrium). The decomposition identifies which sector is contributing most to total welfare loss.

#### 6.3. The Bridge Equation

**Theorem (Eigenstructure Bridge).** On the slow manifold:

$$\nabla^2\Phi|_{\text{slow}} = W^{-1} \cdot \nabla^2 V$$

where W = diag(W_{nn}) with W_{nn} = P_cycle/(|σ_n|·ε_{T_n}).

*Economic interpretation:* Three objects, three meanings:
- Φ (production technology): what the economy *can* do
- V (welfare distance): how far the economy *is* from efficiency
- W (institutional quality): how efficiently the economy *adjusts*

The Bridge says: the curvature of the technology landscape determines the curvature of the welfare landscape, up to a level-specific scaling factor W_{nn} that depends on institutional quality. Countries with better institutions (lower W_{nn}) have tighter correspondence between technological possibility and welfare realization.

The production technology (ρ) determines *which* adjustments are fast and which are slow (eigenvectors). The institutional parameters (σ_n, φ_n) determine *how fast* (eigenvalues). Different countries have different σ_n and φ_n, so they converge at different rates, but along the same directions. This is a Lucas-critique-compatible statement.

#### 6.4. Welfare Loss Decomposition (Under Power-Law Gains)

**Proposition.** With power-law gain functions φ_n(z) = a_n z^{β_n}:

$$V = \frac{P_{\text{cycle}}}{\sigma J}\sum_{n=1}^{N}\frac{1}{\beta_n}\;g\!\left(\frac{F_n}{\bar{F}_n}\right)$$

(under uniform damping), where g(z) = z − 1 − log z ≥ 0. The contribution of level n to welfare loss is proportional to g(F_n/F̄_n)/β_n.

*Economic interpretation:* Sectors with *inelastic* gain functions (small β_n — cross-sector coupling responds weakly to upstream improvements) contribute more welfare loss per unit of disequilibrium. The binding welfare constraint is the most institutionally rigid sector, not the most visibly disrupted one.

**Current implication:** The welfare-relevant bottleneck is more likely at the capability layer (slow-moving training pipelines, regulatory barriers to AI deployment) than at the settlement layer (fast-moving fintech). A policymaker focused on stablecoin regulation is optimizing the wrong margin.

### 7. The Damping Cancellation and Policy (~4 pages)

#### 7.1. The Damping-Speed Tradeoff

**Proposition.** At level n:
- Convergence speed = σ_n/ε_n (increasing in σ_n — more friction means faster convergence)
- Equilibrium output = φ_n(F̄_{n-1})/(σ_n J) (decreasing in σ_n — more friction means lower output)
- **Lyapunov dissipation rate = P_cycle · σ_{n-1} / (β_n · J · F̄_n) · (δF_n)²/F̄_n — independent of σ_n itself.**

*Economic interpretation:* Tightening regulation at sector n speeds up convergence to equilibrium but lowers the equilibrium itself. These two effects *exactly cancel* in the welfare dissipation. The net welfare effect of local regulation is zero. The welfare dissipation at sector n depends on σ_{n-1} (upstream friction) and β_n (the sector's own responsiveness), not on σ_n.

#### 7.2. The Upstream Reform Principle

**Theorem.** To accelerate welfare-relevant adjustment at sector n:
- Increase β_n (make the sector more responsive to upstream improvements), OR
- Reduce σ_{n-1} (reduce friction at the upstream sector)
- Do NOT increase σ_n (tighten local regulation) — it has zero net welfare effect.

**Policy chain:**
- Fix settlement (n=4): reform capability aggregation (σ_3) or increase settlement elasticity (β_4).
- Fix capability (n=3): reform network recruitment (σ_2) or increase training elasticity (β_3).
- Fix network (n=2): reform hardware investment (σ_1) or increase recruitment elasticity (β_2).
- Fix hardware (n=1): reduce γ_c directly (CHIPS Act, semiconductor subsidies).

**Corollary:** Stablecoin regulation (σ_4) has zero marginal welfare effect. Capability-layer reform (σ_3 or β_4) has positive marginal welfare effect. This is a theorem, not a heuristic.

#### 7.3. The Global Welfare Ordering

**Corollary.** Under the partial order β ≽ β' (all gain elasticities weakly higher): V(β) ≤ V(β') at every non-equilibrium state. More responsive cross-sector coupling is unambiguously welfare-improving, regardless of the current state.

### 8. Transition Dynamics (~3 pages)

#### 8.1. The Hierarchical Ceiling Cascade

Each level is bounded by its parent:
- F_4 ≤ S̄(F_3) (settlement bounded by capability)
- F_3 ≤ (φ_eff/δ_C)·F_CES(N*(F_1)) (capability bounded by network and hardware)
- F_2 ≤ N*(F_1) (network bounded by hardware)

The long-run growth rate equals the hardware improvement rate (the Baumol bottleneck). The Baumol cost disease and the Triffin squeeze are the same mathematical object at adjacent layers.

#### 8.2. The Canard: How Long Does the Transition Take?

**Theorem (Canard Duration).** If the bifurcation parameter drifts at rate ε_drift, the transition duration is:

$$\Delta t_{\text{crisis}} = \frac{\pi}{\sqrt{|a|\,\varepsilon_{\text{drift}}}}$$

If the drift is in institutional friction (γ_c improving): a = −1, duration = π/√ε_drift, independent of all other parameters.

At Wright's Law semiconductor improvement rates (≈15% annual cost decline): ≈8 years.

*Economic interpretation:* After conditions become favorable for the high-activity equilibrium, the economy lingers near the old equilibrium for O(1/√ε_drift) time before snapping to the new one. This is the "crisis duration" — the period of structural transition. It is computable from observable drift rates.

Where K enters: K controls the *sharpness* of the transition (how quickly the economy accelerates once it begins transitioning), not the *duration*. Higher complementarity means a faster snap, with less overshooting.

#### 8.3. Dispersion as Leading Indicator

At the structural transition, the spectral gap between diversity and aggregate modes closes. Within-sector heterogeneity stops being slaved to the aggregate. **Prediction:** cross-sectional variance of agent performance widens *before* aggregate statistics move.

### 9. Empirical Predictions (~2 pages)

Present 11 predictions (P1–P11) with falsification criteria and time horizons. Organized by which theoretical result they test:

- **P1–P3:** Test the Triple Role (from Paper A, applied to each level)
- **P4:** Test the spectral threshold (cross-layer acceleration with 6–10 year delay)
- **P5–P6:** Test monetary policy degradation (forward guidance → QE → repression sequence)
- **P7–P8:** Test settlement feedback (stablecoin Treasury share, dollarization)
- **P9:** Test the hierarchical ceiling (mesh growth → frontier training rate)
- **P10:** Test damping cancellation (stablecoin regulation has no persistent welfare effect)
- **P11:** Test dispersion indicator (cross-sectional variance leads aggregate)

### 10. Limitations (~2 pages)

Be ruthlessly honest. Two categories:

**Mathematical:**
1. Gain functions are genuinely free (proved impossible to determine from ρ alone)
2. Timescale separation cannot be eliminated (required for the slow manifold reduction)
3. The system is not a gradient flow (no welfare theorems)
4. Local stability only (global stability requires boundary analysis)
5. Symmetric weights for quantitative bounds (general weights via secular equation are less clean)
6. O(ε) approximation error in the sufficient statistic (crisis episodes may depend on within-sector composition)

**Empirical:**
1. Gain elasticities uncalibrated
2. Damping rates uncalibrated
3. Predictions span 2027–2040 (long horizon)
4. Which specific layer binds first depends on uncalibrated β_n

### 11. Conclusion (~1 page)

One Hamiltonian, one derived architecture, five results, three policy principles.

Three policy principles from theorems:
1. Reform upstream, not locally (damping cancellation).
2. Increase gain elasticities at any layer (global welfare ordering).
3. The transition takes O(1/√ε_drift) — invest in the weakest cross-level link (canard + cycle product sensitivity).

---

## Source Material

All proofs are fully developed in the following files:

1. **`Complementary_Heterogeneity_v3.tex`** — The complete LaTeX paper (1507 lines). Contains all theorems, proofs, and the four-level application. This is the primary source.

2. **`unified_ces_framework.md`** — The complete unified treatment (Parts 0–10). Contains the proof dependency diagram and detailed proofs of all major results.

3. **`ces_three_extensions.md`** — Three consequences of the Bridge Equation: parametric Lyapunov families (Proposition E.1), monotone comparative statics (Proposition E.2), and canard duration (Theorem E.6).

4. **`research/analysis/generating_system_unified.md`** — How the CES free energy produces all three theorems through a single geometric object. Contains the Eigenstructure Bridge in its most explicit form.

5. **Paper A** (the CES Triple Role, from the companion prompt) — cite this for all within-level static results.

## Critical Instructions

1. **Every theorem gets an "Economic interpretation" paragraph.** This is non-negotiable. The paper fails if an economist cannot read the interpretations alone and understand the contribution.

2. **Use the translation table above consistently.** Never say "Fenichel's theorem" in the main text without immediately saying "the fast sector equilibrates before the slow sector moves." Never say "transcritical bifurcation" without immediately saying "structural transition."

3. **The proofs go in the appendix.** The main text should contain theorem statements, proof sketches (2–3 sentences each), and economic interpretations. Full proofs in the appendix for the referees.

4. **Do NOT reproduce the Triple Role proofs.** Cite Paper A. The within-level results are assumed. This paper is about cross-level dynamics.

5. **The non-gradient obstruction is a feature, not a bug.** Do NOT apologize for the lack of a social planner's problem. Instead, explain that directed hierarchical economies are structurally different from the exchange economies for which welfare theorems were designed. The absence of a potential function is *the reason* the Bridge equation and the damping cancellation are nontrivial results.

6. **The Moduli Space Theorem is a model selection result.** Frame it as: the CES geometry dramatically reduces the space of possible models. This is a feature — it means the qualitative results (topology, threshold, stability) are robust to the specific choice of gain functions.

7. **The four economic applications (Sections 5.1–5.4) should be readable as standalone mini-papers.** Each should have enough economic content that a specialist in that area (semiconductor economics, platform theory, AI research economics, monetary theory) can engage with it.

8. **Keep the main text to ~40 pages.** Proofs in the appendix can be as long as needed.

9. **Figures:** Include at least:
   - The proof dependency diagram (from unified_ces_framework.md)
   - A schematic of the four-level hierarchy with timescales
   - The ceiling cascade (F_1 → h_2 → h_3 → h_4)
   - The canard trajectory showing the transition timing
