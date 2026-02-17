# PROMPT: CORRECT DERIVATION OF THE GENERATING SYSTEM

## The Graph-Theoretic Lyapunov Function for a Hierarchical CES-Coupled System

## Attached files:
## 1. ces_triple_role_v2.md (the proved CES Triple Role theorem — contains the Hessian computation)
## 2. generating_system_analysis.md (the variational framework search — identifies Φ = −log F as candidate)
## 3. This prompt

---

## CONTEXT

A search across nine mathematical frameworks identified the variational framework as the strongest candidate (7/10) for a single generating object unifying three theorems about a hierarchical CES system. The generating object is:

$$\Phi(x) = -\sum_{n=1}^{N} \log F_n(x_n) + V_{\text{coupling}}(x_1, \ldots, x_N)$$

where F_n(x_n) = (Σⱼ aⱼ xⱼᵖ)^{1/ρ} is the CES aggregate at level n.

**The three theorems it should generate:**

1. Strong convexity of Φ_n on each level → CES Triple Role (curvature K controls superadditivity, correlation robustness, strategic independence)
2. Loss of positive-definiteness of ∇²Φ_total at ρ(K) = 1 → Master R₀ (transcritical bifurcation, system activation)
3. Fast minimization of Φ at fixed slow variables → Hierarchical Ceiling (slow manifold, Baumol bottleneck)

**The gap:** The actual ODE dynamics may not be a gradient flow of Φ. A graph-theoretic Lyapunov function (Shuai & van den Driessche 2013) was proposed to close this gap. The proposal contained the correct architecture but incorrect formulas. This prompt requests the correct derivation.

---

## THE ACTUAL SYSTEM

### The Four-ODE System

Four state variables at four levels:

- x₁ = c(t): distributed cost advantage (silicon learning curves). **SLOWEST: years-decades.**
- x₂ = N(t): mesh network density. **MEDIUM-SLOW: months.**
- x₃ = C(t): mesh aggregate capability (CES). **MEDIUM-FAST: weeks-months.**
- x₄ = S(t): stablecoin ecosystem size. **FASTEST: days-weeks.**

Timescale ordering: **ε₄ ≪ ε₃ ≪ ε₂ ≪ ε₁ = 1** (Level 1 is slowest, Level 4 is fastest).

The dynamics:

$$\dot{x}_1 = \underbrace{\delta_c \cdot I(x_4)^{\alpha} \cdot x_1^{\phi_c}}_{\text{amplification: investment} \to \text{learning}} - \underbrace{\gamma_c \cdot x_1}_{\text{damping: saturation}}$$

$$\dot{x}_2 = \underbrace{\beta(x_1) \cdot x_2 \cdot (1 - x_2/N^*(x_1))}_{\text{amplification: recruitment via diversity premium}} - \underbrace{\mu \cdot x_2}_{\text{damping: exit + coordination}}$$

$$\dot{x}_3 = \underbrace{\frac{\phi_0}{1 - \beta_{\text{auto}} \phi_0} \cdot \left(\sum_j a_j C_j^{\rho}\right)^{1/\rho}}_{\text{amplification: autocatalytic CES aggregate}} - \underbrace{\delta_C \cdot x_3}_{\text{damping: depreciation + Baumol}}$$

$$\dot{x}_4 = \underbrace{\eta(x_3, x_2) \cdot x_4 \cdot (1 - x_4/\bar{S}(x_3))}_{\text{amplification: settlement demand + dollarization}} - \underbrace{\nu \cdot x_4}_{\text{damping: Triffin + depreciation}}$$

Cross-level couplings (output of level n feeds input of level n+1, plus one closing term):
- β(x₁): mesh recruitment rate increases with cheaper hardware (Level 1 → Level 2)
- N*(x₁): mesh carrying capacity increases with cheaper hardware (Level 1 → Level 2)
- The CES aggregate at Level 3 depends on mesh density x₂ (Level 2 → Level 3) through the number and diversity of agents
- η(x₃, x₂): settlement demand increases with capability (Level 3 → Level 4)
- I(x₄): investment attracted by settlement quality (Level 4 → Level 1, closing the cycle)

### The Trivial Equilibrium

(x₁, x₂, x₃, x₄) = (0, 0, 0, 0): no mesh economy.

### The Non-Trivial Equilibrium

(x̄₁, x̄₂, x̄₃, x̄₄) with all components positive: active mesh economy. Exists when the system is super-threshold.

### The Linearized System Near the Trivial Equilibrium

ẋ = (T + Σ)x where:

**Transmission matrix T** (amplification, nonnegative):

$$T = \begin{pmatrix} T_{11} & 0 & 0 & T_{14} \\ T_{21} & T_{22} & 0 & 0 \\ 0 & T_{32} & T_{33} & 0 \\ 0 & 0 & T_{43} & T_{44} \end{pmatrix}$$

- T₁₁ = δ_c · I(0)^α · φ_c · 0^{φ_c - 1} — this requires care at x₁ = 0. For linearization, evaluate the Jacobian ∂f₁/∂x₁ at the trivial equilibrium.
- T₂₁ = ∂f₂/∂x₁|₀ = β'(0) · x₂ · (1 − x₂/N*(0)) + ... — again requires careful evaluation at (0,0,0,0).
- T₁₄ = ∂f₁/∂x₄|₀: the feedback from settlement to investment.
- Diagonal entries Tᵢᵢ: within-level amplification rates.
- Sub-diagonal entries Tᵢ,ᵢ₋₁: cross-level amplification rates.

**IMPORTANT:** The linearization at (0,0,0,0) may be degenerate because several terms vanish. The more informative linearization is around the non-trivial equilibrium x̄. For the NGM construction, what matters is the behavior near the disease-free equilibrium, which in this system is (0,0,0,0). But the CES structure is most transparent at the non-trivial equilibrium where all components are positive.

**Transition matrix Σ** (damping, diagonal, negative):

$$\Sigma = \text{diag}(-\gamma_c, -\mu, -\delta_C, -\nu)$$

**Next-generation matrix:**

$$\mathbf{K} = -T\Sigma^{-1}$$

**Master R₀:**

$$\mathcal{R}_0 = \rho(\mathbf{K}) = \text{spectral radius}$$

---

## WHAT WAS WRONG IN THE PREVIOUS ATTEMPT

An attempt to map this system to a graph-theoretic Lyapunov function contained five errors:

### Error 1: The (2−ρ) Dissipation Factor

**Claimed:** The diversity eigenvalue of the dissipation matrix is λ_⊥ = σ_n(2−ρ)/ε_n.

**Actual:** The CES Hessian at the symmetric point (from Lemma 1 of the CES Triple Role proof) is:

$$H_F = \frac{(1-\rho)}{J^2 c}\left[\mathbf{1}\mathbf{1}^T - JI\right]$$

The eigenvalues are:
- Tangent directions (v with 𝟙·v = 0): eigenvalue = −(1−ρ)/(Jc), multiplicity J−1
- Normal direction (proportional to 𝟙): eigenvalue = (1−ρ)(J−1)/(J²c), multiplicity 1

For Φ = −log F, the Hessian ∇²Φ = −H_F/F + (∇F)(∇F)ᵀ/F²:

At the symmetric point (where F = c, ∂ⱼF = 1/J):
- Tangent eigenvalue: (1−ρ)/(Jc²)
- Normal eigenvalue: 1/(Jc²) + (1−ρ)(J−1)/(J²c²) = [J + (1−ρ)(J−1)]/(J²c²) = [1 + K]/(Jc²)

The ratio of diversity to aggregate eigenvalues is:

$$\frac{\lambda_\perp}{\lambda_\parallel} = \frac{(1-\rho)/Jc^2}{[1+K]/Jc^2} = \frac{1-\rho}{1+K} = \frac{KJ/(J-1)}{1+K}$$

For ρ = 0 (Cobb-Douglas): K = (J−1)/J, ratio = 1/(1 + (J−1)/J) = J/(2J−1) ≈ 1/2 for large J.
For ρ = −1: K = 2(J−1)/J, ratio = 2/(1 + 2(J−1)/J) = 2J/(3J−2) ≈ 2/3 for large J.
For ρ → 1: K → 0, ratio → 0 (diversity modes have no dissipation — flat isoquant).

The factor (2−ρ) does not appear anywhere in this structure. The correct "filter strength" is (1−ρ)/(1+K) = KJ/[(J−1)(1+K)].

**Task:** Derive the correct dissipation eigenstructure for each level and express in terms of K.

### Error 2: The Spectral Radius Formula

**Claimed:**

$$\rho(\mathbf{K}) = \sqrt[4]{\frac{T_{21}^* T_{32}^* T_{43}^* \beta_1}{\sigma_1 \sigma_2 \sigma_3 \sigma_4 J^4}}$$

**Actual:** This formula holds ONLY for a purely cyclic matrix with zero diagonal entries:

$$K_{\text{cyclic}} = \begin{pmatrix} 0 & 0 & 0 & k_{14} \\ k_{21} & 0 & 0 & 0 \\ 0 & k_{32} & 0 & 0 \\ 0 & 0 & k_{43} & 0 \end{pmatrix}$$

whose eigenvalues are λ = (k₂₁k₃₂k₄₃k₁₄)^{1/4} · ω where ω ranges over the 4th roots of unity.

The actual NGM has nonzero diagonal entries Kᵢᵢ (within-level reproduction). The characteristic polynomial of the full matrix is:

$$\det(\mathbf{K} - \lambda I) = \prod_{i=1}^{4}(K_{ii} - \lambda) - K_{21}K_{32}K_{43}K_{14} + \text{lower-order cycle terms}$$

This is a quartic in λ. The spectral radius ρ(K) depends on ALL entries, not just the 4-cycle product. By Perron-Frobenius (K is nonneg irreducible via the cycle), ρ(K) > max(Kᵢᵢ), but the exact value requires solving the quartic.

Also: the J⁴ factor in the denominator has no derivation and appears incorrect. The CES output F at the symmetric point is c (not c/J), so the cross-level coupling rates do not include a 1/J factor per level.

**Task:** Compute the correct characteristic polynomial of the 4×4 NGM. Derive ρ(K) as a function of the diagonal entries Kᵢᵢ and the cycle product K₂₁K₃₂K₄₃K₁₄. Express in terms of the physical parameters from each level's dynamics.

### Error 3: The Timescale Ordering is Inverted

**Claimed:** ε₁ ≪ ε₂ ≪ ε₃ ≪ ε₄, "Level 1 equilibrates almost instantly," "Level 4 defines the long-run growth limit."

**Actual:** The correct ordering is:

| Level | Process | Timescale | ε parameter |
|-------|---------|-----------|-------------|
| 4 | Settlement/stablecoin dynamics | days-weeks | ε₄ ≪ 1 (FASTEST) |
| 3 | Autocatalytic capability growth | weeks-months | ε₃ |
| 2 | Mesh network formation | months | ε₂ |
| 1 | Silicon learning curves / institutional adaptation | years-decades | ε₁ = 1 (SLOWEST) |

Level 4 (settlement) equilibrates almost instantly. Level 1 (learning curves) defines the long-run growth limit. This is the Baumol bottleneck: the slowest-adapting sector determines the system growth rate.

The slow manifold structure on the slowest timescale:

$$x_4 = h_4(x_1, x_2, x_3) \quad \text{(settlement equilibrates given other variables)}$$
$$x_3 = h_3(x_1, x_2) \quad \text{(capability equilibrates given mesh and hardware)}$$
$$x_2 = h_2(x_1) \quad \text{(mesh equilibrates given hardware cost)}$$

Effective dynamics: ẋ₁ = F_eff(x₁). Long-run growth = growth of x₁.

**Task:** Correctly assign timescales. Derive the slow manifold functions h₄, h₃, h₂ by setting the fast dynamics to zero and solving for the fast variable in terms of the slower ones.

### Error 4: The Tree Coefficients Use Mass-Action, Not CES

**Claimed:** c_{n+1} = cₙ · T*_{n+1,n} x̄ₙ / (σ_{n+1} x̄²_{n+1})

**Actual:** This formula comes from Shuai & van den Driessche (2013) for compartmental epidemiological models with mass-action incidence (transmission rate proportional to the product of susceptible and infected populations). The CES system has DIFFERENT coupling: the transmission rate at Level 3 depends on the CES aggregate of Level 2's outputs, not on a bilinear product.

The correct tree coefficients for a CES-coupled system must be derived from the CES-specific Jacobian. The Shuai-van den Driessche construction requires:

1. Identify the directed graph G of the transmission network (the support of T).
2. For each spanning tree τ of G rooted at node n, compute the weight w(τ) = Π_{edges (i,j) ∈ τ} ∂fᵢ/∂xⱼ|_{x̄}.
3. Set cₙ = Σ_{spanning trees rooted at n} w(τ).
4. The Lyapunov function is V(x) = Σₙ cₙ gₙ(xₙ) where gₙ(xₙ) = xₙ − x̄ₙ − x̄ₙ log(xₙ/x̄ₙ).

The key difference: ∂fᵢ/∂xⱼ for the CES system involves the CES marginal products ∂F/∂xⱼ = F^{1-ρ}/J · xⱼ^{ρ-1}, not the simple bilinear mass-action terms. The tree coefficients must be computed from THESE derivatives.

**Task:** Compute the correct tree coefficients for the four-level CES-coupled system. The directed graph has edges {1→2, 2→3, 3→4, 4→1} (the cycle) plus self-loops {1→1, 2→2, 3→3, 4→4}. Enumerate the spanning trees rooted at each node. Compute the edge weights from the CES-specific Jacobian at the non-trivial equilibrium.

### Error 5: The Lyapunov Function's Relationship to Φ = −log F is Unstated

The graph-theoretic Lyapunov function V(x) = Σ cₙ(xₙ − x̄ₙ log xₙ) and the variational generating function Φ(x) = −Σ log Fₙ(xₙ) are related but NOT identical.

V is a function of the AGGREGATE state variables (x₁, x₂, x₃, x₄). Φ is a function of the DISAGGREGATED component vectors at each level. V lives in ℝ⁴. Φ lives in ℝ^{4J}.

The relationship: Φ restricted to the slow manifold (where within-level allocation is optimal) should reduce to something proportional to V. That is:

$$\Phi(x_1^*, x_2^*, x_3^*, x_4^*) \approx \sum_n \alpha_n (-\log x_n) + \text{const}$$

where x_n^* is the optimal within-level allocation given aggregate output xₙ, and αₙ are constants related to the tree coefficients cₙ.

**Task:** Show that V is the restriction of Φ to the slow manifold. Derive the relationship between the tree coefficients cₙ and the weights in Φ. This is the Eigenstructure Bridge — the connection between the within-level geometry (Φ, governed by K) and the between-level dynamics (V, governed by the spanning tree structure).

---

## WHAT NEEDS TO BE DERIVED

### Derivation 1: Correct Eigenstructure of the CES Dissipation

At each level n, the full dynamics split into:
- Aggregate dynamics: how the total output xₙ = Fₙ(component vector) evolves
- Diversity dynamics: how the allocation across J components evolves

The Hessian of Φₙ = −log Fₙ at the symmetric point has:
- J−1 tangent eigenvalues: (1−ρ)/(Jc²) — these govern diversity dissipation
- 1 normal eigenvalue: [1+K]/(Jc²) — this governs aggregate dissipation

The diversity modes dissipate FASTER than the aggregate mode (ratio (1−ρ)/(1+K) < 1 for ρ < 1). This means within-level diversity dynamics equilibrate before aggregate dynamics — a WITHIN-level timescale separation, separate from the BETWEEN-level timescale separation (ε₁ through ε₄).

**Derive:** The two-timescale structure at each level. Show that the fast within-level diversity dynamics produce the CES optimal allocation, and the slow within-level aggregate dynamics are what couple between levels. This justifies treating each level as a scalar (its CES aggregate output xₙ) when constructing the between-level NGM and Lyapunov function.

### Derivation 2: Correct Characteristic Polynomial of the NGM

The 4×4 NGM K = −TΣ⁻¹ has the structure:

$$\mathbf{K} = \begin{pmatrix} d_1 & 0 & 0 & k_{14} \\ k_{21} & d_2 & 0 & 0 \\ 0 & k_{32} & d_3 & 0 \\ 0 & 0 & k_{43} & d_4 \end{pmatrix}$$

where dᵢ = Kᵢᵢ = −Tᵢᵢ/Σᵢᵢ (within-level reproduction) and kᵢⱼ = −Tᵢⱼ/Σⱼⱼ (cross-level reproduction).

**Derive:** The characteristic polynomial det(K − λI) = 0 for this specific sparsity pattern. Note that K is not symmetric (kᵢⱼ ≠ kⱼᵢ in general) but IS nonnegative irreducible (the cycle ensures irreducibility). Perron-Frobenius guarantees a unique dominant eigenvalue ρ(K) > 0 with a positive eigenvector.

For the cyclic-plus-diagonal structure, the characteristic polynomial is:

$$\prod_{i=1}^{4}(d_i - \lambda) - k_{21}k_{32}k_{43}k_{14} = 0$$

Wait — verify this. The determinant of (K − λI) for the given sparsity:

$$\det \begin{pmatrix} d_1-\lambda & 0 & 0 & k_{14} \\ k_{21} & d_2-\lambda & 0 & 0 \\ 0 & k_{32} & d_3-\lambda & 0 \\ 0 & 0 & k_{43} & d_4-\lambda \end{pmatrix}$$

Expanding along the first row:

$$(d_1-\lambda)\det\begin{pmatrix} d_2-\lambda & 0 & 0 \\ k_{32} & d_3-\lambda & 0 \\ 0 & k_{43} & d_4-\lambda \end{pmatrix} + (-1)^{1+4}k_{14}\det\begin{pmatrix} k_{21} & d_2-\lambda & 0 \\ 0 & k_{32} & d_3-\lambda \\ 0 & 0 & k_{43} \end{pmatrix}$$

$= (d_1-\lambda)(d_2-\lambda)(d_3-\lambda)(d_4-\lambda) - k_{14} \cdot k_{21} \cdot k_{32} \cdot k_{43}$

So the characteristic polynomial IS:

$$p(\lambda) = \prod_{i=1}^{4}(d_i - \lambda) - k_{21}k_{32}k_{43}k_{14} = 0$$

This is a quartic. The spectral radius ρ(K) is the largest real root.

**Special case: all dᵢ equal.** If d₁ = d₂ = d₃ = d₄ = d, then:

$$(d - \lambda)^4 = k_{21}k_{32}k_{43}k_{14} \equiv P_{\text{cycle}}$$

$$\lambda = d + P_{\text{cycle}}^{1/4} \cdot \omega$$

where ω ranges over the 4th roots of unity. The spectral radius is:

$$\rho(\mathbf{K}) = d + P_{\text{cycle}}^{1/4}$$

This exceeds d (the within-level R₀) by exactly the 4th root of the cycle product. The excess amplification from cross-level coupling is P_cycle^{1/4}.

**General case:** Solve or bound the quartic. For small cross-coupling (P_cycle ≪ Π(dᵢ)), perturbation theory gives:

$$\rho(\mathbf{K}) \approx \max_i(d_i) + \frac{P_{\text{cycle}}}{\prod_{i \neq i^*}(\max(d_i) - d_i)} + O(P_{\text{cycle}}^2)$$

For large cross-coupling (P_cycle ≫ Π(dᵢ)), the cycle dominates and ρ(K) ≈ P_cycle^{1/4}.

**Derive:** Express dᵢ and kᵢⱼ in terms of the physical parameters at each level (learning curve exponents, CES parameters, damping rates). Compute P_cycle. Determine whether the current system is in the small-coupling or large-coupling regime.

### Derivation 3: Correct Slow Manifold Functions

With ε₄ ≪ ε₃ ≪ ε₂ ≪ ε₁ = 1:

**Step 1: Equilibrate Level 4 (fastest).** Set ẋ₄ = 0:

$$\eta(x_3, x_2) \cdot x_4 \cdot (1 - x_4/\bar{S}(x_3)) - \nu \cdot x_4 = 0$$

Solutions: x₄ = 0 (trivial) or x₄ = h₄(x₂, x₃) = S̄(x₃) · (1 − ν/η(x₃, x₂)).

This is the slow manifold for Level 4: settlement ecosystem size equilibrates given mesh capability and density. The ceiling S̄(x₃) depends on mesh capability (through the safe asset supply constraint — the Triffin squeeze).

**Step 2: Equilibrate Level 3.** Set ẋ₃ = 0 given x₄ = h₄(x₂, x₃):

$$\frac{\phi_0}{1 - \beta_{\text{auto}}\phi_0} \cdot F_{\text{CES}}(x_2) - \delta_C \cdot x_3 = 0$$

$$x_3 = h_3(x_2) = \frac{\phi_{\text{eff}}}{\delta_C} \cdot F_{\text{CES}}(x_2)$$

The CES aggregate F_CES depends on mesh density x₂ through the number and diversity of agents. The ceiling is δ_C (depreciation + Baumol bottleneck from frontier training).

**Step 3: Equilibrate Level 2.** Set ẋ₂ = 0 given x₃ = h₃(x₂):

$$\beta(x_1) \cdot x_2 \cdot (1 - x_2/N^*(x_1)) - \mu \cdot x_2 = 0$$

Solutions: x₂ = 0 or x₂ = h₂(x₁) = N*(x₁) · (1 − μ/β(x₁)).

The mesh density equilibrates given hardware cost. The ceiling N*(x₁) depends on how cheap distributed inference is.

**Step 4: Effective dynamics of Level 1 (slowest).**

$$\dot{x}_1 = \delta_c \cdot I(h_4(h_2(x_1), h_3(h_2(x_1))))^{\alpha} \cdot x_1^{\phi_c} - \gamma_c \cdot x_1$$

The investment I depends on settlement quality S = h₄, which depends on capability x₃ = h₃, which depends on mesh density x₂ = h₂, which depends on hardware cost x₁. The entire system collapses to one equation in x₁.

**Derive:** The explicit forms of h₂, h₃, h₄. Verify that the composition h₄ ∘ h₃ ∘ h₂ produces a well-defined effective dynamic for x₁. Identify the long-run growth rate as a function of exogenous parameters (frontier training rate g_Z, institutional adaptation speed).

### Derivation 4: Correct Tree Coefficients

The directed graph G of the transmission network has:
- Nodes: {1, 2, 3, 4}
- Edges: {1→2, 2→3, 3→4, 4→1} (cycle) plus {1→1, 2→2, 3→3, 4→4} (self-loops)

For the Shuai-van den Driessche Lyapunov construction, we need spanning trees of the CYCLE GRAPH (ignoring self-loops) rooted at each node.

For a directed 4-cycle, the spanning trees rooted at node n are paths from all other nodes TO node n. There is exactly one spanning tree rooted at each node (the unique path along the cycle). So:

- Tree rooted at 1: edges {2→1, 3→2, 4→3} — wait, the cycle is 1→2→3→4→1. The tree rooted at 1 uses edges pointing toward 1. With the cycle direction 1→2→3→4→1, the tree rooted at 1 has edges {4→1, 3→4, 2→3}: weight = k₁₄ · k₄₃ · k₃₂. Hmm — this depends on how you define the spanning tree.

Actually, for a directed graph, a spanning tree rooted at node r is a directed tree where every node has a unique directed path TO r. For the cycle 1→2→3→4→1, the edges go: 1→2, 2→3, 3→4, 4→1. The only spanning tree rooted at node 1 is: {4→1, 3→4, 2→3}, which uses edges in the REVERSE direction — but these edges don't exist in the cycle 1→2→3→4→1.

Hmm. For the directed cycle 1→2→3→4→1, spanning trees rooted at node 1 must use edges from the cycle. The edges pointing toward 1 is only 4→1. Then from 3, we need to reach 1 via 4: 3→4→1. From 2: 2→3→4→1. The tree is: {2→3, 3→4, 4→1}. Weight = k₃₂ · k₄₃ · k₁₄.

Wait — the edges of the cycle are 1→2, 2→3, 3→4, 4→1, which correspond to weights k₂₁, k₃₂, k₄₃, k₁₄ (transmission FROM j TO i). The spanning tree rooted at 1 uses all edges except the one into 1 (which would create a cycle): it uses {1→2, 2→3, 3→4}. But that's a tree rooted at 1 in the OUT-tree sense. We need IN-trees (all edges pointing toward root).

For the cycle, the unique IN-tree rooted at node 1 is: {2→1, 3→2, 4→3}... but 2→1 is NOT an edge of the cycle (the cycle goes 1→2, not 2→1).

**The issue: the cycle graph 1→2→3→4→1 has no in-tree rooted at any node, because you can't reverse the edges.** The Shuai-van den Driessche construction requires the graph to have a spanning in-tree at each node, which requires the graph to be strongly connected. The cycle IS strongly connected (you can reach any node from any other by going around), but the in-trees use the existing directed edges.

Let me reconsider. In the cycle 1→2→3→4→1, to build an in-tree rooted at 1 using EXISTING edges:
- From 2, we need a path to 1 using directed edges: 2→3→4→1. Uses edges 2→3, 3→4, 4→1.
- From 3: 3→4→1. Uses 3→4, 4→1.  
- From 4: 4→1. Uses 4→1.

The tree is: {2→3, 3→4, 4→1}. Weight = k₃₂ · k₄₃ · k₁₄.

Similarly:
- In-tree rooted at 2: {3→4, 4→1, 1→2}. Weight = k₄₃ · k₁₄ · k₂₁.
- In-tree rooted at 3: {4→1, 1→2, 2→3}. Weight = k₁₄ · k₂₁ · k₃₂.
- In-tree rooted at 4: {1→2, 2→3, 3→4}. Weight = k₂₁ · k₃₂ · k₄₃.

Each in-tree uses exactly 3 of the 4 cycle edges (all except the one entering the root). All four in-trees have the same weight: w = k₂₁ · k₃₂ · k₄₃ · k₁₄ / (entering edge weight). Specifically:

c₁ ∝ k₃₂ · k₄₃ · k₁₄
c₂ ∝ k₄₃ · k₁₄ · k₂₁
c₃ ∝ k₁₄ · k₂₁ · k₃₂
c₄ ∝ k₂₁ · k₃₂ · k₄₃

Note: cₙ = P_cycle / k_{n,n-1} (the cycle product divided by the edge entering node n).

**Task:** Verify this computation. Incorporate the self-loop (diagonal) contributions if they affect the spanning tree enumeration. Compute the explicit cₙ values from the CES-specific kᵢⱼ values. Verify that V(x) = Σ cₙ (xₙ − x̄ₙ log xₙ) is a valid Lyapunov function for the system by checking V̇ ≤ 0 along trajectories.

### Derivation 5: The Eigenstructure Bridge

This is the central new result. Connect the within-level geometry (CES curvature K) to the between-level dynamics (tree-Lyapunov V) through the generating function Φ.

**Step 1:** At each level, the within-level dynamics have a fast (diversity, J−1 dimensions) and slow (aggregate, 1 dimension) component. The diversity dynamics dissipate at rate proportional to (1−ρ)/(Jc²) — proportional to K. The aggregate dynamics dissipate at rate proportional to [1+K]/(Jc²).

**Step 2:** After the diversity modes equilibrate (the WITHIN-level slow manifold), each level reduces to its scalar aggregate xₙ = Fₙ(component vector). The full system on the within-level slow manifold is the four-ODE system in (x₁, x₂, x₃, x₄).

**Step 3:** The Lyapunov function V for the four-ODE system has tree coefficients cₙ derived from the CES-specific Jacobian. These cₙ depend on the CES output at each level, hence on ρ through Fₙ.

**Step 4:** The generating function Φ restricted to the within-level slow manifolds (optimal allocation at each level) reduces to:

$$\Phi|_{\text{slow}} = -\sum_n \log F_n(x_n^*) = -\sum_n \log x_n + \text{const}$$

where x_n^* is the symmetric allocation achieving aggregate output xₙ. Note that −log xₙ ≈ (xₙ − x̄ₙ)/x̄ₙ − (xₙ − x̄ₙ)²/(2x̄ₙ²) + ... near x̄ₙ, so:

$$\Phi|_{\text{slow}} \approx \text{const} + \sum_n \frac{1}{x̄_n}(x̄_n - x_n) + \frac{1}{2x̄_n^2}(x_n - x̄_n)^2$$

while:

$$V = \sum_n c_n(x_n - x̄_n \log x_n) \approx \text{const} + \sum_n \frac{c_n}{2x̄_n}(x_n - x̄_n)^2$$

**The bridge:** V and Φ|_slow have the same quadratic structure near the equilibrium if cₙ/x̄ₙ = 1/x̄ₙ², i.e., cₙ = 1/x̄ₙ. Check whether the tree coefficients cₙ = P_cycle/k_{n,n-1} equal 1/x̄ₙ at the CES equilibrium. If yes, V = Φ|_slow and the generating function IS the Lyapunov function restricted to the within-level slow manifolds.

If not exactly equal, the relationship cₙ = αₙ/x̄ₙ with αₙ determined by the tree structure gives the "weighting" that transforms the free energy Φ into the dynamical Lyapunov function V. The tree coefficients encode how the between-level graph structure modifies the natural free energy weights.

**Task:** Compute whether cₙ = 1/x̄ₙ at the CES equilibrium. If not, derive the exact relationship and interpret the discrepancy. This is the Eigenstructure Bridge.

---

## OUTPUT FORMAT

### Part 1: Corrected Eigenstructure
The full eigenvalue decomposition of the CES dissipation at each level, expressed in terms of K. The two-timescale structure (fast diversity modes, slow aggregate mode). The "filter" interpretation.

### Part 2: Corrected Master R₀
The characteristic polynomial. The spectral radius as a function of diagonal entries and cycle product. The regime (small-coupling vs. large-coupling). The dominant eigenvector and its interpretation.

### Part 3: Corrected Slow Manifold
The functions h₂, h₃, h₄. The effective dynamics of x₁. The long-run growth rate. The Baumol bottleneck as a theorem about the slow manifold.

### Part 4: Corrected Tree Coefficients
The spanning tree enumeration. The edge weights from the CES Jacobian. The explicit cₙ values.

### Part 5: The Eigenstructure Bridge
The relationship between V (Lyapunov) and Φ|_slow (free energy on slow manifold). Whether cₙ = 1/x̄ₙ. The interpretation.

### Part 6: Assessment
Does the corrected construction close the gap identified in the generating system analysis? Is the graph-theoretic Lyapunov function V a valid global potential for the four-ODE system? Does this take the variational framework from 7/10 to 9/10? What remains to reach 10/10?

---

## MATHEMATICAL LEVEL

All derivations should be explicit — equations, not descriptions. The Hessian eigenvalues are computed in the attached CES Triple Role v2 (Lemma 1). Use those results directly. The spanning tree enumeration is finite (four trees) and exact. The characteristic polynomial is a specific quartic. The slow manifold functions are specific algebraic expressions.

This is a computation prompt, not a conceptual prompt. The concepts are established. The numbers need to be correct.
