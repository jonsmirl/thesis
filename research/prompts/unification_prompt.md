# Prompt: Rigorous Unification of the CES Hierarchical Framework

You are a mathematical physicist specializing in port-Hamiltonian systems, singular perturbation theory, and cooperative game theory. You will be given four working documents that develop a hierarchical CES (constant elasticity of substitution) framework from different angles. Your task is to produce a single, rigorous unified document that replaces all four.

## Source Documents

The four documents are:

1. **CES Triple Role Theorem** — Proves that the curvature parameter $K = (1-\rho)(J-1)\Phi^{1/\rho}R_{\min}/J$ controls superadditivity, correlation robustness, and strategic independence through isoquant curvature at the cost-minimizing point. Uses a secular equation for the constrained eigenvalues of the weighted inverse-share matrix.

2. **Port Topology Results** — Proves four claims about CES-forced network topology: (1) coupling passes through $F_n$ (aggregate sufficient statistic), (2) coupling must be directed (feed-forward), (3) port direction is forced but gain is free, (4) nearest-neighbor topology is forced under timescale separation. Scores the result 8.5/10.

3. **Gradient Flow Analysis** — Proves the system is NOT a gradient flow (lower-triangular Jacobian is a topological obstruction), NOT GENERIC ($L\nabla\Phi \neq 0$), but does admit a graph-theoretic Lyapunov function $V = \sum c_n D_{KL}(x_n \| x_n^*)$. Identifies the "Eigenstructure Bridge": $\Phi$ and $V$ share eigenvectors with eigenvalue ratios controlled by $\rho$.

4. **Corrected Generating System** — Derives five results for a 4-level system: (1) eigenstructure with $(2-\rho)$ factor for dynamics vs $(1-\rho)$ for geometry, (2) NGM characteristic polynomial, (3) slow manifold functions, (4) tree coefficients for $V$, (5) the Bridge equation $\nabla^2\Phi|_{\text{slow}} = W^{-1}\nabla^2 V$ where $W_{nn} = P_{\text{cycle}}/(|\sigma_n|\varepsilon_{T_n})$.

## What You Must Produce

A single document with the following structure. Every theorem must have a complete proof. No scoring systems. No "working out loud." No false starts. No "Wait, let me redo this." Clean mathematics throughout.

---

### Part 0: Setup and Notation (2 pages max)

Define everything once. Use these conventions consistently:

- $F_n(x_n) = (\frac{1}{J}\sum_{j=1}^J x_{nj}^\rho)^{1/\rho}$: CES aggregate at level $n$ (equal weights for the main development; state the general-weight extension in remarks)
- $F_n^*$: equilibrium value of $F_n$
- $K = (1-\rho)(J-1)/J$: within-level curvature parameter (equal weights)
- $\mathbf{K}$: next-generation matrix (boldface, to distinguish from scalar $K$)
- $\rho$: CES substitution parameter. Always write $\rho(\mathbf{K})$ for the spectral radius of $\mathbf{K}$ to avoid ambiguity.
- $\Phi = -\sum_n \log F_n$: CES free energy (the Hamiltonian)
- $V = \sum_n c_n D_{KL}(x_n \| x_n^*)$: storage function (the Lyapunov function)
- $W$: supply rate matrix relating $\Phi$ and $V$

State the standing assumptions: $\rho < 1$, $\rho \neq 0$, $J \geq 2$ components per level, $N$ levels, timescale separation $\varepsilon_1 \gg \varepsilon_2 \gg \cdots \gg \varepsilon_N > 0$.

### Part 1: The Curvature Lemma (foundation for everything)

State and prove the curvature structure at the symmetric point. This is Section 2 of the Triple Role document, condensed:

1. **Equal marginal products**: $\nabla F_n = (1/J)\mathbf{1}$ at the symmetric point. State this as the structural fact that makes everything work.
2. **Hessian**: $H_F = \frac{(1-\rho)}{J^2 c}[\mathbf{1}\mathbf{1}^T - JI]$. Eigenvalues: $0$ on $\mathbf{1}$, $-(1-\rho)/(Jc)$ on $\mathbf{1}^\perp$.
3. **Curvature parameter**: $K = (1-\rho)(J-1)/J > 0$ for all $\rho < 1$.

Then state (as a remark, not a full proof) the general-weight extension via the secular equation $\sum_j (w_j - \mu)^{-1} = 0$.

### Part 2: Within-Level Eigenstructure

**Distinguish clearly between two objects.** State upfront:

> There are two matrices with different eigenstructures. Both are correct; they measure different things.
>
> - **Object A**: $\nabla^2\Phi_n$ (Hessian of the free energy). Anisotropy ratio: $(1-\rho)$. This is the *geometric* object.
> - **Object B**: $Df_n$ (Jacobian of the dynamics). Anisotropy ratio: $(2-\rho) = 1 + (1-\rho)$. This is the *dynamical* object. The extra $1$ comes from damping.

Derive both eigenstructures. For Object B:

$$Df_n = \frac{\sigma_n}{\varepsilon_n}\left[\frac{(1-\rho)}{J}\mathbf{1}\mathbf{1}^T - (2-\rho)I\right]$$

Eigenvalues: $-\sigma_n/\varepsilon_n$ (aggregate, on $\mathbf{1}$) and $-\sigma_n(2-\rho)/\varepsilon_n$ (diversity, on $\mathbf{1}^\perp$, multiplicity $J-1$).

State the connection: $(2-\rho) = 1 + KJ/(J-1)$.

### Part 3: Port Topology Theorem

State and prove four claims as a single theorem with four parts. Here are the specific tightening moves for each:

**Claim 1 (Aggregate Coupling).** The current proof establishes equilibrium uniqueness ($x_j^{\rho-2} = \text{const} \Rightarrow x_j = \bar{x}$) but then jumps to "$F_n$ is a sufficient statistic." Close this gap explicitly:

> **Proposition.** The critical manifold $\mathcal{M}_n = \{x_{nj} = F_n/J^{1/\rho}, \; j=1,...,J\}$ is normally hyperbolic: the transverse eigenvalues $-\sigma_n(2-\rho)/\varepsilon_n$ satisfy $|(2-\rho)\sigma_n/\varepsilon_n| > |\sigma_n/\varepsilon_n|$ for all $\rho < 1$. By Fenichel's theorem, a locally invariant slow manifold $\mathcal{M}_n^\varepsilon$ exists within $O(\varepsilon)$ of $\mathcal{M}_n$, smoothly parameterized by $F_n$. On $\mathcal{M}_n^\varepsilon$, the within-level state is determined by $F_n$ up to $O(\varepsilon)$ corrections.

Cite: Fenichel, N. (1979). *Geometric singular perturbation theory for ordinary differential equations.* J. Diff. Eq. 31, 53–98.

**Claim 2 (Directed Coupling).** Extend from power-preserving to passive bidirectional coupling. The current proof handles only the power-preserving (lossless) case. Add:

> Any passive bidirectional coupling contributes a negative-semidefinite term to the Jacobian (by definition of passivity: $\dot{V}_{\text{coupling}} \leq 0$). This strengthens stability, precluding bifurcation. Therefore *any* bidirectional coupling—lossless or lossy—yields an unconditionally stable system. The bifurcation at $\rho(\mathbf{K}) = 1$ requires net energy injection, forcing non-reciprocal (directed) coupling with an external source.

**Claim 3 (Port Gain).** State clearly as a theorem with two parts:
- Port direction $b_n \propto \mathbf{1}$ is **forced** by the CES symmetric equilibrium.
- Port gain function $\phi_n$ is **free** (not determined by $\rho$). The exponents and coefficients of $\phi_n$ are independent degrees of freedom.

Do not score this as a "partial failure." State it as the correct structural result: geometry constrains direction; physics constrains gain.

**Claim 4 (Nearest-Neighbor Topology).** The proof is correct. State the conditional: requires timescale separation. Note that this condition is pre-satisfied by the standing assumption.

### Part 4: The Reduced System on the Slow Manifold

After Parts 2–3 establish the within-level reduction, write the $N$-dimensional reduced system:

$$\varepsilon_n \dot{F}_n = \phi_n(F_{n-1})/J - \sigma_n F_n, \quad n = 1,...,N$$

For the 4-level cyclic system, specialize to the concrete form with Wright's Law, logistic growth, CES capability, and settlement dynamics.

### Part 5: NGM and Spectral Threshold

**Replace the current page-long derivation with this one-paragraph proof:**

> **Proof of the characteristic polynomial.** The matrix $\mathbf{K} - \lambda I$ has the cyclic-plus-diagonal structure: diagonal entries $(d_i - \lambda)$, subdiagonal entries $k_{n+1,n}$, corner entry $k_{14}$, all others zero. In the Leibniz expansion $\det(\mathbf{K}-\lambda I) = \sum_{\pi \in S_4} \text{sgn}(\pi)\prod_i (\mathbf{K}-\lambda I)_{i,\pi(i)}$, only two permutations contribute nonzero products: the identity (contributing $\prod(d_i - \lambda)$) and the 4-cycle $\pi = (1\;2\;3\;4)$ (contributing $\text{sgn}(\pi) \cdot k_{21}k_{32}k_{43}k_{14} = (-1)^3 \cdot P_{\text{cycle}} = -P_{\text{cycle}}$). All other permutations have at least one zero factor. Therefore $p(\lambda) = \prod_{i=1}^4(d_i - \lambda) - P_{\text{cycle}}$. $\blacksquare$

Then state the spectral radius results:
- Equal diagonals: $\rho(\mathbf{K}) = d + P_{\text{cycle}}^{1/4}$
- Zero diagonals: $\rho(\mathbf{K}) = P_{\text{cycle}}^{1/4}$
- General: Perron-Frobenius bounds

State where $K$ enters the NGM: through $k_{32} = \phi_{\text{eff}} \cdot (\partial F_{\text{CES}}/\partial F_2)/|\sigma_2|$, where the CES marginal product at the symmetric allocation is $1/J$.

### Part 6: Hierarchical Ceiling

Derive the slow manifold functions $h_N, h_{N-1}, ..., h_2$ by equilibrating levels fastest-first. State the ceiling inequalities $F_n^* \leq \bar{C}_n(F_{n-1}^*)$ and the economic interpretation (Triffin ceiling for the 4-level application).

### Part 7: Lyapunov Structure and the Eigenstructure Bridge

**This is the key section. Use the port-Hamiltonian framework.**

**Step 1: State the pH structure.**

The hierarchical CES system is a port-Hamiltonian system with dissipation:

$$\dot{x} = [\mathcal{J}(x) - \mathcal{R}(x)]\nabla H(x) + \mathcal{G}(x)u$$

where:
- $H = \Phi = -\sum_n \log F_n$ (Hamiltonian = CES free energy)
- $\mathcal{R} = \text{diag}(\sigma_n I_J)$ (dissipation = damping, positive semidefinite)
- $\mathcal{J}$ encodes the directed coupling (skew part of $Df$; lower-triangular block structure)
- $\mathcal{G}$ encodes the exogenous input $\beta_1$

State explicitly:
- $\mathcal{J}$ is lower-triangular $\Rightarrow$ the system is not a gradient flow ($\mathcal{J} \neq 0$)
- $\mathcal{J}$ has rank-1 cross-level blocks $\Rightarrow$ coupling passes through $F_n$ (Claim 1)
- $\mathcal{J}\nabla H \neq 0$ $\Rightarrow$ the GENERIC degeneracy condition fails (the directed coupling injects free energy from lower to higher levels; this is the mechanism for bifurcation, not a deficiency)

**Step 2: Construct the storage function $V$.**

Use the Li-Shuai-van den Driessche (2010) construction for tree-structured coupling graphs:

$$V = \sum_{n=1}^N c_n \sum_{j=1}^J \left(\frac{x_{nj}}{x_{nj}^*} - 1 - \log\frac{x_{nj}}{x_{nj}^*}\right)$$

with tree coefficients $c_n = P_{\text{cycle}}/k_{n,n-1}$.

Prove $\dot{V} \leq 0$ by showing:
- Within-level terms: $\dot{V}_{\text{within}} = -\sum_n c_n \sigma_n \sum_j (x_{nj} - x_{nj}^*)^2/x_{nj} \leq 0$
- Cross-level terms cancel by the tree condition on $c_n$

**Step 3: Derive the Bridge equation from the pH supply rate.**

This is the crucial tightening move. Do NOT derive the Bridge by separately computing $\nabla^2\Phi$ and $\nabla^2 V$ and then dividing. Instead:

The pH passivity inequality states $\dot{V}_s \leq u^T y$ (supply rate). On the slow manifold (scalar dynamics per level), the supply rate at equilibrium evaluates to:

$$s_n = \frac{P_{\text{cycle}}}{|\sigma_n| \varepsilon_{T_n}}$$

where $\varepsilon_{T_n} = T_n'(\bar{F}_{n-1})\bar{F}_{n-1}/T_n(\bar{F}_{n-1})$ is the elasticity of the coupling at level $n$.

The Bridge matrix $W = \text{diag}(s_1, ..., s_N)$ satisfies:

$$\nabla^2\Phi\big|_{\text{slow}} = W^{-1} \cdot \nabla^2 V$$

**State the special cases:**
- Power-law coupling ($\phi_n(z) = a_n z^{\beta_n}$): $W_{nn} = P_{\text{cycle}}/(\beta_n |\sigma_n|)$
- Linear coupling ($\beta_n = 1$) + uniform damping ($\sigma_n = \sigma$): $W = (P_{\text{cycle}}/\sigma)I$, so $\Phi \propto V$

**State the physical interpretation:** The Bridge matrix encodes how graph topology ($P_{\text{cycle}}$), dissipation ($|\sigma_n|$), and coupling nonlinearity ($\varepsilon_{T_n}$) together deform the natural free-energy geometry into the Lyapunov geometry.

### Part 8: The Triple Role of Curvature

Now prove the three within-level results (superadditivity, correlation robustness, strategic independence) as consequences of the curvature parameter $K$ from Part 1. Follow the proofs in the Triple Role document but:

- Use the notation established in Part 0
- State Part (a) proof in two steps: qualitative (concavity + homogeneity, no curvature needed) then quantitative (curvature comparison gives $\Omega(K)$ gap)
- State Part (b) with the multi-channel effective dimension, making explicit the Cramér-Rao argument
- State Part (c) using the convex game result (Shapley 1971) for the qualitative bound and the constrained Rayleigh quotient for the quantitative bound
- Include the general-weight extension using $K(\rho, \mathbf{a})$ with the secular equation

### Part 9: Moduli Space Theorem (replaces scoring)

State a clean theorem characterizing the degrees of freedom:

> **Theorem (Structural Determination).** Fix the CES parameter $\rho < 1$ and the structural integers $(J, N)$. The hierarchical CES system is determined up to:
>
> **Determined by $\rho$** (qualitative invariants):
> 1. Within-level eigenstructure and curvature $K$
> 2. Coupling topology (aggregate, directed, nearest-neighbor, port-aligned)  
> 3. Existence of a bifurcation threshold at $\rho(\mathbf{K}) = 1$
> 4. Superadditivity, correlation robustness, and strategic independence (the Triple Role)
>
> **Free parameters** (quantitative degrees of freedom):
> 1. Timescales $(\varepsilon_1, ..., \varepsilon_N) \in \mathbb{R}_{++}^N$ with ordering
> 2. Damping rates $(\sigma_1, ..., \sigma_N) \in \mathbb{R}_{++}^N$
> 3. Gain functions $\phi_n: \mathbb{R}_+ \to \mathbb{R}_+$, $C^1$, $\phi_n(0) = 0$, $\phi_n' > 0$
>
> The free parameters determine: the equilibrium cascade $\{F_n^*\}$, the Lyapunov weights $\{c_n\}$, the supply rate matrix $W$, and the convergence rates. The qualitative dynamics are invariant across the moduli space.

### Part 10: Honest Limits

State what the framework cannot do:

1. Determine gain functions from $\rho$ (provably impossible; Claim 3)
2. Eliminate the timescale separation assumption (Fenichel requires it)
3. Make the system a gradient flow ($\mathcal{J} \neq 0$ is a topological obstruction)
4. Prove global stability from $V$ alone (local asymptotic stability is proven; global requires additional boundary analysis)

---

## Style Requirements

- **No scoring.** No "7.5/10", "8/10", "8.5/10". Replace all instances with precise structural statements.
- **No working out loud.** No "Wait, let me reconsider," "Actually," "Hmm." Every derivation should read as a finished proof.
- **No false starts.** If you need to compute something, present the final correct computation, not the journey.
- **Proofs must be complete.** Every step must follow from the previous one. If you cite a result (Fenichel, Perron-Frobenius, Li-Shuai-vdD), state precisely what you are using and what it gives you.
- **LaTeX-ready.** All equations in display math where appropriate. Number all theorems, propositions, lemmas, and corollaries.
- **Cross-references.** When a later result depends on an earlier one, cite it by number (e.g., "by Proposition 3.1").
- **Length.** Aim for 25–35 pages. The current four documents total ~50 pages with significant redundancy. The unified version should be shorter through elimination of overlap, not through loss of content.
- **Dependency diagram.** Include a proof-dependency diagram (as ASCII or described textually) showing which results depend on which.

## Key Mathematical Moves to Get Right

1. **Fenichel's theorem application (Part 3, Claim 1).** You must verify normal hyperbolicity (spectral gap between tangential and transverse eigenvalues) and state the persistence theorem. The spectral gap is $(1-\rho)\sigma_n/\varepsilon_n > 0$ for $\rho < 1$.

2. **Extension of Claim 2 to passive coupling.** Any passive system satisfies $\dot{V}_{\text{coupling}} \leq 0$, which adds a negative-semidefinite term to the effective Jacobian. State this as a one-line argument after the power-preserving case.

3. **One-paragraph NGM proof via permutation expansion.** The key insight: in the Leibniz formula for $\det(\mathbf{K} - \lambda I)$, only the identity permutation and the full $N$-cycle have all nonzero factors. All partial permutations (transpositions, 3-cycles, etc.) hit a zero entry.

4. **Bridge matrix from pH supply rate, not from Hessian comparison.** The passivity inequality $\dot{V} \leq s^T y$ at the equilibrium gives $s_n$ directly. The Bridge equation then follows from $\nabla^2 H = W^{-1}\nabla^2 V_s$ being a standard identity in pH theory relating the Hamiltonian to the storage function through the supply rate.

5. **The $(2-\rho)$ vs $(1-\rho)$ distinction.** State once, clearly, at the start of Part 2: $(2-\rho)$ is the dynamical anisotropy (Jacobian $Df$), $(1-\rho)$ is the geometric anisotropy (Hessian $\nabla^2\Phi$). They differ by 1, which is the damping contribution. Both are correct for their respective objects.

## References to Cite

- Fenichel (1979) — geometric singular perturbation theory
- Li, Shuai, van den Driessche (2010) — graph-theoretic Lyapunov functions
- Diekmann, Heesterbeek, Metz (1990) — next-generation matrix method
- van der Schaft & Jeltsema (2014) — port-Hamiltonian systems
- Shapley (1971) — cores of convex games
- Smith (1995), Hirsch (1985) — monotone dynamical systems
- Shahshahani (1979), Akin (1979) — information geometry on simplices
- do Carmo — Riemannian geometry (curvature comparison)
