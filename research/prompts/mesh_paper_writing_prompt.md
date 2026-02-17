# PROMPT: WRITE THE MESH PAPER

## Attached files:
## 1. Endogenous_Decentralization_v7_3.tex (the prior paper — this new paper begins where it ends)
## 2. This prompt

---

## WHAT THIS IS

This is a **new economics paper**, not a revision. The attached Endogenous Decentralization paper models how concentrated AI infrastructure investment finances the learning curves that enable distributed inference. It ends at the crossing point — the moment x(t) = 0 and distributed architecture becomes cost-competitive. 

This paper answers: **what happens after the crossing?**

The answer is not billions of isolated devices running local inference. It is a self-organizing mesh of heterogeneous specialized agents whose collective capability exceeds centralized provision once the mesh reaches critical mass. The paper formalizes this system, proves the existence and stability of the mesh equilibrium, characterizes the phase transition from centralized to distributed, and derives the conditions under which the mesh dominates.

The paper should be written as a formal economics working paper in LaTeX, at the level of a submission to a top-25 economics journal. Same formatting conventions as the attached paper. Connor Smirl, Department of Economics, Tufts University. Working paper, February 2026.

---

## THE MATHEMATICAL RAW MATERIAL

Prior field searches across statistical mechanics, theoretical biology, network science, computer science, ecology, and economics produced three core results and one unification. Build the paper's formal model from these.

### Core Result 1: Inverse Bose-Einstein Condensation (from Bianconi-Barabási 2001)

Networks with heterogeneous node fitness η_i drawn from distribution ρ(η) exhibit two phases:

- **Winner-Takes-All (BEC condensation):** When ρ(η) is sharply peaked near η_max, the single fittest node captures a macroscopic fraction of all connections. This is the centralized equilibrium — AWS/Google/Azure dominate.
- **Fit-Get-Rich (distributed):** When ρ(η) is broad, many nodes share traffic proportional to fitness. No single node dominates. This is the mesh equilibrium.

The phase boundary depends on ρ(η) ~ (η_max − η)^α near the maximum:
- α ≤ 0: BEC (centralization)  
- α > 0: FGR (distribution)

**The ED paper's learning curve broadens the fitness distribution.** Initially, only datacenter-scale nodes have high fitness → sharply peaked → BEC. As edge device capability improves through the packaging learning curve, the distribution broadens → system crosses the BEC phase boundary → centralized condensate dissolves → traffic distributes across the mesh.

This is **inverse Bose-Einstein condensation** driven exogenously by the technology parameter θ from the ED paper. The crossing point x(t) = 0 in the ED paper corresponds to the system reaching the BEC phase boundary.

The self-consistency equation for the degree distribution:
k_i(t) ~ (t/t_i)^{η_i/C}
where C satisfies: ∫ ρ(η) / (C/η − 1) dη = 1

This has the identical mathematical structure to the Bose gas number equation.

### Core Result 2: The Fortuin-Kasteleyn Unification

The q-state Potts model partition function has an exact cluster representation:

Z_Potts = Σ_{A⊆E} p^|A| (1−p)^{|E|−|A|} q^{c(A)}

At q = 1, this IS bond percolation. Percolation (does the mesh physically connect?) and specialization (does division of labor crystallize?) are **the same mathematical object** at different values of q. They are not separate frameworks to bolt together.

Critical result: For q > 2 specializations, the phase transition is **first-order** (discontinuous). The mesh does not form gradually — it crystallizes abruptly once conditions are met. This is a testable prediction.

### Core Result 3: Vanishing Epidemic Threshold on Scale-Free Networks

Pastor-Satorras and Vespignani (2001): SIS model on networks with degree distribution P(k):

Epidemic threshold: λ_c = ⟨k⟩ / ⟨k²⟩

For scale-free networks with degree exponent γ ≤ 3: ⟨k²⟩ diverges → λ_c → 0.

**Any nonzero rate of knowledge sharing sustains itself indefinitely on a heterogeneous network.** The topology does the work. Once the mesh has a fat-tailed degree distribution (which MoE routing produces endogenously through preferential specialization), capability propagation is guaranteed. You do not need to model knowledge diffusion as a separate mechanism.

### Unification: All Fields Produce the Same Equation

The self-consistency equations across all six fields reduce to the same form:

m = f(m; λ) with transcritical bifurcation at the critical parameter value.

| Field | m | λ | Critical condition |
|-------|---|---|-------------------|
| Percolation | Giant component S∞ | ⟨k⟩ | ⟨k⟩ = 1 |
| Epidemiology | Infected fraction | R₀ | R₀ = 1 |
| Ising | Magnetization | βJq | βJq = 1 |
| Ecology | Productivity | Species richness S | Minimum S for resilience |
| Economics | Market participation | Transaction benefit | Critical liquidity |

The R₀ from the ED paper IS the percolation threshold IS the Ising critical temperature. One equation, six notations. The crossing point x(t) = 0 is the moment R₀^mesh crosses 1.

---

## THE THREE-LAYER MODEL STRUCTURE

Build the formal model as three composable layers:

### Layer 1: Phase Transition and Critical Mass

The R₀ framework from the ED paper provides the boundary condition. Define:

R₀^mesh = N · β · v / D > 1

where N = active nodes, β = connection probability, v = value per interaction, D = attrition rate. The giant component fraction:

S∞ = 1 − exp(−R₀^mesh · S∞)

Existence: positive solution iff R₀ > 1. Uniqueness: one positive fixed point by concavity. Stability: locally asymptotically stable.

### Layer 2: Heterogeneous Specialization

Each agent i has capability vector. The effective capability of the mesh is the CES aggregate:

C_eff(N) = (Σ_i c_i^ρ)^{1/ρ}, 0 < ρ < 1

This rewards diversity because ρ < 1 (imperfect substitutability). Two agents with different specializations contribute more than two identical agents.

Specialization dynamics follow the Bonabeau-Theraulaz Fixed Response Threshold model:
- Agent i has threshold θ_{ij} for task j
- Response probability: P_{ij}(s_j) = s_j^n / (s_j^n + θ_{ij}^n)
- Thresholds adapt: θ̇_{ij} = −ξ · 𝟙[task performed] + φ · 𝟙[idle]

Agents self-sort into specializations without central coordination. This is the Becker-Murphy division of labor emerging from local interactions.

### Layer 3: Knowledge Diffusion

Information propagation on the network: ∂u/∂t = −L·u where L is the graph Laplacian. Convergence rate = λ₂(L), the Fiedler eigenvalue.

Total mesh bandwidth scales as O(N·⟨k⟩) while the centralized hub has fixed B_hub. Once N·⟨k⟩ > B_hub, the mesh serves more total queries per unit time.

Combined with the vanishing epidemic threshold: on scale-free networks, knowledge propagation is self-sustaining at any nonzero transmission rate.

---

## THE CENTRAL THEOREM

Prove the following: **For R₀^mesh > 1 and ρ < 1, there exists a finite N* such that for all N > N*, the mesh equilibrium exists, is unique, is locally asymptotically stable, and C_mesh(N) > C_centralized. N* is decreasing in the diversity of the agent population.**

The proof should use:
- Percolation theory for existence of the connected mesh
- CES properties for the capability comparison
- Supermodular game theory for equilibrium existence (Tarski's fixed point theorem) and stability
- The Bianconi-Barabási framework for the competitive dynamics showing centralized market share declining

---

## THE SETTLEMENT LAYER — LET IT EMERGE

The model requires agents to route queries to specialists and compensate them. This creates a need for:
1. Routing → requires incentives → requires compensation
2. Compensation → requires settlement between arbitrary pairs of agents
3. Settlement at the required volume (millions of micro-transactions/second) and latency (milliseconds) → requires programmable money

**Do not assume the settlement layer.** Derive the need for it from the routing incentive structure. Then note that this connects to a separate paper on programmable monetary infrastructure (The Monetary Productivity Gap) — but do not develop the monetary model here. Just flag the connection.

The Hayek (1945) insight should appear here: the price system in the mesh IS the decentralized information aggregator. Prices encode current demand for each query type, current supply of each specialization, and optimal routing — all in bid-ask spreads. No central coordinator needed.

---

## POST-CROSSING PHASES

Structure the paper's dynamics as three phases after x(t) = 0:

**Phase 1 — Nucleation (R₀ ≈ 1):** Slow stochastic growth. Small specialist clusters. Fragile — exogenous shocks can collapse below threshold. The mesh first wins on the long tail of niche queries that centralized systems underserve.

**Phase 2 — Rapid Growth (R₀ >> 1):** Network effects dominate. C_eff grows superlinearly as new diverse specialists join. First-order Potts crystallization of division of labor. Centralized providers lose market share in progressively more mainstream query types.

**Phase 3 — Maturity:** Growth saturates as capability space is covered. Competition shifts to mesh composition and quality. Centralized providers retain advantage only for frontier-model training and queries requiring capabilities beyond any edge device (this connects back to the ED paper's training-inference bifurcation).

---

## FRAMEWORKS CONSIDERED AND REJECTED

Include a brief discussion of why the following were considered and rejected — this demonstrates rigor:

- **Mean Field Games (Lasry-Lions):** Assumes exchangeable (identical) agents. Your agents are heterogeneous specialists — that's the whole point. The supermodular game framework handles heterogeneity more naturally.
- **Spin glasses:** Require frustrated interactions (mix of positive and negative couplings). The mesh has all positive couplings — everyone benefits from others joining. No frustration. This is a random-field Ising model, not a spin glass.
- **Ecological niche models (Tilman, Loreau-Hector):** Conceptually identical — diverse specialists outperform monocultures. But the formal models are calibrated to plant biomass with resource-competition dynamics that don't transfer cleanly. Use the CES function instead, which appears identically in ecology and economics.
- **Immune system (clonal selection):** Vivid analogy but the formal ODE models of lymphocyte proliferation don't map to inference economics. Stays a metaphor.

---

## RG UNIVERSALITY — METHODOLOGICAL LICENSE

The Renormalization Group result: for real-world networks with spectral dimension d_s > 4, mean-field theory is exact. This is not an approximation — it is a rigorous result. This justifies using the supermodular game framework and Katz-Shapiro network goods model without apology. Include this as a remark after the main model to justify the modeling choices.

---

## FALSIFIABLE PREDICTIONS

Generate predictions with timing and failure conditions, in the same style as the ED paper's Section 7. These should include at minimum:

1. The mesh formation is abrupt (first-order), not gradual — testable from adoption data
2. Specialization precedes generalization — early mesh agents are narrow specialists
3. The mesh first wins on long-tail niche queries before competing on mainstream tasks
4. Hub agents emerge endogenously and handle disproportionate routing traffic
5. Knowledge propagation accelerates nonlinearly once the degree distribution becomes fat-tailed
6. A programmable settlement layer emerges as a binding constraint on mesh growth before other constraints bind

---

## WHAT NOT TO DO

- Do not repeat the ED paper's model. Reference it and use its terminal conditions.
- Do not develop the monetary settlement model. Flag the connection, cite the MPG paper as forthcoming, move on.
- Do not homogenize agents. The heterogeneity IS the contribution. An Ising model alone misses dynamics 3 and 4.
- Do not force the BEC framework if it doesn't compose cleanly with the other layers. Flag any incompatibilities honestly.
- Do not write a survey of cross-disciplinary analogies. This is an economics paper that uses physics and biology where the math demands it, not a review article.

---

## NOTATION COMPATIBILITY

Use notation that is compatible with the ED paper:
- θ for the technology parameter (learning curve output)
- x(t) for the ED paper's state variable (reference only — this paper's dynamics begin at x = 0)
- R₀ for the critical mass threshold (already used in the ED paper's Section 3.9)
- N for number of agents/firms
- α for learning elasticity when referencing the ED paper
- ρ for the CES substitution parameter (new to this paper — do not confuse with the ED paper's use of any similar symbol)

---

## TITLE

Working title: something that captures "self-organizing mesh of specialized agents exceeds centralized capability after the crossing point." Suggest a title and subtitle in the same style as "Endogenous Decentralization: How Concentrated Capital Investment Finances the Learning Curves That Enable Distributed Alternatives."
