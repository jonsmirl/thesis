# FIELD SEARCH: PAPER 4 — THE AUTOCATALYTIC MESH

## Cross-Disciplinary Framework Inventory for Endogenous Capability Growth in a Self-Organizing Agent Mesh

**Prepared for:** Connor Smirl, Tufts University  
**Date:** February 2026  
**Status:** Search output — not the paper

---

## CONTEXT

The mesh paper (Smirl 2026, "The Mesh Equilibrium") proves that a self-organizing mesh of heterogeneous specialized AI agents exceeds centralized provision above a finite critical mass N*. It assumes **fixed-capability nodes**: each agent i has a capability vector **c**_i that redistributes through specialization but does not grow. Knowledge diffusion (∂**u**/∂t = −L**u**) equalizes existing knowledge; it does not create new capability.

Paper 4 removes that assumption. Three mechanisms make capability endogenous:

1. **Training agents improve other agents** (autocatalytic capability growth)
2. **Operation generates training data** (self-referential learning)
3. **The mesh modifies its own composition** (endogenous variety expansion)

The central question: when capability c_i becomes a dynamical variable coupled to the mesh's operation, does C_eff(t) converge to a ceiling, grow polynomially, or grow exponentially/superexponentially?

---

## FRAMEWORK 1: Eigen's Hypercycle

**Source:** Eigen (1971); Eigen & Schuster (1977-1979), *The Hypercycle*, Springer.

**The Math:** For n species with concentrations x_i, the elementary hypercycle ODE system is:

dx_i/dt = k_i · x_i · x_{i-1} − x_i · Φ(t),  i = 1,...,n (indices mod n)

where Φ(t) = Σ_j k_j · x_j · x_{j-1} is the dilution flux (constant total concentration constraint). Each species catalyzes the replication of the next in a cycle. The system is a special case of the replicator equation.

The quasispecies equation for mutation-selection dynamics:

dx_i/dt = Σ_j W_{ij} · x_j · f_j − x_i · Φ

where W_{ij} is the mutation matrix and f_j is fitness.

**Captures:** Dynamic A (autocatalysis with heterogeneous catalysts). The cyclic catalytic structure directly maps to training agents improving specialists who generate data that improves training.

**Ceiling Prediction:** **Convergence.** Under the constant-organization constraint (Σx_i = const), hypercycles reach an internal equilibrium or oscillate with bounded total concentration. The system does not exhibit unbounded growth — it redistributes a fixed total. However, for n ≥ 5 members, hypercycles become unstable to spatial perturbations (limit cycles, chaos).

**Variable Mapping:**
- Species x_i → Agent type populations in the mesh
- Catalytic rate k_i → Training effectiveness of agent i on agent i+1  
- Φ(t) → Mesh capacity constraint (total compute budget)
- Cyclic coupling → Training agent T improves specialist S, S generates data for T

**Breaks Down When:** The constant-organization constraint (Σx_i = const) forces zero-sum dynamics — one species grows only at another's expense. The mesh doesn't have this: adding a new agent increases total capability, not just redistributes it. Also, hypercycles assume cyclic coupling (each species catalyzes exactly one other); the mesh has arbitrary network topology. Finally, hypercycles are vulnerable to parasites (freeloading species that receive but don't give catalysis) — the mesh has price-based selection that addresses this.

**Assessment: 5/10.** The autocatalytic structure is exactly right, but the zero-sum constraint makes the dynamics fundamentally different from the mesh's growth dynamics. The vulnerability to parasites (and the spatial solutions developed by Boerlijst & Hogeweg) is relevant to mesh robustness analysis. The instability for n ≥ 5 is a cautionary result. Best used as a structural template, not a dynamical one.

**Math Prerequisites:** ODEs, phase plane analysis (Connor has this).

---

## FRAMEWORK 2: Kauffman's Autocatalytic Sets / RAF Theory

**Source:** Kauffman (1971, 1986, 1993); Steel (2000); Hordijk & Steel (2004-2017).

**The Math:** A chemical reaction system (CRS) is a tuple Q = {X, R, C} with molecule types X, reactions R, and catalysis assignments C. A RAF (Reflexively Autocatalytic and Food-generated) set is a subset R' ⊆ R satisfying:

1. **Reflexively Autocatalytic (RA):** Every reaction in R' is catalyzed by at least one molecule produced by R' or present in the food set F.
2. **Food-generated (F):** Every reactant of every reaction in R' can be constructed from the food set F by successive application of reactions in R'.

The key quantitative result: in Kauffman's binary polymer model with catalysis probability p per molecule-reaction pair, a RAF set emerges with probability approaching 1 when:

p · |R| ≥ O(log |R|)

i.e., the critical catalysis threshold grows only **logarithmically** with system size, not linearly. This is a percolation-like phase transition.

**Captures:** Dynamic A (autocatalysis), Dynamic D (emergence of self-sustaining sets). The RAF framework captures the moment when the mesh becomes self-sustaining: every agent type is either "food" (base models from centralized training) or is produced/improved by other agents in the set.

**Ceiling Prediction:** **Phase transition from non-self-sustaining to self-sustaining, then depends on further dynamics.** RAF theory tells you *whether* an autocatalytic set exists (binary), not *how fast it grows*. The existence threshold is sharp (phase transition). Growth dynamics require coupling RAF structure to population dynamics (see Framework 3).

**Variable Mapping:**
- Molecule types X → Agent specialization types in the mesh
- Reactions R → Training/improvement operations
- Catalysis C → Which agents can improve which other agents
- Food set F → Base models from centralized training (the mesh paper's "training persistence" assumption)
- RAF existence threshold → The mesh's critical mass N* for self-sustaining capability growth

**Breaks Down When:** RAF theory is fundamentally about existence (graph-theoretic closure), not dynamics (how fast concentrations change). It tells you *whether* the autocatalytic set forms, not whether the system converges, grows polynomially, or explodes. The dynamics require a separate model layered on top.

**Assessment: 7/10.** Extremely useful for establishing the *existence* of the autocatalytic mesh — that there is a finite threshold above which the mesh can sustain and grow its own capabilities without centralized training. The logarithmic scaling of the threshold (more agent types make it *easier*, not harder, to form an autocatalytic set) is a powerful result that directly parallels the mesh paper's N* decreasing in diversity. But it doesn't answer the ceiling question directly.

**Math Prerequisites:** Graph theory, combinatorics (Connor has this).

---

## FRAMEWORK 3: Jain-Krishna Adaptive Network Model

**Source:** Jain & Krishna (1998, 2001), Phys. Rev. Lett. 81, 5684; PNAS 98(2), 543-547.

**The Math:** A directed weighted graph G(t) with s nodes (species). Node populations x_i evolve by:

dx_i/dt = Σ_j c_{ij} · x_j · x_i − x_i · Φ(t)

where c_{ij} is the catalytic interaction from j to i, and Φ maintains constant total population. The key dynamical variable is the Perron-Frobenius eigenvalue of the catalytic matrix: species with the highest eigenvector centrality grow fastest; those with the lowest go extinct and are replaced by new random species.

The graph dynamics: the least-fit species (lowest asymptotic population) is removed and replaced by a new species with random catalytic connections. Starting from sparse random graphs:
1. An autocatalytic set (ACS) appears by chance.
2. The ACS triggers a **cascade of exponentially increasing connectivity**.
3. Eventually the entire network becomes one large ACS (percolation).
4. Catastrophes occur when "keystone" species are randomly lost, followed by recovery.

**Captures:** Dynamics A + D simultaneously. This is the closest existing model to the mesh: a network of catalytically interacting species that self-organizes, grows in complexity, experiences catastrophes, and recovers. The percolation of the ACS through the network is analogous to the mesh's crystallization phase.

**Ceiling Prediction:** **Convergence to a statistical steady state with punctuated equilibria.** The connectivity saturates once the entire network is one ACS. The system then alternates between stasis (all species in the ACS) and catastrophes (keystone loss → cascade → recovery). Total "capability" (average connectivity) reaches a ceiling that fluctuates.

**Variable Mapping:**
- Node i → Agent i in the mesh
- c_{ij} → The degree to which agent j's output improves agent i  
- Population x_i → Traffic/usage share of agent i (Bianconi-Barabási fitness)
- Perron-Frobenius eigenvector → Optimal traffic allocation in the mesh
- Species removal → Low-quality agents losing traffic and exiting
- ACS percolation → Mesh crystallization (Prediction 1 in mesh paper)
- Keystone catastrophes → Hub agent failure cascades

**Breaks Down When:** The constant-population constraint again forces zero-sum dynamics. The replacement mechanism (random new species) doesn't capture directed creation of new agent types to fill identified gaps (the mesh's price-signal mechanism). The model has no notion of "improving" species — only replacing extinct ones with random new ones.

**Assessment: 8/10.** The highest-rated framework for structural analogy. The spontaneous emergence of autocatalytic sets, the exponential cascade to percolation, the punctuated equilibrium dynamics, and the keystone species vulnerability all map beautifully to mesh dynamics. The key limitation (convergence to a ceiling) may actually be the correct prediction, with the ceiling governed by network size and catalytic density. The math is tractable ODEs + graph dynamics — within Connor's toolkit.

**Math Prerequisites:** Linear algebra (Perron-Frobenius), ODEs (Connor has both).

---

## FRAMEWORK 4: Romer (1990) Endogenous Growth with Variety Expansion

**Source:** Romer, P. (1990), "Endogenous Technological Change," *Journal of Political Economy*, 98(5), S71-S102.

**The Math:** Three sectors: final goods, intermediate goods, research.

Final good: Y = H_Y^α · L^β · Σ_{i=1}^{A} x_i^{1-α-β}

where A is the number of intermediate good varieties (ideas), x_i is the quantity of each, H_Y is human capital in production, L is labor.

Idea production: dA/dt = δ · H_A · A

where H_A is human capital allocated to research, and the crucial feature is **linearity in A**: the existing stock of ideas is a direct input to producing new ideas (standing on shoulders).

On the balanced growth path: g_Y = g_A = δ · H_A.

The growth rate is **endogenous** — determined by the allocation of human capital to research — and **constant** (exponential growth, not accelerating).

**Captures:** Dynamic B (endogenous innovation/variety expansion). The CES aggregation in the mesh paper is directly Romer's variety model: each new specialization type j is a new "variety" whose contribution to C_eff is governed by imperfect substitutability (ρ < 1). The mesh paper's J task types are Romer's A varieties.

**Ceiling Prediction:** **Exponential growth** (constant rate). The linearity of the idea production function in A ensures that the growth rate is constant, not diminishing. But this is a knife-edge result: if dA/dt = δ · H_A · A^φ with φ < 1, growth decelerates to zero (Jones critique). If φ > 1, growth explodes (finite-time singularity).

**Variable Mapping:**
- A (number of varieties) → J (number of specialization types in the mesh)
- x_i (quantity of variety i) → C_j (total capability in task type j)
- CES/Dixit-Stiglitz aggregator → CES aggregator in mesh paper (equation 3)
- H_A (research human capital) → Training agents' capability devoted to creating new specialists
- δ · A (idea productivity) → Training productivity scaling with existing mesh knowledge
- Non-rivalry of ideas → Knowledge diffusion on the mesh (Layer 3)

**Breaks Down When:** Romer assumes the number of researchers H_A is exogenous (drawn from a fixed population). In the mesh, the "researchers" (training agents) are themselves products of the mesh. This makes the model self-referential in a way Romer's is not. Also, Romer's balanced growth requires φ = 1 exactly — the Jones critique shows this is a knife-edge condition.

**Assessment: 9/10.** This is the natural economics framework for the mesh's variety expansion. The CES structure is already in the mesh paper. The key contribution of Paper 4 is making A (the number of specialization types) endogenous within the mesh rather than driven by an exogenous research sector. The central question — what determines φ? — is exactly the paper's ceiling question.

**Math Prerequisites:** Optimization, ODEs (Connor has this; Romer's math is within reach).

---

## FRAMEWORK 5: Jones (1995, 2005) Semi-Endogenous Growth

**Source:** Jones, C. I. (1995), "R&D-Based Models of Economic Growth," *JPE* 103(4); Jones (2005), "Growth and Ideas," *Handbook of Economic Growth*.

**The Math:** Modified idea production function:

dA/dt = δ · L_A^λ · A^φ

where λ ∈ (0,1] allows for duplication effects among researchers and φ captures the "standing on shoulders" vs. "fishing out" tradeoff.

Three regimes:
- **φ = 1** (Romer): Constant exponential growth. g_A = δ · L_A. Growth rate depends on *level* of research effort → "strong scale effect."
- **φ < 1** (Jones): "Ideas are getting harder to find." Balanced growth rate: g_A = λn/(1−φ), where n is population growth rate. Without population growth, innovation decelerates to zero.
- **φ > 1**: Superexponential growth → finite-time singularity.

The empirical evidence (Bloom et al. 2020) strongly supports φ < 1: research productivity (TFP growth per researcher) has declined steadily even as research effort has grown enormously.

**Captures:** Dynamic E (the ceiling question directly). Jones's φ parameter is precisely the parameter that governs whether the mesh's capability growth converges, grows exponentially, or explodes. For the mesh: does the marginal productivity of training (in terms of capability improvement per unit of training effort) scale linearly with existing capability (φ = 1), sublinearly (φ < 1), or superlinearly (φ > 1)?

**Ceiling Prediction:** **Depends on φ, and empirical evidence favors φ < 1 (convergence/polynomial).** This is the most important result in the search: the Jones framework predicts that without a growing population of training agents, the mesh's capability growth decelerates. However, the mesh has an endogenous mechanism absent from Jones: the mesh can create new training agents, potentially making L_A endogenous and growing.

**Variable Mapping:**
- A → C_eff (aggregate mesh capability)
- L_A → Number/capability of training agents in the mesh  
- φ → Elasticity of training productivity w.r.t. existing mesh capability
- λ → Diminishing returns to adding more training agents
- n (population growth) → Rate at which new devices join the mesh

**Breaks Down When:** Jones assumes L_A is drawn from an exogenous, growing population. The mesh's "researchers" are endogenous. If the mesh can automate its own training (training agents that improve other training agents), L_A becomes a function of A, potentially restoring φ_effective ≥ 1 even if the "raw" φ < 1. This is exactly the Aghion-Jones-Jones (2018) AI automation argument.

**Assessment: 10/10.** This is the central framework. Paper 4's primary contribution can be stated in Jones's language: what is the effective φ for the autocatalytic mesh, and does the endogenous creation of training agents (automating research) push φ_effective toward or past unity? The Jones framework provides the mathematical structure; the mesh dynamics provide the content.

**Math Prerequisites:** ODEs, basic optimization (Connor has this).

---

## FRAMEWORK 6: Aghion, Jones & Jones (2018) — AI and Economic Growth

**Source:** Aghion, P., Jones, B., & Jones, C. (2018), "Artificial Intelligence and Economic Growth," NBER WP 23928.

**The Math:** Task-based production with automation fraction β:

Y = (∫_0^β A_s · K_s^α ds) · (∫_β^1 A_s · L_s^α ds)^{(1-α)/α}

where β is the fraction of tasks automated by AI. The idea production function incorporates AI automation:

dA/dt = A^φ · [L_A/(1−β)]^λ

Key result: as β → 1 (full automation), the effective number of researchers L_A/(1−β) → ∞ even with fixed human researchers. This can generate a singularity (γ > 1, where γ combines the automation and idea production parameters). Specifically:

If σ > 1/(2√(αβφ)), where σ is the elasticity of substitution between automated and non-automated tasks, explosive growth occurs.

But Baumol's cost disease acts as a brake: as AI automates easy tasks, the remaining hard tasks (requiring human input) become the bottleneck, and their cost share rises toward 100%.

**Captures:** Dynamics A + B + E simultaneously. This is the formal version of "AI improves AI" in an economic growth framework.

**Ceiling Prediction:** **Depends on the elasticity of substitution σ and whether Baumol's disease binds.** Three scenarios:
1. σ < 1: Baumol's disease dominates → convergence. AI accelerates growth temporarily but the non-automated bottleneck eventually stalls it.
2. σ = 1 (Cobb-Douglas): Balanced exponential growth.
3. σ > 1: Growth explosion / singularity possible.

For the mesh: if there are tasks that fundamentally cannot be automated by the mesh (e.g., frontier model training requires centralized infrastructure), these become the binding constraint and growth converges. The mesh paper already identifies this: "training persistence" is the Baumol bottleneck.

**Variable Mapping:**
- Tasks [0,1] → Query types across the mesh's capability space
- β → Fraction of inference tasks served by the mesh (vs. centralized)
- A_s → Capability of the mesh for task s
- σ → CES parameter ρ from the mesh paper (σ = 1/(1−ρ))
- L_A/(1−β) → Training agents' effective capability (growing as mesh automates more)
- Baumol bottleneck → Centralized training persistence

**Breaks Down When:** The model assumes a continuum of tasks with exogenous automation probability. The mesh has endogenous task creation (Dynamic D) — the mesh doesn't just automate existing tasks, it creates new task categories. This can relax the Baumol constraint by expanding the set of tasks faster than automation proceeds.

**Assessment: 9/10.** The most directly applicable economics framework. Paper 4 should be written as an extension of this model, with the mesh providing the microstructure that Aghion-Jones-Jones leave abstract. The σ > 1 condition for explosive growth maps directly to the mesh paper's ρ < 1 condition (complementarity). The training persistence bottleneck is already identified in the mesh paper.

**Math Prerequisites:** This is advanced — requires dynamic optimization, continuum of goods. But the key results can be stated in the simpler Jones (2005) notation.

---

## FRAMEWORK 7: Weitzman (1998) Recombinant Growth

**Source:** Weitzman, M. L. (1998), "Recombinant Growth," *QJE* 113(2), 331-360.

**The Math:** New ideas are produced by combining existing ideas:

ΔA(t) = min{p̄ · [C₂(A(t)) − C₂(A(t−1))], s_b · A(t)}

where C₂(A) = A(A−1)/2 is the number of pairwise combinations of A ideas, p̄ is the probability that a random combination yields a useful idea, and s_b · A(t) is the R&D budget constraint.

Key results:
1. **The combinatorial process eventually outgrows any exponential.** C₂(A) ~ A² grows faster than e^{rt} for any fixed r, once A is large enough.
2. **But growth is constrained by R&D resources.** In steady state, the R&D constraint binds, and growth converges to a constant exponential rate proportional to (s₁, s₂) — the savings rates for physical capital and R&D.
3. **The ultimate limiting cost c*** of R&D determines the growth regime. If c* = 0 (ideas are free to develop), growth is superexponential. If c* > 0 (ideas cost resources), growth converges to exponential.

The long-run growth rate: λ* = F(s₁, s₂; c*) where F is linearly homogeneous in the savings rates.

**Captures:** Dynamic C (self-referential learning — output feeds input). The mesh's training data generation is exactly recombinant: each query-response pair can be combined with other data to produce new training signal. The combinatorial explosion in potential training data parallels Weitzman's combinatorial explosion in potential ideas.

**Ceiling Prediction:** **Exponential growth (not superexponential), constrained by the R&D cost c*.** Even though the *potential* for growth is superexponential (combinatorial), the *actual* growth is bounded by resources devoted to processing the combinatorial explosion. The ultimate limit is not running out of ideas but running out of capacity to evaluate them.

For the mesh: even though training data grows combinatorially with mesh size, the actual capability improvement is limited by training compute. The training agents are the "R&D sector" that processes the combinatorial data explosion.

**Variable Mapping:**
- A (number of ideas) → J (number of effective specialization types) or total knowledge base
- C₂(A) → Number of potential training data combinations
- p̄ → Probability that a random data combination yields useful training signal
- s_b · A → Training compute budget
- c* → Cost of converting potential training data into actual capability improvement

**Breaks Down When:** Weitzman assumes a fixed population allocating resources between consumption and R&D. The mesh has an endogenous and growing "population" of training agents. If the training agents themselves can be improved (autocatalysis), the R&D cost c* may not be constant — it could decline over time, potentially unlocking the superexponential regime.

**Assessment: 8/10.** Excellent for the data layer (Mechanism 2). The key insight — that combinatorial potential doesn't automatically translate to combinatorial growth because of processing constraints — is exactly the right way to think about the mesh's training data advantage. The c* parameter is the bridge to the ceiling question: if autocatalytic training reduces c*, the system transitions from exponential to potentially superexponential.

**Math Prerequisites:** Difference equations, optimization (Connor has this).

---

## FRAMEWORK 8: Chemical Reaction Network Theory (Feinberg)

**Source:** Feinberg, M. (1987, 2019); Horn & Jackson (1972). Feinberg, *Foundations of Chemical Reaction Network Theory*, Springer 2019.

**The Math:** A CRN is specified by a set of species S, complexes C (linear combinations of species), and reactions (directed edges between complexes). Under mass action kinetics, the concentration vector c(t) evolves by:

dc/dt = Y · A_k · Ψ(c)

where Y is the stoichiometric matrix, A_k is the weighted adjacency matrix of the reaction graph, and Ψ(c) is the mass-action vector.

The **deficiency** δ = n − l − s (n = number of complexes, l = linkage classes, s = dimension of stoichiometric subspace) is a purely structural invariant.

**Deficiency Zero Theorem:** If the CRN is weakly reversible and δ = 0, then for any rate constants: (a) there exists exactly one positive equilibrium in each stoichiometric compatibility class, and (b) it is locally asymptotically stable.

**Captures:** Dynamic A (autocatalysis on a reaction network). CRNT provides the mathematical language for asking: given the *structure* of catalytic interactions in the mesh, what can we say about equilibrium existence, uniqueness, and stability — independent of the specific rate constants?

**Ceiling Prediction:** **Convergence to a unique stable equilibrium** (for deficiency-zero, weakly reversible networks). The deficiency zero theorem is a powerful convergence result: regardless of rate constants, the system has a unique stable steady state. This would predict convergence/ceiling for the mesh if the mesh's interaction structure has deficiency zero.

**Variable Mapping:**
- Species → Agent types in the mesh
- Complexes → Combinations of agents involved in training interactions
- Reactions → Training operations (input agents → improved output agents)
- Mass action kinetics → Training rate proportional to product of interacting agents' capabilities
- Stoichiometric compatibility class → Conservation laws in the mesh (total compute budget)
- Deficiency → Structural complexity of the mesh's training network

**Breaks Down When:** CRNT assumes a closed system with conservation laws (stoichiometric compatibility classes). The mesh is open: new agents enter, old ones leave, and the total "mass" (capability) is not conserved. Also, CRNT assumes mass action kinetics (reaction rate proportional to product of reactant concentrations), which may not describe training dynamics accurately. Most importantly, CRNT analyzes equilibrium existence, not growth — it tells you the system converges, but the equilibrium concentration depends on rate constants and initial conditions.

**Assessment: 6/10.** The structural approach (relating network topology to dynamical properties independently of parameters) is methodologically perfect for the mesh, where we don't know the specific training rates. The deficiency zero theorem's power is that it's parameter-free. But the closed-system assumption is a serious limitation. Best used for establishing *equilibrium existence and uniqueness* rather than *growth trajectories*.

**Math Prerequisites:** Linear algebra, graph theory (Connor has this). The proofs use Lyapunov functions — a good stretch.

---

## FRAMEWORK 9: Nordhaus (2021) / Aghion-Jones-Jones — Economic Singularity Conditions

**Source:** Nordhaus, W. D. (2021), "Are We Approaching an Economic Singularity?", *AEJ: Macro* 13(1); Aghion, Jones & Jones (2018).

**The Math:** Nordhaus's singularity condition: if the idea production function is dA/dt = φ(λY)^β / A, and output Y = A · L, then:

g_A = Ȧ/A = φ(λL)^β · A^{β−1}

For β ≥ 1: growth rate increases with A → singularity (growth accelerates without bound, reaching infinite output in finite time for β > 1).

For β < 1: growth rate decreases with A → convergence to a steady state.

Aghion-Jones-Jones define a composite parameter γ that captures the combined effect of automation and idea production:

γ = σφ / [(1−α)(1−β)]

If γ > 1: Type II singularity (infinite output in finite time).  
If γ = 1: Exponential growth.  
If γ < 1: Growth decelerates.

The key question is whether σ (the elasticity of substitution between AI and human inputs) is high enough to overcome the Baumol drag.

**Captures:** Dynamic E (the ceiling question stated as a singularity condition). This directly formalizes: does C_eff(t) → ∞ in finite time?

**Ceiling Prediction:** **Depends on γ relative to 1.** Nordhaus's empirical tests suggest we are not near singularity: research productivity has been declining, not accelerating. But his tests apply to the aggregate economy, not to a self-contained autocatalytic system like the mesh.

**Variable Mapping:**
- γ → The mesh's composite growth parameter
- σ → CES substitution parameter (= 1/(1−ρ) from the mesh paper)
- φ → Training productivity parameter  
- β → Degree of AI automation of training
- Baumol disease → Centralized training bottleneck

**Assessment: 8/10.** Provides the precise mathematical condition for the ceiling question. Paper 4 should derive the mesh's γ parameter from the mesh's microstructure and determine its value.

---

## FRAMEWORK 10: Lotka-Volterra Mutualism with Network Structure

**Source:** Lotka (1925); Volterra (1926); May (1972); Bastolla et al. (2009), Nature 458.

**The Math:** For n mutualistic species on a network:

dx_i/dt = r_i · x_i · (1 − x_i/K_i + Σ_j a_{ij} · x_j / (1 + h · Σ_j a_{ij} · x_j))

where a_{ij} is the mutualistic interaction strength (from adjacency matrix), K_i is carrying capacity, and h is a handling time (saturation parameter).

Without saturation (h = 0): mutualistic systems can exhibit unbounded growth (the "orgy of mutual beneficence" — May 1972). With saturation (h > 0): growth is bounded and the system has a unique stable equilibrium.

On networks, the spectral properties of the interaction matrix A = [a_{ij}] determine stability: if the spectral radius ρ(A) exceeds a threshold, the equilibrium destabilizes.

**Captures:** Dynamic A (mutualistic interactions on a network). The mesh's agents are mutualists: training agents improve specialists, specialists generate data for training agents.

**Ceiling Prediction:** **Convergence (with saturation) or unbounded growth (without saturation).** The ceiling question reduces to: does the improvement function saturate? If fine-tuning an already-good specialist yields diminishing returns (saturation), the system converges. If not, it can grow without bound.

**Variable Mapping:**
- x_i → Capability level of agent i
- a_{ij} → How much agent j's output improves agent i
- K_i → Maximum capability of agent i (hardware/architecture limit)
- h → Diminishing returns to training
- ρ(A) → Spectral radius of the mesh's improvement matrix

**Assessment: 7/10.** The saturation parameter h is exactly the ceiling parameter. Paper 4 should identify whether mesh training has h > 0 or h = 0.

**Math Prerequisites:** ODEs, linear algebra, spectral theory (Connor has the prerequisites).

---

## FRAMEWORK 11: Replicator Dynamics on Networks

**Source:** Taylor & Jonker (1978); Nowak & May (1992); Ohtsuki et al. (2006), Nature 441.

**The Math:** The replicator equation on a graph:

dx_i/dt = x_i · (f_i(**x**) − f̄(**x**))

where f_i is the fitness of strategy/type i (dependent on the population state and the network), and f̄ is average fitness. On a network with adjacency matrix W:

f_i = Σ_j W_{ij} · π(i,j)

where π(i,j) is the payoff from interacting with neighbor j.

On structured populations, cooperation can persist even when it would collapse in well-mixed populations, because cooperators can form clusters. The condition for cooperation: b/c > k (benefit-to-cost ratio exceeds average degree — Ohtsuki et al.).

**Captures:** Dynamic D (competition and selection among agent types on a network). The mesh's fitness dynamics — where high-quality agents attract traffic and low-quality agents lose it — are replicator dynamics on the mesh network.

**Ceiling Prediction:** **Convergence to an evolutionarily stable state.** The replicator equation on graphs has well-characterized fixed points. The population converges to an equilibrium mix of types.

**Assessment: 6/10.** Useful for the selection/composition dynamics (Mechanism 3) but doesn't capture the capability growth that is Paper 4's central question.

---

## FRAMEWORK 12: Model Collapse / Self-Referential Learning Limits

**Source:** Shumailov et al. (2024), Nature 631; Alemohammad et al. (2024); recent formalization by multiple groups (2025-2026).

**The Math:** Let P be the true data distribution and Q_t be the model's learned distribution at iteration t. If training data is a mixture of authentic data (fraction α) and synthetic data from Q_t (fraction 1−α):

KL(Q_{t+1} || P) ≥ KL(Q_t || P)  when α < α_crit

i.e., the model's divergence from truth increases monotonically when self-generated data dominates. The system converges to a degenerate fixed point with reduced diversity.

However, when α > 0 (persistent access to fresh authentic data), convergence to a neighborhood of P can be maintained, though with gradual degradation.

**Captures:** Dynamic C (self-referential learning — the negative case). This is the "garbage in, garbage out" risk: if the mesh trains on its own low-quality outputs, capabilities degrade rather than improve.

**Ceiling Prediction:** **Convergence to a degenerate fixed point** (model collapse) without external data. With persistent external signal (user queries, real-world feedback), the mesh can maintain quality but not necessarily improve unboundedly.

**Variable Mapping:**
- Q_t → Mesh's collective capability distribution at time t
- P → True capability frontier (what perfect agents would achieve)
- α → Fraction of training signal from genuine user interactions (vs. synthetic)
- Model collapse → Mesh capability degradation from self-referential training

**Assessment: 7/10.** Critical counterpoint to optimistic autocatalytic scenarios. Paper 4 must address the model collapse risk: under what conditions does the mesh's self-referential training improve rather than degrade capabilities? The key insight is that the mesh has an advantage over single-model self-training: **diversity**. Different specialists generate diverse outputs, which is the antidote to mode collapse. The mesh paper's CES diversity premium may protect against the degeneracy that kills homogeneous self-training.

---

## FRAMEWORK 13: Open-Ended Evolution / NK Fitness Landscapes

**Source:** Kauffman & Levin (1987); Kauffman (1993), *The Origins of Order*; Standish (2003); Bedau et al. (2000).

**The Math:** The NK model: N loci, each with K epistatic interactions. Fitness of genotype:

f(s₁,...,s_N) = (1/N) · Σ_i f_i(s_i, s_{i1},...,s_{iK})

where f_i depends on locus i and K other loci. For K = 0, the landscape is smooth (single peak, easy optimization). For K = N−1, the landscape is random (exponentially many local optima).

Open-ended evolution occurs when the fitness landscape is not fixed but co-evolves with the population. Bedau's "evolutionary activity" metric measures whether genuinely novel adaptations continue to emerge:

A(t) = number of novel adaptations persisting at time t

If A(t) → const: bounded novelty (ceiling).  
If A(t) ~ t^α: unbounded polynomial novelty.  
If A(t) ~ e^{rt}: unbounded exponential novelty.

**Captures:** Dynamic D (endogenous niche creation, co-evolving fitness landscape). The mesh creates new specialization types that create new demands that create new specialization types — the fitness landscape co-evolves with the agent population.

**Ceiling Prediction:** **Open question — depends on landscape structure.** Kauffman's NK model at intermediate K (the "edge of chaos") produces the longest adaptive walks. No formal proof exists for whether co-evolutionary dynamics produce bounded or unbounded novelty. This is one of the deepest open questions in complexity science.

**Assessment: 5/10.** Conceptually perfect but mathematically incomplete. The NK model is well-defined, but the co-evolutionary extension lacks the convergence/divergence results needed for Paper 4's ceiling question. Useful as motivation, not as a formal framework.

---

## FRAMEWORK 14: Control Theory — Positive Feedback Systems with Saturation

**Source:** Khalil (2002), *Nonlinear Systems*; Sontag (2005); Angeli, De Leenheer & Sontag (2007).

**The Math:** Consider the dynamical system:

dC/dt = g(f · C) − δ · C

where C is total capability, f is the fraction invested in self-improvement, g is the improvement function, and δ is depreciation. Three regimes:

1. **g concave** (g'' < 0): Diminishing returns. dC/dt = 0 has a unique stable fixed point C*. The system converges: C(t) → C*.
2. **g linear** (g(x) = r·x): Constant returns. C(t) = C(0)·e^{(rf−δ)t}. Exponential growth if rf > δ.
3. **g convex** (g'' > 0): Increasing returns. C(t) → ∞ in finite time (blow-up). This is the intelligence explosion.

The phase transition between regimes occurs at the inflection point of g. If g is sigmoidal (concave at high C, convex at low C), there's a threshold C* below which growth decelerates and above which it accelerates — a tipping point.

**Captures:** Dynamic E directly. This is the most stripped-down formalization of the ceiling question.

**Ceiling Prediction:** **Determined by the shape of g.** The entire paper reduces to: what is the shape of the mesh's improvement function g?

**Variable Mapping:**
- C → C_eff from the mesh paper
- f → Fraction of mesh capacity devoted to training
- g → The improvement function: how much capability improvement results from investing capability in training
- δ → Capability depreciation (model staleness, hardware degradation, competitive obsolescence)
- g concave → Diminishing returns to training already-good specialists
- g convex → Training improvements compound (better trainers produce better specialists produce better data produce better trainers)

**Assessment: 9/10.** This is the minimal formal model for Paper 4. Everything else (Romer, Jones, Weitzman, hypercycles) can be seen as providing microstructure for the function g. Paper 4's contribution is deriving g from the mesh's three mechanisms and determining whether it is concave, linear, or convex — and what parameters govern the transitions.

**Math Prerequisites:** ODEs, Lyapunov stability (Connor has the foundations).

---

## FRAMEWORK 15: Bloom et al. (2020) — Are Ideas Getting Harder to Find?

**Source:** Bloom, N., Jones, C. I., Van Reenen, J., & Webb, M. (2020), "Are Ideas Getting Harder to Find?", *AER* 110(4), 1104-1144.

**The Math:** Define research productivity:

TFP_growth / Research_effort

Bloom et al. show empirically that this ratio has been declining at ~5% per year across multiple domains (Moore's Law, agricultural yields, medical innovation). This implies φ < 1 in the Jones equation.

The "effective number of researchers" needed to maintain constant TFP growth doubles every 13 years.

**Captures:** Dynamic E (empirical evidence on the shape of g). This is the empirical anchor for calibrating the theoretical models above.

**Ceiling Prediction:** **φ < 1 in the current economy** — ideas are getting harder to find. But: this is measured for *human* researchers. If AI training agents have different scaling properties (perhaps because they can process the combinatorial explosion that Weitzman identifies), the mesh's effective φ could differ from the human economy's.

**Assessment: 8/10.** Essential empirical calibration. Paper 4 should argue that the mesh's autocatalytic structure changes the effective φ — and specify the mechanism.

---

## SYNTHESIS

### Which frameworks compose naturally?

The frameworks form three layers that map directly to the mesh paper's three-layer structure:

**Layer 1 — Existence of the Autocatalytic Set (Does self-sustaining improvement form?)**
- RAF Theory (Framework 2): Establishes existence threshold
- Jain-Krishna (Framework 3): Provides dynamics of autocatalytic set formation and percolation
- Maps to: Mesh paper's Layer 1 (percolation / giant component)

**Layer 2 — Growth Dynamics (How fast does capability grow?)**
- Jones (Framework 5) / Romer (Framework 4): The φ parameter governs the growth regime
- Aghion-Jones-Jones (Framework 6): AI automation of research can push φ_effective toward 1
- Weitzman (Framework 7): Combinatorial data explosion constrained by processing capacity
- Control Theory (Framework 14): Minimal formalization: dC/dt = g(fC) − δC
- Maps to: Paper 4's central question

**Layer 3 — Composition Dynamics (What determines the agent population?)**
- Lotka-Volterra mutualism (Framework 10): Network mutualism with saturation
- Replicator dynamics (Framework 11): Selection among agent types
- Open-ended evolution (Framework 13): Endogenous niche creation
- Maps to: Mesh paper's Layer 2 (specialization) extended with endogenous J

**Counterpoint Layer — Degradation Risk:**
- Model Collapse (Framework 12): Self-referential training degrades without external signal
- Eigen's error threshold (Framework 1): Information fidelity limits

### What does the combined mathematical structure predict about C_eff(t)?

The synthesis points to a **regime-dependent answer** governed by three parameters:

1. **φ** (training productivity elasticity w.r.t. existing capability): From Jones/Romer.
   - φ < 1: Convergence. Training gets harder as the mesh improves.
   - φ = 1: Exponential growth. Each improvement enables the next at the same rate.
   - φ > 1: Superexponential. Self-improvement accelerates.

2. **h** (saturation in mutualistic training interactions): From Lotka-Volterra.
   - h > 0: Ceiling exists. Diminishing returns to training already-good specialists.
   - h → 0: No ceiling from training saturation.

3. **α** (fraction of training signal from authentic external sources): From model collapse.
   - α > α_crit: Mesh maintains and improves capability.
   - α < α_crit: Model collapse degrades capability.

The **most likely regime** for the mesh is **exponential growth bounded by a Baumol bottleneck**: the mesh's autocatalytic structure pushes φ_effective toward 1 (by automating training), but the centralized training bottleneck (frontier model training cannot be distributed) acts as the non-automatable sector that Baumol's disease predicts will eventually dominate costs. The growth rate of C_eff is then proportional to the growth rate of frontier model capability — which is exogenous to the mesh.

This is a falsifiable prediction: the mesh improves at the rate frontier models improve (because frontier models are the "food set" in RAF terms), with a multiplicative amplifier from autocatalytic training and a diversity premium from CES aggregation.

### Where are the gaps?

1. **No existing framework captures all three mechanisms simultaneously on a network.** The closest is Jain-Krishna, but it has the wrong (zero-sum) growth dynamics. A new model is needed that combines:
   - RAF-like autocatalytic structure (which agents improve which)
   - Romer-like variety expansion (endogenous J)
   - Jones-like diminishing/constant/increasing returns (the φ question)
   - Network structure (not well-mixed)

2. **The interaction between model collapse and autocatalytic growth is uncharted.** No framework addresses: when does diversity in the mesh protect against the mode collapse that kills homogeneous self-training?

3. **The Baumol bottleneck for the mesh is identified but not formalized.** The mesh paper identifies centralized training persistence. Paper 4 needs to formalize this as the non-automatable sector and derive the conditions under which it binds.

### What is the minimal formal model for Paper 4?

**The Autocatalytic Mesh Growth Equation:**

dC_eff/dt = g(f · C_eff, J(t), α(t)) − δ · C_eff

where:
- g is the improvement function with three arguments:
  - f · C_eff: training effort (fraction f of capability devoted to self-improvement)  
  - J(t): number of specialization types (endogenous, growing with unmet demand)
  - α(t): fraction of training signal from external sources (declining as mesh grows)
- δ: depreciation/obsolescence

With J(t) governed by a RAF-like threshold:
- dJ/dt > 0 when mesh coverage gaps exist (unmet demand signals)
- J(t) bounded by the diversity premium (CES structure)

And α(t) governed by the model collapse constraint:
- α(t) = (external queries) / (total training data)
- Must exceed α_crit for non-degenerate learning

**Paper 4's central theorem** should characterize the regime of this system:

**Theorem (Autocatalytic Mesh Growth):** For the mesh with endogenous capability and training agents:
- If φ < 1 and h > 0: C_eff(t) → C_max (ceiling, with C_max depending on J, α, and the Baumol bottleneck).
- If φ = 1 and h > 0: C_eff(t) ~ e^{rt} (exponential growth at rate r = f · g'(0) − δ, bounded by the non-automatable sector).
- If φ > 1 and h = 0: C_eff(t) → ∞ in finite time (singularity, prevented only by external constraints).

The parameter φ is determined by the shape of the training improvement function, which in turn depends on:
- Whether training agents can be improved by other training agents (autocatalytic coupling)
- Whether the combinatorial data advantage (Weitzman) overcomes the fishing-out effect (Jones)
- Whether diversity (CES) protects against model collapse

This is the paper.
