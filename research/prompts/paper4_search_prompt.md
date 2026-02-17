# PROMPT: FIELD SEARCH FOR PAPER 4 — AUTOCATALYTIC MESH

## Attached files:
## 1. Mesh_Equilibrium_v1.tex (the mesh paper — Paper 4 begins where it ends)
## 2. This prompt

---

## WHAT THIS IS

This is a **cross-disciplinary field search** — the first step in developing a new economics paper. Do NOT write the paper yet. The goal is to find formal mathematical frameworks from ANY discipline that capture the dynamics described below.

The attached mesh paper (Smirl 2026, "The Mesh Equilibrium") proves that a self-organizing mesh of heterogeneous specialized AI agents exceeds centralized provision above a finite critical mass N*. It uses three layers: percolation for connectivity, CES aggregation for heterogeneous specialization, and Laplacian diffusion for knowledge propagation.

**The mesh paper assumes fixed-capability nodes.** Each agent i has a capability vector c_i that redistributes through specialization but does not grow. C_eff(N) increases only by adding diverse agents. Knowledge diffusion (∂u/∂t = −Lu) is consensus-seeking — it equalizes existing knowledge, it does not create new capability.

**Paper 4 removes that assumption.** It asks: what happens when the mesh can improve itself?

---

## THE DYNAMICS TO FORMALIZE

Three mechanisms make capability endogenous within the mesh. Describe each precisely so you can search for formal frameworks.

### Mechanism 1: Agents That Improve Agents

Some nodes in the mesh are not inference-specialists — they are **training agents**. They fine-tune, distill, quantize, and adapt other agents. When a training agent T operates on specialist S:

- S's capability vector c_S changes: c_{Sj} increases for the specialist's domain
- T's effectiveness at training improves with experience (learning-by-doing in training itself)
- The improved S generates better query-response pairs, which become training data for T to improve other agents

This is **autocatalytic**: improved agents produce better data, which produces better training, which produces better agents. The capability vector c_i is no longer a parameter — it's a dynamical variable coupled to the mesh's operation.

The structural properties that matter:
- Positive feedback loop (output of process feeds input)
- Heterogeneous agents (not all agents improve at the same rate or in the same way)
- The "catalyst" (training agent) is not consumed — it can improve many specialists
- Network structure matters: a training agent improves agents it's connected to
- Possible saturation (diminishing returns to further fine-tuning of an already-good specialist)

### Mechanism 2: Operation Generates Training Data

Every routed query-response pair in the mesh is information. Specifically:
- Successful routings (user satisfied) are positive training signal
- Failed routings (user re-queries or escalates to centralized) are negative signal
- The diversity of queries across the mesh exceeds what any single centralized provider sees, because the mesh serves the long tail

The mesh's **learning rate scales with throughput**. More queries → more training signal → faster capability growth → better routing → more queries attracted → more training signal. This is a second autocatalytic loop, operating on the data layer rather than the agent layer.

Structural properties:
- Information is non-rival (one query-response pair can train many agents)
- Information quality depends on agent quality (garbage in, garbage out — low-quality agents generate low-quality training signal)
- There's a bootstrapping problem: early mesh has few queries, so little training data, so slow improvement
- The centralized provider also learns from its operations — but its training data is narrower (optimized for high-volume queries, misses long tail)

### Mechanism 3: The Mesh Modifies Its Own Composition

The mesh doesn't just improve existing agents — it creates new ones and deprecates old ones:
- Unmet demand signals (high bids, no specialist available) trigger creation of new specialists
- The number of specialization types J becomes endogenous — it grows as the mesh identifies gaps
- Low-quality agents lose traffic (Bianconi-Barabási fitness dynamics) and effectively exit
- Routing topology itself evolves — hub agents emerge, merge, split

Structural properties:
- The "species" of agents are not fixed — new types emerge
- The fitness landscape is co-evolving with the population
- Selection operates continuously through the market mechanism (price signals from Section 8 of mesh paper)
- There's a possibility of open-ended evolution — new niches create new selection pressures that create new niches

### THE CENTRAL QUESTION

When you make capability endogenous through these three mechanisms, what happens to C_eff(t)?

Three possibilities:
1. **Convergence to a ceiling.** C_eff(t) → C_max as t → ∞. The autocatalytic loops have diminishing returns that dominate. The mesh gets better but plateaus.
2. **Polynomial growth.** C_eff(t) ~ t^α for some α > 0. Steady improvement but subexponential. Each doubling takes longer than the last.
3. **Exponential or superexponential growth.** C_eff(t) ~ e^{rt} or faster. Each improvement enables the next improvement at least as fast. No ceiling within the model.

Which of these obtains — and what parameters govern the regime — is the paper's central question. Note that this is structurally the question of whether recursive self-improvement has a fixed point, stated in the mathematical language of the mesh paper.

---

## SEARCH INSTRUCTIONS

Search across ALL disciplines — mathematics, physics, chemistry, biology, ecology, computer science, economics, control theory, dynamical systems, complex systems, network science — for formal frameworks that capture the dynamics above. Do NOT limit yourself to economics or AI.

For EACH framework you find, provide:

1. **Name and source**: The canonical reference(s)
2. **The math**: The actual equations, not just a description
3. **What structural property it captures**: Which of the three mechanisms above, or their combination
4. **What it predicts about the ceiling question**: Convergence, polynomial, or exponential?
5. **What maps to what**: Explicit mapping between the framework's variables and the mesh's variables
6. **What doesn't transfer**: Be precise about where the analogy breaks down

### SPECIFIC DYNAMICS TO SEARCH FOR

Here are precise structural descriptions. Find formal frameworks for each:

**Dynamic A: Autocatalysis with heterogeneous catalysts on a network.**
A set of interacting species where some species catalyze the improvement of others, the catalysts have different efficiencies, and the interaction structure is a network (not well-mixed). The output of catalyzed reactions feeds back as input. Does the system converge, grow polynomially, or grow exponentially? Under what conditions?

**Dynamic B: Endogenous innovation / variety expansion in a network economy.**
An economy where the number of product types (specializations) is not fixed but grows endogenously through investment. Each new variety has CES complementarity with existing varieties. Agents are heterogeneous. The Romer (1990) / Aghion-Howitt (1992) endogenous growth tradition addresses this — but what specific mathematical structures govern whether growth is bounded or unbounded? How does network structure affect the answer?

**Dynamic C: Self-referential learning systems.**
A system that generates data from its own operation, uses that data to improve itself, and then generates better data from the improved operation. This is the online learning / reinforcement learning structure, but at the system level rather than the agent level. Are there convergence results for systems that learn from their own output? What governs the rate?

**Dynamic D: Open-ended evolution with endogenous niche creation.**
A population where new "species" (agent types) emerge endogenously, creating new ecological niches that support further speciation. The fitness landscape co-evolves with the population. Does the system reach an equilibrium number of species, or does speciation continue indefinitely? Under what conditions?

**Dynamic E: The recursive self-improvement question formalized.**
A system with capability C(t) that invests fraction f of its capability in self-improvement. The improvement function is C'(t) = g(f · C(t)), where g captures the productivity of self-improvement. Three regimes: g concave (diminishing returns → convergence), g linear (constant returns → exponential), g convex (increasing returns → superexponential / finite-time singularity). What determines which regime obtains? Are there phase transitions between regimes?

### WHAT I AM NOT LOOKING FOR

- General AI safety arguments about recursive self-improvement (I need math, not philosophy)
- Vague complexity theory without specific equations
- Frameworks that assume well-mixed populations (the network structure matters)
- Anything that requires homogeneous agents (heterogeneity is the mechanism)

### FRAMEWORKS I SUSPECT MAY BE RELEVANT (but search broadly beyond these)

- Chemical reaction network theory (Feinberg, Horn-Jackson) — autocatalytic sets
- Kauffman's autocatalytic sets / RAF theory
- Eigen's quasispecies and hypercycle theory
- Romer (1990) endogenous growth with variety expansion
- Aghion-Howitt (1992) Schumpeterian growth
- Weitzman (1998) recombinant growth
- Jones (2005) growth with increasing research difficulty
- Open-ended evolution literature (Standish, Bedau, Taylor)
- Self-organized criticality (Bak-Tang-Wiesenfeld)
- Stuart Russell / MIRI work on recursive self-improvement (if formalized)

But DO NOT limit to these. The ISP method is to search ALL of human knowledge for the solved version of this problem. The solved version probably exists in a field I haven't guessed.

---

## OUTPUT FORMAT

For each framework found, use this structure:

### Framework N: [Name]
**Source:** [Canonical reference(s)]
**The Math:** [Actual equations]
**Captures:** [Which dynamic(s) A-E]
**Ceiling Prediction:** [Convergence / polynomial / exponential / depends on parameter X]
**Variable Mapping:** [mesh variable → framework variable]
**Breaks Down When:** [Precise limitation]
**Assessment:** [How useful for Paper 4, 1-10]

After all frameworks, provide:

### SYNTHESIS
- Which frameworks compose naturally?
- What does the combined mathematical structure predict about C_eff(t)?
- Where are the gaps — dynamics that no framework captures?
- What is the minimal formal model for Paper 4?

---

## CONSTRAINTS

- This is a search prompt, not a paper prompt. Do not write the paper.
- Be exhaustive. I'd rather have 15 frameworks rated 4/10 than miss the 10/10 framework because you stopped early.
- The math matters. If a framework is conceptually relevant but doesn't have equations, say so and move on.
- Prioritize frameworks that address the CEILING QUESTION specifically. That's the central contribution of Paper 4.
- Connor is a Tufts sophomore. The math he knows: Calc III, Differential Equations, Linear Algebra, Discrete Math. Frameworks requiring measure theory, functional analysis, or stochastic calculus can still be found — we'll handle the translation later. But note the mathematical prerequisites.
- Notation should be compatible with the attached mesh paper where possible.
