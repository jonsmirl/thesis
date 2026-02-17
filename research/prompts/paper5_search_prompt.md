# PROMPT: FIELD SEARCH FOR PAPER 5 — THE SETTLEMENT FEEDBACK

## Attached files:
## 1. Autocatalytic_Mesh.tex (Paper 4 — this paper builds on its conclusions)
## 2. This prompt

---

## WHAT THIS IS

This is a **cross-disciplinary field search** — the first step in developing a new economics paper. Do NOT write the paper yet. The goal is to find formal mathematical frameworks from ANY discipline that capture the dynamics described below.

## THE PAPER SEQUENCE SO FAR

Four papers form a closed loop about the internal dynamics of distributed AI infrastructure:

1. **Endogenous Decentralization** (Smirl 2026a): Concentrated AI infrastructure investment ($1.3T hyperscaler capex) endogenously finances the learning curves that enable distributed alternatives. The crossing occurs when R₀ > 1.

2. **The Mesh Equilibrium** (Smirl 2026): After the crossing, heterogeneous specialized agents self-organize into a mesh whose collective capability exceeds centralized provision at N > N*. Three layers: percolation (connectivity), CES aggregation (specialization), Laplacian diffusion (knowledge propagation). The mesh requires a programmable settlement layer for routing compensation (Section 8).

3. **The Monetary Productivity Gap** (Smirl 2026b): Analyzes the cost differential between fiat and cryptocurrency monetary infrastructure. Transfer cost gap (remittance fees) and yield access gap (unbanked populations denied Treasury yields). Six-stage country classification. Provides the monetary infrastructure analysis the mesh paper points to.

4. **The Autocatalytic Mesh** (Smirl 2026, attached): Removes the fixed-capability assumption. Three mechanisms make capability endogenous: autocatalytic training, self-referential learning, variety expansion. Growth regime governed by φ_eff, h, and α. Most likely regime: convergence to ceiling governed by Baumol bottleneck (frontier training as non-automatable sector). The mesh is a multiplier, not a generator.

## THE MISSING PIECE

All four papers treat the mesh's external financial environment as exogenous. The settlement layer is needed (mesh paper), the MPG analyzes monetary infrastructure (MPG paper), but neither formalizes **what happens when the mesh's agents become the dominant participants in the capital markets that fund and enable them.**

A separate analysis (J. Smirl 2026, "The Last Currency") identifies the endgame qualitatively:
- Autonomous AI agents become the marginal buyers/sellers of sovereign debt
- Market pricing efficiency approaches theoretical limits (every asset priced continuously at fundamentals)
- Monetary policy tools that depend on market friction (forward guidance, QE, financial repression) lose effectiveness
- Financial repression collapses as tokenized dollar assets give captive savers an exit
- Spontaneous dollarization accelerates as stablecoins provide exit from mismanaged currencies
- The US dollar strengthens as unit of account while losing control as transaction medium

**Paper 5 formalizes this.** The central question: when the mesh's autonomous agents enter capital markets, does this create a positive or negative feedback loop for the mesh's own growth?

---

## THE DYNAMICS TO FORMALIZE

### The Coupled System

The mesh and the financial system are coupled:

**Forward link (mesh → financial system):**
- The mesh grows → demands more settlement infrastructure → mesh agents enter capital markets
- Mesh agents process information faster and allocate capital more efficiently than human participants
- Market efficiency increases toward the theoretical limit
- Monetary policy tools that depend on friction (information asymmetry, institutional inertia, captive savers) lose effectiveness
- Financial repression collapses → sovereign debt repriced on fundamentals
- Capital flows permissionlessly across borders → geographic segmentation of capital markets collapses

**Backward link (financial system → mesh):**
- More efficient capital markets allocate resources better to productive uses, including mesh infrastructure
- Collapse of geographic capital market segmentation → capital flows to wherever mesh returns are highest
- Settlement infrastructure improvement → mesh growth constraint relaxes (Prediction 6 of mesh paper)
- Sovereign repricing → some currencies collapse → more adoption of programmable money → larger settlement ecosystem → mesh grows faster

**The feedback question:** Is this loop positive (self-reinforcing, mesh growth accelerates) or negative (self-limiting)? Under what conditions does it produce stability vs. instability?

### Mechanism 1: Market Efficiency as a Function of Participant Composition

Current capital markets have human-speed participants with institutional constraints (prime brokerage, DTCC clearing, margin accounts, quarterly rebalancing, career risk, herding behavior). The marginal price-setter is a human institution responding to a mix of fundamentals and frictions.

When mesh agents become the marginal participants:
- Information processing: real-time vs. human-speed
- Portfolio rebalancing: continuous vs. periodic
- Institutional constraints: none (no prime broker, no clearinghouse) vs. heavy
- Response to forward guidance: data-driven vs. narrative-driven
- Cross-border mobility: permissionless vs. regulated

**Structural question:** How does market efficiency (in the Fama sense) change as the fraction of autonomous agent participants increases? Is the transition smooth or is there a phase transition at some critical fraction?

### Mechanism 2: Monetary Policy Effectiveness Degradation

Central bank tools depend on market frictions:
- **Forward guidance:** Works because human participants debate verbal signals for weeks, giving the central bank time to adjust. AI agents compare statements to real-time data and reprice in seconds.
- **QE (quantitative easing):** Works because buying pressure suppresses yields when participants are slow to arbitrage. AI agents sell to the central bank at suppressed prices and redeploy elsewhere — the central bank becomes a predictable liquidity source.
- **Financial repression:** Works because captive savers have no practical alternative to negative-real-return government bonds. Tokenized dollar assets give every saver an exit.

**Structural question:** Is there a critical fraction of autonomous participants above which each monetary policy tool loses effectiveness? What is the mapping between participant composition and policy tool effectiveness?

### Mechanism 3: The Dollarization Spiral

Stablecoins are dollar-denominated. Stablecoin adoption is spontaneous dollarization. For countries with weak currencies (inflation > 15-20%, the "Tier 3" in "The Last Currency"):

- Stablecoin access → marginal saver exits local currency → local currency depreciates → inflation rises → more savers exit → spiral accelerates
- This is the classic hyperinflationary spiral, but stablecoins lower the threshold at which it begins and accelerate its pace
- Each currency collapse increases the stablecoin ecosystem → more settlement infrastructure → mesh benefits

**Structural question:** Is there a critical "Fiat Quality Index" below which the dollarization spiral is inevitable once stablecoin access reaches a threshold? Can this be formalized as a tipping point / phase transition?

### Mechanism 4: The Triffin Paradox in Stablecoin Form

The on-chain economy needs dollar-denominated risk-free assets (Treasuries) as collateral backing. The US must supply them. This means:
- Stablecoin growth increases Treasury demand (good for US borrowing costs)
- But also makes the on-chain economy structurally dependent on continuous US debt issuance
- The US fiscal position becomes the collateral quality of the entire on-chain financial system
- AI agents pricing sovereign risk in real time make it impossible to mask fiscal unsustainability

**Structural question:** Is this a stable equilibrium or does it contain a contradiction? Does the system require US fiscal deficits to function (structural demand for Treasuries as collateral) while simultaneously making those deficits harder to sustain (AI agents pricing fiscal risk accurately)?

### Mechanism 5: r > g at Machine Speed

Piketty's r > g assumes human-speed capital markets with intermediation frictions. When AI agents manage tokenized portfolios:
- r accelerates (continuous optimization, zero intermediation friction, full arbitrage)
- g is bounded by human productivity growth and physical constraints
- The r-g wedge widens from 2-3% to potentially 5-10%
- Wealth concentration accelerates unless AI portfolio management is democratized

**Structural question:** Does the mesh's entry into capital markets amplify or reduce inequality? The answer likely depends on whether mesh-quality capital allocation becomes a consumer product (democratized, like index funds) or remains a technical capability (concentrated, like early HFT).

### THE CENTRAL QUESTION FOR PAPER 5

When the mesh's autonomous agents transform the capital markets they depend on, the coupled system has the following properties:

1. The mesh needs settlement infrastructure (programmable money)
2. The mesh's agents make capital markets more efficient
3. More efficient capital markets allocate resources better to the mesh
4. But: more efficient capital markets also destroy the monetary policy tools that governments use to manage economies
5. And: the mesh's settlement infrastructure (stablecoins) triggers dollarization that collapses weak currencies
6. And: the mesh's collateral needs require continuous US debt issuance while simultaneously making that debt harder to manage

**Does this coupled system have a stable equilibrium? What does it look like? Is the transition to it smooth or discontinuous?**

---

## SEARCH INSTRUCTIONS

Search across ALL disciplines for formal frameworks capturing these dynamics. The economics literature is primary here but don't limit to it.

For EACH framework found, provide:

1. **Name and source**: The canonical reference(s)
2. **The math**: The actual equations, not just a description
3. **What structural property it captures**: Which mechanism(s) above
4. **What it predicts about the feedback question**: Positive/negative loop, stability/instability
5. **What maps to what**: Explicit variable mapping
6. **What doesn't transfer**: Precise limitations
7. **Assessment**: 1-10 for Paper 5

### SPECIFIC DYNAMICS TO SEARCH FOR

**Dynamic A: Market microstructure transition from human to algorithmic participants.**
Models of how market properties (efficiency, liquidity, volatility, price discovery) change as the composition of participants shifts from human to algorithmic. Not just HFT literature (which is about speed at the margin) but fundamental change in who the marginal price-setter is. Does efficiency improve smoothly or is there a phase transition?

**Dynamic B: Monetary policy effectiveness as a function of market structure.**
Formal models of how central bank tools (open market operations, forward guidance, yield curve control) depend on market participant characteristics. The Lucas critique applied to monetary policy: when the participants change, the policy transmission mechanism changes. Are there models that formalize when specific tools lose effectiveness?

**Dynamic C: Currency substitution / dollarization dynamics.**
Formal models of currency substitution where agents choose between domestic currency and a foreign alternative. Threshold models where the transition accelerates beyond a tipping point. The literature on dollarization in Latin America, hyperinflationary spirals, and currency competition. How do switching costs and network effects interact?

**Dynamic D: Coupled technology-institution co-evolution.**
Systems where a technology transforms the institutional environment it operates in, and the changed environment feeds back to the technology's development. Not just "technology adoption" but genuine co-evolution where the technology changes the rules of the game. Historical examples: the internet transforming commerce, railways transforming capital markets, the printing press transforming information markets.

**Dynamic E: Endogenous financial infrastructure.**
Models where the financial system's structure is not given but emerges from the interaction of market participants. Market microstructure theory, endogenous market formation, the conditions under which intermediaries (banks, brokers, exchanges) emerge or become obsolete. When does disintermediation occur?

**Dynamic F: The Triffin dilemma formalized.**
Formal models of the reserve currency provider's constraint: needing to supply reserve assets while maintaining their quality. Does the stablecoin-mediated version of this dilemma have different stability properties than the original?

**Dynamic G: Wealth concentration dynamics with heterogeneous capital allocation technology.**
Models of wealth distribution where agents have access to different capital allocation technologies (some human-speed, some machine-speed). How does the wealth distribution evolve when a fraction of agents have strictly better allocation technology? Is there a steady-state Gini coefficient, or does inequality grow without bound?

### FRAMEWORKS I SUSPECT MAY BE RELEVANT

- Kyle (1985) — Informed trading, market microstructure
- Grossman-Stiglitz (1980) — Information and market efficiency paradox
- Calvo-Reinhart — Fear of floating, dollarization
- Obstfeld-Rogoff — Open economy macroeconomics, capital flows
- Brunnermeier-Sannikov — Money in general equilibrium
- Eichengreen — Exorbitant Privilege, reserve currency dynamics
- Piketty-Saez-Zucman — Wealth distribution dynamics
- Duffie — Dark markets, OTC market microstructure
- Gabaix — Variable rare disasters, power law in finance
- Acemoglu-Robinson — Why Nations Fail, institutional dynamics
- Tirole — Bubbles, rational asset pricing
- Diamond-Dybvig — Bank runs, financial fragility
- Minsky — Financial instability hypothesis (if formalized)
- Brian Arthur — Increasing returns and path dependence
- Evolutionary game theory applied to currency competition

But DO NOT limit to these. Search broadly.

### WHAT I AM NOT LOOKING FOR

- Crypto/blockchain enthusiasm without formal models
- Bitcoin maximalist arguments
- Generic "AI will change everything" commentary
- Policy recommendations without formal mechanisms
- Anything that doesn't have equations

### FRAMEWORKS FROM THE PRIOR PAPERS THAT MAY CARRY FORWARD

- The BEC phase transition framework (mesh paper) may apply to capital market concentration/deconcentration
- The CES diversity premium may apply to portfolio diversification benefits
- The R₀ threshold may apply to stablecoin adoption dynamics
- The Baumol bottleneck (Paper 4) may have a financial analog
- The model collapse constraint (Paper 4) may apply to market efficiency — can markets be "too efficient" (Grossman-Stiglitz paradox)?

---

## OUTPUT FORMAT

For each framework found:

### Framework N: [Name]
**Source:** [Canonical reference(s)]
**The Math:** [Actual equations]
**Captures:** [Which dynamic(s) A-G]
**Feedback Prediction:** [Positive loop / negative loop / stable equilibrium / instability]
**Variable Mapping:** [mesh/financial variable → framework variable]
**Breaks Down When:** [Precise limitation]
**Assessment:** [1-10 for Paper 5]

After all frameworks:

### SYNTHESIS
- Which frameworks compose?
- What does the combined structure predict about the coupled mesh-financial system?
- What are the gaps?
- What is the minimal formal model for Paper 5?
- Is this one paper or two? (market efficiency + monetary policy might be separable from dollarization + Triffin)

---

## CONSTRAINTS

- This is a search prompt, not a paper prompt. Do not write the paper.
- Be exhaustive. 15 frameworks is better than 5.
- The math matters — concepts without equations are noted but not rated highly.
- This paper sits at the intersection of macro, monetary economics, market microstructure, and the mesh sequence. It needs to be credible in all four.
- Connor's math: Calc III, Differential Equations, Linear Algebra, Discrete Math. Note prerequisites.
- The paper should be Connor's — formal, falsifiable, building on the prior four. Jon's "Last Currency" provides the qualitative target; the formal paper provides the machinery.
- Notation compatible with the mesh paper sequence where possible.
- The key novelty claim must be identified: what does this paper prove that the combination of existing monetary economics + existing market microstructure + existing dollarization theory does NOT already establish?
