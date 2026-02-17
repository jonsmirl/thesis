# FIELD SEARCH RESULTS: PAPER 5 — THE SETTLEMENT FEEDBACK

**Prepared for:** Connor Smirl, Tufts University  
**Date:** February 2026  
**Scope:** Cross-disciplinary search for formal frameworks capturing the coupled mesh–financial system dynamics

---

## Framework 1: Grossman–Stiglitz (1980) — The Information Efficiency Paradox

**Source:** Grossman, S.J. and J.E. Stiglitz (1980). "On the Impossibility of Informationally Efficient Markets." *American Economic Review* 70(3): 393–408.

**The Math:** Fraction λ of agents pay cost c to become informed. Risky asset return u = ρ + θ + ε, where informed agents observe signal θ. In equilibrium with CARA utility:

- Informed demand: X_I = (E[u|θ, P] − RP) / (aσ²_I)
- Uninformed demand: X_U = (E[u|P] − RP) / (aσ²_U)
- Market clearing determines price P
- Equilibrium informativeness: h ≡ τ_θ/(τ_ε) where τ denotes precision
- Equilibrium condition: E[U_I] − c = E[U_U], determining λ*
- Key result: As c → 0, λ* → some interior value, not 1. Prices never fully reveal information.

**Captures:** Dynamic A (market efficiency transition), Dynamic E (endogenous financial infrastructure)

**Feedback Prediction:** Negative loop / stability. As information costs fall (mesh agents have near-zero marginal info processing cost), the model predicts: (1) equilibrium λ → 1 (all become "informed"), (2) but then no one earns rents from information → paradox. The market approaches a limit where prices almost fully reveal, but the last basis point of inefficiency is preserved as the equilibrium "noise" that incentivizes participation.

**Variable Mapping:**
- c (info cost) → mesh agents' marginal cost of processing information (approaches zero)
- λ (fraction informed) → fraction of capital managed by autonomous agents
- σ²_ε (noise) → remaining human/institutional friction in markets
- h (informativeness) → market efficiency as function of mesh participation

**Breaks Down When:** The model assumes a binary informed/uninformed distinction. Mesh agents don't just become "informed" — they change the speed and structure of information processing. Also assumes exogenous noise trading; if noise traders are eliminated (captive savers exit via stablecoins), the equilibrium collapses entirely. The model is static, single-period.

**Assessment:** 8/10. This is foundational for the market efficiency mechanism. The paradox is precisely the "model collapse" analog for markets — can markets be "too efficient" to function? The key insight: mesh agents drive c → 0, which pushes the system toward the paradox. Paper 5 needs to resolve what happens at the limit. Connor's math: accessible (CARA utility, normal distributions).

---

## Framework 2: Kyle (1985) — Strategic Informed Trading

**Source:** Kyle, A.S. (1985). "Continuous Auctions and Insider Trading." *Econometrica* 53(6): 1315–1335.

**The Math:** One informed trader, noise traders (order flow u ~ N(0, σ²_u)), and a market maker.

- Asset value: v ~ N(p₀, Σ₀)
- Informed trader submits order: x = β(v − p₀) where β = σ_u/σ_v
- Market maker sets price: p = p₀ + λ(x + u) where λ = σ_v/(2σ_u)
- Kyle's lambda (price impact): λ = (1/2)√(Σ₀/σ²_u)
- Informed trader's expected profit: E[π] = σ_u·σ_v / 2
- In multi-period: information revealed gradually; price converges to v by terminal date

**Captures:** Dynamic A (market microstructure transition)

**Feedback Prediction:** Transitional dynamics. As mesh agents replace noise traders (σ_u → 0), Kyle's lambda → ∞. Market depth collapses. In the limit where all participants are informed or quasi-informed, the model predicts market breakdown: no liquidity, no trade. This mirrors the Grossman-Stiglitz paradox from the microstructure side.

**Variable Mapping:**
- β (informed trading intensity) → mesh agent order aggressiveness
- λ (price impact / market depth) → liquidity as function of participant composition
- σ_u (noise volume) → residual human/institutional trading volume
- Σ₀ (prior uncertainty) → information advantage of mesh over non-mesh participants

**Breaks Down When:** Model assumes one informed trader. With many informed agents (the mesh), competition among informed traders changes the equilibrium structure entirely (Holden and Subrahmanyam 1992 show information is revealed immediately with multiple insiders). Also, no endogenous entry/exit.

**Assessment:** 7/10. The scaling laws (Kyle-Obizhaeva invariance) provide testable microstructure predictions. The key contribution to Paper 5: as mesh fraction increases, λ's behavior characterizes the phase transition in market structure. Connor's math: accessible (linear equilibrium, normal distributions).

---

## Framework 3: Brunnermeier–Sannikov (2014, 2015a) — Endogenous Financial Instability and the I Theory of Money

**Source:** Brunnermeier, M.K. and Y. Sannikov (2014). "A Macroeconomic Model with a Financial Sector." *American Economic Review* 104(2): 379–421. And (2015a). "The I Theory of Money." NBER Working Paper 22533.

**The Math:** Continuous-time general equilibrium with financial frictions. State variable η_t = expert net worth / total wealth.

- Capital dynamics: dK_t/K_t = (Φ(ι_t) − δ)dt + σdZ_t
- Expert wealth: dη_t = (drift)dt + (σ_η)dZ_t where σ_η is *endogenous*
- Price of capital q(η) solves an ODE with endogenous volatility
- Volatility paradox: as exogenous σ → 0, endogenous σ_η does NOT vanish
- Money value: θ(η) = p_t·M_t / (q_t·K_t), ratio of money to capital
- Monetary policy redistributes: affects η trajectory, mitigates disinflationary spirals
- Fisher disinflation: negative shock → η falls → inside money contracts → money demand rises → deflation → further balance sheet damage

**Captures:** Dynamic B (monetary policy effectiveness), Dynamic D (coupled technology-institution co-evolution), Dynamic E (endogenous financial infrastructure)

**Feedback Prediction:** Instability with endogenous crises. The model's central insight — the volatility paradox — directly applies: even if the mesh reduces exogenous market volatility, endogenous financial instability can persist or worsen. The system spends most time near a stable steady state but occasionally enters volatile crisis regimes. This predicts the mesh-financial system has a *bimodal* distribution: mostly stable, occasionally catastrophic.

**Variable Mapping:**
- η (expert net worth share) → mesh-aligned capital fraction (capital under autonomous management)
- σ (exogenous risk) → pre-mesh market volatility
- σ_η (endogenous risk) → post-mesh financial instability
- θ (money-capital ratio) → stablecoin ecosystem size relative to real capital
- Inside money creation → stablecoin issuance by intermediaries
- Monetary policy tools → central bank interventions in the mesh-transformed economy

**Breaks Down When:** Assumes intermediaries are essential for risk allocation. If the mesh disintermediates financial intermediaries entirely, the two-sector structure (experts vs. households) may not apply. Also, the model treats monetary policy as fully operational; Paper 5's question is precisely what happens when it isn't.

**Assessment:** 9/10. This is the strongest candidate for the backbone of Paper 5. The I Theory already has money, intermediaries, monetary policy, and endogenous crises. The extension needed: make the intermediary sector's technology endogenous (mesh agents replace human intermediaries) and analyze what happens to monetary transmission. Connor's math: requires ODEs and stochastic calculus — at the edge of his current preparation (Diff Eq + some extension). The shooting method for solving the equilibrium is computationally tractable.

---

## Framework 4: Uribe (1997) — Currency Substitution with Hysteresis

**Source:** Uribe, M. (1997). "Hysteresis in a Simple Model of Currency Substitution." *Journal of Monetary Economics* 40: 185–202.

**The Math:** Agents accumulate "dollarization capital" k through network externalities.

- Transaction cost: ψ(π, k) where π is inflation rate, k is dollarization capital stock
- Dollarization capital evolves: dk/dt = v(π, k) − δ·k where v is velocity of foreign currency use
- Key thresholds: π̄ (upper) and π_ (lower) define bifurcation points
- For π ∈ (π_, π̄): multiple steady states exist (dollarized and non-dollarized)
- Phase diagram: saddle-path dynamics with catastrophe points
- Hysteresis: temporary high inflation permanently shifts economy to dollarized steady state
- Seignorage Laffer curve is modified by dollarization capital

**Captures:** Dynamic C (dollarization spiral)

**Feedback Prediction:** Positive loop with tipping point. Once dollarization capital k exceeds a threshold, the system irreversibly shifts to the dollarized equilibrium. Critically, the threshold depends on inflation history (hysteresis), not just current inflation. Stablecoins reduce the effective threshold by lowering the cost of building dollarization capital.

**Variable Mapping:**
- k (dollarization capital) → stablecoin adoption infrastructure (wallets, on-ramps, merchant acceptance)
- π (inflation rate) → domestic currency quality / "Fiat Quality Index"
- π̄, π_ (bifurcation thresholds) → critical inflation rate above which stablecoin adoption is irreversible
- δ (depreciation of dollarization capital) → how quickly stablecoin infrastructure decays without use
- v (foreign currency velocity) → stablecoin transaction volume

**Breaks Down When:** Model treats dollarization as costly (learning-by-doing). Stablecoins change this: the "dollarization capital" arrives pre-built via smartphone apps. The thresholds π̄ and π_ shift dramatically downward. The model also doesn't account for the speed differential — traditional dollarization took decades; stablecoin dollarization can happen in months.

**Assessment:** 8/10. The hysteresis structure is exactly right for the dollarization spiral. Paper 5 needs to: (1) endogenize the threshold as a function of stablecoin infrastructure quality, (2) connect stablecoin growth to mesh settlement infrastructure (creating the feedback loop), (3) add the speed acceleration that digital adoption enables. The catastrophe/bifurcation framework maps to the R₀ threshold from the mesh sequence. Connor's math: accessible (ODEs, phase diagrams).

---

## Framework 5: Farhi–Maggiori (2018) — A Model of the International Monetary System

**Source:** Farhi, E. and M. Maggiori (2018). "A Model of the International Monetary System." *Quarterly Journal of Economics* 133(1): 295–355.

**The Math:** A Hegemon issues reserve assets to the Rest of World (RoW). Key equations:

- Hegemon issues b units of debt, earning monopoly rents from safety/liquidity premium
- Three zones: Safety (b ≤ b̄), Instability (b̄ < b ≤ b̃), and Collapse (b > b̃)
- Safety premium: R_s(b) < R̄_r (safe rate below risky rate; difference is liquidity premium)
- In Safety zone: debt is always safe, Hegemon earns exorbitant privilege
- In Instability zone: multiple equilibria — sunspot can trigger run
- Hegemon's value function: V(b) = (1 − α(b))·b·(R̄_r − R_s(b)) − α(b)·λ·τ·(1 − e_L)
- Triffin Dilemma formalized: Hegemon chooses between (a) issue b̄ safely, or (b) issue b_FC > b̄ with positive crisis probability α
- Proposition 2: Under limited commitment, the Hegemon may choose the instability zone when rents are high enough

**Captures:** Dynamic F (Triffin dilemma formalized)

**Feedback Prediction:** Conditional instability. The stablecoin version of the Triffin dilemma: stablecoin growth increases b (demand for Treasuries as collateral), pushing the Hegemon (US) deeper into the instability zone. But unlike Farhi-Maggiori's setting, the mesh's agents price sovereign risk in real time, which makes the sunspot probability α endogenous and potentially increasing in b. This creates a contradiction: stablecoin growth requires more Treasuries, but the resulting fiscal expansion makes those Treasuries less safe.

**Variable Mapping:**
- b (Hegemon debt issuance) → US Treasury issuance to back stablecoin ecosystem
- b̄ (safety threshold) → maximum debt level consistent with safe-asset status
- α(b) (crisis probability) → probability of sovereign confidence crisis, now endogenous to mesh pricing
- R_s(b) (safety premium) → stablecoin-era Treasury yield advantage
- RoW demand for safe assets → global stablecoin demand

**Breaks Down When:** The model treats the safety/instability boundary as exogenous to the market structure. With mesh agents continuously pricing sovereign risk, the boundary itself becomes endogenous and potentially unstable. Also, the model has only one Hegemon; the stablecoin world might fragment across multiple stablecoin-backing currencies.

**Assessment:** 9/10. This is the canonical model for the Triffin mechanism in Paper 5. The extension: make α endogenous to the fraction of mesh agents pricing sovereign risk (connecting Dynamic F to Dynamic A). The key novelty claim: in the stablecoin era, the instability zone *shrinks* because mesh agents make the sunspot coordination problem easier to resolve (faster information → sharper threshold). Connor's math: accessible (optimization, simple game theory).

---

## Framework 6: Gabaix–Lasry–Lions–Moll (2016) — The Dynamics of Inequality

**Source:** Gabaix, X., J.-M. Lasry, P.-L. Lions, and B. Moll (2016). "The Dynamics of Inequality." *Econometrica* 84: 2071–2111.

**The Math:** Wealth evolves via random growth with mean-field interactions:

- Log wealth process: dx_t = μdt + σdW_t (canonical random growth)
- Stationary distribution: double Pareto with tail exponent ζ satisfying σ²ζ²/2 + μζ − δ = 0
- Kolmogorov Forward equation: ∂p/∂t = −∂(μp)/∂x + (σ²/2)∂²p/∂x²
- Key innovation: "scale dependence" — μ(x) varies with wealth level x
- Speed of transition: s₁ = (σ²/2)(ζ₊ − ζ₋) where ζ₊, ζ₋ are tail exponents
- Standard random growth generates transitions that are *too slow* — need scale dependence or type dependence to match data
- With heterogeneous growth types: "superstar" agents shift the tail faster

**Captures:** Dynamic G (wealth concentration dynamics)

**Feedback Prediction:** Concentration with heterogeneous speed. When a fraction of agents have higher μ (mean growth rate) — which is precisely what mesh-quality capital allocation provides — the wealth distribution's upper tail fattens. The speed of concentration depends on the *difference* between mesh-agent μ and non-mesh μ. If mesh access is democratized (like index funds), μ converges across agents; if concentrated (like early HFT), the tail fattens rapidly.

**Variable Mapping:**
- μ (mean wealth growth rate) → return on capital under different allocation technologies
- σ (wealth volatility) → investment risk
- μ_mesh − μ_human → the "allocation technology premium" from AI portfolio management
- ζ (Pareto tail exponent) → wealth concentration measure
- s₁ (speed of transition) → how fast inequality adjusts to the mesh era
- "Scale dependence" → mesh agents' returns may increase with scale (more data → better allocation)

**Breaks Down When:** Model is partial equilibrium — doesn't feed back to asset prices or growth rates. In Paper 5's coupled system, wealth concentration affects capital markets, which affect mesh growth, which affects allocation technology access. Also treats returns as exogenous random variables rather than endogenous to market structure.

**Assessment:** 7/10. Provides the formal machinery for the r > g mechanism at machine speed. The contribution to Paper 5: calibrate the speed of transition under two scenarios (democratized vs. concentrated mesh access) and show the resulting steady-state Gini coefficients. The "scale dependence" mechanism is the key connection — mesh agents may have scale-dependent returns. Connor's math: requires Laplace transforms and PDEs (at the edge of current preparation; operator methods would need some study).

---

## Framework 7: Duffie–Gârleanu–Pedersen (2005, 2007) — OTC Market Search and Disintermediation

**Source:** Duffie, D., N. Gârleanu, and L.H. Pedersen (2005). "Over-the-Counter Markets." *Econometrica* 73(6): 1815–1847.

**The Math:** Search-and-matching model of decentralized markets:

- Agents meet pairwise at Poisson rate λ (search intensity)
- Meet market makers at rate ρ
- Four agent types: {high, low} × {owner, non-owner}
- Population masses μ evolve: dμ/dt = (flow in) − (flow out), governed by matching rates
- Equilibrium price negotiated via Nash bargaining with search frictions
- Bid-ask spread: s = (1 − z)·Δ where z is investor bargaining power, Δ is gains from trade
- As λ → ∞ (frictionless search): spread → 0, prices converge to Walrasian
- Information percolation: fraction of informed agents grows as f(t) = 1 − e^(−λt) (exponential diffusion through encounters)

**Captures:** Dynamic E (endogenous financial infrastructure), Dynamic A (market microstructure transition)

**Feedback Prediction:** Smooth disintermediation. As mesh agents increase λ (search/matching speed) toward infinity, the OTC market converges to a frictionless exchange. Market makers' rents disappear. This is the formal model of how the mesh eliminates the need for traditional financial intermediaries. Transition is smooth (exponential convergence), not discontinuous.

**Variable Mapping:**
- λ (search intensity) → mesh agent's ability to find counterparties (approaches ∞ on programmable settlement)
- ρ (dealer contact rate) → relevance of traditional market makers
- s (bid-ask spread) → intermediation cost that mesh eliminates
- z (bargaining power) → shifts toward investors as outside options improve
- f(t) (information percolation) → speed at which prices incorporate information in mesh-era markets

**Breaks Down When:** The model treats search frictions as the only source of intermediation. Real intermediaries provide other services: credit evaluation, warehousing risk, regulatory compliance. Also, the model has exogenous asset supply; in Paper 5's setting, the assets themselves (stablecoins, tokenized securities) are endogenous.

**Assessment:** 7/10. Provides the formal disintermediation mechanism. The key contribution: as mesh agents make λ → ∞, the equilibrium *continuously* transitions from OTC to exchange-like, providing the smooth market structure change that other frameworks predict discontinuously. Connor's math: accessible (Poisson processes, ODEs, Nash bargaining).

---

## Framework 8: Piketty–Saez–Zucman / Jones (2015) — r − g and Wealth Dynamics

**Source:** Piketty, T. (2014). *Capital in the Twenty-First Century*. Formalized by Jones, C. (2015). "Pareto and Piketty: The Macroeconomics of Top Income and Wealth Inequality." *Journal of Economic Perspectives* 29(1): 29–46.

**The Math:** Individual wealth evolves:

- db_t = [w_t + r·b_t − c_t]dt where w = labor income, r = return, c = consumption
- With constant consumption fraction α: individual wealth grows at rate r − α
- Average wealth grows at rate g
- Relative wealth (b/average) grows at rate r − g − (α − s_avg)
- Steady-state wealth share of top percentile: increasing in r − g
- Pareto tail exponent: ζ determined by r − g and idiosyncratic return shocks
- With heterogeneous returns (Fagereng et al. 2020): top wealth holders earn r_top > r_avg

**Captures:** Dynamic G (wealth concentration dynamics with heterogeneous capital allocation technology)

**Feedback Prediction:** Positive loop for concentration if r_mesh > r_human. The mesh widens the r − g gap by: (1) continuous portfolio optimization → higher r, (2) zero intermediation friction → higher r, (3) full arbitrage across markets → higher r. If g remains bounded by physical/human productivity, the wedge grows. Whether this concentrates wealth depends on access: if r_mesh is available to all (democratized), it raises r uniformly and the distributional impact is neutral. If r_mesh accrues to mesh operators, concentration accelerates.

**Variable Mapping:**
- r → return on capital (now a function of allocation technology: r_human vs. r_mesh)
- g → GDP growth (bounded by Baumol bottleneck from Paper 4)
- r − g → the "wedge" that drives wealth concentration
- α (consumption rate) → unchanged by technology
- Heterogeneous r → mesh-quality returns vs. traditional returns

**Breaks Down When:** As noted by Krusell and Smith (2015), the savings rate s responds endogenously to r − g, potentially offsetting concentration. Also, r and g are interdependent in general equilibrium (higher capital accumulation → diminishing returns → lower r). The framework treats r as exogenous.

**Assessment:** 6/10. Conceptually central but formally weak — the r − g relationship is more of an accounting identity than a predictive model. Paper 5 needs to embed this in a general equilibrium framework (like Brunnermeier-Sannikov) where r is endogenous. Connor's math: accessible (basic calculus, no advanced tools needed).

---

## Framework 9: Calvo (1998) / Obstfeld (1996) — Speculative Attack Models and Currency Crises

**Source:** Calvo, G. (1998). "Capital Flows and Capital-Market Crises." *Journal of Applied Economics* 1(1): 35–54. Obstfeld, M. (1996). "Models of Currency Crises with Self-Fulfilling Features." *European Economic Review* 40: 1037–1047.

**The Math:** Three generations of currency crisis models:

- First generation (Krugman 1979): R_t = R₀ − D·t (reserves depleted by fiscal deficit D). Crisis at T* when R_{T*} = 0. Deterministic.
- Second generation (Obstfeld 1996): Government minimizes loss L = (π − π*)² + θ·(ε fix cost). Multiple equilibria when θ is in intermediate range. Sunspot triggers switch.
- Third generation (Chang and Velasco 2001): Bank runs + currency crises interact. Diamond-Dybvig mechanism in open economy.
- Key equation for stablecoin era: If s = stablecoin adoption rate, then reserves drain at rate D + f(s) where f is increasing and potentially convex in s (network effects)

**Captures:** Dynamic C (dollarization/currency collapse spiral), Dynamic B (monetary policy degradation)

**Feedback Prediction:** Threshold instability. The stablecoin-mediated crisis combines all three generations: (1) persistent fiscal deficits drain reserves (1st gen), (2) self-fulfilling expectations of devaluation can trigger runs (2nd gen), (3) banking sector fragility interacts with currency pressure (3rd gen). Stablecoins accelerate all three: reserves drain faster (capital exits permissionlessly), sunspot coordination is easier (information travels faster), and bank deposits flee to stablecoins.

**Variable Mapping:**
- R (reserves) → central bank reserves available to defend currency
- D (fiscal deficit) → reserve drain rate (now augmented by stablecoin-mediated capital flight)
- T* (crisis time) → accelerated by stablecoin infrastructure
- s (stablecoin adoption) → new state variable that feeds back to reserve dynamics
- θ (government commitment) → weakened by loss of monetary policy tools

**Breaks Down When:** Classical models assume a defending central bank. In the stablecoin era, the "attack" is passive — savers simply download an app. The game theory is fundamentally different from a coordinated speculative attack. Also, the models are mostly small-open-economy; the mesh is global.

**Assessment:** 7/10. Provides the formal crisis mechanics. Paper 5 contribution: extend the 2nd-generation model to include stablecoin adoption as a state variable that endogenously determines the multiple-equilibria region. Show that the "intermediate θ" region where self-fulfilling crises are possible *expands* with stablecoin availability. Connor's math: accessible (optimization, basic game theory).

---

## Framework 10: Caballero–Farhi–Gourinchas (2017) — The Safe Assets Shortage Conundrum

**Source:** Caballero, R.J., E. Farhi, and P.-O. Gourinchas (2017). "The Safe Assets Shortage Conundrum." *Journal of Economic Perspectives* 31(3): 29–46.

**The Math:** Global demand for safe assets exceeds supply:

- Demand for safety: D(r_safe) = D₀ − b·r_safe (decreasing in safe rate)
- Supply of safe assets: S = S_US + S_other (largely US Treasuries)
- Equilibrium: D(r*) = S determines the safe rate r*
- When D₀ grows faster than S: r* falls, potentially hitting ZLB
- At ZLB: excess demand → safety trap → output gap
- Risk premium: R_risky − r_safe = φ(shortage severity)
- The "safe asset" is defined by information-insensitivity (Gorton 2017)

**Captures:** Dynamic F (Triffin / stablecoin collateral demand)

**Feedback Prediction:** Stablecoins transform the safe asset equilibrium in two ways: (1) They massively increase D₀ (global demand for Treasury-backed safe assets from stablecoin reserves — current estimates: $1–3 trillion by 2030 per Fed Governor Miran's speech), which puts further downward pressure on r*. (2) They potentially increase S by tokenizing Treasury access. Net effect: r* declines, strengthening the US borrowing position but deepening the dependency.

**Variable Mapping:**
- D₀ (baseline safe asset demand) → augmented by stablecoin reserve requirements
- S (safe asset supply) → US Treasury issuance capacity (now with fiscal sustainability constraint from Dynamic F)
- r* (safe rate) → pushed down by stablecoin demand
- "Information insensitivity" → precisely what mesh agents destroy (they make everything information-sensitive)

**Breaks Down When:** The framework treats safe assets as exogenously defined. Mesh agents performing real-time sovereign risk pricing may *destroy* the information-insensitivity that makes assets "safe." This is the deepest contradiction: stablecoins need safe assets, but the mesh erodes the conditions that make assets safe.

**Assessment:** 8/10. Provides the formal safe-asset supply/demand framework. The key novelty: stablecoins are simultaneously the largest new source of safe-asset demand AND the mesh's agents are the greatest threat to safe-asset status. This contradiction should be the centerpiece of the Triffin analysis in Paper 5.

---

## Framework 11: Diamond–Dybvig (1983) / Gorton (2022) — Bank Runs and Stablecoin Fragility

**Source:** Diamond, D. and P. Dybvig (1983). "Bank Runs, Deposit Insurance, and Liquidity." *Journal of Political Economy* 91: 401–419. Extended by Gorton, G.B. et al. (2022). "Leverage and Stablecoin Pegs." NBER Working Paper.

**The Math:** Standard Diamond-Dybvig:

- Depositors are type-1 (early, probability π) or type-2 (late, probability 1−π)
- Bank invests in long-term asset: $1 at t=0 yields R > 1 at t=2, but only r₁ < 1 if liquidated at t=1
- Efficient allocation: c₁* > 1 (insurance for early types), c₂* = R(1−πc₁*)/(1−π) 
- Bank run equilibrium: if all withdraw early, bank liquidates everything, c₁ = r₁ < 1
- Two Nash equilibria: no-run (efficient) and run (inefficient)
- Gorton extension to stablecoins: stablecoin = demand deposit backed by reserves; levered traders compensate holders for foregone interest; run dynamics apply when reserve quality is questioned

**Captures:** Dynamic F (stablecoin fragility as Triffin manifestation)

**Feedback Prediction:** Fragility is endemic. Stablecoins are structurally isomorphic to demand deposits — they promise par redemption backed by assets that can lose value. A systemic stablecoin run forces simultaneous Treasury sales, depressing Treasury prices, which undermines other stablecoins' reserves — a self-reinforcing run. The mesh's dependence on stablecoins as settlement infrastructure means a stablecoin run is simultaneously a mesh infrastructure failure.

**Variable Mapping:**
- Bank deposits → stablecoin balances
- Long-term illiquid asset → Treasury bills (liquid individually, illiquid in mass liquidation)
- Deposit insurance (prevents runs) → absent for stablecoins (no FDIC equivalent)
- Type-1 agents → stablecoin holders needing immediate liquidity
- Run equilibrium → systemic stablecoin de-peg event

**Breaks Down When:** Smart contracts potentially eliminate the sequential-service constraint that drives runs in Diamond-Dybvig. If redemption is algorithmically managed (circuit breakers, pro-rata liquidation), run dynamics may be fundamentally different. Also, Treasury bills are far more liquid than the long-term assets in Diamond-Dybvig.

**Assessment:** 7/10. Provides the fragility analysis for the settlement layer. Paper 5 needs this to formalize the downside: the mesh's settlement infrastructure is vulnerable to exactly the kind of run that Diamond-Dybvig describes, and the mesh has no lender of last resort. Connor's math: fully accessible.

---

## Framework 12: Dou–Goldstein–Ji (2025) — AI-Powered Trading and Algorithmic Collusion

**Source:** Dou, W.W., D. Goldstein, and Y. Ji (2025). "AI-Powered Trading, Algorithmic Collusion, and Price Efficiency." NBER Working Paper 34054.

**The Math:** Extension of Kyle (1985) with reinforcement-learning informed traders:

- N informed AI speculators observe signals s_i,t about asset value v_t
- Each uses Q-learning: Q_{t+1}(s,a) = (1−α)Q_t(s,a) + α·[r + γ·max_{a'}Q_t(s',a')]
- Nash equilibrium with "price-trigger strategies" (collusive regime + punishment regime)
- Collusion condition: discount factor δ ≥ δ* threshold
- Key result: AI agents autonomously learn to sustain supra-competitive profits without communication
- Market efficiency degradation: prices reveal less information under collusion

**Captures:** Dynamic A (market efficiency as function of participant composition)

**Feedback Prediction:** Ambiguous for efficiency. AI agents may *reduce* market efficiency through algorithmic collusion, contradicting the naive prediction that more AI = more efficiency. This creates a non-monotone relationship: some AI participation improves efficiency, but high AI concentration may degrade it through strategic coordination. This is a novel prediction that links market structure to competitive dynamics.

**Variable Mapping:**
- N (number of AI speculators) → mesh agent count in capital markets
- α (learning rate) → mesh agent adaptation speed
- δ* (collusion threshold) → minimum patience for sustained coordination
- Price efficiency measures → calibrated against mesh participation fraction

**Breaks Down When:** Model uses simple Q-learning, not the kind of sophisticated agents the mesh would deploy. The collusion result depends on repeated game structure with discrete periods; continuous-time mesh operation may have different properties. Also assumes homogeneous AI agents; the mesh is heterogeneous by design.

**Assessment:** 6/10. Important cautionary result — the assumption that more AI always improves market efficiency is wrong. Paper 5 should acknowledge this as a constraint on the "market efficiency improvement" narrative. However, the model is too stylized for direct adoption. Connor's math: Q-learning formalism is accessible; game theory prerequisite satisfied.

---

## Framework 13: Brian Arthur (1994) — Increasing Returns, Path Dependence, and Lock-In

**Source:** Arthur, W.B. (1994). *Increasing Returns and Path Dependence in the Economy*. University of Michigan Press. Formalized in Arthur (1989). "Competing Technologies, Increasing Returns, and Lock-In by Historical Events." *Economic Journal* 99: 116–131.

**The Math:** Urn model of technology adoption:

- Two technologies A and B. After n adoptions, fraction x_n = n_A/n
- Adoption probability: P(choose A) = f(x_n) where f is increasing (network effects)
- Multiple equilibria: x = 0, x = 1, and possibly interior fixed points
- Dynamics: x_n is a Pólya urn process; converges to a random limit
- With heterogeneous agents: if type-R agents prefer A and type-S prefer B, the winner depends on the sequence of arrivals
- Lock-in: once one technology gains sufficient lead, switching costs prevent reversal

**Captures:** Dynamic C (currency competition), Dynamic D (technology-institution co-evolution)

**Feedback Prediction:** Path-dependent lock-in. Applied to currency competition: once stablecoins gain sufficient adoption in a region, network effects prevent reversion to local currency. The sequence of adoption matters: early stablecoin penetration in a few countries can trigger cascading adoption globally. This generates the same hysteresis as Uribe (1997) but from a different mechanism (network effects rather than learning-by-doing).

**Variable Mapping:**
- Technology A → stablecoin (dollar-denominated programmable money)
- Technology B → local fiat currency
- f(x_n) → adoption probability as function of stablecoin network size
- Lock-in → dollarization irreversibility
- Heterogeneous agents → countries at different stages of the MPG paper's six-stage classification

**Breaks Down When:** Arthur's model treats technologies as substitutes. Stablecoins and local currency may coexist (dual-currency systems). Also, the model has no central authority that can change the rules — governments can ban stablecoins or impose capital controls, changing f.

**Assessment:** 6/10. Provides the path-dependence framework for currency competition. More conceptual than quantitative. The Uribe (1997) model is more directly applicable. Main contribution to Paper 5: justifies why stablecoin adoption is likely irreversible once thresholds are crossed, supporting the "dollarization spiral" mechanism.

---

## Framework 14: Ahmed–Aldasoro (2025) — Stablecoins and Safe Asset Prices

**Source:** Ahmed, R. and I. Aldasoro (2025). "Stablecoins and Safe Asset Prices." BIS Working Paper 1270.

**The Math:** Instrumented local projections of stablecoin flows on Treasury yields:

- y_{t+h} − y_t = α_h + β_h·ΔSC_t + γ_h·X_t + ε_{t+h} (local projection specification)
- Instrument: crypto market shocks orthogonal to macro fundamentals (BCGI index purged of financial correlations)
- Key finding: 2-standard-deviation stablecoin inflow lowers 3-month T-bill yields by 2–2.5 basis points within 10 days
- Asymmetric effects: outflows raise yields 2–3× as much as inflows lower them
- Limited spillover to longer tenors (preferred habitat story)
- Tether contributes most, followed by USDC (consistent with relative size)

**Captures:** Dynamic F (stablecoin demand for Treasuries — empirical validation)

**Feedback Prediction:** Quantifies the forward link. Stablecoin growth measurably lowers short-term Treasury yields, confirming the mechanism in Frameworks 5 and 10. The asymmetry (outflows hurt more) creates fragility: a stablecoin crisis would spike yields disproportionately. This provides empirical calibration targets for Paper 5's theoretical model.

**Variable Mapping:**
- ΔSC_t → stablecoin market cap changes (proxy for Treasury demand flows)
- y_{t+h} → Treasury yields (3-month T-bill)
- β_h → elasticity of yields to stablecoin flows (the parameter Paper 5 needs to calibrate)
- Asymmetry → stablecoin fragility creates nonlinear yield risk

**Breaks Down When:** Short sample (2021–2025), small stablecoin market relative to Treasury market. Effects may not scale linearly as stablecoin market grows by 10× (Fed projects $1–3T by 2030). Also, reduced-form — no structural model of the mechanism.

**Assessment:** 7/10. Provides the empirical grounding for the Treasury-demand channel. Paper 5 should cite this for calibration but needs a structural model for the feedback loop. Connor's math: local projections are standard econometrics (accessible).

---

## Framework 15: Moll–Achdou–Han–Lasry–Lions (2022) — Continuous-Time Heterogeneous Agent Models

**Source:** Achdou, Y., J. Han, J.-M. Lasry, P.-L. Lions, and B. Moll (2022). "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *Review of Economic Studies* 89(1): 45–86.

**The Math:** Coupled Hamilton-Jacobi-Bellman (HJB) and Kolmogorov Forward (KF) system:

- HJB (individual optimization): ρv(a,z) = max_c {u(c) + v_a·(ra + z − c) + (1/2)σ²v_aa + ...}
- KF (distribution evolution): ∂g/∂t = −∂(s(a,z)·g)/∂a + (1/2)σ²·∂²g/∂a² + ...
- Market clearing: ∫a·g(a,z)da·dz = K (aggregate capital)
- The HJB–KF system is "transpose": once HJB is solved, KF follows mechanically
- Finite difference methods on the HJB yield sparse matrix that transposes directly to KF
- Mean Field Game structure: individual optimizes given distribution, distribution emerges from individual optimization

**Captures:** Dynamic G (wealth distribution dynamics), Dynamic D (technology-institution co-evolution)

**Feedback Prediction:** Provides the computational architecture for Paper 5's general equilibrium. The HJB–KF framework can accommodate heterogeneous agents with different allocation technologies (mesh vs. non-mesh), endogenous asset returns, and distributional dynamics. This is the natural framework for embedding all the other mechanisms.

**Variable Mapping:**
- Agent state (a, z) → (wealth, allocation technology type: mesh or human)
- s(a,z) (saving policy) → endogenous saving behavior under different return profiles
- g(a,z) (distribution) → wealth distribution across technology types
- Market clearing → determines endogenous r that feeds back to all agents

**Breaks Down When:** Computationally demanding for multi-state problems. The mesh–financial system has many interacting state variables (mesh participation, stablecoin adoption, monetary policy stance, sovereign debt). May need dimensional reduction.

**Assessment:** 8/10 (as computational framework). This is not a stand-alone theory but the *toolkit* for implementing Paper 5's model. Moll's codes are publicly available at benjaminmoll.com/codes/. Connor's math: requires PDEs (current preparation includes Diff Eq; would need to extend to numerical PDE methods — feasible with guided study).

---

## Framework 16: Lucas Critique Applied to Monetary Policy Transmission

**Source:** Lucas, R.E. (1976). "Econometric Policy Evaluation: A Critique." In *Carnegie-Rochester Conference Series on Public Policy* 1: 19–46. Extended by Woodford, M. (2003). *Interest and Prices*.

**The Math:** The core insight formalized: policy parameters enter agents' decision rules. When agents change (human → mesh), decision rules change, and policy transmission changes.

- New Keynesian Phillips Curve: π_t = β·E_t[π_{t+1}] + κ·x_t
- Forward guidance works because E_t[π_{t+1}] responds to central bank communication with a lag (human processing delay)
- If agents update E_t[π_{t+1}] instantaneously (mesh agents): forward guidance has zero duration of effect
- QE transmission: central bank buys bonds → portfolio balance effect → requires *slow* arbitrage to work
- If arbitrage is instantaneous (mesh agents): QE has zero portfolio balance effect; only signaling channel remains

**Captures:** Dynamic B (monetary policy effectiveness degradation)

**Feedback Prediction:** Monotonic degradation. Each monetary policy tool depends on specific frictions that mesh agents eliminate: forward guidance needs processing delays, QE needs slow arbitrage, financial repression needs captive savers. As mesh participation increases, each tool's effectiveness declines monotonically. No phase transition — just gradual erosion.

**Variable Mapping:**
- Agent expectation formation speed → mesh information processing rate
- Portfolio rebalancing frequency → continuous (mesh) vs. periodic (human)
- Arbitrage speed → mesh approaches instantaneous
- Each policy tool → maps to a specific friction that the mesh eliminates

**Breaks Down When:** Not all monetary transmission works through frictions. The interest rate channel (lowering rates makes borrowing cheaper) works regardless of market efficiency. The model overstates the loss of policy tools.

**Assessment:** 7/10. The Lucas critique is the *meta-framework* — it says that any model of monetary policy must account for the changing nature of participants. Paper 5's contribution: systematically identify which frictions each policy tool depends on, and model each friction as a function of mesh participation. Connor's math: fully accessible (expectation operators, basic macro).

---

## SYNTHESIS

### Which Frameworks Compose?

The frameworks form a natural layered structure:

**Layer 1 — Market Microstructure Transition (Frameworks 1, 2, 7, 12):** Grossman-Stiglitz provides the efficiency paradox, Kyle provides the microstructure mechanics, Duffie provides the disintermediation dynamics, Dou-Goldstein-Ji provides the collusion caution. These compose to describe how market properties change as mesh agents become dominant participants. Key combined prediction: market efficiency improves non-monotonically with possible "too-efficient-to-function" limit.

**Layer 2 — Monetary Policy Degradation (Frameworks 3, 16):** Brunnermeier-Sannikov provides the monetary macro framework; Lucas critique provides the meta-principle. These compose to show that monetary policy tools degrade as the market structure changes. Key combined prediction: the volatility paradox persists — even as exogenous risk falls, endogenous instability may increase because monetary policy tools lose effectiveness.

**Layer 3 — Dollarization Dynamics (Frameworks 4, 9, 13):** Uribe provides the hysteresis mechanism, Calvo/Obstfeld provides the crisis dynamics, Arthur provides the path dependence. These compose to describe the stablecoin-mediated dollarization spiral. Key combined prediction: there exists a critical "Fiat Quality Index" below which stablecoin access triggers irreversible dollarization, and this threshold is lower (and the dynamics faster) than historical dollarization episodes.

**Layer 4 — Reserve Currency / Triffin (Frameworks 5, 10, 11, 14):** Farhi-Maggiori provides the Triffin formalization, Caballero-Farhi-Gourinchas provides the safe asset demand framework, Diamond-Dybvig/Gorton provides the stablecoin fragility, Ahmed-Aldasoro provides empirical calibration. Key combined prediction: stablecoins create a contradiction — they increase demand for Treasuries while simultaneously (via the mesh) making Treasury risk pricing more transparent, shrinking the "instability zone" margin.

**Layer 5 — Wealth Distribution (Frameworks 6, 8, 15):** Gabaix-Lasry-Lions-Moll provides the inequality dynamics, Piketty/Jones provides the r−g framework, Achdou-Han-Lasry-Lions-Moll provides the computational architecture. Key combined prediction: the r−g wedge widens but whether this concentrates wealth depends on the accessibility of mesh-quality allocation technology.

### What Does the Combined Structure Predict?

**The coupled system has a conditionally stable equilibrium with the following properties:**

1. **Market efficiency improves but hits a Grossman-Stiglitz ceiling.** Perfect efficiency is impossible because of the information paradox. The mesh pushes efficiency close to but not through the ceiling. The remaining inefficiency is the equilibrium "noise" that sustains market function.

2. **Monetary policy tools degrade gradually, not discontinuously.** Each tool depends on specific frictions; each friction is eroded at a rate proportional to mesh participation. Forward guidance degrades first (depends on information processing delay), QE next (depends on arbitrage speed), financial repression last (depends on capital controls, which are institutional rather than informational).

3. **The dollarization spiral has a threshold and is irreversible.** Below a critical Fiat Quality Index (estimated around 15-20% inflation in the pre-stablecoin era, likely lower with stablecoin access), currency substitution becomes self-reinforcing. Each currency collapse expands the stablecoin ecosystem, which is the mesh's settlement layer — creating a positive feedback loop for mesh growth.

4. **The Triffin contradiction is the system's fundamental instability.** Stablecoin growth requires safe assets (Treasuries). The mesh's agents make safe-asset status harder to maintain. This is a slowly building tension that the system resolves either through fiscal adjustment, through the emergence of alternative safe assets (e.g., diversified stablecoin backing), or through a crisis that reprices the entire system.

5. **Wealth concentration accelerates during the transition but may stabilize.** The r−g wedge widens during the period when mesh-quality allocation is concentrated. If democratized (the "index fund" scenario from the mesh paper), the steady-state Gini converges to a new value that is not dramatically different from today's. If concentrated, inequality grows without bound until political intervention.

### What Are the Gaps?

1. **No existing model couples all five layers.** Each framework treats the others as exogenous. Paper 5's core contribution must be the *coupling*.

2. **The transition path is unknown.** Existing frameworks characterize steady states but not the dynamics of moving from the current (human-dominated) equilibrium to the new (mesh-dominated) one. Is the transition smooth, or does the system pass through an unstable intermediate regime?

3. **The speed mismatch is unmodeled.** The mesh operates at machine speed; institutions (central banks, regulators, legislatures) operate at human speed. No framework captures the consequences of this speed differential formally.

4. **Stablecoin-specific models are nascent.** The Diamond-Dybvig application to stablecoins (Gorton 2022) is promising but young. There is no accepted general equilibrium model with stablecoins.

### What Is the Minimal Formal Model for Paper 5?

A Brunnermeier-Sannikov (2014) style continuous-time model with the following modifications:

1. **Two agent types:** mesh-agents (fraction φ, growing endogenously) and human-agents (fraction 1−φ)
2. **Endogenous market efficiency:** Kyle's λ as a function of φ
3. **Monetary policy effectiveness:** policy tools parametrized by φ-dependent frictions
4. **Settlement layer:** stablecoin ecosystem size S as a state variable, growing with mesh demand
5. **Dollarization dynamics:** country-level currency quality as a state variable, with stablecoin-mediated switching
6. **Safe asset constraint:** US fiscal position as a state variable, with stablecoin demand and mesh pricing both affecting it

The minimal model might focus on the *coupled ODE system* in (φ, S, η) where:
- φ (mesh participation) evolves based on technology learning curves (from Paper 1)
- S (stablecoin ecosystem size) evolves based on mesh demand and dollarization dynamics
- η (financial sector capitalization) evolves per Brunnermeier-Sannikov, but with φ-dependent parameters

### Is This One Paper or Two?

**Recommendation: Two papers.**

- **Paper 5a:** The market efficiency + monetary policy degradation story (Layers 1 + 2). Central question: what happens to market structure and monetary policy when the marginal participant changes from human to mesh? Backbone: Brunnermeier-Sannikov with Kyle microstructure, parametrized by mesh participation φ. This is the "internal" dynamics story.

- **Paper 5b:** The dollarization + Triffin story (Layers 3 + 4). Central question: what happens to the international monetary system when stablecoins become the dominant settlement infrastructure? Backbone: Farhi-Maggiori extended with stablecoin dynamics and Uribe-style dollarization. This is the "external" dynamics story.

The wealth distribution story (Layer 5) can be a section in either paper, or a brief standalone analysis, depending on how much it interacts with the other layers.

**The coupling between 5a and 5b** is the key novelty claim: market efficiency improvement (5a) degrades monetary policy tools, which accelerates dollarization (5b), which grows the stablecoin ecosystem, which feeds back to market efficiency (5a). This circular structure mirrors the R₀ logic from Paper 1.

### Key Novelty Claim

What Paper 5 (or 5a/5b) proves that existing literature does NOT:

**Existing literature treats market structure, monetary policy, dollarization, and safe-asset dynamics as separate problems.** Paper 5 shows they are coupled through the settlement layer: the mesh needs programmable money → stablecoins grow → dollarization accelerates → monetary policy degrades → capital markets restructure → mesh grows faster. The formal demonstration that this feedback loop has R₀ > 1 (is self-reinforcing) under specific conditions — and the identification of those conditions — is the novel contribution.

### Prerequisites Beyond Connor's Current Math

- **Stochastic calculus:** Required for Brunnermeier-Sannikov. Itô's lemma, SDEs, Fokker-Planck/Kolmogorov Forward. Can be learned efficiently with "Shreve, Stochastic Calculus for Finance II" or Moll's lecture notes.
- **Numerical PDE methods:** Required for computing equilibria. Finite difference methods for HJB equations. Moll's publicly available codes provide templates.
- **Basic game theory:** Already covered. Useful for Farhi-Maggiori and speculative attack models.
- **Laplace transforms:** Required if using Gabaix-Lasry-Lions-Moll inequality framework. Can be deferred if wealth distribution is secondary.

**Estimated preparation time:** 4-6 weeks of focused study on stochastic calculus and numerical methods, using the existing Diff Eq and Linear Algebra foundation.
