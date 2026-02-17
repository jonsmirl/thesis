# ENDOGENOUS DECENTRALIZATION

*How Concentrated Capital Investment Finances the Learning Curves That Enable Distributed Alternatives*

Connor Smirl

*Department of Economics, Tufts University*

February 2026

**Working Paper**

Data repository: github.com/[forthcoming]

---

## Abstract

This paper identifies and formalizes endogenous decentralization: a mechanism by which concentrated capital investment in centralized infrastructure finances the learning curves that enable distributed alternatives. The mechanism's distinctive property is ∂T*/∂I < 0: increased centralized investment accelerates the crossing time at which distributed architectures become cost-competitive. Unlike Arrow's (1962) learning-by-doing, where cost reduction benefits the same production paradigm, endogenous decentralization produces architectural substitution—the learning investments finance a different organizational form.

Five contributions are new. First, I formalize the mechanism as a continuous-time differential game in which the distance to the crossing point is a common-pool state variable depleted by cumulative production. Competing centralized firms, each maximizing individual rents in symmetric Markov Perfect Equilibrium, produce aggregate output that strictly exceeds the cooperative optimum at every interior state, accelerating T* beyond what any firm would individually prefer (Proposition 1). The overinvestment decomposes into a Cournot channel and a learning externality channel unique to this setting. Three comparative statics show the result is robust to increasing competition, cost asymmetry, and capacity constraints. Both Nash and cooperative value functions admit exact closed-form solutions, verified to machine precision.

Second, the pure cost-parity crossing condition generalizes to a self-sustaining adoption threshold: the distributed ecosystem's basic reproduction number R₀ must exceed unity, incorporating network effects, coordination friction, and latency advantage alongside cost advantage. This R₀ threshold maps to a modified cumulative production threshold Q̄* that preserves the model's one-state-variable structure and all closed-form solutions, while endogenizing the 3–5 year coordination layer lag observed in historical transitions.

Third, the model incorporates a structural distinction between training and inference workloads. Training—the creation of frontier AI models—requires tightly synchronized GPU clusters at scales that remain architecturally incompatible with distributed execution. Inference—the deployment of trained models to serve user requests—is atomizable, latency-sensitive, and follows precisely the decentralization trajectory the model predicts. The mechanism's primary empirical domain is inference, which constitutes an estimated 80–90% of AI compute cycles and is projected to represent a $255 billion market by 2030.

Fourth, cross-domain empirical analysis reveals that the effective crossing threshold is being approached from two directions simultaneously: hardware cost decline from below along semiconductor learning curves (α ∈ [0.18, 0.25] across DRAM, HBM, microprocessors, NAND, and solar PV) and a reduction in effective hardware requirements from above through algorithmic efficiency gains—quantization, mixture-of-experts architectures, and distillation. The algorithmic path is driven primarily by an ecosystem of open-weight model developers operating under binding compute constraints imposed by semiconductor export controls, who have a structural incentive to minimize hardware requirements per unit of inference capability. This dual convergence accelerates crossing beyond what hardware learning curves alone predict, and the export-control setting provides a natural experiment: exogenous constraints on centralized scale predict a shift to efficiency-maximizing strategies, which is what the data show.

Fifth, the model generates nine falsifiable predictions with specific timing and failure conditions for the current AI infrastructure buildout ($1.3 trillion cumulative hyperscaler capex, 2018–2025). Calibrated to HBM learning rates, observed capex, and the dual convergence of hardware and software efficiency, the model predicted inference crossing by approximately 2027–2028 under Nash competition—a 79% acceleration relative to the cooperative benchmark. As of Q1 2026, the technology threshold has been met at professional price points, but the 2025–26 DRAM supercycle—a demand shock driven by the same concentrated investment the model identifies as Stage 1—has temporarily reversed consumer memory cost trends, inflating GPU prices 50–150% above MSRP and delaying the cost threshold by an estimated 1–2 years. This boom-phase deviation is predicted by Corollary 3 (capacity constraint and boom-bust) and does not alter the long-run learning elasticity (α = 0.22, stable across four prior DRAM supercycles over 41 years). The capacity expansion the supercycle has triggered is expected to produce overcapacity and below-trend consumer pricing by 2028–2029, potentially accelerating crossing beyond the pre-boom projection.

**Keywords:** endogenous decentralization, learning curves, Markov Perfect Equilibrium, architectural substitution, AI infrastructure, training-inference bifurcation, open-weight models, basic reproduction number

**JEL:** O33, L16, D43, C73

---

# 1. Introduction

Between 2018 and 2025, the five largest US technology companies—together with Oracle and the Stargate joint venture—committed an estimated $1.3 trillion in cumulative capital expenditure to construct centralized AI infrastructure. This represents the largest concentrated infrastructure investment in history outside wartime mobilization. The near-term revenue objective is to sell AI inference—running trained models to serve user requests—as a cloud service at premium margins. A second, longer-horizon objective is frontier model training at scales that may produce discontinuous capability advances. The mechanism identified in this paper applies to the inference objective; the training objective is addressed as an alternative specification of the firms' objective function (Section 3.1).

This paper argues that this investment is *endogenously self-disrupting*: the very act of building centralized AI datacenters finances the component learning curves—particularly in high-bandwidth memory, model compression, and advanced packaging—that enable distributed alternatives to replicate datacenter-class inference on consumer hardware. As of Q1 2026, the technology threshold for interactive 70B-class inference has been met at professional and enthusiast price points. Paradoxically, the same concentrated investment has also triggered the most severe DRAM supercycle in two decades, temporarily reversing consumer memory cost trends and inflating GPU prices far above MSRP—a boom-phase deviation that the model's capacity-constraint corollary predicts will resolve into overcapacity and below-trend pricing as new fab capacity ramps. The remaining constraint is price migration from professional to mass-market form factors—a market structure transition compounded by, but not permanently altered by, the current supply shock. The mechanism is not novel in observation—economists studying the mainframe-to-PC transition have noted analogous dynamics—but it has not been previously formalized as a general theoretical result with testable predictions.

Two structural features of the current AI landscape sharpen the mechanism beyond what prior transitions exhibited.

First, AI workloads bifurcate into *training* (creating models via massive synchronized GPU clusters) and *inference* (running models to serve user requests on independent, atomizable tasks). The endogenous decentralization mechanism applies directly and powerfully to inference, which already constitutes 80–90% of AI compute cycles. Training may remain permanently centralized—not because learning curves fail to reduce its costs, but because the synchronization and bandwidth requirements are architectural constraints that cost reduction alone cannot address. The post-crossing equilibrium is partial decentralization: inference distributes while training persists centrally. Historical precedent confirms this pattern: IBM's mainframe business continues to generate $3–4 billion annually, decades after the PC revolution, serving workloads where centralized architecture retains genuine advantages.

Second, the effective crossing threshold is being approached from two directions simultaneously. From below, semiconductor learning curves reduce hardware costs along the trajectory this paper models (α = 0.22–0.23). From above, algorithmic efficiency gains—mixture-of-experts architectures, aggressive quantization, and distillation—reduce the effective hardware requirement for a given inference capability level. These software-side gains are driven primarily by open-weight model developers operating under binding compute constraints: US semiconductor export controls deny these firms access to frontier datacenter GPUs, creating a structural incentive to maximize inference capability per unit of available hardware. The result is a dual convergence in which cumulative hardware production Q(t) rises toward the crossing threshold while the threshold itself Q̄_eff(t) falls. The formal model captures both paths.

The contribution is five-fold. First, the formal mechanism: a continuous-time differential game with exact closed-form solutions, in which symmetric Markov Perfect Equilibrium produces overinvestment that accelerates crossing time T* (Proposition 1), extended to incorporate asymmetric crossing valuations (Corollary 2*). Second, a generalized crossing condition: the cost-parity threshold extends to a self-sustaining adoption threshold R₀ > 1 that incorporates network effects, coordination friction, and latency advantage, endogenizing the empirically observed coordination layer lag. Third, the training-inference bifurcation and its implications for post-crossing continuation values. Fourth, dual-convergence empirical evidence: hardware learning curves from below and algorithmic efficiency from above. Fifth, nine falsifiable predictions with timing for the current AI infrastructure cycle.

The paper is organized as follows. Section 2 develops the mechanism. Section 3 presents the formal model. Section 4 establishes the training-inference structural distinction. Section 5 presents the empirical evidence, organized around the dual convergence of hardware cost decline and algorithmic efficiency gains. Section 6 validates parameter consistency across historical transitions. Section 7 offers predictions. Section 8 concludes.

---

# 2. The Endogenous Decentralization Mechanism

## 2.1 Three-Stage Structure

**Stage 1: Centralized Investment.** Firms with market power invest I(t) in centralized infrastructure to capture scale economies, producing cumulative component production Q(t). The investment is motivated by standard rent-seeking—no new theory is required.

**Stage 2: Component Cost Decline.** Cumulative production drives unit costs along Wright's (1936) learning curve:

> c(Q) = c₀ · Q^(–α)     (1)

where α is the learning elasticity. The critical property is that α is a *technology* parameter, not a *firm* parameter: learning embodied in manufacturing process improvements transfers across applications.

**Stage 3: Architectural Recombination.** When component costs cross a threshold c*, the same components can be recombined into distributed architectures exhibiting network externalities. Beyond a crossing time T*, the distributed paradigm dominates for workloads amenable to distributed execution. The key qualifier is "amenable": workloads requiring massive synchronization across thousands of processors (such as frontier model training) are not candidates for architectural recombination regardless of component cost. Workloads that are atomizable and latency-sensitive (such as inference) are the primary candidates.

## 2.2 The Self-Undermining Investment Property

The mechanism's distinctive feature is that each stage causally enables the next, and the final stage undermines the first. Define T* as the first date at which distributed architecture cost-performance matches centralized provision for the marginal inference user. Then:

> ∂T*/∂I < 0     (2)

Increased centralized investment accelerates displacement of the centralized paradigm's inference revenue. The self-undermining property applies to the inference revenue pool; if centralized firms retain training capabilities and develop model-licensing businesses, the post-crossing continuation value is materially higher than under full displacement (Section 3.7). However, as Section 5.2 documents, the open proliferation of near-frontier models may erode this training moat, pushing the system toward the pure displacement case.

## 2.3 Dual Convergence

The current AI transition exhibits a feature absent from prior technological transitions: the effective crossing threshold is being approached from two directions simultaneously.

**From below: hardware cost decline.** Semiconductor learning curves reduce the cost of memory bandwidth, compute, and packaging along the trajectory c(Q) = c₀ · Q^(–α). This is the classical mechanism and the focus of the formal model.

**From above: algorithmic efficiency gains.** Advances in model architecture and compression reduce the hardware *required* to achieve a given inference capability level. Mixture-of-experts (MoE) architectures activate only a fraction of total parameters per token, reducing effective memory bandwidth requirements by 3–6×. Quantization (INT4, INT2) reduces model memory footprint by 4–16×. Distillation transfers capability from large models to smaller ones. These techniques reduce the effective crossing threshold—equivalently, they reduce the cumulative production Q̄ required for crossing.

Define Q̄_eff(t) = Q̄ · f(η(t)), where η(t) indexes cumulative algorithmic efficiency gains and f is decreasing. The state variable becomes x(t) = Q̄_eff(η(t)) – Q(t), and the rate of depletion exceeds what hardware learning curves alone would predict. Under the quasi-static approximation (Appendix C), if algorithmic efficiency evolves slowly relative to quarterly production decisions, the model's propositions apply pointwise.

The two convergence paths are driven by different actors with different incentive structures. Hardware cost decline is driven by centralized firms' cumulative production (the endogenous mechanism). Algorithmic efficiency is driven primarily by open-weight model developers facing binding compute constraints, who have a structural incentive to minimize hardware requirements. This incentive structure is examined empirically in Section 5.2.

## 2.4 Distinction from Adjacent Theory

Table 1 summarizes the positioning. The distinctions are precise: Arrow's (1962) learning-by-doing benefits the same paradigm; Bresnahan and Trajtenberg's (1995) GPT spillovers enable applications across sectors rather than architectural self-replacement; Schumpeter's (1942) creative destruction comes from external entrants.

Several adjacent frameworks describe facets of a single dynamic that endogenous decentralization unifies. David's (1990) diffusion lag corresponds to the coordination layer maturation period. Christensen's (1997) displacement is financed by the incumbent's own learning-curve investment. Bresnahan and Greenstein's (1994) adjustment cost is the gap between hardware crossing and coordination-layer readiness. Thompson and Spanuth's (2023) GPT specialization is Stage 1 generating the learning curves that enable Stage 3.

**Table 1. Theoretical positioning of endogenous decentralization.**

| Framework | Learning Scope | Beneficiary | Disruption Source | Self-Undermining? |
|:---|:---|:---|:---|:---|
| Arrow (1962) | Same paradigm | Same firms | N/A | No |
| Bresnahan-Trajtenberg (1995) | Cross-sector | Other sectors | External applications | No |
| Schumpeter (1942) | External | Entrant firms | External entrant | No |
| Christensen (1997) | Cross-market | Entrant firms | New value network | Partial |
| **This paper** | **Cross-paradigm** | **Different architecture** | **Self-financed** | **Yes (∂T*/∂I<0)** |

---

# 3. Formal Model

## 3.1 Environment

Consider N ≥ 2 symmetric centralized firms indexed by i ∈ {1,...,N}. Time is continuous. The *state variable* is x(t) = Q̄_eff – Q(t) ∈ [0, x₀], measuring the remaining cumulative production until the effective crossing threshold at which distributed architecture becomes cost-competitive for inference workloads. When x reaches zero, inference crossing occurs. The state evolves as:

> dx/dt = –Σᵢ qᵢ(t)     (3)

where qᵢ(t) ≥ 0 is firm i's output rate. Each unit of output serves the centralized market and simultaneously depletes the remaining distance to crossing—this dual role is the formal expression of the self-undermining investment property.

Flow profits for firm i are determined by linear inverse demand P = a – bQ, where Q = Σqⱼ is total output rate:

> πᵢ(t) = (a – bQ)qᵢ     (4)

with a > 0, b > 0. Upon crossing (x = 0), each firm receives continuation value:

> S = S_T + S_I/(N(r + δ))     (5)

where S_T represents the persistent training and model-licensing revenue that survives inference decentralization, S_I = π̄_I is the pre-crossing inference profit level, r is the discount rate, and δ > 0 is the post-crossing inference displacement rate. The decomposition reflects the training-inference structural distinction (Section 4): training revenue S_T does not decay with inference displacement, while inference rents S_I decay at rate δ.

When S_T = 0, the model reduces to the pure displacement case. When S_T > 0, incumbents retain value post-crossing, moderating the overinvestment externality. Section 3.7 discusses the conditions under which S_T erodes toward zero.

**Remark on the objective function.** The model assumes firms maximize discounted revenue from infrastructure services. An alternative specification treats centralized investment as purchasing an option on a discontinuous payoff: the first firm to achieve a capability threshold (e.g., artificial general intelligence) captures a prize V* that dwarfs cumulative investment. Under this specification, S_T is not persistent training *revenue* but the option value of retaining frontier training *capability*—the infrastructure required to attempt training runs at scales no distributed architecture can replicate. The mechanism's core result (∂T*/∂I < 0) is invariant to the firms' objective: regardless of whether firms invest to capture inference revenue or to win a capability race, cumulative production drives the same learning curves and inference crossing occurs at the same threshold. What changes is the welfare interpretation and the magnitude of investment. Under the option-value specification, equilibrium investment may exceed the Nash revenue-maximization prediction, since the marginal value of being first to the capability threshold can be arbitrarily large. The revenue-maximization model thus provides a *lower bound* on aggregate investment and a correspondingly conservative estimate of T*. Section 3.7 calibrates both specifications.

The game has a *common-pool* structure: the state x is a shared resource (remaining time before inference disruption) that all firms deplete through production. This structure is analogous to the fishery or oil extraction commons (Levhari and Mirman 1980), with the critical distinction that the "resource" being depleted is the incumbent paradigm's remaining inference viability.

## 3.2 Markov Perfect Equilibrium

I restrict attention to symmetric stationary Markov strategies qᵢ = q(x), where output depends only on the current state. Each firm's value function V(x) satisfies the Hamilton-Jacobi-Bellman equation:

> rV(x) = max_{qᵢ} {(a – b(qᵢ + (N–1)q(x)))qᵢ – V′(x)·(qᵢ + (N–1)q(x))}     (6)

The first-order condition under symmetry is:

> a – b(N+1)q – V′(x) = 0     (7)

yielding the equilibrium strategy:

> qᴺ(x) = [a – Vᴺ′(x)] / [b(N+1)]     (8)

Substituting back into the HJB yields the ODE:

> rVᴺ(x) = (a – Vᴺ′(x))(a – N²Vᴺ′(x)) / [b(N+1)²]     (ODE-N)

with boundary condition Vᴺ(0) = S.

## 3.3 Cooperative Benchmark

The cooperative planner maximizes total producer surplus W(x) = NVᴾ(x), choosing total output rate Q. This is a cartel benchmark—it maximizes incumbent welfare, not total social welfare including consumers:

> rVᴾ(x) = (a – NVᴾ′(x))² / (4bN)     (ODE-C)

with boundary condition Vᴾ(0) = S.

## 3.4 Analytical Solutions

Both ODEs are autonomous and separable. The cooperative ODE yields the exact implicit solution:

> x(V) = [a·ln((a – 2√(bnrS))/(a – 2√(bnrV))) + 2(√(bnrS) – √(bnrV))] / (2br)     (C-exact)

The Nash ODE is solved by the substitution u = √(D + EV):

> x(V) = (4N²/E)·[(u₀ – u) + A·ln((A – u₀)/(A – u))]     (N-exact)

Both solutions share the same functional form—√ + log—differing only in the constants governing shadow cost internalization. Both are implicit in V but invertible numerically, yielding exact crossing times to machine precision (max |x_exact – x_num| < 10⁻¹²; see Figure 1).

[Figure 1]

*Figure 1. Verification of closed-form solutions against RK4 numerical integration.*

## 3.5 The Overinvestment Result

**Proposition 1 (Overinvestment in Markov Perfect Equilibrium).** *In the symmetric MPE, aggregate output Qᴺ(x) = Nqᴺ(x) strictly exceeds cooperative output Qᶜ(x) for all x > 0. Consequently, T*ᴺᵃˢʰ < T*ᶜᵒᵒᵖ: Nash equilibrium crossing occurs strictly earlier than the cooperative optimum.*

**Proof.** The proof compares shadow costs across regimes.

*Step 1.* At x = 0, Vᴺ(0) = Vᴾ(0) = S. Evaluating the boundary derivatives from (ODE-N) and (ODE-C), the planner's total shadow cost Nμ strictly exceeds the Nash firm's private shadow cost λ for N ≥ 2. This gap reflects the learning externality: each Nash firm internalizes only its own future profit loss from approaching crossing.

*Step 2.* By a standard comparison theorem for ODEs (Walter 1998, Theorem I.9.1), the ordering N·Vᴾ′(x) > Vᴺ′(x) propagates to all x > 0.

*Step 3.* From the output expressions, both the smaller numerator (higher shadow cost) and larger denominator of Qᶜ relative to Qᴺ ensure Qᴺ(x) > Qᶜ(x) for all x > 0. ■

**Remark 1 (Irreversibility).** At Q = Q̄, a new basin of attraction—the distributed inference equilibrium—becomes accessible. In the language of Morse theory (Milnor 1963), this topological change persists under all sufficiently small perturbations. Reversing the crossing would require cumulative production to decrease—which contradicts monotonicity. Once Q crosses Q̄, the inference transition is topologically irreversible.

**Remark 2 (Niche Persistence).** Irreversibility of inference crossing does not imply extinction of the centralized paradigm. IBM's mainframe business continues to generate approximately $3–4 billion annually as of 2025—decades after the PC revolution—serving high-reliability transaction processing (Greenstein 1997, updated). The addressable market expanded so dramatically that mainframes went from 100% of computing to a stable niche. The analog for AI: centralized datacenters persist for frontier model training while the inference market expands around them.

**Economic interpretation.** The overinvestment decomposes into a Cournot channel (price-depressing rival output) and a learning externality channel (private shadow cost = 1/N of social shadow cost). The decomposition of S into S_T + S_I/(N(r + δ)) reveals a moderating effect: when S_T is large, crossing is less catastrophic and the overinvestment gap narrows. The strategic interaction—each firm internalizing only 1/N of the social shadow cost—persists regardless of S_T.

Through niche construction (Odling-Smee, Laland, and Feldman 2003), centralized firms' cumulative production modifies the competitive landscape to favor distributed inference entrants. Unlike true evolutionary suicide (Gyllenberg and Parvinen 2001), the centralized firms are not rendered extinct—they undergo *evolutionary metamorphosis*. Training capabilities persist and may appreciate; inference revenue erodes. The investment equilibrium constitutes an evolutionarily stable strategy (Maynard Smith 1982), but the ESS is collectively self-undermining for inference rents.

**Welfare loss.** At baseline calibration (N = 5, S_T = 0), the per-firm welfare loss under Nash competition is 34.1%. With S_T calibrated to estimated training revenue persistence, the loss moderates to approximately 22–28%. The qualitative result is unchanged.

[Figures 2–3]

## 3.6 Comparative Statics

**Corollary 1 (Increasing N).** *Nash equilibrium aggregate output is strictly increasing in N for all x > 0.* As N increases, each firm internalizes a smaller share of the social shadow cost.

**Corollary 2 (Asymmetric firms).** *If firm 1 has marginal cost c₁ – ε, aggregate equilibrium output is strictly increasing in ε.* Any asymmetry that shifts one firm's best response outward increases aggregate output.

**Corollary 2* (Asymmetric crossing valuation).** *If firm j has post-crossing value S_j > S (i.e., firm j benefits from crossing relative to symmetric rivals), firm j produces strictly more than symmetric competitors, and aggregate output increases.*

**Proof sketch.** Let Sⱼ = S + Δ with Δ > 0. Firm j's boundary condition Vⱼ(0) = Sⱼ > S = Vᵢ(0) for i ≠ j. By the first-order condition (8), higher continuation value at crossing implies lower shadow cost of approaching crossing. Firm j produces more at every interior state. Aggregate output increases; T* decreases. ■

Corollary 2* applies when a subset of firms captures value from the distributed equilibrium—for instance, through edge hardware sales or ecosystem platform effects. Such firms' best responses are shifted outward, accelerating T* beyond the symmetric Nash prediction. More generally, the corollary applies whenever the player set includes agents whose post-crossing payoff exceeds the symmetric case, whether through direct market capture or through complementary goods (Cournot 1838; Economides 1996).

**Corollary 3 (Capacity constraint and boom-bust).** *Crossing time delay is bounded by the construction lag Δ for new capacity. The long-run learning rate α is unaffected.* The 41-year DRAM panel confirms: at least four boom-bust cycles left α = 0.22 unchanged. The 2025–26 DRAM supercycle provides a fifth test: consumer DDR5 prices have risen 300–400% above trend in under six months (Table 4, † note), driven by AI datacenter demand reallocating wafer capacity from consumer to HBM formats. The corollary predicts that (a) this deviation is temporary, bounded by the construction lag for new fab capacity (Samsung P4, SK Hynix M15X, Micron Idaho), and (b) the long-run α is unaffected.

**Why not collude?** The N symmetric firms face a prisoner's dilemma: each firm's incentive to produce exceeds the collectively optimal rate. But even if a subset of firms could enforce restraint, the overinvestment result is robust to the presence of asymmetric players (Corollary 2*) who benefit from crossing and therefore have no incentive to participate in restraint. The presence of such players makes collusion among symmetric firms strictly less valuable, since the crossing time under mixed collusion-competition exceeds T*ᴺᵃˢʰ by less than under full collusion.

**Remark 3 (Option-value amplification).** Under the option-value objective function specification (Section 3.1), the overinvestment result is amplified. If firms invest to maximize the probability of achieving a discontinuous capability threshold, the marginal value of additional investment is governed by the prize V* rather than by discounted market revenue. When V* is sufficiently large, equilibrium investment exceeds the Nash revenue-maximization level: firms treat inference revenue as a financing vehicle for the capability race rather than as the terminal objective. The model's quantitative predictions (Qᴺ/Qᶜ ≈ 3–4×, T* ≈ 2028) are then conservative—actual aggregate investment may exceed even the Nash prediction, and crossing may arrive sooner. The inference decentralization mechanism operates identically in either case; the distinction matters only for interpreting the magnitude of investment and the post-crossing welfare calculation.

[Figures 4–6]

## 3.7 Calibration

The learning elasticity α = 0.23 from the HBM packaging curve (Table 4). Current HBM cost is approximately $12/GB (HBM3E, 2025); the crossing threshold is $5–7/GB. The calibration uses the conservative bound Q̄ ≈ 112 EB ($5/GB target).

**Post-crossing continuation value.** The inference displacement rate δ ≈ 0.30 from the IBM trajectory (Section 6.1). The training persistence value S_T admits two interpretations corresponding to the two objective function specifications.

*Revenue-maximization specification.* S_T is calibrated from structural economics of frontier model training. A conservative estimate assumes training and model-licensing revenue at 20–40% of current total revenue per firm. The key uncertainty is whether open-weight model proliferation erodes this revenue (Section 5.2). Three scenarios:

- *S_T high (closed-model dominance)*: Frontier training remains proprietary; hyperscalers license models. Welfare loss moderates to ~22%.
- *S_T moderate (open-weight competition)*: Near-frontier open-weight models compress licensing margins. Welfare loss ~28%.
- *S_T ≈ 0 (open-weight commoditization)*: Open models match frontier capability; training output approaches a public good. Pure displacement case; welfare loss ~34%.

Current trajectory favors the moderate-to-low S_T scenario under the revenue-maximization specification. Evidence is presented in Section 5.2.

*Option-value specification.* Under the alternative objective function, S_T represents the option value of maintaining frontier training capability—the ability to attempt training runs at scales (10⁵–10⁶ synchronized GPUs, months of continuous operation) that no distributed architecture can replicate. This value depends on the probability that such runs produce a discontinuous capability advance and on the expected payoff conditional on success. Neither parameter is directly observable, but the revealed preference of firm behavior is informative: the $1.3 trillion in cumulative capex, the multi-year forward commitments (Stargate's $500B over four years), and the explicit public statements by firm leaders citing artificial general intelligence as the objective all suggest that the option-value component of S_T is large relative to the revenue component. Under this specification, S_T does not erode with open-weight proliferation: open-weight models commoditize *current-generation* capability but do not replicate the frontier training infrastructure. The open-weight ecosystem has not attempted synchronized training runs at Stargate-class scale, and the export-control regime makes such attempts structurally difficult for constrained developers. The option-value specification thus implies S_T is both higher and more durable than the revenue specification—but it carries a different risk: if the discontinuous capability advance does not materialize (scaling laws plateau, alignment constraints bind, or the required breakthrough is paradigmatic rather than computational), the option expires and S_T collapses to zero retroactively. The infrastructure becomes the largest stranded asset in commercial history.

The two specifications bracket the range of outcomes. Under revenue maximization with open-weight erosion, S_T → 0 and the pure displacement case applies. Under the option-value specification with successful capability scaling, S_T may exceed the entire pre-crossing firm value. The model's mechanism and crossing predictions are identical under both; the specifications diverge only on the welfare interpretation and the post-crossing equilibrium for incumbent firms.

**Quantitative predictions.** Under Nash competition with N = 5, the model's hardware-only convergence path predicted crossing in approximately 3.8 years from 2024 (~2028). As of Q1 2026, discrete consumer GPUs with sufficient memory bandwidth for 70B-class inference at interactive speeds exist in shipping hardware (Section 5.1.4). However, the 2025–26 DRAM supercycle has driven consumer memory costs sharply above the long-run trend line (Table 4, † note), inflating GPU street prices 50–150% above MSRP and triggering production rationing. The remaining constraint—memory capacity and cost at consumer price points—thus reflects both the structural market segmentation identified earlier and a cyclical demand shock operating through Corollary 3's boom-bust mechanism.

The net effect on T* timing is a short-term delay of approximately 1–2 years during the boom phase, with potential acceleration during the subsequent bust phase as new fab capacity (Samsung P4, SK Hynix M15X, Micron Idaho) produces overcapacity and below-trend consumer pricing. The model's central prediction—crossing under Nash competition approximately 14 years earlier than under cooperation—remains unchanged, as the supercycle does not alter the long-run learning elasticity α = 0.22 (Corollary 3). What changes is the short-run trajectory: crossing now tracks a non-monotonic path (below-trend approach through mid-2025, above-trend deviation 2025–2027, accelerated approach post-2028) rather than the smooth exponential the calibration assumed. Under cooperation, ~2042. Competition accelerates by 79%. The dual convergence path (Section 5.2) continues to compress timing through the algorithmic channel, which is unaffected by the DRAM supercycle.

## 3.8 Note on Identification

The DRAM learning curve is estimated by OLS regression of log price on log cumulative output. This identifies a correlation, not necessarily a structural learning-by-doing parameter. Endogeneity concerns (demand shocks driving both output and investment in cost reduction) are standard in the learning-curve literature (Irwin and Klenow 1994). The mechanism's predictions are robust to this ambiguity: the key property—that cumulative production is associated with cost decline at a stable elasticity—holds across the full 41-year panel. This paper follows the quantitative calibration tradition in growth economics rather than claiming causal identification. The self-undermining property (∂T*/∂I < 0) requires only that centralized investment contributes to cumulative Q and that c(Q) is decreasing and stable.

## 3.9 Generalized Crossing Condition

The model defines crossing at cost parity, but empirical evidence shows hardware crossing *precedes* architectural dominance by 3–5 years (Section 6.4). What is actually required is that the distributed ecosystem's basic reproduction number exceeds unity:

> R₀ ≡ β(c, λ) · γ / (κ + μ) > 1     (9)

where β(c, λ) is the adoption rate, increasing in cost advantage and latency advantage λ; γ is the network effect multiplier; κ is the coordination friction; and μ is the churn rate.

The latency advantage λ captures a structural property specific to inference: edge devices achieve <10ms response versus 50–200ms for cloud round-trip. This is a quality dimension independent of cost. The inclusion of λ means R₀ > 1 can be achieved even before strict cost parity—consistent with the empirical observation that enterprises already deploy edge inference at higher cost for latency-critical applications.

### 3.9.1 Equivalence to a Modified Production Threshold

R₀ = 1 maps to a modified cost threshold c**, which maps to a modified cumulative production threshold:

> Q̄* = Q̄ · (1 – (κ + μ)/(γ · c*) + λ/c*)⁻¹/ᵅ     (10)

Replace Q̄ with Q̄*(κ) throughout the model. All propositions and solutions carry through identically.

**Proposition 1* (Generalized Overinvestment).** *Under fixed κ: in the symmetric MPE with R₀ = 1 crossing condition, Qᴺ(x) > Qᶜ(x) for all x > 0.* Proof identical to Proposition 1.

### 3.9.2 The Model Is Unchanged

The R₀ generalization preserves the one-state-variable structure. Replacing Q̄ with Q̄*(κ) changes the boundary condition's *location* without altering the ODEs, their solutions, or Proposition 1. The coordination layer dynamics modify the distance to crossing (via Q̄*) while the Nash competition dynamics drive how fast that distance is traversed (via aggregate output).

### 3.9.3 The Coordination Layer Lag Becomes Endogenous

**Hardware crossing** occurs at Q(t) = Q̄ (cost parity). In the friction-dominated regime, R₀ < 1 at this point because κ is high. **R₀ crossing** occurs at Q(t) = Q̄*(κ(t)). The lag ΔT depends on coordination maturation.

The open-weight model ecosystem directly affects κ. When models are released with optimized deployment runtimes, documented APIs, and hardware-specific inference engines, the coordination layer matures *in parallel with* hardware cost decline rather than *after* it. This pre-crossing coordination maturity lowers κ₀ at the moment of hardware crossing, compressing the lag. Empirical evidence for this mechanism is presented in Section 5.2.

**Table 2. Coordination layer lag across transitions.**

| Transition | Hardware T* | R₀ T* | ΔT |
|:---|:---|:---|:---|
| Mainframe → PC | 1987 | 1990–92 | 3–5 yr |
| ARPANET → Internet | ~1989 | 1993–94 | 4–5 yr |
| Cloud → Edge AI | 2027–29† | ? | 2–3 yr (predicted) |

*† Hardware capability threshold met at professional price points Q1 2026; consumer cost threshold delayed by 2025–26 DRAM supercycle (Corollary 3). Range reflects supercycle resolution uncertainty.*

The predicted compression to 2–3 years reflects two features absent from prior transitions: the structural latency advantage λ, and pre-crossing coordination maturity from the open-weight ecosystem.

---

# 4. The Training-Inference Structural Distinction

The $1.3 trillion in centralized AI infrastructure investment builds capacity for two structurally distinct workloads. Conflating them overstates the mechanism's scope; separating them sharpens it.

## 4.1 Two Workloads, Two Architectures

**Training** teaches models by processing massive datasets across tightly synchronized GPU clusters. It requires 10,000–100,000+ GPUs communicating at terabits per second via InfiniBand or NVLink, running continuously for weeks to months on petabytes of proximate data. Power density: 100–1,000 kW/rack. Latency-insensitive (100ms between regions is tolerable). Frequency: one-time per model generation.

**Inference** runs trained models to serve real-time user requests. Tasks are independent and atomizable—each query can be served by a single device with no inter-device communication. Latency-sensitive: users benefit from <10ms local execution versus 50–200ms cloud round-trip. Power density: 30–150 kW/rack. Frequency: continuous, scales with every user and every query.

**Table 3. Training vs. inference structural comparison.**

| Dimension | Training | Inference |
|:---|:---|:---|
| Share of AI compute (2023) | ~67% | ~33% |
| Share of AI compute (2025) | ~50% | ~50% |
| Share of AI compute (2026, proj.) | ~33% | ~67% |
| Synchronization requirement | Massive (10K+ GPUs tightly coupled) | None (atomizable) |
| Interconnect requirement | InfiniBand/NVLink, TB/s | Standard networking |
| Latency sensitivity | Low (100ms tolerable) | High (<10ms for UX) |
| Location constraint | Remote, power-rich | Metro-adjacent, near users |
| Cost trajectory | Rising per frontier model | Declining ~280× in 2 years |
| Frequency | One-time per model | Continuous, scales with users |
| Edge-viable? | No (architectural constraint) | Yes (this paper's thesis) |
| Revenue model | Amortized capital expenditure | Ongoing per-query revenue |

*Sources: Deloitte (2025), McKinsey (2025), MIT Technology Review (2025), Stanford AI Index (2025), Epoch AI (2025).*

## 4.2 Training Does Not Decentralize

Frontier model training requires synchronized clusters of 10,000–100,000+ GPUs communicating at terabit-per-second speeds, running continuously for weeks to months on petabytes of proximate data. These requirements are architectural, not merely economic. The learning curves that reduce memory and compute costs per unit do not address the synchronization, bandwidth, and scale requirements. A consumer device with excellent memory bandwidth cannot participate in a distributed training run because the inter-device communication latency (milliseconds over WiFi vs. nanoseconds over NVLink) creates a performance gap of 5–6 orders of magnitude. No plausible learning curve closes this gap because the constraint is topological (network diameter and synchronization protocol) rather than cost-based.

## 4.3 Inference Decentralizes

Inference is the workload where the endogenous decentralization mechanism applies with full force:

Inference tasks are *atomizable*: each user query is independent and can be served by a single device with no inter-device communication. Inference is *latency-advantaged*: local execution at <10ms outperforms cloud round-trip at 50–200ms on a quality dimension independent of cost. Inference is *bandwidth-bound*: token generation speed is determined almost entirely by the ratio of memory bandwidth to model size in memory—exactly the constraint whose learning curve the model tracks. Inference *scales with users*: every new AI application, user, and agentic action generates inference demand.

## 4.4 The Inference Revenue Pool

OpenAI's 2024 compute spending illustrates the economics: approximately $5 billion on R&D/training compute versus approximately $2 billion on inference compute (Epoch AI 2025). But inference costs scale with users while training is amortized. GPT-4's inference costs were projected at approximately $2.3 billion in 2024—roughly 15× its training cost. At scale, inference dominates both compute cycles (80–90%, MIT Technology Review 2025) and ongoing revenue. The inference market is projected to grow from $106 billion (2025) to $255 billion by 2030 (MarketsandMarkets 2025). This is the revenue pool that the $1.3 trillion infrastructure buildout targets, and the revenue pool that edge devices will intercept.

## 4.5 Implications for the Model

The post-crossing continuation value S = S_T + S_I/(N(r + δ)) captures the bifurcation: training revenue S_T persists while inference revenue S_I decays. Under the option-value specification (Section 3.1), S_T represents the value of maintaining the only infrastructure capable of frontier training runs—a value that may exceed inference revenue if such runs produce discontinuous capability advances. The endogenous decentralization mechanism applies to inference with full force under either specification. The crossing condition (Section 5.1.4) is an inference benchmark. The predictions (Section 7) are inference predictions. Training provides incumbents either a revenue floor (revenue specification) or an option on a discontinuous payoff (option-value specification), but does not alter the mechanism driving inference decentralization.

---

# 5. Empirical Evidence: Dual Convergence

The inference crossing condition—≥70B-class output quality at ≥20 tok/s under $1,500—is being approached from two directions: hardware costs declining from below (Section 5.1) and algorithmic efficiency reducing the effective threshold from above (Section 5.2).

## 5.1 Convergence from Below: Hardware Cost Decline

### 5.1.1 DRAM Learning Curve (1984–2024)

A 41-year panel through 2024: α = 0.22 (SE = 0.03, n = 41, R² = 0.96). Prices declined from $870,000/GB (1984) to $2.00/GB (2024).

**Table 4. DRAM learning curve (selected years).**

| Year | Generation | Price ($/GB) | Cum. Prod. (EB) | ln(Price) | ln(Cum.) |
|:---|:---|---:|---:|---:|---:|
| 1984 | 64Kb | 870,000 | <0.001 | 13.68 | –11.51 |
| 1990 | 4Mb | 100,000 | 0.003 | 11.51 | –5.81 |
| 1995 | 16Mb | 30,000 | 0.10 | 10.31 | –2.30 |
| 2000 | 256Mb | 1,200 | 2.0 | 7.09 | 0.69 |
| 2005 | 1Gb | 90 | 17 | 4.50 | 2.83 |
| 2010 | 2Gb | 10 | 95 | 2.30 | 4.55 |
| 2015 | 8Gb | 3.20 | 400 | 1.16 | 5.99 |
| 2020 | 16Gb | 2.80 | 1,400 | 1.03 | 7.24 |
| 2024 | 32Gb | 2.00 | 3,200 | 0.69 | 8.07 |
| 2025–26 | 32Gb† | 10–16 | ~4,200 | 2.30–2.77 | 8.34 |

*OLS through 2024: α = 0.22, SE = 0.03, R² = 0.96.*

**† The 2025–26 DRAM supercycle.** Consumer DDR5 prices reversed sharply beginning Q4 2025, rising from approximately $2–3/GB to $10–16/GB by Q1 2026—a 3–5× increase in under six months. The 2025–26 data point lies 1.5–2.0 log units *above* the 41-year trend line at the corresponding cumulative output. This is a demand-driven price shock, not a change in the production function. AI datacenter buildout (Table 6) is absorbing an estimated 40% of global DRAM wafer capacity, with HBM production requiring approximately 3× the wafer area per bit of standard DDR5. Samsung, SK Hynix, and Micron have simultaneously curtailed DDR4 production, reallocated wafer starts to HBM and server DDR5, and—in Samsung's case—doubled contract prices in a single quarter. TrendForce projects DRAM contract prices rising 55–60% QoQ in Q1 2026, with shortages persisting through at least Q4 2027.

This is Corollary 3 in real time: the boom-bust cycle induced by centralized overinvestment. The 41-year DRAM panel contains at least four prior supercycles (1987–88, 1994–95, 1999–2001, 2017–18), none of which altered the long-run learning elasticity α = 0.22. The current cycle is distinguished by its magnitude (driven by AI-specific demand that is structurally unprecedented) and by its direct connection to the mechanism: the same concentrated capital investment that finances learning curves (Stage 1) has temporarily reversed them in the consumer segment by reallocating capacity to datacenter formats. The capacity expansion underway—Samsung P4, SK Hynix M15X, Micron Idaho—will, on historical precedent, produce overcapacity and below-trend pricing within 2–3 years of full ramp, exactly as the mechanism predicts. The fabs built for datacenter HBM and DDR5 will pivot to consumer memory formats when datacenter demand moderates (Section 5.3).

### 5.1.2 HBM Cost Trajectory

HBM prices declined from $120/GB (2015) to $12/GB (2025). α = 0.23 (SE = 0.06, n = 6). The packaging knowledge transfers to consumer form factors—the learning externality central to the mechanism. Note: HBM pricing has diverged from consumer DRAM during the 2025–26 supercycle. HBM contracts are long-term negotiated and rose approximately 20% for 2026 deliveries (Samsung, SK Hynix), while consumer DDR5 spot prices rose 300–400%. This divergence reflects the capacity allocation dynamic: HBM wafer starts crowd out consumer DRAM, amplifying consumer price spikes while HBM itself follows a more moderate trajectory shaped by intensifying three-way supplier competition (Samsung, SK Hynix, Micron).

**Table 5. HBM pricing trajectory.**

| Year | Generation | $/GB | Capacity/Stack (GB) |
|:---|:---|---:|---:|
| 2015 | HBM1 | 120 | 4 |
| 2016 | HBM2 | 60 | 8 |
| 2018 | HBM2E | 35 | 8 |
| 2020 | HBM2E | 25 | 16 |
| 2022 | HBM3 | 20 | 24 |
| 2024 | HBM3E | 15 | 36 |
| 2025 | HBM3E+ | 12 | 48 |
| 2026 | HBM3E/HBM4 | 10–14† | 48 |

*α = 0.23 (SE = 0.06).*

*† HBM3E contract prices rose ~20% for 2026 deliveries but are projected to decline 25–30% YoY under three-way supplier competition (Goldman Sachs, TrendForce). HBM4 carries an estimated 40% premium over HBM3E at introduction. The 2026 range reflects this uncertainty.*

### 5.1.3 Hyperscaler Capital Expenditure

**Table 6. Hyperscaler capex ($B).**

| Company | 2018 | 2020 | 2022 | 2024 | 2025E |
|:---|---:|---:|---:|---:|---:|
| Microsoft | 11.6 | 15.4 | 23.9 | 44.5 | 80 |
| Alphabet | 25.1 | 22.3 | 31.5 | 52.5 | 75 |
| Amazon | 13.4 | 35.0 | 58.3 | 78.0 | 100 |
| Meta | 13.9 | 15.7 | 31.4 | 39.2 | 65 |
| Stargate JV | — | — | — | — | 100 |
| **Industry Total** | **64** | **88** | **148** | **232** | **436** |

*Cumulative 2018–2025: $1,298B. Sources: company filings and guidance.*

Each firm's guidance explicitly cited competitive pressure from rivals—qualitative evidence of the Nash dynamic. The stated near-term objective is inference revenue from AI-as-a-service cloud APIs. However, the scale of forward commitments—particularly Stargate's $500B over four years and the construction of million-GPU synchronized clusters—exceeds what inference revenue projections alone would justify at standard discount rates, suggesting the option-value specification (Section 3.1) may better characterize the actual objective function.

### 5.1.4 Consumer Silicon and the Inference Crossing Condition

Stage 2 materializes as component migration from datacenter to consumer form factors. Token generation speed for inference is determined almost entirely by the ratio of memory bandwidth to model size in memory, making memory bandwidth—not compute throughput—the binding constraint on consumer AI inference. Four tiers of consumer and professional AI silicon now in mass production reveal both the trajectory and the binding constraint's shift from technology to market structure.

**Edge tier.** The most significant evidence of the hardware trajectory comes from the progression of edge AI processors from vision-only capability to LLM-capable inference. Rockchip's NPU progression illustrates the transition: the RK1808 (2019, 3 TOPS, 2MB on-die SRAM) performed vision tasks only; the RK3588 (2022, 6 TOPS, external DRAM) barely ran 1.1B-parameter models at 14 tok/s; the RK1828 (2025, 20 TOPS, 5GB 3D stacked DRAM co-processor) runs 7B-parameter models at 59 tok/s in W4A16 quantization.¹ Hailo's 10H processor (2025, 40 TOPS INT4, 2.5W) provides a complementary data point: the Raspberry Pi AI HAT+ 2—chip plus 8GB LPDDR4 on a single PCB—retails at $130 and runs 2B-parameter models at 10+ tok/s. Both architectures trace the same transition from on-die SRAM (vision only) to stacked or co-packaged DRAM (LLM-capable). These are Stage 2 deployments at embedded scale in form factors that never previously contained AI inference capability.

**Consumer desktop tier.** AMD's Ryzen AI Max+ 395 ("Strix Halo") ships at approximately $2,000 with 128GB unified LPDDR5X, delivering ~215 GB/s bandwidth. On dense 70B models (Q4, ~40GB): ~5 tok/s, matching the theoretical ceiling of 215/40 ≈ 5.4 tok/s. MoE architectures provide an end-run around the dense-model crossing condition: models with 70B-class output quality and ~20B active parameters per token achieve ~31 tok/s on this system at interactive speeds.

**Discrete GPU tier.** NVIDIA's RTX 5090 (Q1 2026, 32GB GDDR7, ~1,792 GB/s, $1,999 MSRP) exceeds the speed threshold on models that fit in 32GB: a 32B Q4 model (~18GB) runs at ~100 tok/s (1,792/18 ≈ 99.6). For 70B-class output quality, MoE architectures with ≤32GB memory footprint meet both the speed and capability thresholds. The professional variant—NVIDIA RTX PRO 6000 (96GB GDDR7, 1,792 GB/s, ~$8,000 MSRP)—runs 70B dense models at ~45 tok/s and 120B+ models comfortably. Both cards use the same GB202 die and identical GDDR7 memory chips; the professional card simply populates more memory on a double-sided PCB. The technology for consumer 70B-class inference at interactive speeds exists in shipping hardware as of early 2026.

However, the DRAM supercycle has driven actual transaction prices far above MSRP. RTX 5090 street prices range from $3,000 to $5,000+ as of February 2026, with NVIDIA reportedly planning 30–40% production cuts to its GeForce lineup due to GDDR7 supply constraints. Memory now accounts for an estimated 80% of GPU bill-of-materials cost, up from approximately 30–40% pre-shortage. The practical effect is that the hardware *capability* threshold has been met while the *cost* threshold has moved sharply in the wrong direction—exactly the boom-phase deviation from trend that Corollary 3 predicts.

**Performance tier.** Apple's Mac Studio with M4 Max (128GB, $3,949) delivers 546 GB/s bandwidth, yielding ~14 tok/s on 70B Q4 models. Apple's unified memory architecture accommodates larger models than discrete GPUs at equivalent price points, but at lower bandwidth per dollar.

**The gap is now three constraints, not one.** Before the DRAM supercycle, the gap between demonstrated capability and consumer affordability was purely market segmentation: the RTX PRO 6000 proved that 96GB GDDR7 works in a desktop form factor with air cooling, while the consumer RTX 5090 used the same die but populated only 32GB—a business decision, not an engineering constraint. At MSRP, GDDR7 cost approximately $83/GB at professional pricing versus $37.50/GB at consumer pricing, a 2.2× segmentation premium.

The DRAM supercycle has added two compounding constraints. First, input cost inflation: GDDR7 procurement costs have risen sharply as memory manufacturers prioritize HBM for AI accelerators, with DRAM now reported at 80%+ of GPU BOM versus ~30–40% pre-shortage. Second, supply rationing: NVIDIA is reportedly cutting GeForce production 30–40% to allocate scarce GDDR7 to higher-margin professional and datacenter products, prioritizing revenue per GB of GDDR7 allocated (the RTX PRO 6000 generates approximately 41% more revenue per GB of onboard GDDR7 than the RTX 5090). The combined effect is that the inference crossing condition for 70B-class models at ≥20 tok/s under $1,500 faces not one but three barriers: (1) the segmentation premium on memory capacity, which is structural; (2) the supercycle premium on memory cost, which is cyclical; and (3) supply rationing, which is strategic.

The econometric distinction matters: constraints (2) and (3) are temporary. The supercycle is a demand shock operating through Corollary 3's boom-bust mechanism. The capacity expansion it has triggered (Samsung P4, SK Hynix M15X, Micron Idaho) will, on the 41-year historical pattern, produce overcapacity and below-trend pricing within 2–3 years of full ramp. When the bust phase arrives, it will compress not only the supercycle premium but also the segmentation premium, as overcapacity in GDDR7 production gives NVIDIA less economic reason to ration consumer memory capacity. Constraint (1) remains, but weakens under competitive pressure from AMD and Intel. Historical precedent suggests professional-tier memory capacity migrates to consumer price points within 2–3 product generations. The net effect of the supercycle on T* is a short-term delay (1–2 years) followed by a potential acceleration, as the post-bust pricing environment will be more favorable to consumer adoption than the pre-boom trend line projected.

¹ The RK1828 development kit (Firefly) lists at $1,029, but this carries substantial markup over production cost and is not indicative of mass-market pricing. Rockchip ships hundreds of millions of SoC units annually across consumer electronics and IoT; the architectural datum—stacked DRAM enabling LLM inference at edge power levels—is what matters for the mechanism, not the development kit price.

## 5.2 Convergence from Above: Algorithmic Efficiency

### 5.2.1 The Incentive Structure

A binding compute constraint on a subset of model developers creates a structural incentive to maximize inference capability per unit of available hardware. US semiconductor export controls, beginning in October 2022 and progressively tightened through 2025, denied a significant population of AI developers access to frontier datacenter GPUs. The theoretical prediction is that constrained firms should optimize for efficiency and pursue deployment strategies compatible with available hardware—including edge devices. This is a testable implication independent of any particular policy assessment.

The empirical outcome is consistent with the prediction. Model developers operating under these constraints have converged on open-weight release strategies paired with aggressive efficiency optimization, producing models that approach frontier capability at a fraction of the compute budget. This is not unique to any single firm: by late 2025, the open-weight ecosystem included participants across a wide range of organizational scales, from large technology conglomerates to venture-backed startups.

### 5.2.2 Scale and Adoption

The open-weight model ecosystem has achieved substantial adoption. Total downloads of the Qwen model family (Alibaba) exceeded 700 million on Hugging Face by January 2026, surpassing Meta's Llama as the most-downloaded model family (Alibaba Cloud 2026). By August 2025, Qwen-derived models accounted for over 40% of all new Hugging Face language model derivatives, compared to approximately 15% for Llama (Lambert 2025). An empirical study of 100 trillion tokens processed through the OpenRouter aggregator found open-weight model share surging from 1.2% of weekly token volume in late 2024 to peaks of approximately 30% within months, averaging 13% of weekly volume through 2025 (OpenRouter/Andreessen Horowitz 2025).

At the capability frontier, Moonshot AI's Kimi K2.5 (open-weight, 2026) approached the performance of Anthropic's Claude Opus on early benchmarks at approximately one-seventh the API price. DeepSeek R1 (open-weight, January 2025) matched OpenAI's o1 on reasoning benchmarks at a reported fraction of the training cost (MIT Technology Review 2025).

### 5.2.3 Mechanisms Reducing the Effective Crossing Threshold

Three classes of algorithmic improvement reduce the hardware required for a given inference capability level, effectively lowering Q̄_eff:

**Mixture-of-Experts (MoE).** MoE architectures activate only a fraction of total parameters per token. DeepSeek V3 (671B total, ~37B active) and GPT-OSS 120B (116.8B total, ~20B active) demonstrate that 70B-class output quality is achievable with 20–37B active parameters. This reduces the memory bandwidth required per token by 3–6×, directly compressing the crossing condition. MoE has become the dominant architecture in the open-weight ecosystem precisely because it maximizes capability per unit of constrained hardware—an architectural choice consistent with the binding-constraint incentive structure.

**Quantization.** INT4 quantization reduces model memory footprint by approximately 4× with modest quality loss. More aggressive techniques (INT2, mixed-precision) push further. Open-weight models are routinely released with quantized variants optimized for constrained-hardware deployment.

**Distillation.** Smaller models trained to mimic larger ones. DeepSeek's distilled models (1.5B, 7B, 14B variants of R1) explicitly target edge deployment, maintaining reasoning capability at dramatically reduced hardware requirements.

The combined effect is substantial. Stanford's 2025 AI Index documented a 280-fold drop in inference costs between November 2022 and October 2024. This reflects hardware learning curves AND algorithmic efficiency gains. The paper's α = 0.22–0.23 captures hardware alone; the effective cost decline including algorithmic optimization is significantly steeper.

### 5.2.4 Implications for the Effective Crossing Threshold

The algorithmic convergence path enters the formal model through Q̄_eff. When MoE reduces active parameters by half, the effective memory bandwidth requirement halves, and Q̄_eff drops accordingly. The state variable x(t) = Q̄_eff(η(t)) – Q(t) is depleted from both sides: Q(t) rises from below (hardware learning curves) while Q̄_eff falls from above (algorithmic efficiency).

Under the quasi-static approximation (Appendix C), the model's propositions and solutions apply pointwise when algorithmic efficiency evolves slowly relative to production decisions. The key empirical question is whether algorithmic gains are smooth or lumpy. The evidence from 2024–2025—with multiple discrete capability jumps from open-weight releases—suggests lumpiness, which the model handles as discrete threshold shifts.

### 5.2.5 Open Weights and the S_T Parameter

Open-weight model proliferation affects the model not only through Q̄_eff but also through S_T. If the training moat erodes—if open-weight models consistently approach frontier capability—then the post-crossing continuation value S_T declines toward zero, and the pure displacement case applies. The current trajectory of open-weight capability convergence (Section 5.2.2) is relevant for calibrating S_T scenarios in Section 3.7.

### 5.2.6 Open Weights and the Coordination Layer

The open-weight ecosystem also affects the coordination friction parameter κ. Model releases increasingly ship with optimized deployment runtimes, hardware-specific inference engines, and standardized APIs. In 2025, open-weight model releases began to include day-zero support for multiple hardware targets, including edge-specific runtimes (RKNN for Rockchip, Core ML for Apple, ONNX for cross-platform). This is coordination layer construction occurring *before* hardware crossing—the mechanism identified in Section 3.9.3 as compressing the coordination layer lag ΔT.

## 5.3 The Demand Shock as Nash Overinvestment

The Stargate project alone demands approximately 40% of global DRAM output. Combined hyperscaler commitments exceed current manufacturing capacity by 5–10×. The memory industry's response is the largest capacity expansion in semiconductor history. Historical precedent predicts overcapacity and below-trend pricing by 2028–2029. The fabs built for datacenter demand will pivot to consumer stacked DRAM and LPDDR6 when datacenter demand moderates—accelerating the very edge inference capability that drives the moderation. This is the Nash overinvestment dynamic operating through the supply side.

**The 2025–26 DRAM supercycle as mechanism validation.** The model predicts that concentrated capital investment simultaneously (a) finances learning curves and (b) generates boom-bust dynamics through capacity overshoot (Corollary 3). The current DRAM supercycle is this prediction unfolding in real time. Consumer DDR5 prices have risen 300–400% since October 2025 (Table 4, † note). GDDR7 shortages have driven consumer GPU street prices 50–150% above MSRP, with NVIDIA rationing GeForce production to protect professional margins. The causation runs exactly through the model's channel: hyperscaler investment in centralized AI infrastructure (Table 6) → surge in HBM and server DDR5 demand → reallocation of DRAM wafer capacity from consumer to datacenter formats → consumer memory shortage → above-trend consumer pricing.

The critical empirical observation is that this is a *demand allocation* shock, not a *production function* shock. Total DRAM bit output continues to rise. The supercycle premium reflects not a loss of manufacturing capability but a revealed preference by memory manufacturers to serve higher-margin datacenter customers at the expense of consumer supply. Samsung, SK Hynix, and Micron have simultaneously curtailed DDR4 production, reallocated advanced-node capacity to HBM (which requires ~3× the wafer area per bit of standard DDR5), and—rationally—allowed consumer prices to rise toward datacenter contract levels. TrendForce reports Q4 2025 DRAM contract prices rising 55–60% QoQ, with industry sources projecting shortages through Q4 2027.

This is Corollary 3 operating through a novel channel: the boom phase temporarily *reverses* the consumer cost trajectory even as it finances the capacity expansion that will eventually crash consumer prices below the pre-boom trend. The 41-year DRAM panel contains at least four prior cycles with this signature. What distinguishes the current cycle is (a) its magnitude, driven by structurally unprecedented AI-specific demand, and (b) its direct theoretical relevance: the same overinvestment that the model identifies as accelerating T* in the long run is *delaying* T* in the short run by inflating consumer memory costs. The resolution is temporal: the boom phase adds 1–2 years to the hardware crossing timeline, but the bust phase—when Samsung P4, SK Hynix M15X, and Micron Idaho reach full production—may compress the post-bust crossing timeline by a comparable amount, as below-trend pricing during overcapacity makes the T* threshold easier to reach than the pre-boom trend projected.

The training-inference distinction sharpens this dynamic. As inference migrates to edge devices, datacenter demand for memory *moderates* (inference was the volume driver), but the fabs are already built. The overcapacity pivots to consumer memory formats—accelerating the edge inference capability that caused the moderation. This positive feedback loop is the mechanism's supply-side amplifier.

---

# 6. Historical Validation and Parameter Consistency

## 6.1 Mainframe → Personal Computer (1975–2000)

IBM dominated mainframe computing with 75–80% market share through the 1970s. IBM's semiconductor investment drove the learning curves that reduced microprocessor and memory costs (Flamm 1993: α = 0.24 for Intel microprocessors, 1974–1989).

**Table 7. IBM financial trajectory.**

| Year | Revenue ($B) | Net Income ($B) | R&D ($B) | Employees (K) | Stage |
|:---|---:|---:|---:|---:|:---|
| 1984 | 46.3 | 6.6 | 3.7 | 395 | Peak profits |
| 1987 | 54.2 | 5.3 | 5.0 | 390 | 386 crosses |
| 1990 | 69.0 | 6.0 | 5.6 | 374 | Revenue peak |
| 1991 | 64.8 | (–2.8) | 5.5 | 345 | First loss |
| 1993 | 62.7 | (–8.1) | 5.1 | 257 | Record loss |
| 1995 | 71.9 | 4.2 | 5.5 | 225 | Services pivot |

The $15.8B in cumulative losses (1991–93) reflected a business model adaptation failure, not technology extinction. IBM's mainframe division persists today at $3–4B annual revenue, serving banking, airline, and government high-reliability systems. The mainframe's share went from 100% of enterprise computing to a stable niche—not by extinction but by market expansion. The δ ≈ 0.30 calibration should be understood as the inference-equivalent displacement rate: IBM lost ~60% of its compute-service profit in three years. Total firm displacement was moderated by IBM's eventual services pivot—the analog of hyperscalers pivoting to training and model licensing.

Hardware crossing preceded platform dominance by 3–5 years. The R₀ framework explains this: R₀ < 1 in 1987 (no dominant OS, fragmented software ecosystem); R₀ crossed 1 circa 1990–92 as Windows reduced coordination friction.

## 6.2 ARPANET → Commercial Internet (1969–2000)

Government investment drove TCP/IP development and router cost reduction. Proprietary online service share collapsed from 60% (1989) to 2% (2000). Coordination layer lag: 3–5 years (TCP/IP technically viable by late 1980s; commercial viability arrived with the browser in 1993–94).

## 6.3 The Export-Control Natural Experiment

The October 2022 US semiconductor export controls and subsequent tightenings provide an exogenous shock useful for testing the model's comparative statics. The controls created a binding constraint: a significant population of AI developers lost access to frontier datacenter GPUs. The model predicts that constrained firms should shift toward efficiency-maximizing strategies and distributed deployment—precisely the response to a capacity constraint that reduces the feasible set for centralized competition while leaving distributed alternatives unaffected.

The observed response is consistent with the prediction. Constrained developers converged on: (i) open-weight release strategies, (ii) efficiency-optimizing architectures (MoE), (iii) aggressive quantization and distillation, and (iv) model-hardware co-optimization for edge deployment. The resulting ecosystem of open-weight models functions as an exogenous accelerator of Stage 3: it reduces Q̄_eff from above (Section 5.2), reduces κ through pre-crossing coordination layer construction (Section 3.9.3), and potentially erodes S_T by commoditizing training output (Section 5.2.5).

For the formal model, this natural experiment tests the asymmetric-player extension (Corollary 2*). The constrained ecosystem includes players whose post-crossing value exceeds the symmetric case—firms that capture edge hardware revenue, cloud platform revenue from open-model deployment, or ecosystem network effects. The prediction that aggregate output should exceed the symmetric Nash level in the presence of such players is consistent with the observed acceleration of inference cost decline (280× in two years, Stanford 2025) beyond what hardware learning curves alone would predict.

## 6.4 Cross-Domain Parameter Consistency

**Table 8. Cross-domain learning rates.**

| Industry | Product | α | SE | Period |
|:---|:---|---:|---:|:---|
| Semiconductor | DRAM (full panel) | 0.22 | 0.03 | 1984–2024 |
| Semiconductor | HBM | 0.23 | 0.06 | 2015–2024 |
| Semiconductor | Intel microprocessors | 0.24 | 0.04 | 1974–1989 |
| Semiconductor | NAND Flash | 0.24 | 0.05 | 2003–2023 |
| Semiconductor | Solar PV cells | 0.23 | 0.02 | 1976–2023 |
| Energy | Lithium-ion batteries | 0.21 | 0.03 | 1995–2023 |
| Internet | Cloud compute (AWS) | 0.25 | 0.03 | 2006–2023 |

*α ∈ [0.19, 0.25] across products, firms, and decades.*

## 6.5 The Coordination Layer Pattern

**Table 9. Coordination layer lag across transitions.**

| Transition | Hardware Crossing | Coordination Layer | Dominance | Lag |
|:---|:---|:---|:---|:---|
| Mainframe → PC | 386 at $2,200 (1987) | Windows 3.0/3.1 (1990–92) | Win95 (1995) | 3–5 yr |
| ARPANET → Internet | Routers/TCP (~1989) | Mosaic/Netscape (1993–94) | Web (1995–97) | 3–5 yr |
| Cloud → Edge AI | Consumer GPU VRAM (2027–29†) | Emerging pre-crossing | Proj: 2030–32 | 2–3 yr? |

The AI coordination layer is emerging in: model serving frameworks (llama.cpp, ONNX, vLLM), edge deployment runtimes (TensorRT, Core ML, RKNN), open-weight model distribution (Hugging Face), and economic settlement infrastructure (stablecoin rails at $3.9T/quarter). Uniquely among transitions studied, the coordination layer is maturing *before* hardware crossing—driven by the open-weight ecosystem's co-optimization of models and deployment infrastructure.

---

# 7. Falsifiable Predictions

The model generates nine predictions with timing. If these fail, the theory is wrong.

**Prediction 1: Consumer Stacked Memory ≥16GB by 2027.** HBM-derived 3D stacking in consumer products with ≥16GB on-chip stacked memory below $200. The Rockchip RK1828 (2025, 5GB 3D stacked) and Hailo-10H (2025, 8GB on-module at $130) confirm component migration is underway; the prediction is capacity scaling. Evidence against: ≤8GB through 2028.

**Prediction 2: 70B-Class Inference On-Device by 2028–2029 (Hardware Crossing).** Consumer devices under $1,500 running inference at 70B-class output quality at ≥20 tok/s. The "70B-class output quality" formulation accounts for MoE architectures that achieve equivalent capability with fewer active parameters. As of Q1 2026, the technology capability threshold has been met at professional and enthusiast price points (Section 5.1.4), but the 2025–26 DRAM supercycle has temporarily inflated consumer memory costs 300–400% above trend, delaying the cost threshold. The prediction is that competitive pressure plus post-supercycle overcapacity drives the price threshold below $1,500. Training of such models remains centralized. Evidence against: not achieved by 2031.

**Prediction 2* (Refined): R₀ > 1 for Distributed AI Inference by 2030–2032.** Self-sustaining distributed inference adoption arrives 2–3 years after hardware crossing—compressed from the historical 3–5 year lag by latency advantage and pre-crossing coordination maturity. Evidence for: quarterly distributed inference share growing >5% per quarter without subsidies. Evidence against: distributed share stalling below 20% by 2033.

**Prediction 3: Inference Capex Deceleration with Training Persistence by 2028–2029.** At least one top-four US hyperscaler reduces inference-oriented capex by ≥20% YoY while maintaining or increasing training-oriented capex, accompanied by public repositioning toward hybrid inference architectures and/or model licensing. Pure cyclical deceleration without strategic repositioning does not confirm. Aggregate capex reduction without differential treatment of training vs. inference does not confirm. The DRAM supercycle may mask strategic capex shifts behind cost-driven reductions; the differentiating evidence is explicit reallocation from inference to training, not aggregate cost management.

**Prediction 4: Stablecoin-Treasury Holdings Exceed $300B by 2027.** Tests coordination layer formation for distributed economic settlement. Evidence against: plateau below $200B.

**Prediction 5: Learning Rate Stability.** DRAM/HBM α remains in [0.18, 0.28] through 2030. The 2025–26 DRAM supercycle provides a real-time test: consumer DDR5 prices have deviated 1.5–2.0 log units above the 41-year trend line (Table 4, † note). If α measured over a rolling 5-year window including 2025–26 data falls below 0.15, this prediction fails and all timing predictions shift outward. If prices return to or below trend by 2028–2029 as the capacity expansion model predicts, the supercycle confirms rather than refutes learning rate stability. If α < 0.15, all timing predictions shift outward.

**Prediction 6: Distributed Inference Tipping Point at ~40% of Inference Workloads.** Network effects reverse at ~40% distributed inference share, making centralized inference non-viable for holdout use cases. The mechanism is indirect: distributed adoption above this threshold shifts supplier priority (memory fabs pivot to consumer formats), developer mindshare (tooling investment shifts to edge deployment), and complementary service density. The ~40% threshold follows from calibrating the self-sustaining growth condition to observed network-effect elasticities in cloud infrastructure markets. Applies to inference only; centralized training is unaffected. Evidence against: centralized inference remaining commercially stable with distributed inference share exceeding 50% through 2032.

**Prediction 7: Non-Monotonic Inference Adoption with Coordination-Layer Trough.** Two-wave pattern: initial surge (2027–2030), coordination fragmentation trough (2031–2032), standardization-driven second wave (2033–2035). The R₀ framework provides the structural mechanism: the trough corresponds to the period when hardware has crossed but R₀ < 1.

**Prediction 8: Open-Weight Models Exceed 50% of Global Inference Token Volume by 2028.** This tests the algorithmic convergence path and its interaction with hardware crossing. The open-weight ecosystem's trajectory—from <2% to peaks of ~30% of token volume within 2025 alone (OpenRouter/a16z 2025)—suggests continued rapid adoption. The prediction is that open-weight models become the majority deployment mode as edge hardware reaches capability thresholds. Evidence against: proprietary closed models maintaining >60% of inference token volume through 2029.

**Prediction 9: Training Remains Centralized Through 2035.** Frontier model training (>10,000 synchronized GPUs, >7 days continuous operation) remains exclusively performed in centralized clusters. Fewer than 20 organizations globally are capable of frontier training. This tests the training-inference bifurcation. Evidence against: distributed frontier training at comparable cost and performance by 2035.

---

# 8. Conclusion

This paper has identified and formalized endogenous decentralization: a mechanism by which concentrated capital investment in centralized infrastructure finances the learning curves that enable distributed alternatives. The self-undermining investment property (∂T*/∂I < 0) is distinct from learning-by-doing, GPT spillovers, and Schumpeterian creative destruction. In symmetric Markov Perfect Equilibrium, competing centralized firms produce aggregate output that strictly exceeds the cooperative optimum at every interior state, accelerating crossing time—and the result is robust to increasing competition, cost asymmetry, capacity constraints, and asymmetric crossing valuations.

The mechanism operates through two convergence paths that are novel relative to prior technological transitions. Hardware cost decline from below follows the classical learning curve (α = 0.22–0.23, stable across 41 years and at least four prior boom-bust cycles). Algorithmic efficiency gains from above reduce the effective crossing threshold Q̄_eff through MoE architectures, quantization, and distillation. The algorithmic path is driven by developers operating under binding compute constraints who have a structural incentive to maximize capability per unit of hardware—an incentive amplified by the export-control regime (Section 6.3). The two paths interact: hardware cost decline makes edge deployment progressively more viable, while algorithmic efficiency makes it viable *sooner* and on *cheaper* hardware. The 2025–26 DRAM supercycle illustrates both the mechanism's predictions and its non-monotonic short-run dynamics: the same concentrated investment that finances long-run learning curves has temporarily reversed consumer cost trends by reallocating memory capacity to datacenter formats—exactly the boom-phase deviation Corollary 3 predicts. The capacity expansion this demand shock has triggered will, on historical precedent, produce overcapacity and below-trend consumer pricing within 2–3 years, resolving the deviation and potentially accelerating the hardware crossing beyond the pre-boom trajectory.

The training-inference bifurcation sharpens the mechanism's empirical scope. AI workloads divide into training (architecturally centralized for structural synchronization and bandwidth reasons) and inference (atomizable, latency-sensitive, bandwidth-bound—following the endogenous decentralization trajectory). The post-crossing equilibrium is partial decentralization: inference distributes to edge devices while training persists in centralized clusters. This coexistence is stable because the architectural constraints on training are topological, not cost-based, and no plausible learning curve bridges a 5–6 order-of-magnitude latency gap. Historical precedent confirms niche persistence: IBM's mainframe business continues decades after the PC revolution.

The generalized crossing condition (R₀ > 1) endogenizes the 3–5 year coordination layer lag observed in historical transitions and predicts compression to 2–3 years for the current AI transition, driven by two features absent from prior transitions: the structural latency advantage of local inference and pre-crossing coordination maturity from the open-weight ecosystem. The open-weight ecosystem simultaneously reduces the effective crossing threshold (through algorithmic efficiency), the coordination friction (through deployment infrastructure), and potentially the training moat (through capability convergence)—three effects operating through distinct model parameters (Q̄_eff, κ, and S_T respectively).

The welfare implications depend on the objective function specification. Under revenue maximization, the overinvestment externality imposes a 22–34% per-firm welfare loss relative to the cooperative benchmark (range reflects S_T calibration uncertainty). Under the option-value specification, the welfare calculation is fundamentally different: if firms are purchasing an option on a discontinuous capability advance, the "overinvestment" relative to inference revenue is not a market failure but a rational bet on a different payoff entirely. The 22–34% loss becomes the cost of the option rather than a deadweight loss. In either case, the same investment generates consumer surplus through lower prices and earlier access to distributed AI, plus positive externalities of distributed inference—privacy, resilience, and innovation ecosystem diversity.

The two specifications converge on the mechanism and diverge on the post-crossing equilibrium. Under revenue maximization with open-weight erosion of S_T, hyperscalers face the pure displacement case: inference decentralizes and the training moat is commoditized, leaving incumbents structurally diminished. Under the option-value specification, the post-crossing equilibrium depends on whether the capability advance materializes before inference revenue erodes. If it does, the centralized training infrastructure justifies its cost regardless of inference decentralization—the inference revenue was never the point. If it does not—if scaling laws plateau, if the required breakthrough is paradigmatic rather than computational, or if the advance arrives but proves distributable on the same learning curves that decentralized inference—then the option expires and the infrastructure becomes a stranded asset of historically unprecedented scale. The model cannot resolve which specification is correct. It establishes that inference decentralization proceeds identically under both, and that the distinction matters for firm survival but not for the mechanism.

The mechanism predicts a qualitative shift in innovation dynamics post-crossing. A distributed ecosystem of many small producers exploring architectural variations searches the design space differently than a concentrated ecosystem of large producers optimizing along established trajectories. The current proliferation of edge AI silicon—over 50 distinct neural processing unit architectures announced in 2024–2026—is consistent with this prediction.

What the mechanism predicts unambiguously is that concentrated investment endogenously produces inference decentralization, that this process accelerates with the number of competitors and is amplified by asymmetric players who benefit from crossing, and that training centralization and inference decentralization will coexist as stable features of the AI economic landscape.

---

# Appendix A. Two-Period Pedagogical Model

**Setup.** Period 1: N symmetric firms choose investment Iᵢ, earning Cournot profits. Period 2: if ∑Iⱼ exceeds Q̄, distributed inference entry occurs. Incumbents earn S = S_T + S_I per firm.

**Nash equilibrium.** I* = (a–c)/[b(N+1)]. Total investment NI* exceeds Q̄ whenever N is sufficiently large.

**Cooperative benchmark.** Iᶜ = (a–c)/(2b) < NI* for N ≥ 2.

**Corollary A1 (Asymmetric cost).** Lower-cost firm invests more; total investment rises.

**Corollary A2 (Vertical integration).** Firm capturing θ of consumer surplus from decentralization invests more aggressively.

**Corollary A3 (Capacity constraints).** Delay bounded; α invariant.

# Appendix B. Overinvestment in Dollar Terms

**Table B1. Overinvestment calibration.**

| | 2024 | 2025 (prelim.) |
|:---|---:|---:|
| Actual AI capex ($B) | ~230 | ~436 |
| Model Qᴺ/Qᶜ ratio | 3–4× | 3–4× |
| Implied cooperative ($B) | ~65–75 | ~110–145 |
| Excess investment ($B) | ~155–165 | ~291–326 |

The excess is not deadweight loss—it transfers surplus to consumers through the learning curve. A portion builds training infrastructure whose value persists post-crossing (S_T > 0). Under the option-value specification, the excess may understate actual investment: if firms optimize against a discontinuous capability payoff rather than inference revenue, the Qᴺ/Qᶜ ratio from the revenue-maximization model is a lower bound on the true overinvestment ratio.

# Appendix C. Semi-Endogenous Coordination Dynamics

Under declining κ and declining Q̄_eff, the state variable x(t) = Q̄_eff(η(t)) – Q(t) evolves as:

> dx/dt = (dQ̄_eff/dη)·(dη/dt) – Σqᵢ

The first term is negative: the effective threshold drifts toward firms as algorithmic efficiency improves. Under the quasi-static approximation (|dQ̄_eff/dt| ≪ |Σqᵢ|), the Nash equilibrium applies pointwise. This approximation is reasonable for smooth algorithmic progress but breaks down at discrete jumps—e.g., major model releases that shift Q̄_eff substantially within a single quarter. The model handles such events as discrete regime shifts.

## C.1 The Two-Gap Decomposition

At any date t, x(t) = [Q̄_eff(η(t)) – Q̄*(κ(t))] + [Q̄*(κ(t)) – Q(t)], where the first bracket is the "coordination gap" (how much Q̄ exceeds the R₀-adjusted threshold) and the second is the "production gap" (how far production is from the R₀-adjusted threshold).

## C.2 Quasi-Static Approximation

If κ evolves on a timescale τ_κ ≫ 1/(Nq̄), where q̄ is the average production rate, firms treat Q̄*(κ) as locally constant. The formal model applies at each "freeze" of κ. Timescale separation: coordination dynamics evolve over years (standards bodies, developer ecosystem formation); production decisions are quarterly.

## C.3 Feedback Loop

Larger Q → lower prices → broader adoption → faster κ decline (more developers, more tooling investment) → lower Q̄* → faster perceived approach to crossing → more aggressive production. This positive feedback loop amplifies the Nash overinvestment dynamic, compressing T* beyond the static model's prediction. Algorithmic efficiency gains from the open-weight ecosystem operate through the same channel: each major model release that reduces Q̄_eff narrows the effective gap, which under the Nash dynamic induces more aggressive production from symmetric firms.

---

# References

Acemoglu, D., & Guerrieri, V. (2008). Capital deepening and nonbalanced economic growth. *Journal of Political Economy*, 116(3), 467–498.

Alibaba Cloud (2026). Qwen model family surpasses 700 million downloads. Press release, January 2026.

Arrow, K. J. (1962). The economic implications of learning by doing. *Review of Economic Studies*, 29(3), 155–173.

Bass, F. M. (1969). A new product growth for model consumer durables. *Management Science*, 15(5), 215–227.

Bresnahan, T. F., & Greenstein, S. (1994). The competitive crash in large-scale commercial computing. NBER Working Paper No. 4901.

Bresnahan, T. F., & Trajtenberg, M. (1995). General purpose technologies: Engines of growth? *Journal of Econometrics*, 65(1), 83–108.

Christensen, C. M. (1997). *The Innovator's Dilemma.* Harvard Business School Press.

Cournot, A. A. (1838). *Recherches sur les principes mathématiques de la théorie des richesses.* Paris: Hachette.

David, P. A. (1990). The dynamo and the computer. *American Economic Review*, 80(2), 355–361.

Deloitte (2025). More compute for AI, not less. *Technology, Media, and Telecom Predictions 2026.*

Economides, N. (1996). The economics of networks. *International Journal of Industrial Organization*, 14(6), 673–699.

Epoch AI (2025). Most of OpenAI's 2024 compute went to experiments. *Data Insights.*

Flamm, K. (1993). *Mismanaged Trade?* Brookings Institution.

Greenstein, S. (1997). Lock-in and the costs of switching mainframe computer vendors. *Industrial and Corporate Change*, 6(2), 247–273.

Gyllenberg, M., & Parvinen, K. (2001). Necessary and sufficient conditions for evolutionary suicide. *Bulletin of Mathematical Biology*, 63(5), 981–1015.

Irwin, D. A., & Klenow, P. J. (1994). Learning-by-doing spillovers in the semiconductor industry. *Journal of Political Economy*, 102(6), 1200–1227.

Jackson, M. O., & Yariv, L. (2007). Diffusion of behavior and equilibrium properties in network games. *American Economic Review*, 97(2), 92–98.

Kermack, W. O., & McKendrick, A. G. (1927). A contribution to the mathematical theory of epidemics. *Proceedings of the Royal Society A*, 115(772), 700–721.

Lambert, N. (2025). The ATOM Project: American Truly Open Models. *Hugging Face Analysis.*

Levhari, D., & Mirman, L. J. (1980). The great fish war: An example using a dynamic Cournot-Nash solution. *Bell Journal of Economics*, 11(1), 322–334.

MarketsandMarkets (2025). AI inference market size, share & growth, 2025 to 2030.

Maynard Smith, J. (1982). *Evolution and the Theory of Games.* Cambridge University Press.

McKinsey (2025). The next big shifts in AI workloads and hyperscaler strategies.

Milnor, J. (1963). *Morse Theory.* Princeton University Press.

MIT Technology Review (2025). We did the math on AI's energy footprint.

MIT Technology Review (2026). What's next for Chinese open-source AI. February 12, 2026.

Odling-Smee, F. J., Laland, K. N., & Feldman, M. W. (2003). *Niche Construction.* Princeton University Press.

OpenRouter & Andreessen Horowitz (2025). Open-source LLM usage analysis: 100 trillion tokens.

Rosen, J. B. (1965). Existence and uniqueness of equilibrium points for concave N-person games. *Econometrica*, 33(3), 520–534.

Schumpeter, J. A. (1942). *Capitalism, Socialism and Democracy.* Harper & Brothers.

Smirl, C. (2026). The monetary productivity gap. Unpublished thesis, Tufts University.

Stanford University (2025). AI Index Report 2025. *Human-Centered Artificial Intelligence.*

Stokey, N. L. (1988). Learning by doing and the introduction of new goods. *Journal of Political Economy*, 96(4), 701–717.

Thompson, N. C., & Spanuth, S. (2023). The decline of computers as a general purpose technology. *Communications of the ACM*, 64(3), 64–72.

Tirole, J. (1988). *The Theory of Industrial Organization.* MIT Press.

Walter, W. (1998). *Ordinary Differential Equations.* Springer.

Wright, S. (1932). The roles of mutation, inbreeding, crossbreeding and selection in evolution. *Proceedings of the Sixth International Congress of Genetics*, 1, 356–366.

Wright, T. P. (1936). Factors affecting the cost of airplanes. *Journal of the Aeronautical Sciences*, 3(4), 122–128.

Young, A. (1991). Learning by doing and the dynamic effects of international trade. *Quarterly Journal of Economics*, 106(2), 369–405.

Young, H. P. (2009). Innovation diffusion in heterogeneous populations. *American Economic Review*, 99(5), 1899–1924.
