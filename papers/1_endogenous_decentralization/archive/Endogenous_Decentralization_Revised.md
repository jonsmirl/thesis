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

Third, cross-domain empirical analysis reveals structural parameter consistency: semiconductor learning elasticities cluster at α ∈ [0.18, 0.25] across DRAM (41 years, α = 0.22, R² = 0.96), HBM, Intel microprocessors, NAND flash, and solar PV. Historical transitions (mainframe→PC, ARPANET→Internet) exhibit a coordination layer pattern: hardware crossing precedes market dominance by 3–5 years, mediated by a platform that converts device capability into ecosystem value.

Fourth, the model incorporates a critical structural distinction between training and inference workloads. Training—the creation of frontier AI models—requires tightly synchronized GPU clusters at scales that remain architecturally incompatible with distributed execution. Inference—the deployment of trained models to serve user requests—is atomizable, latency-sensitive, and follows precisely the decentralization trajectory the model predicts. The mechanism's primary empirical domain is the inference workload, which already constitutes an estimated 80–90% of AI compute cycles and is projected to represent a $255 billion market by 2030. The post-crossing equilibrium is partial decentralization: inference distributes while training persists centrally, analogous to mainframe computing's stable niche persistence decades after the PC revolution.

Fifth, the model generates nine falsifiable predictions with specific timing and failure conditions for the current AI infrastructure buildout ($1.3 trillion cumulative hyperscaler capex, 2018–2025). Calibrated to HBM learning rates and observed capex, the model predicts inference crossing by approximately 2028 under Nash competition—a 79% acceleration relative to the cooperative benchmark.

**Keywords:** endogenous decentralization, learning curves, Markov Perfect Equilibrium, architectural substitution, AI infrastructure, training-inference bifurcation, basic reproduction number

**JEL:** O33, L16, D43, C73

---

# 1. Introduction

Between 2018 and 2025, the five largest technology companies—together with Oracle and the Stargate joint venture—committed an estimated $1.3 trillion in cumulative capital expenditure to construct centralized AI infrastructure. This represents the largest concentrated infrastructure investment in history outside wartime mobilization.

This paper argues that this investment is *endogenously self-disrupting*: the very act of building centralized AI datacenters finances the component learning curves—particularly in high-bandwidth memory, model compression, and advanced packaging—that will enable distributed alternatives to replicate datacenter-class inference on consumer devices. The mechanism is not novel in observation—economists studying the mainframe-to-PC transition have noted analogous dynamics—but it has not been previously formalized as a general theoretical result with testable predictions.

A critical structural distinction governs the mechanism's scope. AI datacenter workloads divide into two fundamentally different activities: *training* (teaching models by processing massive datasets across tightly synchronized GPU clusters) and *inference* (running trained models to serve real-time user requests). Training requires massive interconnect bandwidth, weeks of continuous synchronized operation, and scales that remain architecturally incompatible with distributed execution. Inference, by contrast, is atomizable—individual tasks can be handled independently—and latency-sensitive, creating a natural advantage for processing close to users. The endogenous decentralization mechanism applies directly and powerfully to inference. Training may remain permanently centralized, not because learning curves fail to reduce its costs, but because the synchronization and bandwidth requirements are architectural constraints that cost reduction alone cannot overcome.

This distinction is not a limitation of the theory—it sharpens it. Inference already constitutes an estimated 80–90% of AI compute cycles (MIT Technology Review 2025) and is projected to grow from approximately $106 billion (2025) to $255 billion by 2030 (MarketsandMarkets 2025). The inference revenue pool is the primary target of the $1.3 trillion infrastructure investment, and it is the revenue pool that edge devices will intercept. Training, while essential for producing frontier models, represents a declining share of total AI compute (from approximately two-thirds in 2023 to one-third by 2026, per Deloitte 2025) and serves a structurally different function in the value chain.

The contribution is five-fold. First, the formal mechanism: a continuous-time differential game with exact closed-form solutions, in which symmetric Markov Perfect Equilibrium produces overinvestment that accelerates crossing time T* (Proposition 1). Second, a generalized crossing condition: the cost-parity threshold extends to a self-sustaining adoption threshold R₀ > 1 that incorporates network effects, coordination friction, and latency advantage, endogenizing the empirically observed coordination layer lag. Third, cross-domain parameter consistency: semiconductor learning elasticities cluster at α ∈ [0.18, 0.25] across products, firms, and decades, and historical transitions exhibit a 3–5 year coordination layer lag. Fourth, the training-inference bifurcation: a structural analysis showing that the mechanism's primary domain is inference, with implications for post-crossing continuation values and incumbent pivot strategies. Fifth, nine falsifiable predictions with timing for the current AI infrastructure cycle.

The paper is organized as follows. Section 2 develops the three-stage mechanism. Section 3 presents the formal model, proves the overinvestment result in continuous time, establishes robustness, and extends the crossing condition to a self-sustaining adoption threshold. Section 4 presents the primary empirical domain, including the training-inference structural distinction. Section 5 validates parameter consistency across historical transitions. Section 6 offers predictions. Section 7 concludes.

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

Increased centralized investment accelerates the centralized paradigm's own displacement—specifically, displacement of its inference revenue. This follows from the learning curve's dependence on cumulative production but has not been formalized in the existing literature.

The self-undermining property applies to the inference revenue pool, not to total datacenter value. If centralized firms retain training capabilities and develop model-licensing businesses, the post-crossing continuation value is materially higher than under full displacement. Section 3.7 calibrates this distinction.

## 2.3 Distinction from Adjacent Theory

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

Consider N ≥ 2 symmetric centralized firms indexed by i ∈ {1,...,N}. Time is continuous. The *state variable* is x(t) = Q̄ – Q(t) ∈ [0, x₀], measuring the remaining cumulative production until the crossing threshold Q̄ at which distributed architecture becomes cost-competitive for inference workloads. When x reaches zero, inference crossing occurs. The state evolves as:

> dx/dt = –Σᵢ qᵢ(t)     (3)

where qᵢ(t) ≥ 0 is firm i's output rate. Each unit of output serves the centralized market and simultaneously depletes the remaining distance to crossing—this dual role is the formal expression of the self-undermining investment property.

Flow profits for firm i are determined by linear inverse demand P = a – bQ, where Q = Σqⱼ is total output rate:

> πᵢ(t) = (a – bQ)qᵢ     (4)

with a > 0, b > 0. Upon crossing (x = 0), each firm receives continuation value:

> S = S_T + S_I/(N(r + δ))     (5)

where S_T represents the persistent training and model-licensing revenue that survives inference decentralization, S_I = π̄_I is the pre-crossing inference profit level, r is the discount rate, and δ > 0 is the post-crossing inference displacement rate. The decomposition S = S_T + S_I/(N(r + δ)) reflects a structural distinction: training revenue S_T is a persistent asset that does not decay with inference displacement, while inference rents S_I decay at rate δ as distributed alternatives capture inference workloads. The discount rate is r > 0.

When S_T = 0, the model reduces to the pure displacement case. When S_T > 0, incumbents retain value post-crossing, moderating (but not eliminating) the overinvestment externality.

The game has a *common-pool* structure: the state x is a shared resource (remaining time before inference disruption) that all firms deplete through production. Each firm's output generates private flow rents but simultaneously brings the crossing point closer for all firms. This structure is analogous to the fishery or oil extraction commons (Levhari and Mirman 1980), with the critical distinction that the "resource" being depleted is the incumbent paradigm's remaining inference viability.

## 3.2 Markov Perfect Equilibrium

I restrict attention to symmetric stationary Markov strategies qᵢ = q(x), where output depends only on the current state. Each firm's value function V(x) satisfies the Hamilton-Jacobi-Bellman equation:

> rV(x) = max_{qᵢ} {(a – b(qᵢ + (N–1)q(x)))qᵢ – V′(x)·(qᵢ + (N–1)q(x))}     (5)

The first term is flow profit under Cournot competition. The second captures the cost of state depletion: V′(x) > 0 because value increases in distance to crossing, so production that reduces x destroys value. The first-order condition under symmetry is:

> a – b(N+1)q – V′(x) = 0     (6)

yielding the equilibrium strategy:

> qᴺ(x) = [a – Vᴺ′(x)] / [b(N+1)]     (7)

Substituting (7) back into the HJB and simplifying yields a first-order nonlinear ODE for Vᴺ. The algebra: flow profit in symmetric equilibrium is (a – bNq)q where a – bNq = (a + NV′)/(N+1); the depletion cost is V′·Nq. Combining:

> rVᴺ(x) = (a – Vᴺ′(x))(a – N²Vᴺ′(x)) / [b(N+1)²]     (ODE-N)

with boundary condition Vᴺ(0) = S. This ODE is solved numerically via backward shooting from x = 0.

## 3.3 Cooperative Benchmark

The cooperative planner maximizes total producer surplus W(x) = NVᴾ(x), choosing total output rate Q. This is a cartel benchmark—it maximizes incumbent welfare, not total social welfare including consumers—and therefore understates the socially optimal investment rate if consumer surplus from decentralization is large:

> rW(x) = max_Q {(a – bQ)Q – W′(x)·Q}     (8)

The planner's FOC yields cooperative total output:

> Qᶜ(x) = [a – NVᴾ′(x)] / (2b)     (9)

and per-firm value satisfies:

> rVᴾ(x) = (a – NVᴾ′(x))² / (4bN)     (ODE-C)

with boundary condition Vᴾ(0) = S.

## 3.4 Analytical Solutions

Both ODEs are autonomous (no explicit x dependence) and therefore separable. The cooperative ODE V′ = (a – √(4bnrV))/n separates to dx = n·dV/(a – √(4bnrV)). Integration yields the exact implicit solution:

> x(V) = [a·ln((a – 2√(bnrS))/(a – 2√(bnrV))) + 2(√(bnrS) – √(bnrV))] / (2br)     (C-exact)

where S = V(0) is the boundary value. The Nash ODE is solved by the substitution u = √(D + EV). Defining A = a(1+N²) and u₀ = √(D + ES):

> x(V) = (4N²/E)·[(u₀ – u) + A·ln((A – u₀)/(A – u))]     (N-exact)

where u(V) = √(D + EV). Both solutions share the same functional form—√ + log—differing only in the constants governing shadow cost internalization. The cooperative solution embeds the full social shadow cost NVᴾ′; the Nash solution embeds only the private cost Vᴺ′. Both are implicit in V but invertible numerically, yielding exact crossing times to machine precision (max |x_exact – x_num| < 10⁻¹²; see Figure 1).

[Figure 1]

*Figure 1. Verification of closed-form solutions against RK4 numerical integration. Left: exact (dotted) and numerical (solid) value functions overlay perfectly. Right: verification error on log scale; all errors at floating-point precision.*

## 3.5 The Overinvestment Result

**Proposition 1 (Overinvestment in Markov Perfect Equilibrium).** *In the symmetric MPE, aggregate output Qᴺ(x) = Nqᴺ(x) strictly exceeds cooperative output Qᶜ(x) for all x > 0. Consequently, T*ᴺᵃˢʰ < T*ᶜᵒᵒᵖ: Nash equilibrium crossing occurs strictly earlier than the cooperative optimum.*

**Proof.** The proof compares equilibrium output levels by comparing shadow costs of state depletion across the Nash and cooperative regimes.

*Step 1: Shadow cost comparison at the boundary.* At x = 0, both value functions satisfy Vᴺ(0) = Vᴾ(0) = S. The boundary derivatives are determined by evaluating (ODE-N) and (ODE-C) at x = 0. Define λ = Vᴺ′(0) and μ = Vᴾ′(0). From (ODE-N):

> rS = (a – λ)(a – N²λ) / [b(N+1)²]     (10)

From (ODE-C):

> rS = (a – Nμ)² / (4bN)     (11)

Setting (10) equal to (11) and solving yields Nμ > λ for N ≥ 2. The planner's *total* shadow cost of state depletion (Nμ, aggregated across all N firms) strictly exceeds the Nash firm's *private* shadow cost (λ). This gap reflects the learning externality: each Nash firm internalizes only its own future profit loss from approaching crossing, while the planner internalizes the full N-fold social cost.

*Step 2: Propagation to the interior.* Both (ODE-N) and (ODE-C) are autonomous first-order ODEs in V(x). By a standard comparison theorem for ODEs (see, e.g., Walter 1998, Theorem I.9.1), the ordering N·Vᴾ′(x) > Vᴺ′(x) established at the boundary propagates to all x > 0, provided the right-hand sides of the ODEs satisfy a Lipschitz condition—which holds because a, b, N, r > 0 and V′ is bounded on compact intervals.

*Step 3: Output comparison.* From (7) and (9):

> Qᴺ(x) = N(a – Vᴺ′(x)) / [b(N+1)]

> Qᶜ(x) = (a – NVᴾ′(x)) / (2b)

Since NVᴾ′(x) > Vᴺ′(x), the numerator of Qᶜ is smaller than a – Vᴺ′(x). Moreover, the denominator of Qᴺ satisfies b(N+1)/N < 2b for N ≥ 2. Both effects push Qᴺ above Qᶜ. Therefore Qᴺ(x) > Qᶜ(x) for all x > 0. Since crossing time is T* = inf{t : x(t) = 0} and x evolves as dx/dt = –Q, faster aggregate output implies earlier crossing. ■

**Remark 1 (Irreversibility).** The crossing possesses a robustness property stronger than the comparative statics of Corollaries 1–3. At Q = Q̄, the topology of the viable configuration space changes: a new basin of attraction—the distributed inference equilibrium—becomes accessible. In the language of Morse theory (Milnor 1963), this corresponds to a handle attachment of index 1, creating a path between the centralized and distributed equilibria that did not previously exist. By the stability theorem for Morse functions, this topological change persists under all sufficiently small perturbations of the cost function C(Q). A temporary supply shock that raises distributed costs does not reverse the crossing; it shifts the saddle point but does not remove it. Reversing the handle attachment would require cumulative production Q(t) to decrease—which contradicts the monotonicity of cumulative output. Once Q crosses Q̄, the inference transition is topologically irreversible: no continuous change in market parameters (α, N, demand elasticity) can restore the pre-crossing configuration space.

**Remark 2 (Partial Displacement and Niche Persistence).** Irreversibility of inference crossing does not imply extinction of the centralized paradigm. The historical record demonstrates that centralized architectures persist indefinitely in niche applications even after distributed alternatives capture the majority of the addressable market. IBM's mainframe business continues to generate approximately $3–4 billion annually in revenue as of 2025—decades after the PC revolution—serving high-reliability transaction processing in banking, airline reservations, and government systems. The mainframe did not die; the *addressable market* expanded so dramatically that mainframes went from 100% of computing to a stable but small fraction. The analog for AI infrastructure is that centralized datacenters will persist for frontier model training, the most demanding enterprise inference workloads, and applications requiring petabyte-scale data proximity—while the enormous growth in AI inference demand (billions of edge devices, every phone, vehicle, and appliance) will be served primarily by distributed architectures.

The overinvestment result does not depend on the linear demand specification. Under any downward-sloping inverse demand P(Q) with P′ < 0, symmetric Nash firms underprice the shadow cost of state depletion by a factor of 1/N relative to the cooperative benchmark, because each firm's FOC includes only its private V′(x) rather than the social cost NVᴾ′(x). The linear form permits closed-form solutions but the common-pool externality driving Proposition 1 is structural.

**Economic interpretation.** The overinvestment decomposes into two channels visible in the FOC (6). The term b(N+1)q includes bNq, the price-depressing effect of total rival output—the standard Cournot externality. The term V′(x) is the private shadow cost of approaching crossing, which is 1/N of the social shadow cost NVᴾ′(x)—the learning externality unique to endogenous decentralization. Both channels push Nash output above the cooperative level. In the limit δ → 0 (no post-crossing displacement), V′ → 0 and the model reduces to standard Cournot. In the limit b → 0 (perfectly elastic demand), only the learning externality remains.

The decomposition of S into S_T + S_I/(N(r + δ)) reveals a moderating effect: when S_T is large (incumbents retain substantial training revenue post-crossing), the shadow cost V′(x) is lower because crossing is less catastrophic. This *reduces* the overinvestment gap relative to the pure displacement case (S_T = 0), but does not eliminate it. The strategic interaction—each firm internalizing only 1/N of the social shadow cost—persists regardless of S_T. The overinvestment result holds for any S ≥ 0.

[Figure 2]

*Figure 2. Aggregate output: Nash overinvestment. The shaded region shows the gap Q_N(x) – Q_C(x). Nash output exceeds cooperative output at all interior states.*

[Figure 3]

*Figure 3. The learning externality. Each Nash firm's private shadow cost V_N′(x) (blue) captures only a fraction of the social shadow cost NV_P′(x) (red).*

**Welfare loss.** The overinvestment externality imposes a quantifiable welfare cost. At the baseline calibration (N = 5, S_T = 0), per-firm value under Nash competition Vᴺ(x₀) = 19.22 falls 34.1% below the cooperative value Vᴾ(x₀) = 29.16. The total welfare loss is ΔW = N·(Vᴾ – Vᴺ) = 49.7 in normalized units. Each firm would prefer to slow production, but unilateral restraint is not individually rational: reducing output merely transfers learning-curve contributions to rivals while forgoing current revenue.

When S_T > 0, the welfare loss moderates. At S_T calibrated to the estimated training revenue floor (Section 3.7), the per-firm welfare loss falls to approximately 22–28%, reflecting the reduced stakes of inference crossing when training revenue persists. The qualitative result—Nash welfare strictly below cooperative welfare—is unchanged.

This welfare comparison captures producer surplus only. It does not account for consumer gains from lower prices during the transition or from earlier access to decentralized technology. Whether the Nash outcome is socially preferable depends on the relative weight of producer versus consumer surplus—a question the model identifies but does not resolve.

## 3.6 Comparative Statics

**Corollary 1 (Increasing N).** *Nash equilibrium aggregate output Qᴺ(x) is strictly increasing in N for all x > 0.*

As N increases, each firm internalizes a smaller share (1/N) of the social shadow cost. The private shadow cost Vᴺ′(x) falls relative to the social cost, expanding equilibrium output at every state. From (7), ∂Qᴺ/∂N > 0 follows by differentiating the equilibrium ODE system with respect to N and applying the implicit function theorem. Empirically, the number of entities making $10B+ AI infrastructure commitments has expanded from approximately five (2023) to ten or more, including sovereign wealth funds and national programs. The model predicts this accelerates T*.

**Corollary 2 (Asymmetric firms).** *If firm 1 has marginal cost c₁ – ε with ε > 0 and all other firms retain cost c₁, aggregate equilibrium output is strictly increasing in ε.*

Firm 1's best response shifts outward; rivals contract by less than firm 1 expands. Summing the N asymmetric FOCs and applying the IFT on the resulting system yields dQᴺ/dε > 0. Any asymmetry in capital endowment or productivity that shifts one firm's best response outward increases aggregate output. Capital concentration among AI investors compresses T*.

**Corollary 3 (Capacity constraint and boom-bust).** *If aggregate output is constrained by fabrication capacity κ(t) such that Σqᵢ ≤ κ(t), crossing time delay is bounded by the construction lag Δ for new capacity. The long-run learning rate α is unaffected.*

When the constraint binds, all firms produce at capacity and the Nash overshooting is temporarily suppressed. But excess demand creates incentives for capacity investment. New capacity arrives with lag Δ, at which point Nash overshooting resumes and may itself overshoot due to coordination failure among capacity investors. The learning rate α—a cumulative relationship between cost and total output—is invariant to the rate of production because it depends only on cumulative Q (Equation 1). The 41-year DRAM panel confirms this empirically: at least four boom-bust cycles left the long-run α = 0.22 unchanged across sub-periods.

Taken together, all three perturbations either preserve or compress T*. The self-undermining property is robust precisely because the learning curve is a *cumulative* relationship: any force increasing total output over time moves the system along the same invariant cost-reduction trajectory. No plausible market structure perturbation reverses the sign of ∂T*/∂I.

**Why not collude?** If incumbents recognize they are financing their own inference disruption, why don't they tacitly coordinate to slow investment? The Nash equilibrium provides the answer: unilateral restraint is not individually rational. A firm that reduces output forfeits current revenue and market share while rivals continue producing, and the learning curve advances regardless—driven by rivals' cumulative output. The common-pool structure ensures that the only way to slow crossing is for all firms to restrain simultaneously, which requires enforceable coordination. Antitrust law prohibits exactly this. The result is a prisoner's dilemma in which each firm's dominant strategy is to invest aggressively, collectively accelerating the transition that each would prefer to delay.

The investment equilibrium is in fact stronger than a prisoner's dilemma. In the language of evolutionary game theory, the Nash investment intensity constitutes an evolutionarily stable strategy (Maynard Smith 1982): no alternative investment rule can invade a population playing this strategy, even in small numbers. Yet the ESS is collectively self-undermining for inference rents. Through what evolutionary biologists call niche construction (Odling-Smee, Laland, and Feldman 2003)—organisms altering their environment through normal metabolic activity—centralized firms' cumulative production modifies the competitive landscape in a direction that favors distributed inference entrants. The parallel is instructive but imperfect: unlike true evolutionary suicide (Gyllenberg and Parvinen 2001), the centralized firms are not rendered extinct. They undergo *evolutionary metamorphosis*—their training and model-creation capabilities persist and may even appreciate in value, while their inference revenue erodes. The ESS ensures that the inference transition proceeds at maximum speed, but the organism survives in altered form, much as IBM persisted through its mainframe-to-services metamorphosis.

The R₀ framing developed in Section 3.9 provides the epidemiological mechanism for this result: the centralized firms' ESS production strategy simultaneously increases the distributed alternative's transmissibility β (by reducing costs through the learning curve) and contributes to reducing recovery rate κ + μ (by funding coordination layer maturation through the ecosystem investment it stimulates).

[Figure 4]

*Figure 4. Crossing time as a function of N. Left: absolute crossing times under Nash and cooperative regimes. Right: Nash crossing as a fraction of cooperative.*

## 3.7 Calibration

The model is calibrated from observable data and yields predictions in calendar years. The learning elasticity α = 0.23 from the HBM packaging curve (Table 2). Current HBM cost is approximately $12/GB (HBM3E, 2024); the crossing threshold is $5–7/GB, the price point at which 50GB consumer stacked memory becomes viable. From the learning curve c(Q) = c₀·Q⁻ᵅ, fitting to current cost and cumulative production (~2.5 exabytes through 2024) gives a crossing threshold range of Q̄ ≈ 26–112 EB (10–45× current cumulative production), depending on the target price point within the $5–7/GB range. The calibration uses the conservative bound Q̄ ≈ 112 EB ($5/GB) for all timing predictions; if the effective crossing threshold is $7/GB, the required cumulative production falls to ~26 EB and all timing predictions compress accordingly.

**Calibrating the post-crossing continuation value.** The decomposition S = S_T + S_I/(N(r + δ)) requires separate calibration of training persistence and inference displacement.

The inference displacement rate δ ≈ 0.30 from the IBM inference analogy: IBM lost approximately 60% of its compute-service profit pool within three years of the coordination layer maturing (1990–1993). However, IBM's revenue losses reflected both training-equivalent and inference-equivalent workloads migrating simultaneously. In the AI case, only inference migrates; training persists.

The training persistence value S_T is calibrated from the structural economics of frontier model training. As of 2025, the largest training runs cost $100M–$1B+ in compute alone. The number of organizations capable of training frontier models (>10T parameters by 2028) is structurally limited by capital requirements, data access, and talent concentration. A conservative estimate of S_T assumes training-as-a-service and model-licensing revenue of 20–40% of current total revenue per firm, capitalizing at S_T ≈ 0.2–0.4 times the pre-crossing per-firm value. The key uncertainty is whether open-source model proliferation (Llama, DeepSeek) erodes this training moat, which would push S_T toward zero and restore the pure displacement case.

The discount rate r = 0.05. The baseline competitive structure N = 5 matches the hyperscaler set (Microsoft, Google, Amazon, Meta, and one rotating entrant) through 2023–24; the expanded set N = 10–12 reflects the entry of sovereign funds and additional investors in 2025–26.

**Quantitative predictions.** Under Nash competition with N = 5 hyperscalers and S_T = 0 (pure displacement), the model predicts inference crossing in approximately 3.8 years from 2024—that is, by ~2028. Under the cooperative benchmark, crossing would not occur until ~2042 (18.3 years). Competition accelerates the transition by 79%. Expanding the investor set to N = 12 compresses the Nash crossing time by an additional ~13%, to approximately 3.3 years (~2027).

With S_T > 0 (training persistence), the acceleration ratio moderates to approximately 70–75% but remains substantial. The qualitative result—Nash crossing dramatically preceding cooperative crossing—is robust to the S_T calibration.

**Overinvestment magnitude.** The model's output ratio Qᴺ/Qᶜ ≈ 3–4× at the baseline calibration provides a structural interpretation of the observed capex trajectory. Against projected 2025 AI infrastructure spending of ~$436B (Table 4), the model implies a "rational monopolist" benchmark of ~$145B, with the excess ~$291B attributable to the overinvestment externality. This estimate is illustrative; a precise mapping requires demand elasticity estimates from cloud inference pricing data, which I leave to future work.

[Figure 5]

*Figure 5. Crossing time sensitivity to displacement rate δ. Nash crossing time (blue) is nearly invariant to δ, while cooperative crossing (red) is highly sensitive.*

[Figure 6]

*Figure 6. Per-firm value functions. The cooperative value V_P(x) exceeds the Nash value V_N(x) at every interior state.*

## 3.8 Note on Identification

The DRAM learning curve is estimated by OLS regression of log price on log cumulative output. This identifies a correlation, not necessarily a structural learning-by-doing parameter in Arrow's (1962) strict sense. The observed α could reflect pure learning, scale economies, or correlated exogenous technical change. The mechanism's predictions are robust to this ambiguity: regardless of whether α reflects learning, scale, or technical change, the key property—that cumulative production is associated with cost decline at a stable elasticity—holds across the full 41-year panel and across sub-periods.

This paper follows the quantitative calibration tradition in growth economics (e.g., Stokey 1988; Young 1991; Acemoglu and Guerrieri 2008) rather than claiming causal identification of the learning parameter. The calibration approach takes c(Q) = c₀Q⁻ᵅ as an empirical regularity—stable across four decades, robust to boom-bust cycles, consistent across semiconductor generations—and derives equilibrium implications. The self-undermining property (∂T*/∂I < 0) requires only that centralized investment contributes to cumulative Q and that c(Q) is decreasing and stable.

If future work identifies instruments for cumulative production (candidates include the CHIPS Act as a supply-side shock, or geopolitical disruptions to fab location), the structural learning parameter could be estimated separately. The current paper's contribution is the equilibrium mechanism, not the identification of α.

## 3.9 Generalized Crossing Condition

The model as developed in Sections 3.1–3.8 defines crossing at cost parity: distributed inference becomes viable when x(t) = 0, equivalently when c(Q) ≤ c*. But the empirical evidence in Section 5 shows that hardware crossing *precedes* architectural dominance by 3–5 years. The 386 was cost-competitive by 1987; Windows did not dominate until 1990–95. TCP/IP was viable by ~1989; the browser arrived 1993–94. This lag is documented in Table 7 but not explained by the game-theoretic structure.

The lag exists because cost parity is *necessary but not sufficient* for self-sustaining distributed adoption. What is actually required is that the distributed ecosystem's basic reproduction number exceeds unity:

> R₀ ≡ β(c, λ) · γ / (κ + μ) > 1     (12)

where β(c, λ) is the adoption rate, increasing in cost advantage and latency advantage λ; γ is the network effect multiplier (each adopter's contribution to ecosystem value); κ is the coordination friction (cost of building and using the coordination layer); and μ is the churn rate (users reverting to centralized). The latency advantage λ captures a structural property specific to inference: edge devices serving local users achieve response times of <10ms, compared to 50–200ms for cloud round-trip. For interactive AI applications (chat, code completion, real-time translation, agentic workflows), this latency advantage is a direct quality improvement independent of cost. When R₀ < 1, distributed adoption dies out even if hardware is cost-competitive. When R₀ > 1, adoption is self-sustaining. The crossing that matters is R₀ = 1, not c = c*. The threshold terminology is drawn from the SIR epidemic model (Kermack and McKendrick 1927); the technology adoption analog connects to the Bass (1969) diffusion model with network effects, which generates identical threshold dynamics.

The inclusion of latency advantage λ in the adoption function β(c, λ) means that R₀ > 1 can be achieved even before strict cost parity if the latency advantage is sufficiently large. This is empirically relevant: enterprises are already deploying edge inference at higher per-unit cost than cloud inference specifically because sub-10ms latency enables use cases (real-time manufacturing inspection, autonomous vehicle decision-making, medical device AI) that cloud inference cannot serve.

### 3.9.1 Equivalence to a Modified Production Threshold

The key tractability result is that R₀ = 1 is equivalent to a modified cost threshold c**, which maps back to a modified cumulative production threshold Q̄*. Setting R₀ = 1 and solving for the critical distributed cost:

| Adoption function | β specification | c** (from R₀ = 1) |
|:---|:---|:---|
| Linear | β = (c* – c) + λ | c* – (κ+μ)/γ + λ |
| Power law | β = ((c* – c) + λ)ᵖ | c* – ((κ+μ)/γ)^{1/p} + λ |
| Logistic | β = 1/(1+e^{–k(c*–c+λ–c₀)}) | Implicit; numerical |

*Table 8. Critical cost c under alternative adoption function specifications. Linear case used for closed-form exposition.*

Under the linear specification, the learning curve c(Q) = c₀ · Q⁻ᵅ and the condition c = c** imply:

> Q̄* = Q̄ · (1 – (κ + μ)/(γ · c*) + λ/c*)⁻¹/ᵅ     (13)

where Q̄ is the original cost-parity threshold. When λ is sufficiently large, Q̄* < Q̄: the latency advantage pulls the effective crossing threshold *closer*, so less cumulative production is needed before the distributed ecosystem becomes self-sustaining.

**Definition 3 (Regime boundary).** The transition is *friction-dominated* if β(0, λ)·γ < κ + μ and *network-dominated* if β(0, λ)·γ > κ + μ.

The relationship between Q̄* and Q̄ is regime-dependent. In the friction-dominated regime, R₀ < 1 at cost parity, so self-sustaining adoption requires costs to fall *below* parity—a cost cushion to overcome switching costs—and Q̄* > Q̄. The gap Q̄* – Q̄ is the coordination layer deficit. This is the empirically relevant case during the pre-crossing period when the coordination layer is immature. In the network-dominated regime, R₀ > 1 *before* cost parity: network effects and latency advantage compensate for the cost disadvantage, and Q̄* < Q̄. Self-sustaining adoption arrives sooner than pure cost parity predicts.

### 3.9.2 The Model Is Unchanged

Replace Q̄ with Q̄*(κ) throughout the model, treating κ as a fixed parameter reflecting coordination infrastructure that evolves on institutional timescales outside the game. The state variable becomes:

> x(t) = Q̄*(κ) – Q(t)     (14)

All propositions, comparative statics, and closed-form solutions carry through identically with Q̄* in place of Q̄. Firms are myopic with respect to coordination dynamics: they observe the current state x = Q̄*(κ) – Q and optimize production rates taking Q̄* as fixed. This is natural when coordination investment is diffuse—built by thousands of independent developers, standards bodies, and infrastructure providers—and no firm can individually influence κ.

**Proposition 1* (Generalized Overinvestment).** *Under fixed κ: in the symmetric MPE with generalized crossing condition R₀ = 1, aggregate output Qᴺ(x) strictly exceeds cooperative output Qᶜ(x) for all x > 0, where x = Q̄*(κ) – Q. Nash equilibrium crossing occurs strictly before cooperative crossing: T*ᴺᵃˢʰ < T*ᶜᵒᵒᵖ.*

*Proof.* Identical to Proposition 1 with Q̄ replaced by Q̄*(κ). The HJB equation, FOCs, and shadow cost comparison are functions of the state x, not of the threshold's composition. The boundary condition V(0) = S is unchanged because post-crossing displacement dynamics are independent of how the threshold is defined. ■

*Remark 3.* In the friction-dominated regime (Q̄* > Q̄), crossing occurs later in absolute time than the cost-parity model predicts, but the acceleration ratio T*ᴺᵃˢʰ/T*ᶜᵒᵒᵖ is preserved: Nash competition accelerates crossing relative to cooperation regardless of which regime applies. In the network-dominated regime (Q̄* < Q̄), crossing arrives earlier than cost parity predicts. The regime structure enriches the model's predictions without altering its strategic core.

### 3.9.3 The Coordination Layer Lag Becomes Endogenous

This framework explains the 3–5 year lag structurally rather than by assertion. The explanation applies in the friction-dominated regime, which characterizes all historical transitions during their early phases.

**Hardware crossing** occurs when Q(t) = Q̄ (cost parity). In the friction-dominated regime, R₀ < 1 at this point because κ is still high—the coordination layer has not matured. **R₀ crossing** occurs when Q(t) = Q̄*(κ(t)). The lag between hardware crossing and R₀ crossing is:

> ΔT = T*_{R₀} – T*_{hardware} > 0     (friction-dominated)     (15)

This lag depends on the initial coordination friction κ₀, the coordination learning rate, and the responsiveness of ecosystem investment to cost advantage. While the model treats κ as fixed within the game, the economic motivation for why κ declines over time is straightforward: as hardware costs fall, the expected returns to coordination-layer investment rise, attracting ecosystem development. Standards bodies convene, developer tools mature, settlement infrastructure grows. The coordination friction κ is not fixed in the real economy—it declines as the coordination layer matures, and the cost advantage created by centralized overinvestment drives that maturation.

The qualitative dynamics require only that κ is monotonically decreasing in cumulative ecosystem investment, that ecosystem investment is increasing in the cost advantage created by centralized investment, and that declining κ reduces the effective threshold Q̄*(κ). These three properties create a two-sided convergence: Q(t) rises toward Q̄*(t) while Q̄*(t) falls toward Q(t). The specific functional form of κ's decline is not pinned down by the theory—exponential, power-law, and discontinuous threshold functions (where κ drops when a dominant standard emerges) all satisfy these conditions. A formal treatment of declining κ requires a timescale separation assumption: coordination infrastructure evolves slowly (institutional years) relative to production decisions (quarterly cycles), so that firms can treat Q̄* as locally constant when solving their HJB. This quasi-static approximation is developed in Appendix C; the fixed-κ results of Propositions 1 and 1* are exact and require no such assumption.

| Transition | Hardware T* | R₀ T* | ΔT | Implied κ₀/λ |
|:---|:---|:---|:---|:---|
| Mainframe → PC | 1987 | 1990–92 | 3–5 yr | ~15 |
| ARPANET → Internet | ~1989 | 1993–94 | 4–5 yr | ~18 |
| Cloud → Edge AI | 2028–30 | ? | ? | ? |

*Table 9. Coordination layer lag across technology transitions. Two historical data points cannot establish a structural regularity; the similar implied κ₀/λ ratios are suggestive but not conclusive.*

The more interesting falsifiable implication concerns the AI transition specifically. The coordination layer for distributed AI inference is already substantially more developed before hardware crossing than in either prior case. Stablecoin settlement ($3.9T/quarter), containerized model serving (ONNX, llama.cpp), and edge deployment frameworks (TensorRT, Core ML) are advancing in parallel with—not after—hardware cost decline. In the regime framework, this pre-crossing coordination maturity lowers κ₀ and raises γ, pushing the system toward the network-dominated regime. Additionally, the latency advantage λ for inference workloads has no parallel in prior transitions: there was no inherent latency advantage to running a spreadsheet locally versus on a mainframe terminal. For AI inference, the latency advantage is structural—local execution eliminates network round-trip—and significant (5–20× improvement for interactive applications). If the R₀ framework is correct, this predicts a compressed lag: ΔT ≈ 2–3 years rather than 3–5, or potentially ΔT ≈ 0 if the transition reaches the network-dominated regime before hardware crossing. This prediction breaks the historical pattern based on a structural argument (pre-crossing coordination investment reduces κ₀; latency advantage λ > 0 pulls Q̄* closer), which makes it a sharper test of the theory than extrapolating the 3–5 year regularity.

---

# 4. Primary Empirical Domain: AI Semiconductor Infrastructure

## 4.1 DRAM Learning Curve (1984–2024)

The empirical foundation is a 41-year panel of DRAM prices and cumulative production. Regressing log price on log cumulative output yields α = 0.22 (SE = 0.03, n = 41, R² = 0.96). Prices declined from $870,000/GB (1984) to $2.00/GB (2024). Sub-period analysis reveals modest flattening: α = 0.25 (1984–2000) versus α = 0.19 (2000–2024), consistent with mature-industry deceleration.

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

*Table 2. DRAM learning curve (selected years). OLS: α = 0.22, SE = 0.03, R² = 0.96.*

## 4.2 HBM Cost Trajectory

High-bandwidth memory stacks conventional DRAM dies using through-silicon vias and microbump interconnects. The technology was developed for datacenter GPUs but the packaging knowledge transfers to consumer form factors—the learning externality central to the mechanism. HBM prices declined from $120/GB (2015) to $12/GB (2025). The estimated learning elasticity α = 0.23 (SE = 0.06, n = 6) is consistent with the broader DRAM rate.

| Year | Generation | $/GB | Capacity/Stack (GB) | ln($/GB) |
|:---|:---|---:|---:|---:|
| 2015 | HBM1 | 120 | 4 | 4.79 |
| 2016 | HBM2 | 60 | 8 | 4.09 |
| 2018 | HBM2E | 35 | 8 | 3.56 |
| 2020 | HBM2E | 25 | 16 | 3.22 |
| 2022 | HBM3 | 20 | 24 | 3.00 |
| 2024 | HBM3E | 15 | 36 | 2.71 |
| 2025 | HBM3E+ | 12 | 48 | 2.48 |
| 2026 | HBM4 (proj.) | 9 | 48 | 2.20 |

*Table 3. HBM pricing trajectory. α = 0.23 (SE = 0.06).*

## 4.3 Hyperscaler Capital Expenditure

Industry capex accelerated from $64B (2018) to $232B (2024), with 2025 guidance totaling $436B—a 6.8× increase over the full 2018–2025 period. Cumulative investment 2018–2025 reaches $1.3 trillion. Each firm's guidance explicitly cited competitive pressure from rivals, qualitative evidence of the Nash dynamic.

| Company | 2018 | 2020 | 2022 | 2024 | 2025G | Source |
|:---|---:|---:|---:|---:|---:|:---|
| Microsoft | 11.6 | 15.4 | 23.9 | 44.5 | 80 | 10-K |
| Alphabet | 25.1 | 22.3 | 31.5 | 52.5 | 75 | 10-K |
| Amazon | 13.4 | 35.0 | 58.3 | 78.0 | 100 | 10-K |
| Meta | 13.9 | 15.7 | 31.4 | 39.2 | 65 | 10-K |
| Stargate JV | — | — | — | — | 100 | Annc. |
| **Industry Total** | **64** | **88** | **148** | **232** | **436** | |

*Table 4. Hyperscaler capex ($B). 2025G = guidance. Cumulative 2018–2025: $1,298B.*

## 4.4 The Training-Inference Structural Bifurcation

The $1.3 trillion in cumulative hyperscaler capex builds infrastructure that serves two structurally distinct workloads with different decentralization trajectories. Conflating them produces an overstated displacement prediction; separating them sharpens the model's empirical scope.

**Table 4a. Training vs. inference: structural comparison.**

| Dimension | Training | Inference |
|:---|:---|:---|
| **Function** | Teaching models via massive datasets | Running trained models for user requests |
| **Share of AI compute (2023)** | ~67% | ~33% |
| **Share of AI compute (2025)** | ~50% | ~50% |
| **Share of AI compute (2026, proj.)** | ~33% | ~67% |
| **Projected share (2030)** | ~20–30% | ~70–80% |
| **Power density (kW/rack)** | 100–1,000 | 30–150 |
| **Latency sensitivity** | Low (100ms tolerable) | High (<10ms for interactive UX) |
| **Synchronization requirement** | Massive (10K–100K+ GPUs tightly coupled) | None (tasks atomizable) |
| **Interconnect requirement** | InfiniBand/NVLink, TB/s between GPUs | Standard networking sufficient |
| **Location constraint** | Power-rich, remote acceptable | Metro-adjacent, near users |
| **Cost trajectory** | Rising per frontier model generation | Declining ~280× in 2 years |
| **Frequency** | One-time per model generation | Continuous, scales with every user |
| **Edge-viable?** | No (architectural constraint) | Yes (this paper's thesis) |
| **Revenue model** | Capital expenditure, amortized | Ongoing service revenue per query |

*Sources: Deloitte (2025), McKinsey (2025), MIT Technology Review (2025), Stanford AI Index (2025), Epoch AI (2025).*

**Training does not decentralize.** Frontier model training requires synchronized clusters of 10,000–100,000+ GPUs communicating at terabit-per-second speeds via InfiniBand or NVLink, running continuously for weeks to months on petabytes of proximate data. These requirements are architectural, not merely economic: the learning curves that reduce memory and compute costs per unit do not address the synchronization, bandwidth, and scale requirements. A consumer device with excellent memory bandwidth still cannot participate in a distributed training run because the inter-device communication latency (milliseconds over WiFi vs. nanoseconds over NVLink) creates a performance gap of 5–6 orders of magnitude that no plausible learning curve closes. Training remains centralized for structural reasons, not cost reasons.

**Inference decentralizes.** Inference tasks are atomizable: each user query is independent and can be served by a single device with no inter-device communication. Inference is latency-sensitive: users benefit from local execution (<10ms) over cloud round-trip (50–200ms). Inference scales with users: every new AI application, every new user, every new agentic action generates inference demand. And inference cost is governed by the same memory bandwidth constraint the model tracks—token generation speed is determined almost entirely by the ratio of memory bandwidth to model size in memory (Section 4.5).

**The inference market is the revenue pool at stake.** OpenAI's 2024 compute spending illustrates the economics: approximately $5 billion on R&D/training compute versus approximately $2 billion on inference compute—but inference costs scale with users while training is amortized. GPT-4's inference costs were projected at approximately $2.3 billion in 2024, roughly 15× its training cost. At scale, inference dominates both compute cycles (80–90% per MIT Technology Review) and revenue. The inference market is projected to grow from $106 billion (2025) to $255 billion by 2030 (MarketsandMarkets). This is the revenue pool that the $1.3 trillion infrastructure buildout targets, and it is the revenue pool that edge devices will intercept.

**Implications for the model.** The endogenous decentralization mechanism applies to inference with full force and greater precision than a monolithic treatment would suggest. The crossing condition (Section 4.5) is an inference benchmark. The post-crossing continuation value S incorporates training persistence (Section 3.7). The predictions (Section 6) are inference predictions. The training workload provides a revenue floor for incumbents but does not alter the fundamental mechanism driving inference decentralization.

## 4.5 Consumer Silicon and the Inference Crossing Condition

Stage 2 materializes as component migration from datacenter to consumer form factors. The inference crossing condition—≥70B parameters at ≥20 tok/s under $1,500—has not yet been met, but three tiers of consumer AI silicon now in mass production reveal both the trajectory and the binding constraint.

At the edge, Hailo's 10H processor (2025, 40 TOPS INT4, 2.5W) runs 2B-parameter LLMs at 10+ tokens per second. The Raspberry Pi AI HAT+ 2 board—chip plus 8GB LPDDR4 on a single PCB—retails for $130, yielding 15.4M parameters per dollar. Hailo's architectural progression is itself evidence of the DRAM constraint: the prior-generation Hailo-8 solved edge vision AI with on-die memory alone, requiring no external DRAM. The Hailo-10H added a dedicated DDR interface and on-module LPDDR4 specifically to handle LLMs—a design change driven by memory bandwidth requirements that maps directly onto the component cost migration the model predicts. HP has committed Hailo-10H for point-of-sale systems; automotive integration begins in 2026. This is Stage 2 deployment at embedded scale in form factors that never previously contained AI inference capability.

At the consumer desktop tier, AMD's Ryzen AI Max+ 395 ("Strix Halo") ships in complete systems—Framework Desktop, Beelink GTR9 Pro, GMKtec EVO-X2—at approximately $2,000 with 128GB of unified LPDDR5X memory. The 256-bit bus delivers ~215 GB/s measured bandwidth. On dense 70B models in Q4 quantization (~40GB file size), token generation runs at approximately 5 tok/s—almost exactly the bandwidth-limited theoretical ceiling of 215/40 ≈ 5.4 tok/s. At 35M parameters per dollar, the AMD platform is the most cost-efficient path to 70B-class inference. More significantly, Mixture-of-Experts architectures provide an end-run around the dense-model crossing condition: GPT-OSS 120B (116.8B total parameters, ~20B active per token) achieves ~31 tok/s on the same $2,000 system—120B-class output quality at interactive speeds. If the crossing condition is reframed in terms of output capability rather than raw parameter count, the AMD platform arguably crosses today.

At the performance tier, Apple's Mac Studio with M4 Max and 128GB unified memory ($3,949) delivers 546 GB/s bandwidth, yielding ~14 tok/s on 70B Q4 models (theoretical ceiling: 546/40 ≈ 13.7 tok/s) and ~30 tok/s on 32B Q4 (546/18 ≈ 30.3 tok/s), at 17.7M parameters per dollar. The M3 Ultra, with 819 GB/s bandwidth, reaches ~20 tok/s on 70B—barely meeting the speed target at a price point exceeding $5,000.

The bandwidth calculations reveal a precise physical constraint: token generation speed for inference is determined almost entirely by the ratio of memory bandwidth to model size in memory. This makes memory bandwidth—not compute throughput, not parameter count, not model architecture—the binding constraint on consumer AI inference. The M3 Ultra (819 GB/s) outperforms the newer M4 Pro (273 GB/s) on identical models despite being an older silicon generation, because bandwidth alone determines inference throughput. The constraint is exactly the component whose learning curve the model tracks: stacked and high-bandwidth memory. Each system's performance ceiling can be calculated from a single ratio, and the gap between current capability and the crossing condition can be expressed as a bandwidth deficit denominated in dollars per GB/s—a quantity whose trajectory is governed by the learning elasticity α = 0.23.

No current system meets the inference crossing condition as stated (≥70B, ≥20 tok/s, under $1,500). The AMD platform at $2,000 achieves 70B capability at 5 tok/s—meeting parameters but not speed. The Apple M3 Ultra at $5,000+ meets both parameters and speed but at 3.3× the price target. The gap is dominated by memory cost: at current LPDDR5X pricing (~$3.50/GB), 128GB of memory alone costs approximately $450, roughly 30% of a $1,500 target system. At the HBM-derived learning rate (α = 0.23) applied to consumer memory packaging, this gap closes by 2028–2030.

## 4.6 The Demand Shock as Nash Overinvestment

The Stargate project alone demands approximately 40% of global DRAM output. Combined hyperscaler commitments likely exceed current manufacturing capacity by 5–10×. The memory industry's response—the largest capacity expansion in semiconductor history—follows the boom-bust pattern visible in four prior DRAM cycles. Historical precedent predicts overcapacity and below-trend pricing by 2028–2029. The fabs built for datacenter demand do not shut down when demand moderates; they pivot to consumer stacked DRAM and LPDDR6. This is the Nash overinvestment dynamic from Section 3 operating through the supply side.

The training-inference distinction sharpens this dynamic. As inference migrates to edge devices, the datacenter demand for memory *moderates* (inference was the volume driver), but the fabs are already built. The overcapacity pivots to consumer memory formats—accelerating the very edge inference capability that caused the datacenter inference demand moderation. This is a positive feedback loop that the monolithic model understates.

---

# 5. Historical Validation and Parameter Consistency

## 5.1 Mainframe → Personal Computer (1975–2000)

IBM dominated mainframe computing with 75–80% market share through the 1970s. IBM's investment in semiconductor manufacturing drove the learning curves that reduced microprocessor and memory costs (Flamm 1993 estimates α = 0.24 for Intel microprocessors, 1974–1989). The IBM financial trajectory provides direct calibration of ∂T*/∂I < 0: R&D averaged $5–5.6B annually through 1987–1993, and the company posted $15.8B in cumulative losses (1991–1993)—the largest corporate losses in American history at that time.

| Year | Revenue ($B) | Net Income ($B) | R&D ($B) | Employees (K) | Stage |
|:---|---:|---:|---:|---:|:---|
| 1984 | 46.3 | 6.6 | 3.7 | 395 | Peak profits |
| 1987 | 54.2 | 5.3 | 5.0 | 390 | 386 crosses |
| 1990 | 69.0 | 6.0 | 5.6 | 374 | Revenue peak |
| 1991 | 64.8 | (–2.8) | 5.5 | 345 | First loss |
| 1992 | 64.5 | (–4.9) | 5.1 | 302 | Akers fired |
| 1993 | 62.7 | (–8.1) | 5.1 | 257 | Record loss |
| 1995 | 71.9 | 4.2 | 5.5 | 225 | Services pivot |

*Table 5. IBM financial trajectory. Sources: IBM Annual Reports, 10-K filings.*

The critical timing: PC hardware was capable by 1987 (386 at $2,200), but dominance arrived only with the coordination layer—Windows 3.0 (1990), Windows 95 (1995). Hardware crossing preceded platform explosion by 3–5 years. The R₀ framework (Section 3.9) provides the structural explanation: R₀ < 1 in 1987 because κ was high (no dominant operating system, fragmented software ecosystem); R₀ crossed 1 circa 1990–92 as Windows reduced coordination friction.

**The mainframe persists.** The IBM case is more nuanced than a simple displacement narrative. IBM's mainframe business was not extinguished—it was *marginalized by market expansion*. As of 2025, IBM's mainframe division (IBM Z) continues to generate approximately $3–4 billion in annual revenue, serving high-reliability transaction processing in banking, government, and airline systems. The mainframe's share of total computing revenue went from effectively 100% (1970s) to a small single-digit percentage—not because mainframes stopped working or became uneconomic for their use cases, but because the addressable market for computing expanded by orders of magnitude into domains (personal productivity, consumer internet, mobile) that mainframes were architecturally unsuited to serve.

This niche persistence is the correct analog for AI datacenter training infrastructure. The prediction is not that datacenters will be decommissioned, but that the explosive growth in AI inference demand (billions of edge devices, every phone and vehicle and appliance) will dwarf centralized inference, while centralized training persists as a smaller but stable and possibly growing business. IBM's $15.8B in losses (1991–93) reflected a company that was slow to adapt its *business model* to a world where mainframes became one option among many—not the inherent extinction of the technology. Hyperscalers that proactively pivot to model licensing and training-as-a-service face a more favorable transition than IBM's traumatic 1990s restructuring.

The δ ≈ 0.30 calibration from IBM therefore overstates total firm displacement and should be understood as the inference-revenue displacement rate. IBM's mainframe revenue decline was approximately 60% over three years in the early 1990s, but this reflected both compute and application migration. In the AI case, where training persists, the effective firm-level displacement rate δ_eff is lower than δ_inference because training revenue provides a floor. Section 3.7 addresses this through the S_T parameter.

## 5.2 ARPANET → Commercial Internet (1969–2000)

Government investment in ARPANET and NSFNET drove TCP/IP and router technology development. Once standardized, mesh networks exceeded centralized alternatives: proprietary online service share collapsed from 60% (1989) to 2% (2000). The coordination layer pattern repeats: TCP/IP was viable by the late 1980s; the browser (Mosaic 1993, Netscape 1994) provided the common interface. Lag: 3–5 years.

## 5.3 Cross-Domain Parameter Consistency

Semiconductor learning elasticities cluster at α ∈ [0.19, 0.25] with mean 0.23 across products, firms, and decades—suggesting a structural property of semiconductor manufacturing.

| Industry | Product | α | SE | Period |
|:---|:---|---:|---:|:---|
| Semiconductor | DRAM (full panel) | 0.22 | 0.03 | 1984–2024 |
| Semiconductor | HBM | 0.23 | 0.06 | 2015–2024 |
| Semiconductor | Intel microprocessors | 0.24 | 0.04 | 1974–1989 |
| Semiconductor | NAND Flash | 0.24 | 0.05 | 2003–2023 |
| Semiconductor | Solar PV cells | 0.23 | 0.02 | 1976–2023 |
| Energy | Lithium-ion batteries | 0.21 | 0.03 | 1995–2023 |
| Internet | Cloud compute (AWS) | 0.25 | 0.03 | 2006–2023 |

*Table 6. Cross-domain learning rates. Semiconductor products cluster at α ∈ [0.19, 0.25].*

## 5.4 The Coordination Layer Pattern

Across both transitions, hardware crossing precedes architectural dominance by 3–5 years, mediated by a coordination layer providing: (1) a common API abstracting heterogeneous hardware; (2) a developer ecosystem; and (3) an economic settlement mechanism enabling value exchange between distributed participants.

| Transition | Hardware Crossing | Coordination Layer | Dominance | Lag |
|:---|:---|:---|:---|:---|
| Mainframe → PC | 386 at $2,200 (1987) | Windows 3.0/3.1 (1990–92) | Win95 (1995) | 3–5 yr |
| ARPANET → Internet | Routers/TCP/IP (~1989) | Mosaic/Netscape (1993–94) | Web (1995–97) | 3–5 yr |
| Cloud → Edge AI | 50GB stacked (2028–30) | Emerging | Proj: 2031–35 | 3–5 yr? |

*Table 7. The coordination layer pattern across three technology transitions.*

The coordination layer for distributed AI inference is emerging in stablecoin settlement rails ($3.9T quarterly, 2025-Q1), micropayment protocols, and tokenized asset infrastructure. Current adoption trajectories are consistent with the pre-explosion phase observed in both historical cases. Smirl (2026) develops the full monetary treatment.

---

# 6. Falsifiable Predictions

The model generates nine predictions with timing. If these fail, the theory is wrong.

**Prediction 1: Consumer Stacked Memory ≥16GB by 2027.** HBM-derived 3D stacking in consumer products with ≥16GB on-chip stacked memory below $200. The Rockchip RK1828 (2025, 5GB 3D stacked DRAM) and Hailo-10H (2025, 8GB LPDDR4 on-module at $130) confirm component migration is underway; the prediction is capacity scaling. Evidence against: ≤8GB through 2028.

**Prediction 2: 70B Inference On-Device by 2028–2030 (Hardware Crossing).** Consumer devices under $1,500 running 70B-parameter models at ≥20 tok/s for inference. This is the inference cost-parity T*. Training of such models remains centralized. Evidence against: not achieved by 2031.

**Prediction 2* (Refined): R₀ > 1 for Distributed AI Inference by 2030–2032.** The R₀ extension (Section 3.9) refines Prediction 2: hardware crossing (cost parity) is necessary but not sufficient for self-sustaining adoption. Self-sustaining distributed inference adoption (R₀ > 1) arrives 2–3 years after hardware crossing—compressed from the historical 3–5 year lag by the latency advantage λ and pre-crossing coordination maturity. Evidence for: quarterly distributed inference share growing >5% per quarter without subsidies. Evidence against: distributed share stalling below 20% despite hardware capability by 2033.

**Prediction 3: Inference Capex Deceleration with Training Persistence by 2027–2028.** At least one top-four hyperscaler reduces inference-oriented capex by ≥20% YoY before 2029 while maintaining or increasing training-oriented capex, accompanied by public statements citing strategic repositioning toward edge/hybrid inference architectures and/or model licensing. Pure cyclical deceleration without strategic repositioning would not confirm the prediction. Pure aggregate capex reduction without evidence of differential treatment of training vs. inference investment would not confirm the prediction.

**Prediction 4: Stablecoin-Treasury Holdings Exceed $300B by 2027.** From the current $150B+ base. This tests coordination layer formation for distributed economic settlement. Evidence against: plateau below $200B.

**Prediction 5: Learning Rate Stability.** DRAM/HBM α remains in [0.18, 0.28] through 2030. If α < 0.15, all timing predictions shift outward.

**Prediction 6: Distributed Inference Adoption Tipping Point at Approximately 40% of Inference Workloads.** Once distributed architectures serve approximately 40% of AI inference workloads by volume, network effects reverse and the centralized *inference* model becomes non-viable for holdout use cases. The mechanism is indirect: as distributed adoption crosses this threshold, the centralized inference ecosystem loses supplier priority (memory fabs pivot to consumer stacked DRAM), developer mindshare (tooling investment shifts to edge deployment), and complementary service density. This is the economic analog of the epidemiological herd immunity threshold. The ~40% threshold follows from calibrating the self-sustaining growth condition to observed network-effect elasticities in cloud infrastructure markets. This prediction applies to inference only; centralized training infrastructure is unaffected by inference market share shifts. Evidence against: centralized inference provision remaining commercially stable with distributed inference share exceeding 50% through 2032.

**Prediction 7: Non-Monotonic Inference Adoption with Coordination-Layer Trough.** Distributed inference adoption follows a two-wave pattern: initial surge during first-generation coordination-layer deployment, a trough during coordination fragmentation and standardization competition, and a second wave after consolidation around winning protocols. Both historical transitions exhibit this structure. The model predicts a similar structure for AI: initial distributed inference adoption wave 2027–2030, coordination-layer fragmentation trough 2031–2032, and standardization-driven second wave 2033–2035. Evidence against: monotonic quarterly growth in distributed AI inference market share exceeding 5% per quarter throughout 2028–2034 without interruption.

**Prediction 7* (Refined).** The R₀ framework provides the structural mechanism for Prediction 7: the coordination-layer trough corresponds to the period when hardware has crossed (c ≤ c*) but R₀ < 1. Early adopters face high κ (immature coordination layer) and adoption is not yet self-sustaining. The second wave arrives when κ falls enough for R₀ > 1.

**Prediction 8: Coordination Friction Is Observable and Declining.** The R₀ framework requires that coordination friction κ decreases over time, which is empirically testable through proxies: (a) *deployment friction*—time-to-deploy for a standardized model on new edge hardware; (b) *interoperability friction*—number of mutually incompatible model interchange formats weighted by market share; (c) *settlement friction*—transaction cost for micropayment-scale AI inference settlements. The prediction is directional: all three proxies should show sustained decline through the transition period, with acceleration following hardware crossing. Evidence against: any proxy showing sustained increase or stagnation through 2030 despite hardware costs continuing to decline.

**Prediction 9: Training Remains Centralized Through 2035.** Frontier model training (defined as models requiring >10,000 synchronized GPUs for >7 days of continuous operation) remains exclusively performed in centralized datacenter clusters through 2035. No distributed or edge-based training of frontier-scale models achieves cost parity or capability parity with centralized training. The number of organizations capable of training frontier models remains structurally limited (<20 globally). This prediction tests the training-inference bifurcation: if it fails (distributed frontier training emerges), the paper's scope extends to total displacement rather than inference-only displacement. Evidence against: a frontier-scale model trained on distributed consumer hardware by 2035 at comparable cost and performance to centralized training.

---

# 7. Conclusion

This paper has identified and formalized endogenous decentralization: a mechanism by which concentrated capital investment in centralized infrastructure finances the learning curves that enable distributed alternatives. The self-undermining investment property (∂T*/∂I < 0) is distinct from learning-by-doing, GPT spillovers, and Schumpeterian creative destruction. In symmetric Markov Perfect Equilibrium, competing centralized firms produce aggregate output that strictly exceeds the cooperative optimum at every interior state, accelerating crossing time—and this result is robust to increasing competition, cost asymmetry, and capacity constraints.

The generalized crossing condition (Section 3.9) extends the model's reach: the cost-parity threshold becomes a self-sustaining adoption threshold R₀ > 1 that incorporates network effects, coordination friction, and the structural latency advantage of local inference over cloud round-trip, endogenizing the 3–5 year coordination layer lag observed in historical transitions. The regime structure—friction-dominated versus network-dominated—generates the sharper prediction that the AI transition's lag may be compressed to 2–3 years or eliminated entirely if pre-crossing coordination maturity and latency advantage push the system into the network-dominated regime.

The training-inference bifurcation (Section 4.4) sharpens the mechanism's empirical scope. AI workloads divide into training (which remains centralized for structural architectural reasons) and inference (which follows the endogenous decentralization trajectory). The mechanism's primary domain is inference: already 80–90% of AI compute cycles, projected to reach a $255 billion market by 2030, and governed by the memory bandwidth constraint whose learning curve the model tracks. Training persists centrally, providing incumbents with a revenue floor (S_T > 0) that moderates but does not eliminate the overinvestment externality.

The empirical grounding spans 41 years of DRAM data (α = 0.22, R² = 0.96), $1.3 trillion in hyperscaler capex, and two historical transitions exhibiting consistent learning parameters (α ∈ [0.18, 0.25]) and a 3–5 year coordination layer lag. The current AI buildout is the mechanism operating in real time. All predictions carry specific timing and failure conditions, and Prediction 9 directly tests the training-inference bifurcation itself.

The post-crossing equilibrium is *partial decentralization*: inference distributes to edge devices while training persists in centralized clusters. This coexistence is stable because training and inference have fundamentally different synchronization, bandwidth, and scale requirements—a distinction that learning curves do not bridge. Historical precedent confirms the pattern: IBM's mainframe business persists at approximately $3–4 billion annually, decades after the PC revolution, serving workloads where centralized architecture retains genuine advantages. The datacenters will not be decommissioned; the addressable market will expand around them.

This partial decentralization does not rescue incumbent business models in their current form. Inference represents the majority of AI compute cycles and the primary source of ongoing revenue. A business model that retains training but loses inference is viable but structurally diminished—analogous to a record label that owns the recording studio but not the distribution channel. Whether hyperscalers can successfully pivot to model licensing and training-as-a-service—becoming the "studios" of the AI era rather than the "theaters"—is an open strategic question that the model identifies but does not resolve. The viability of this pivot depends on whether open-source model proliferation (Llama, DeepSeek) erodes the training moat, whether training costs continue to scale or hit diminishing returns, and whether model-as-product revenue can replace inference-as-service revenue at comparable margins.

The welfare implications incorporate the bifurcation. The overinvestment externality imposes a 22–34% per-firm welfare loss relative to the cooperative benchmark (depending on S_T calibration). Yet this captures only producer surplus: the same overinvestment that reduces incumbent inference profits generates consumer surplus through lower prices and earlier access to decentralized AI. The social welfare calculation is further complicated by the positive externalities of distributed AI—privacy benefits from local inference, resilience from architectural redundancy, and innovation from ecosystem diversity.

The mechanism also predicts a qualitative shift in innovation dynamics post-crossing. A distributed ecosystem of many small producers, each exploring architectural variations with limited resources, searches the design space differently than a concentrated ecosystem of large producers optimizing along established trajectories. The current proliferation of edge AI silicon—over 50 distinct neural processing unit architectures announced in 2024–2025—is consistent with this prediction. Post-crossing, the distributed ecosystem does not merely replicate centralized inference at lower cost; it explores faster and along dimensions the centralized ecosystem cannot reach. This exploration advantage reinforces irreversibility even as centralized training persists.

What the mechanism predicts unambiguously is that concentrated investment endogenously produces inference decentralization, that this process accelerates with the number of competitors, and that training centralization and inference decentralization will coexist as stable features of the AI economic landscape.

---

# Appendix A. Two-Period Pedagogical Model

This appendix presents a simplified two-period version of the model that provides intuition for the continuous-time results.

**Setup.** Period 1: N symmetric firms choose investment Iᵢ ≥ 0, earning Cournot profits πᵢ = (a – b∑Iⱼ)·Iᵢ – cIᵢ at unit cost c. Period 2: if cumulative investment ∑Iⱼ exceeds threshold Q̄, the technology "crosses": unit costs fall to c̄ < c, enabling distributed inference entry. Incumbents earn continuation value S = S_T + S_I per firm, where S_T is persistent training revenue and S_I < π̄ is residual inference value. If ∑Iⱼ < Q̄, incumbents retain full profits π̄ > S.

**Nash equilibrium.** Each firm maximizes πᵢ + β·Vᵢ(∑Iⱼ), where β is the discount factor and Vᵢ = π̄ if ∑I < Q̄, Vᵢ = S otherwise. The FOC is a – b(N+1)I* – c = 0 (interior solution), giving I* = (a–c)/[b(N+1)]. Total investment NI* exceeds Q̄ whenever N is sufficiently large or a–c is sufficiently high, even though each firm would prefer total investment below Q̄.

**Cooperative benchmark.** A planner choosing total I to maximize Nπ(I/N) + βNV(I) sets Iᶜ = (a–c)/(2b) < NI* for N ≥ 2. The cooperative solution internalizes the discontinuity at Q̄ and may choose Iᶜ < Q̄ even when NI* > Q̄.

**Corollary A1 (Asymmetric cost advantage).** If firm 1 has cost c₁ = c – ε while rivals have cost c, then firm 1's equilibrium investment I₁* > I* and total investment rises. Capital concentration compresses T*.

**Corollary A2 (Vertical integration).** A vertically integrated firm that captures fraction θ of consumer surplus from decentralized technology has Vᵢ = S + θ·CS. This firm invests more aggressively, accelerating crossing. Empirical analog: cloud hyperscalers that both produce and consume AI inference hardware.

**Corollary A3 (Capacity constraints).** If per-period investment is bounded by Iᵢ ≤ κ, crossing is delayed by at most ⌈Q̄/(Nκ)⌉ periods. The learning elasticity α is invariant to production rate, so capacity constraints delay but do not alter the cost trajectory. The DRAM panel confirms: four boom-bust cycles left α = 0.22 unchanged.

---

# Appendix B. Overinvestment in Dollar Terms

The model's output ratio Qᴺ/Qᶜ provides a structural interpretation of observed AI infrastructure spending. Table B1 compares actual aggregate capex to the model's implied "rational monopolist" benchmark.

| | 2024 | 2025 (proj.) | Cumulative |
|:---|---:|---:|---:|
| Actual AI capex ($B) | ~230 | ~436 | ~1,300 |
| Model Qᴺ/Qᶜ ratio | 3–4× | 3–4× | — |
| Implied cooperative ($B) | ~65–75 | ~110–145 | — |
| Excess investment ($B) | ~155–165 | ~291–326 | — |

*Table B1. Actual vs. model-implied cooperative AI infrastructure investment.*

The excess investment is not waste in the social welfare sense. As noted in Section 3.5, the overinvestment that reduces incumbent inference producer surplus simultaneously generates consumer surplus through lower prices and earlier inference decentralization. The $291–326B excess for 2025 is a transfer mediated by the learning curve, not a deadweight loss. Moreover, a portion of the "excess" investment builds training infrastructure whose value persists post-crossing—the excess attributable to the learning externality applies primarily to the inference-oriented component of capex.

---

# Appendix C. Semi-Endogenous Coordination Dynamics

This appendix develops the declining-κ extension sketched in Section 3.9. The results here use a quasi-static approximation; the full two-state treatment is left to future work.

## C.1 The Two-Gap Decomposition

Define the **hardware gap** Gᴴ(t) = Q̄ – Q(t) (distance to cost parity, the state variable x in the original model); the **coordination gap** Gᶜ = Q̄*(κ) – Q̄ (positive in the friction-dominated regime, negative in the network-dominated regime; constant under fixed κ, declining under semi-endogenous κ); and the total gap x*(t) = Gᴴ(t) + Gᶜ = Q̄*(κ) – Q(t).

Under fixed κ (Section 3.9), Gᶜ is a constant and x* is a standard state variable—the tractability result is exact. Under declining κ, Gᶜ shrinks over time (becoming less positive or more negative) as the coordination layer matures, and the total gap x* contracts from both directions.

## C.2 Quasi-Static Approximation

If κ declines as the coordination layer matures, Q̄*(t) is moving. The state x(t) = Q̄*(κ(t)) – Q(t) evolves as:

> dx/dt = (dQ̄*/dκ) · (dκ/dE) · Ė – Σqᵢ

The first term is negative (Q̄* falls as κ falls as E rises), meaning the threshold drifts toward the firms even when they produce nothing. The one-state-variable structure survives under a timescale separation assumption: if coordination infrastructure evolves slowly relative to the production game (formally, if |dQ̄*/dt| ≪ |Σqᵢ|), firms can treat Q̄* as locally constant when solving their HJB. This is a standard quasi-static approximation, analogous to the Born-Oppenheimer approximation in physics. Under quasi-statics, the Nash equilibrium at each instant is the solution to the fixed-Q̄* game evaluated at the current Q̄*(t), and the analysis of Section 3.9 applies pointwise.

The quasi-static assumption is empirically reasonable: coordination layers mature over years (standards committees, developer ecosystem formation, regulatory adaptation), while production decisions operate on quarterly cycles. But it breaks down if coordination friction drops discontinuously—as when a dominant standard suddenly emerges. At such moments, Q̄* jumps downward, x drops discretely, and the smooth HJB solution is interrupted. The model handles this as a regime shift rather than a continuous dynamic.

*Remark 4.* Under the quasi-static approximation, Proposition 1* holds pointwise at each Q̄*(t), with approximation error bounded by the timescale separation ratio |dQ̄*/dt|/|Σqᵢ|. The overinvestment result is robust to slow threshold drift because the strategic mechanism—each firm internalizing only 1/N of the social shadow cost—operates at the production timescale regardless of boundary movement.

## C.3 Feedback Loop

The coordination friction κ declines as the coordination layer matures. The qualitative dynamics require only that κ is monotonically decreasing in cumulative ecosystem investment E(t): κ(t) = κ(E(t)), with κ′(E) < 0. The specific functional form is not pinned down by the theory. Exponential decay (κ = κ₀·e⁻λE) is analytically convenient but poorly motivated: the economics of coordination—standards wars, winner-take-all platform competition, committee processes—are lumpy, not smooth. A threshold function where κ drops discontinuously when a dominant standard emerges (as Windows did for PCs, or HTTP/HTML did for the web) may better describe the actual dynamics. A power law κ = κ₀·E⁻η accommodates heavy-tailed adoption patterns.

The rate at which E responds to the cost gap is Ė = φ · max{0, c_centralized – c(Q)}, where φ converts cost advantage into ecosystem investment incentive. This creates a two-sided convergence: Q(t) rises toward Q̄*(t) while Q̄*(t) falls toward Q(t). The speed of convergence depends on the functional form of κ(E) and the magnitude of φ, but the direction is structural.

For the κ dynamics to truly require a second state variable in the HJB, firms would need to strategically invest in coordination layer development as a separate choice variable, and coordination investment would need to interact with production decisions. The lumpy, discontinuous nature of real coordination dynamics further argues against embedding κ as a smooth state variable in the current model—it would impose false precision on an inherently discrete process. A full two-state treatment (companion paper territory) would model adoption dynamics dn/dt = β(x)·n·(1–n) – μ·n with firms' HJB depending on both x and n, likely requiring numerical solution.

---

# References

Acemoglu, D., & Guerrieri, V. (2008). Capital deepening and nonbalanced economic growth. *Journal of Political Economy*, 116(3), 467–498.

Arrow, K. J. (1962). The economic implications of learning by doing. *Review of Economic Studies*, 29(3), 155–173.

Bass, F. M. (1969). A new product growth for model consumer durables. *Management Science*, 15(5), 215–227.

Bresnahan, T. F., & Greenstein, S. (1994). The competitive crash in large-scale commercial computing. NBER Working Paper No. 4901.

Bresnahan, T. F., & Trajtenberg, M. (1995). General purpose technologies: Engines of growth? *Journal of Econometrics*, 65(1), 83–108.

Christensen, C. M. (1997). *The Innovator's Dilemma.* Harvard Business School Press.

David, P. A. (1990). The dynamo and the computer. *American Economic Review*, 80(2), 355–361.

Deloitte (2025). More compute for AI, not less. *Technology, Media, and Telecom Predictions 2026.*

Epoch AI (2025). Most of OpenAI's 2024 compute went to experiments. *Data Insights.*

Flamm, K. (1993). *Mismanaged Trade?* Brookings Institution.

Greenstein, S. (1997). Lock-in and the costs of switching mainframe computer vendors. *Industrial and Corporate Change*, 6(2), 247–273.

Gyllenberg, M., & Parvinen, K. (2001). Necessary and sufficient conditions for evolutionary suicide. *Bulletin of Mathematical Biology*, 63(5), 981–1015.

Jackson, M. O., & Yariv, L. (2007). Diffusion of behavior and equilibrium properties in network games. *American Economic Review*, 97(2), 92–98.

Kermack, W. O., & McKendrick, A. G. (1927). A contribution to the mathematical theory of epidemics. *Proceedings of the Royal Society A*, 115(772), 700–721.

Levhari, D., & Mirman, L. J. (1980). The great fish war: An example using a dynamic Cournot-Nash solution. *Bell Journal of Economics*, 11(1), 322–334.

MarketsandMarkets (2025). AI inference market size, share & growth, 2025 to 2030. Market Report.

Maynard Smith, J. (1982). *Evolution and the Theory of Games.* Cambridge University Press.

McKinsey (2025). The next big shifts in AI workloads and hyperscaler strategies. *TMT Practice Report.*

Milnor, J. (1963). *Morse Theory.* Annals of Mathematics Studies, No. 51. Princeton University Press.

MIT Technology Review (2025). We did the math on AI's energy footprint.

Odling-Smee, F. J., Laland, K. N., & Feldman, M. W. (2003). *Niche Construction: The Neglected Process in Evolution.* Princeton University Press.

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

Young, H. P. (2009). Innovation diffusion in heterogeneous populations: Contagion, social influence, and social learning. *American Economic Review*, 99(5), 1899–1924.
