# The Training/Inference Gap in "Endogenous Decentralization"
## Proposed Revisions to Strengthen the Paper

**Prepared for Connor Smirl — February 2026**

---

## The Core Problem

The paper treats "centralized AI infrastructure" as a monolithic category that decentralization displaces wholesale. But datacenter AI workloads split into two fundamentally different activities with different economics, different hardware requirements, and critically different decentralization trajectories:

- **Training**: Teaching models by processing massive datasets. Compute-intensive, frontloaded (one-time per model generation), latency-insensitive, requires tightly synchronized GPU clusters. Power density: 100–200+ kW/rack, with frontier systems approaching 1 MW/rack.

- **Inference**: Running trained models to serve real-time user requests. Continuous, scales with every user interaction, latency-sensitive (must be close to users), and highly "atomizable" — individual tasks can be handled independently. Power density: 30–150 kW/rack.

The paper's mechanism — learning curves driving component costs down until edge devices can replicate datacenter-class performance — applies almost perfectly to **inference**. But it has limited applicability to **training**, which may remain permanently centralized. This creates the possibility of a **stable coexistence equilibrium** rather than the full architectural substitution the paper currently predicts.

---

## The Empirical Landscape

### The Shift Is Already Happening

The data strongly supports a structural divergence between training and inference:

- **2023**: ~1/3 of AI compute was inference, ~2/3 training/research
- **2025**: Roughly 50/50 split (Deloitte)
- **2026**: ~2/3 inference, ~1/3 training (Deloitte projection)
- **2030**: McKinsey projects inference as the dominant AI workload; ~70% of datacenter demand from inference (industry projections)
- **Current**: MIT Technology Review estimates 80–90% of computing power for AI is already used for inference

### The Revenue Story Confirms It

OpenAI's 2024 compute spending breakdown (via Epoch AI) is revealing:

- ~$5B on R&D/training compute (including experimental runs)
- ~$2B on inference compute
- But: GPT-4's inference costs alone were projected at ~$2.3B in 2024 — roughly **15× its training cost**

Training is a one-time capital expenditure per model generation. Inference is the ongoing operational cost that scales with every user, every query, every agentic action. The entire business model depends on inference revenue recouping training investment.

### Inference Costs Are Collapsing

Stanford's 2025 AI Index Report documented a **280-fold** drop in inference costs between November 2022 and October 2024. This is consistent with the paper's learning curve framework — but the cost decline is happening faster for inference than for training because inference benefits from both hardware learning curves AND algorithmic optimization (quantization, distillation, mixture-of-experts, speculative decoding).

### The Infrastructure Is Bifurcating

Hyperscalers are already building different facilities for different workloads:

- **Training campuses**: Remote, power-rich locations. Massive synchronized GPU clusters. Liquid cooling. Can tolerate 100ms latency between regions. Think Stargate, Abilene TX.
- **Inference facilities**: Metro-adjacent, close to population centers. Smaller GPU clusters or specialized inference chips (AWS Inferentia, Google TPUs, Groq LPUs). Latency-sensitive. Think regional data centers, edge nodes.

This isn't speculative — it's current procurement and construction strategy.

---

## What This Means for the Paper

### What the Paper Gets Right (and Understates)

The endogenous decentralization mechanism is actually **stronger** for inference than the paper currently argues, because:

1. **Inference is atomizable.** Unlike training, which requires massive synchronized clusters, inference tasks are independent. This is precisely the property that enables distributed execution on heterogeneous edge hardware. The paper should emphasize this structural distinction.

2. **Inference is latency-sensitive.** Users want fast responses. Moving inference closer to users via edge devices isn't just cost-competitive — it's latency-superior. This creates an additional adoption driver beyond cost parity that the current R₀ framework could incorporate.

3. **The inference market is the revenue pool being disrupted.** The $255B projected inference market by 2030 is what hyperscalers are building to capture. Edge inference directly threatens this revenue stream. The paper's $1.3T capex figure is building capacity for inference revenue that edge devices will intercept.

4. **Inference cost decline is faster than the paper's learning curve alone predicts.** The 280× cost reduction in two years reflects hardware learning curves PLUS algorithmic efficiency gains (quantization, MoE, distillation). The paper's α = 0.22–0.23 captures hardware alone. Effective inference cost decline includes a software multiplier.

### What the Paper Gets Wrong (or Omits)

1. **Training may not decentralize — and doesn't need to.** Frontier model training requires:
   - Synchronized clusters of 10,000–100,000+ GPUs
   - Massive interconnect bandwidth between GPUs (InfiniBand/NVLink)
   - Weeks to months of continuous operation
   - Petabytes of training data in close proximity

   None of these requirements are satisfied by distributed edge hardware. The learning curves that reduce inference costs do not help with the synchronization, bandwidth, and scale requirements of training. This is not a temporary gap — it's an architectural constraint.

2. **The "displacement rate δ" is too blunt.** The paper models post-crossing displacement as a single exponential decay. But if only inference revenue is threatened, the actual displacement trajectory is:
   - **Inference revenue**: declines sharply (consistent with δ ≈ 0.30 from IBM analogy)
   - **Training-as-a-service / model licensing**: potentially grows or remains stable
   - **Net effect**: δ_effective < δ_inference because training revenue provides a floor

3. **The IBM analogy is imperfect.** IBM's mainframe decline was total — both compute and the applications running on it migrated to PCs. In the AI case, the "compute" (training) may stay centralized while the "applications" (inference) distribute. A closer analogy might be the music industry: recording studios (training) remained centralized and capital-intensive; playback (inference) decentralized from concert halls to Walkmans to phones. Studios didn't disappear — they changed their business model.

4. **A viable pivot strategy exists for incumbents.** The paper's prisoner's dilemma argument assumes that crossing eliminates incumbent rents. But if hyperscalers can:
   - Retain training as a core competency
   - Sell/license frontier models rather than selling inference
   - Offer fine-tuning and post-training services
   - Provide hybrid cloud-edge orchestration

   ...then post-crossing continuation value S may be much higher than the paper assumes, which changes the Nash equilibrium dynamics significantly.

---

## Proposed Revisions

### 1. Decompose the Model into Training and Inference Channels

**Where**: Section 3.1 (Environment) and throughout

Split the state variable and value function into two components:

- **Inference crossing** x_I(t) = Q̄_I − Q(t): The distance to the point where edge inference becomes cost-competitive. This is what the current model tracks.
- **Training persistence**: Training workloads remain centralized, providing a revenue floor.

The post-crossing continuation value S should be reformulated as:

> S = S_training + S_inference(δ)

where S_training reflects ongoing training/model-licensing revenue (possibly growing) and S_inference reflects the decaying inference revenue pool. The key insight is that S_training > 0 even as S_inference → 0, giving incumbents a nonzero continuation value that materially affects the equilibrium.

**Effect on results**: Higher S means lower V′(x) — firms are less threatened by crossing, which slightly reduces Nash overinvestment relative to cooperation. The acceleration ratio T*_Nash/T*_Coop may moderate, though the qualitative result (∂T*/∂I < 0) is preserved.

### 2. Add a Section on Training/Inference Bifurcation

**Where**: New Section 4.5 or expand current 4.4

Present the empirical evidence:

| Metric | Training | Inference |
|--------|----------|-----------|
| Share of AI compute (2023) | ~67% | ~33% |
| Share of AI compute (2025) | ~50% | ~50% |
| Share of AI compute (2026, proj.) | ~33% | ~67% |
| Power density (kW/rack) | 100–1,000 | 30–150 |
| Latency sensitivity | Low (100ms ok) | High (<10ms for UX) |
| Synchronization requirement | Massive (10K+ GPUs) | None (atomizable) |
| Location constraint | Power-rich, remote ok | Metro-adjacent, near users |
| Cost trajectory | Growing per frontier model | Declining 280× in 2 years |
| Edge-viable? | No (architectural constraint) | Yes (paper's thesis) |

This table makes explicit that the paper's mechanism applies to the inference column, which is the growing majority of the market.

### 3. Refine the Crossing Condition

**Where**: Section 4.4

The current crossing condition (≥70B parameters at ≥20 tok/s under $1,500) is an inference benchmark. Make this explicit:

> **Inference Crossing Condition**: Consumer devices under $1,500 running 70B-parameter models at ≥20 tok/s for inference.

And add:

> **Training Non-Crossing Condition**: Frontier model training (10T+ parameter scale by 2028) requires synchronized clusters exceeding $1B in hardware cost, with no plausible consumer-device alternative. Training does not cross.

This sharpens the predictions and makes the paper more resistant to the critique that "AI needs big datacenters."

### 4. Modify Prediction 3 (Hyperscaler Capex Deceleration)

**Where**: Section 6

The current prediction says at least one top-four hyperscaler reduces AI capex by ≥20% YoY before 2029. This is too blunt given the training/inference split. Revise to:

> **Prediction 3 (Revised)**: By 2028, at least one top-four hyperscaler (a) reduces inference-oriented capex by ≥20% YoY while (b) maintaining or increasing training-oriented capex, accompanied by public statements about hybrid architectures and/or model licensing. Pure aggregate capex reduction without evidence of strategic inference-to-edge migration would not confirm the prediction.

### 5. Add a "Coexistence Equilibrium" Discussion

**Where**: Section 7 (Conclusion) or new Section 6.5

Acknowledge the most likely medium-term outcome explicitly:

> The endogenous decentralization mechanism predicts full architectural substitution for inference workloads but not for training. The equilibrium outcome may be *partial decentralization*: training remains centralized while inference distributes. This coexistence is stable because training and inference have fundamentally different synchronization, bandwidth, and scale requirements. The paper's welfare analysis applies to the inference revenue pool (~$255B projected by 2030) rather than to total AI infrastructure spending.

> However, this coexistence does not rescue incumbent business models in their current form. Inference currently represents 80–90% of AI compute cycles and is projected to generate the majority of ongoing revenue. A business model that retains training but loses inference is analogous to a record label that owns the recording studio but not the distribution channel — viable but structurally diminished.

### 6. Strengthen the R₀ Framework with Latency Advantage

**Where**: Section 3.9

The R₀ threshold currently incorporates cost advantage, network effects, and coordination friction. Add latency advantage as a fourth component:

> β(c, λ): adoption rate increasing in both cost advantage AND latency advantage λ of local inference over cloud round-trip.

This is empirically grounded: edge inference at <10ms versus cloud inference at 50–200ms is a significant UX improvement for interactive applications. It also creates a mechanism by which R₀ > 1 could be reached even before strict cost parity — the latency advantage compensates for remaining cost disadvantage.

---

## The "Model Store" Scenario

The paper should briefly address the scenario you raised: datacenters shed inference to the edge and pivot to selling frontier models. This is actually consistent with the paper's framework, but with a twist:

**How it works**: Hyperscalers train frontier models using massive centralized clusters, then sell/license the models (or fine-tuning access) to edge devices and enterprises. Revenue shifts from inference-as-a-service to model-as-a-product.

**Why it partially validates the paper**: This IS endogenous decentralization — centralized investment financed the learning curves that made edge inference viable, forcing the business model pivot. The mechanism is correct.

**Why it complicates the paper**: The post-crossing value S is much higher than the IBM analogy suggests, because the training moat persists. This means:

- The welfare loss from overinvestment may be smaller (higher S → lower V′ gap → smaller overinvestment)
- The "evolutionary suicide" language (Section 3.6) is too strong — it's more like "evolutionary metamorphosis"
- The 34.1% per-firm welfare loss estimate at N=5 would need recalibration with a higher S

**Whether it's viable**: Unknown, and the paper should say so. Key uncertainties include:
- Whether open-source models (Llama, DeepSeek) eliminate the training moat
- Whether training costs continue to scale or hit diminishing returns
- Whether the model-as-product revenue can replace inference-as-service revenue at comparable margins
- Whether enterprise fine-tuning creates sufficient lock-in

The honest answer is that nobody knows if "selling models instead of inference" is a viable strategy. The paper should flag this as a critical uncertainty rather than assuming total displacement.

---

## Summary of Recommended Changes

| Section | Change | Priority |
|---------|--------|----------|
| Abstract | Add "inference" qualifier to displacement claims | High |
| 3.1 | Decompose S into S_training + S_inference(δ) | High |
| 3.5 | Soften "evolutionary suicide" to acknowledge training persistence | Medium |
| 3.9 | Add latency advantage to R₀ framework | Medium |
| 4.4 | Make crossing condition explicitly about inference | High |
| New 4.5 | Training/inference bifurcation evidence table | High |
| 6, Pred 3 | Revise to distinguish inference vs training capex | Medium |
| 7 | Add coexistence equilibrium discussion | High |
| Throughout | Replace "centralized AI infrastructure" with "centralized AI inference" where appropriate | Medium |

None of these changes invalidate the core mechanism. They sharpen it by identifying exactly what decentralizes (inference) and what doesn't (training), making the predictions more precise and the paper more robust to the obvious counterargument that "we'll always need big datacenters for training."

The irony is that this refinement actually **strengthens** the paper's empirical relevance: the inference market is the larger, faster-growing revenue pool, and it's the one where the endogenous decentralization mechanism applies most directly.
