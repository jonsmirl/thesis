# Casimir Functions and Network Renormalization

## Two related ideas from the arxiv scan that could extend the framework.

---

## Part 1: Casimir Functions as Conservation Law Tests

### Source
van der Schaft, A.J. "Port-Hamiltonian Systems: Network Modeling and Control of Nonlinear Physical Systems." arXiv:2412.19673v1, Dec 2024.

### The Idea

Paper 13 (Conservation Laws) derives conservation laws for the (ρ,T) system but doesn't have a sharp criterion for distinguishing *exact* conservation laws from *approximate* ones. van der Schaft provides this.

A **Casimir function** C(x) satisfies:

    ∇C(x)^T · [J(x) - R(x)] = 0  for all x

If R = 0 (no dissipation), this reduces to:

    ∇C(x)^T · J(x) = 0

i.e., C lives in the kernel of the structure matrix J. This is a purely algebraic condition — you can check it without solving any ODEs.

### Application to Paper 13

Paper 13 identifies several conservation laws:
- Aggregate CES output (under balanced growth)
- Information capacity (Shannon bound)
- Cross-level coupling constants

The Casimir test: write the (ρ,T) system in port-Hamiltonian form ẋ = [J-R]∇H + gu, then check which of these are in ker(J^T). Those are *exact* Casimirs. Others may be approximate (broken by dissipation R or by port inputs gu).

### What This Buys

1. **Classification**: Exact Casimirs vs. broken symmetries. Exact ones hold regardless of parameter values; broken ones hold only approximately and the breaking rate depends on R.

2. **Noether connection**: Each Casimir corresponds to a symmetry of the structure matrix J. Identifying the symmetry tells you *why* the conservation law holds — what structural feature of the economy guarantees it.

3. **Policy implications**: Casimirs constrain achievable policy outcomes (you can't violate a conservation law). Knowing which are exact vs. approximate tells you which constraints are hard (structural) vs. soft (can be relaxed by changing dissipation or ports).

### Specific Test

For the 4-level hierarchy (Papers 1-4), write:

    x = (c, N, φ, S)  (hardware cost, mesh density, capability, stablecoin size)

    J = | 0    J₁₂   0     0   |
        |-J₁₂  0    J₂₃    0   |
        | 0   -J₂₃   0    J₃₄  |
        | 0    0   -J₃₄    0   |

This is tridiagonal antisymmetric (nearest-neighbor coupling from timescale separation). Its kernel is 2-dimensional for even-dimensional systems. Find the two Casimirs explicitly — they should correspond to the "slow + fast" and "even + odd" level combinations identified in Paper 13.

---

## Part 2: Network Renormalization for Multi-Scale CES

### Source
From arxiv scan: network renormalization / coarse-graining literature. Key papers:
- Villegas, P., et al. "Laplacian Renormalization Group for Heterogeneous Networks." Nature Physics 19: 445-450, 2023.
- Garcia-Perez, G., et al. "Multiscale unfolding of real networks by geometric renormalization." Nature Physics 14: 583-589, 2018.

### The Problem

The thesis uses nested CES at multiple scales:
- Firm level: F_firm = (Σ x_j^ρ_firm)^(1/ρ_firm)
- Sector level: F_sector = (Σ F_firm^ρ_sector)^(1/ρ_sector)
- Economy level: F_economy = (Σ F_sector^ρ_economy)^(1/ρ_economy)

Paper 15 (Endogenous ρ) allows ρ to vary across levels. But the nesting is assumed, not derived. **Why should coarse-graining a fine-grained CES economy produce a CES economy at the coarser scale?**

### The Idea

Network renormalization provides a rigorous answer. The procedure:

1. Start with a network of N agents, each producing x_i, connected by a weighted graph W_ij
2. Define CES aggregate at the micro level
3. Coarse-grain: merge groups of agents into super-nodes, defining effective inputs and effective ρ
4. Ask: under what conditions does the coarse-grained system have CES form?

### What's Known

For **linear** aggregation (ρ = 1), coarse-graining preserves linearity trivially. For **Cobb-Douglas** (ρ → 0), coarse-graining preserves the form because log-linear functions are closed under averaging. For **general CES**, the situation is more subtle.

**Conjecture**: CES is the unique aggregation function that is (approximately) preserved under renormalization for a broad class of coarse-graining schemes. This would make CES the "fixed point of the renormalization group" for economic aggregation — analogous to how the Gaussian is the fixed point of the CLT.

### Why This Matters

1. **Justifies the axiom**: Paper 16's Axiom A1 (CES production) would follow from a deeper principle — CES is what survives coarse-graining, so any theory that operates at multiple scales must use CES.

2. **Determines how ρ changes across scales**: The RG flow equation dρ/d(log scale) = β(ρ) would tell you how complementarity evolves as you zoom out. If β has a fixed point, that's the natural ρ for macroeconomic aggregation.

3. **Connects to universality**: In physics, RG fixed points classify universality classes. If CES is the fixed point, then the specific micro-level production function doesn't matter — anything "close enough" flows to CES at macro scales. This would explain why CES works empirically despite being a parametric assumption.

### Sketch of the Calculation

For a symmetric CES economy with J agents:

    F = (1/J Σ x_j^ρ)^(1/ρ)

Coarse-grain into J/k blocks of k agents each. Within block b, agents have inputs {x_j : j ∈ b}. The block-level aggregate is:

    Y_b = (1/k Σ_{j∈b} x_j^ρ)^(1/ρ)

The economy-level aggregate of blocks:

    F = (1/(J/k) Σ_b Y_b^ρ')^(1/ρ')

For this to equal the original F, we need ρ' = ρ. So CES with uniform ρ is exactly self-similar under block coarse-graining. But with heterogeneous ρ within blocks, the effective ρ' ≠ micro ρ, and the RG flow begins.

### Open Questions

1. **Is the CES fixed point stable?** If you start with a non-CES production function and coarse-grain repeatedly, do you converge to CES? Under what conditions?

2. **What is β(ρ)?** The RG flow function for complementarity. Does it have fixed points other than ρ = 1 (perfect substitutes) and ρ → -∞ (Leontief)?

3. **Connection to Paper 15**: The endogenous ρ dynamics (firm optimization, selection, standardization) operate *within* a scale. The RG flow operates *across* scales. Do they commute? If not, the order of limits matters — a point that could have empirical implications.

4. **Empirical test**: If CES is the RG fixed point, then aggregate production functions should be "more CES" (better fit) at higher levels of aggregation. Test: estimate Kmenta approximation residuals at establishment, firm, industry, and sector levels. Residuals should shrink with aggregation.

## Potential Use

Not a standalone paper, but could strengthen Paper 16 (Unified Theory) significantly by grounding Axiom A1 in a deeper principle. Could also extend Paper 15 by adding the cross-scale dimension to ρ dynamics.
