# Research Prompt: General-Weights Extension of the CES Triple Role Theorem

## The Problem

The CES Triple Role Theorem has been proved at the symmetric point with equal weights $a_j = 1/J$. The theorem states that for $F(\mathbf{x}) = (\sum a_j x_j^\rho)^{1/\rho}$ with $\rho < 1$, the single curvature parameter $K = (1-\rho)(J-1)/J$ simultaneously controls:

**(a) Superadditivity gap:** $F(\mathbf{x}+\mathbf{y}) - F(\mathbf{x}) - F(\mathbf{y}) \geq \Omega(K) \cdot \text{diversity}$

**(b) Correlation robustness:** $d_{\text{eff}} \geq \text{linear baseline} + \Omega(K^2) \cdot \text{idiosyncratic bonus}$

**(c) Strategic independence:** $\Delta(S) \leq -\Omega(K) \cdot \text{deviation}^2$

**Goal:** Extend all three results to **arbitrary weights** $a_j > 0$ with $\sum a_j = 1$. Find the generalized curvature parameter $K_{\text{gen}}(\rho, \mathbf{a})$ that replaces $K$ in all three bounds, with $K_{\text{gen}}(\rho, (1/J, \ldots, 1/J)) = (1-\rho)(J-1)/J$.

---

## What We Already Know

### The equal-weights proof strategy (all verified, see attached ces_triple_role_v2a.md)

1. **Curvature Lemma:** At the symmetric point $\bar{\mathbf{x}} = (c, \ldots, c)$ with $a_j = 1/J$, the Hessian of $F$ is $H_F = \frac{(1-\rho)}{J^2 c}[\mathbf{1}\mathbf{1}^T - JI]$. All $J-1$ principal curvatures of the isoquant equal $\kappa^* = (1-\rho)/(c\sqrt{J})$. The permutation symmetry of equal weights makes all tangent directions equivalent.

2. **Superadditivity** follows from concavity + degree-1 homogeneity (qualitative), with the quantitative bound from isoquant curvature comparison.

3. **Correlation robustness** follows from a second-order expansion around the symmetric point, decomposing into common mode and idiosyncratic modes. The curvature bonus arises because $Y_2 = -\frac{(1-\rho)}{2J\mu}\|\boldsymbol{\eta}\|^2$ depends only on idiosyncratic variation.

4. **Strategic independence** follows from convexity of the CES cooperative game (Shapley 1971) and the Hessian's negative definiteness restricted to coalition strategy spaces.

### Why equal weights matters in the current proof

- The symmetric point $x_j = c$ for all $j$ exists only with equal weights. With general weights, the "natural" point on the isoquant is $x_j^* = c \cdot (a_j)^{1/(1-\rho)} / (\sum_k a_k^{1/(1-\rho)})^{1/\rho}$ (the cost-minimizing bundle).
- At the symmetric point with equal weights, all principal curvatures coincide. With general weights, the curvatures differ across tangent directions.
- The spectral decomposition into $\mathbf{1}$ and $\mathbf{1}^\perp$ relies on equal weights. With general weights, the relevant decomposition is into $\nabla F$ and $(\nabla F)^\perp$.

### What we believe is true but haven't proved

For general weights, define the **weighted symmetric point** $\mathbf{x}^*(\mathbf{a})$ as the cost-minimizing allocation on the unit isoquant. The minimum principal curvature $\kappa_{\min}$ at $\mathbf{x}^*(\mathbf{a})$ should control all three bounds. The conjecture is:

$$K_{\text{gen}} = (1-\rho) \cdot h(\mathbf{a})$$

where $h(\mathbf{a})$ is some measure of weight dispersion satisfying $h(1/J, \ldots, 1/J) = (J-1)/J$ and $h(\mathbf{a}) \to 0$ as weights concentrate on a single component.

---

## Promising Attack Paths

### Path 1: Minimum principal curvature bound

Compute the full set of principal curvatures at the weighted symmetric point $\mathbf{x}^*(\mathbf{a})$. The Hessian of $F$ at a general point has a known form. The principal curvatures are the eigenvalues of the shape operator $S = -H_F/\|\nabla F\|$ restricted to the tangent space of the isoquant.

**Key computation needed:** At $\mathbf{x}^*(\mathbf{a})$, compute $H_F$ explicitly, project onto the tangent space $\{v : \nabla F \cdot v = 0\}$, and find eigenvalue bounds.

**Expected result:** $\kappa_{\min} \geq (1-\rho) \cdot g(\mathbf{a}) / (c \cdot \|\nabla F\|)$ where $g(\mathbf{a})$ involves the minimum weight $a_{\min}$ or the Herfindahl index $\text{HHI} = \sum a_j^2$.

**Literature to check:**
- Riemannian geometry of sublevel sets of homogeneous functions
- Curvature of production isoquants (Sato 1977, "Self-dual preferences")
- Principal curvatures of CES isoquants (any differential geometry treatment of these standard surfaces)

### Path 2: Strong concavity approach

Instead of computing curvatures directly, establish strong concavity of $F$ restricted to the isoquant with respect to an appropriate metric.

$F$ is concave but NOT strongly concave on $\mathbb{R}^J_+$ (it's homogeneous of degree 1, so it's linear along rays). But restricted to the isoquant $\{F = c\}$, the function $G(\mathbf{x}) = F(\mathbf{x}) - c$ is strongly concave with parameter depending on the curvature.

**Key idea:** Use the fact that for homogeneous-of-degree-1 concave functions, the restriction to any compact subset of the isoquant is strongly concave with modulus determined by the minimum principal curvature. This gives the superadditivity bound directly.

**Literature to check:**
- Nesterov's "Lectures on Convex Optimization" — strong convexity of level set restrictions
- Boyd & Vandenberghe — curvature of sublevel sets
- Rockafellar "Convex Analysis" — curvature of convex sets

### Path 3: Weighted Poincaré inequality

For the correlation robustness result (Part b), the equal-weights proof uses the decomposition into common mode $\bar{\epsilon}$ and idiosyncratic modes $\boldsymbol{\eta}$ with $\mathbf{1} \cdot \boldsymbol{\eta} = 0$. With general weights, the natural decomposition is:

$$\boldsymbol{\epsilon} = \bar{\epsilon}_a \cdot \mathbf{w} + \boldsymbol{\eta}_a$$

where $\mathbf{w} = \nabla F / \|\nabla F\|$ (the gradient direction at the weighted symmetric point) and $\boldsymbol{\eta}_a \perp \mathbf{w}$.

The curvature bonus then involves $\text{Var}[\|\boldsymbol{\eta}_a\|^2_W]$ where $W$ is a weight matrix determined by the Hessian of $F$ at $\mathbf{x}^*(\mathbf{a})$.

**Key computation:** Work out $H_F$ at $\mathbf{x}^*(\mathbf{a})$, decompose into the $\nabla F$ direction and its orthogonal complement, and compute the variance of the quadratic form in the idiosyncratic modes.

### Path 4: Convex game approach for Part (c)

The strategic independence result uses Shapley (1971): CES cooperative games are convex (the characteristic function $v(S) = \max_{\mathbf{x}_S} F(\mathbf{x}_S, \mathbf{0}_{-S})$ is convex). This holds for ALL weights, not just equal weights. The Shapley value lies in the core for convex games regardless of weights.

**This part may already generalize without additional work.** The strong concavity of $F$ restricted to the coalition's strategy space gives the manipulation bound. The minimum eigenvalue of $H_F$ restricted to the coalition's tangent directions (with $\nabla F \cdot \delta = 0$) is the key quantity.

For general weights, the minimum eigenvalue is:
$$\lambda_{\min}(H_F|_{\text{tangent}}) = -\frac{(1-\rho)}{c} \cdot \min_j \frac{a_j x_j^{*\rho-2}}{(\sum a_k x_k^{*\rho})^{(2-\rho)/\rho}}$$

This should simplify at the weighted symmetric point to something involving $a_{\min}$ or $\text{HHI}$.

### Path 5: Entropy / information-geometric approach

The CES function is related to the Rényi entropy: $F(\mathbf{x})^\rho = \sum a_j x_j^\rho$ is a weighted power mean. The isoquant geometry is the geometry of Rényi divergence level sets.

The information geometry literature (Amari, Ay, Jost, Lê, Schwachhöfer) has results on the curvature of statistical manifolds parameterized by Rényi divergence. If the CES isoquant can be mapped to a Rényi divergence level set, curvature bounds from information geometry may apply directly.

**Literature to check:**
- Amari "Information Geometry and Its Applications" — curvature of alpha-families
- van Erven & Harremoës "Rényi Divergence and Kullback-Leibler Divergence" (2014)
- Naudts "Generalised Thermostatistics" — deformed exponential families and CES-like structures

---

## What Success Looks Like

A complete solution would state and prove:

**Theorem (General CES Triple Role).** Let $F(\mathbf{x}) = (\sum_{j=1}^J a_j x_j^\rho)^{1/\rho}$ with $\rho < 1$, weights $a_j > 0$, $\sum a_j = 1$, and $J \geq 2$. Define

$$K_{\text{gen}} = (1-\rho) \cdot h(\mathbf{a})$$

where $h(\mathbf{a})$ is explicitly computable. Then:

**(a)** $F(\mathbf{x}+\mathbf{y}) - F(\mathbf{x}) - F(\mathbf{y}) \geq \Omega(K_{\text{gen}}) \cdot \text{diversity}$

**(b)** $d_{\text{eff}} \geq \text{linear baseline} + \Omega(K_{\text{gen}}^2) \cdot \text{idiosyncratic bonus}$

**(c)** $\Delta(S) \leq -\Omega(K_{\text{gen}}) \cdot \text{deviation}^2$

with $h(1/J, \ldots, 1/J) = (J-1)/J$.

**The most likely candidate for $h(\mathbf{a})$:**

$$h(\mathbf{a}) = 1 - \sum_{j=1}^J a_j^{2/(2-\rho)}  \Big/ \left(\sum_{j=1}^J a_j^{1/(2-\rho)}\right)^2$$

or more simply:

$$h(\mathbf{a}) = 1 - \text{HHI}(\mathbf{a}) = 1 - \sum a_j^2$$

which equals $(J-1)/J$ at equal weights. But verify — the correct $h$ must emerge from the actual minimum curvature computation, not from a guess.

---

## Attached Files

- **ces_triple_role_v2a.md** — The complete equal-weights proof. All notation, lemmas, and proof structure are established here. The general-weights extension should follow the same structure with modifications to the Curvature Lemma and subsequent steps.

## Instructions

1. Start with Path 1 (minimum principal curvature). Compute $H_F$ at the weighted symmetric point $\mathbf{x}^*(\mathbf{a})$ explicitly. Find the eigenvalues of the shape operator. Identify $\kappa_{\min}$.

2. If Path 1 yields $\kappa_{\min}$ in closed form, propagate through all three parts of the theorem using the same proof structure as v2a.

3. If Path 1 stalls, try Path 2 (strong concavity) — the bound may be looser but sufficient.

4. For Part (c), check whether the convex game argument generalizes without additional curvature computation.

5. For Part (b), the Fisher information bridge is the hardest step. The weighted decomposition into common and idiosyncratic modes requires care with the weight-dependent inner product.

6. Throughout: track exactly where equal weights was used in v2a and what changes with general weights. The qualitative results (superadditivity holds, $d_{\text{eff}}$ exceeds linear baseline, manipulation gain is negative) should follow from concavity and homogeneity alone. The quantitative bounds are where $K_{\text{gen}}$ enters.
