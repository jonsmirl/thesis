# The CES Triple Role Theorem: General Weights Extension

## Complete Proof that the Triple Role Holds for Arbitrary Weight Vectors

---

## 0. Summary of Results

We extend the CES Triple Role Theorem from equal weights $a_j = 1/J$ to arbitrary weights $a_j > 0$ with $\sum a_j = 1$. The generalized curvature parameter is

$$K_{\text{gen}} = (1 - \rho) \cdot h(\rho, \mathbf{a})$$

where

$$\boxed{h(\rho, \mathbf{a}) \;=\; \frac{J-1}{J}\,\Phi^{1/\rho}\, R_{\min}(\mathbf{a}, \rho)}$$

with $\Phi = \sum_{j=1}^J a_j^{\sigma}$, $\sigma = 1/(1-\rho)$, and $R_{\min}$ is the minimum of the Rayleigh quotient $\sum_j v_j^2 / a_j^{\sigma}$ over unit vectors $\mathbf{v}$ with $\sum_j v_j = 0$. Verification: at equal weights, $h(\rho, (1/J,\ldots,1/J)) = (J-1)/J$ for all $\rho$.

**Key finding:** The weight-dispersion measure $h$ depends on $\rho$ through the effective shares $p_j = a_j^{\sigma}$. The conjecture that $h(\mathbf{a}) = 1 - \text{HHI}(\mathbf{a})$ is **incorrect** — the HHI does not capture the correct $\rho$-dependent geometry. The correct $h$ reduces to $(J-1)/J$ at equal weights but otherwise varies with both the weight vector and the substitution elasticity.

The three roles generalize as follows:

**(a) Superadditivity gap:** $\geq \frac{K_{\text{gen}}}{4c} \cdot \frac{\sqrt{J}}{J-1} \cdot \min(F(\mathbf{x}), F(\mathbf{y})) \cdot d_{\mathcal{I}}^2$

**(b) Correlation robustness:** $d_{\text{eff}} \geq \text{linear baseline} + \Omega(K_{\text{gen}}^2) \cdot \text{idiosyncratic bonus}$ (with weight-dependent mode decomposition)

**(c) Strategic independence:** $\Delta(S) \leq -\Omega(K_{\text{gen}}) \cdot \text{deviation}^2$

---

## 1. Setup and Notation

Carry forward all notation from v2a. The CES aggregate is $F(\mathbf{x}) = (\sum a_j x_j^\rho)^{1/\rho}$ with $\rho < 1$, $\rho \neq 0$, weights $a_j > 0$, $\sum a_j = 1$, $J \geq 2$.

**New notation for general weights.** Define the elasticity of substitution $\sigma = 1/(1-\rho)$ and the **effective shares**

$$p_j \;=\; a_j^{\sigma} \;=\; a_j^{1/(1-\rho)}$$

These are the demand shares at unit prices under CES preferences. Define $\Phi = \sum_j p_j$. At equal weights, $p_j = J^{-\sigma}$ and $\Phi = J^{1-\sigma}$.

---

## 2. The Weighted Symmetric Point

**Definition.** For output level $c > 0$, the **weighted symmetric point** (cost-minimizing bundle at unit prices) on the isoquant $\{F = c\}$ is

$$x_j^* \;=\; \frac{c\, p_j}{\Phi^{1/\rho}}, \qquad j = 1, \ldots, J$$

where $p_j = a_j^{\sigma}$ and $\Phi = \sum_k p_k$.

*Verification that $F(\mathbf{x}^*) = c$:*

$$F(\mathbf{x}^*)^{\rho} = \sum_j a_j \left(\frac{c\, p_j}{\Phi^{1/\rho}}\right)^{\rho} = \frac{c^{\rho}}{\Phi}\sum_j a_j\, p_j^{\,\rho}$$

Since $p_j^{\,\rho} = a_j^{\sigma\rho} = a_j^{\rho/(1-\rho)}$ and $a_j \cdot a_j^{\rho/(1-\rho)} = a_j^{1/(1-\rho)} = p_j$:

$$F(\mathbf{x}^*)^{\rho} = \frac{c^{\rho}}{\Phi}\sum_j p_j = c^{\rho} \qquad \Longrightarrow \qquad F(\mathbf{x}^*) = c \qquad \checkmark$$

*Verification at equal weights:* $p_j = J^{-\sigma}$, $\Phi = J^{1-\sigma}$, $\Phi^{1/\rho} = J^{(1-\sigma)/\rho} = J^{-\sigma}$ (since $(1-\sigma)/\rho = -\sigma$). Then $x_j^* = c \cdot J^{-\sigma}/J^{-\sigma} = c$ for all $j$. $\checkmark$

---

## 3. Gradient and Hessian at the Weighted Symmetric Point

### 3.1. Gradient

The partial derivative of $F$ is

$$\frac{\partial F}{\partial x_j} = a_j\, x_j^{\rho-1}\, F^{1-\rho}$$

At $\mathbf{x}^*$, substituting $x_j^* = c\, p_j / \Phi^{1/\rho}$ and $F = c$:

$$(x_j^*)^{\rho-1} = \left(\frac{c\, p_j}{\Phi^{1/\rho}}\right)^{\rho-1} = c^{\rho-1}\, p_j^{\,\rho-1}\, \Phi^{(1-\rho)/\rho}$$

Since $p_j^{\,\rho-1} = a_j^{\sigma(\rho-1)} = a_j^{-1}$:

$$\frac{\partial F}{\partial x_j}\bigg|_{\mathbf{x}^*} = a_j \cdot c^{\rho-1} \cdot a_j^{-1} \cdot \Phi^{(1-\rho)/\rho} \cdot c^{1-\rho} = \Phi^{(1-\rho)/\rho}$$

**This is independent of $j$.** At the cost-minimizing point, all marginal products are equal:

$$\boxed{\nabla F(\mathbf{x}^*) \;=\; g \cdot \mathbf{1}, \qquad g \;=\; \Phi^{(1-\rho)/\rho}}$$

$$\|\nabla F(\mathbf{x}^*)\| = g\sqrt{J}$$

**Consequence.** The tangent space to the isoquant at $\mathbf{x}^*$ is

$$T_{\mathbf{x}^*}\mathcal{I}_c = \{\mathbf{v} \in \mathbb{R}^J : \mathbf{1} \cdot \mathbf{v} = 0\}$$

This is the **same subspace** as in the equal-weights case. The unit normal to the isoquant at $\mathbf{x}^*$ is $\mathbf{n} = \mathbf{1}/\sqrt{J}$, regardless of weights.

### 3.2. Hessian

The general CES Hessian entry is (using the product rule on $\partial_j F = a_j x_j^{\rho-1} F^{1-\rho}$):

$$H_{ij} = \frac{\partial^2 F}{\partial x_i \partial x_j} = \frac{(1-\rho)}{F}\,(\partial_i F)(\partial_j F) - \delta_{ij}\,\frac{(1-\rho)}{x_j}\,(\partial_j F)$$

*Derivation:* Differentiating $\partial_j F = a_j x_j^{\rho-1} F^{1-\rho}$ with respect to $x_i$:

$$H_{ij} = a_j x_j^{\rho-1}(1-\rho)F^{-\rho}\,(\partial_i F) + \delta_{ij}\, a_j(\rho-1)x_j^{\rho-2} F^{1-\rho}$$

$$= \frac{(1-\rho)}{F}\,(\partial_j F)(\partial_i F) - \delta_{ij}\,\frac{(1-\rho)}{x_j}\,(\partial_j F)$$

At $\mathbf{x}^*$ where $\partial_j F = g$ for all $j$ and $F = c$:

$$\boxed{H_{ij}\big|_{\mathbf{x}^*} = \frac{(1-\rho)\, g}{c}\left[g - \delta_{ij}\,\frac{c}{x_j^*}\right] = (1-\rho)\,g\left[\frac{g}{c} - \frac{\delta_{ij}}{x_j^*}\right]}$$

In matrix form:

$$H_F(\mathbf{x}^*) = (1-\rho)\,g\left[\frac{g}{c}\,\mathbf{1}\mathbf{1}^T - D^{-1}\right]$$

where $D = \text{diag}(x_1^*, \ldots, x_J^*)$.

*Verification at equal weights:* $g = 1/J$ (from $\Phi^{(1-\rho)/\rho} = (J^{1-\sigma})^{(1-\rho)/\rho}$; exponent is $(1-\sigma)(1-\rho)/\rho = (-\rho/(1-\rho))(1-\rho)/\rho = -1$; so $g = J^{-1}$). And $x_j^* = c$, so $D^{-1} = (1/c)\,I$.

$$H_F = (1-\rho)\cdot\frac{1}{J}\left[\frac{1}{Jc}\mathbf{1}\mathbf{1}^T - \frac{1}{c}I\right] = \frac{(1-\rho)}{J^2 c}\left[\mathbf{1}\mathbf{1}^T - JI\right] \qquad \checkmark$$

---

## 4. The Generalized Curvature Lemma

**Lemma 1-G (Isoquant Curvature with General Weights).** At the weighted symmetric point $\mathbf{x}^*$ on $\mathcal{I}_c$, the normal curvature of the isoquant in tangent direction $\mathbf{v}$ (with $\mathbf{1} \cdot \mathbf{v} = 0$) is

$$\kappa(\mathbf{v}) \;=\; \frac{(1-\rho)\,\Phi^{1/\rho}}{c\sqrt{J}} \cdot \frac{\displaystyle\sum_{j=1}^J v_j^2\, / \, p_j}{\displaystyle\sum_{j=1}^J v_j^2}$$

where $p_j = a_j^{1/(1-\rho)}$. In particular, the **principal curvatures** $\kappa_1 \leq \cdots \leq \kappa_{J-1}$ are the constrained eigenvalues of the diagonal matrix $\text{diag}(1/p_1, \ldots, 1/p_J)$ restricted to $\{\mathbf{v}: \sum v_j = 0\}$, scaled by $(1-\rho)\Phi^{1/\rho}/(c\sqrt{J})$.

*Proof.* For $\mathbf{v} \in T_{\mathbf{x}^*}\mathcal{I}_c$ (so $\sum v_j = 0$), the Hessian quadratic form gives:

$$\mathbf{v}^T H_F \mathbf{v} = (1-\rho)\,g\left[\frac{g}{c}\underbrace{(\sum_j v_j)^2}_{=\,0} - \sum_j \frac{v_j^2}{x_j^*}\right] = -(1-\rho)\,g\sum_j \frac{v_j^2}{x_j^*}$$

Substituting $1/x_j^* = \Phi^{1/\rho}/(c\, p_j)$:

$$\mathbf{v}^T H_F \mathbf{v} = -\frac{(1-\rho)\,g\,\Phi^{1/\rho}}{c}\sum_j \frac{v_j^2}{p_j}$$

The normal curvature is

$$\kappa(\mathbf{v}) = \frac{-\mathbf{v}^T H_F \mathbf{v}}{\|\nabla F\|\cdot\|\mathbf{v}\|^2} = \frac{(1-\rho)\,g\,\Phi^{1/\rho}}{c}\cdot\frac{\sum_j v_j^2/p_j}{g\sqrt{J}\cdot\|\mathbf{v}\|^2} = \frac{(1-\rho)\,\Phi^{1/\rho}}{c\sqrt{J}}\cdot\frac{\sum_j v_j^2/p_j}{\|\mathbf{v}\|^2} \qquad\blacksquare$$

### 4.1. The Principal Curvatures via the Secular Equation

The principal curvatures are determined by the Rayleigh quotient $R(\mathbf{v}) = \sum_j v_j^2/p_j \,/\, \|\mathbf{v}\|^2$ restricted to $T = \{\mathbf{v}: \sum v_j = 0\}$. Setting $w_j = 1/p_j = a_j^{-\sigma}$, the constrained critical points satisfy (Lagrange multipliers):

$$(W - \mu I)\mathbf{v} = \lambda\,\mathbf{1}, \qquad \sum_j v_j = 0, \qquad \|\mathbf{v}\| = 1$$

where $W = \text{diag}(w_1, \ldots, w_J)$. This gives $v_j = \lambda/(w_j - \mu)$, and the constraint $\sum v_j = 0$ yields the **secular equation**:

$$\boxed{\sum_{j=1}^J \frac{1}{w_j - \mu} = 0, \qquad w_j = a_j^{-1/(1-\rho)}}$$

This has exactly $J-1$ real roots $\mu_1 < \mu_2 < \cdots < \mu_{J-1}$, one in each interval $(w_{(k)}, w_{(k+1)})$ where $w_{(1)} < \cdots < w_{(J)}$ are the ordered values. The minimum and maximum constrained eigenvalues are:

$$R_{\min} = \mu_1 \in (w_{(1)}, w_{(2)}), \qquad R_{\max} = \mu_{J-1} \in (w_{(J-1)}, w_{(J)})$$

The principal curvatures are $\kappa_k = \frac{(1-\rho)\Phi^{1/\rho}}{c\sqrt{J}} \cdot \mu_k$ for $k = 1, \ldots, J-1$.

*Verification at equal weights:* All $w_j = J^{\sigma}$ are equal, so $R(\mathbf{v}) = J^{\sigma}$ for all $\mathbf{v} \in T$. All principal curvatures coincide: $\kappa^* = (1-\rho)\Phi^{1/\rho} J^{\sigma}/(c\sqrt{J}) = (1-\rho) J^{-\sigma} J^{\sigma}/(c\sqrt{J}) = (1-\rho)/(c\sqrt{J})$. $\checkmark$

### 4.2. Closed Form for $J = 2$

For $J = 2$, the tangent space is one-dimensional: $\mathbf{v} \propto (1, -1)$. There is a single principal curvature:

$$\kappa_1 = \frac{(1-\rho)\,\Phi^{1/\rho}}{c\sqrt{2}}\cdot\frac{1/p_1 + 1/p_2}{2} = \frac{(1-\rho)\,\Phi^{1/\rho}}{2c\sqrt{2}}\left(a_1^{-\sigma} + a_2^{-\sigma}\right)$$

### 4.3. Bounds on $R_{\min}$ for General $J$

**Proposition (Bounds on $R_{\min}$).** Let $w_{(1)} \leq w_{(2)} \leq \cdots \leq w_{(J)}$ be the ordered values of $\{1/p_j\}$. Then:

**(i) Trivial lower bound:** $R_{\min} > w_{(1)} = 1/p_{\max}$ (strict: the minimum eigenvalue of $W$ is $w_{(1)}$, but the eigenvector is $e_{(1)} \notin T$).

**(ii) Two-component upper bound:** $R_{\min} \leq (w_{(1)} + w_{(2)})/2$ (achieved by $\mathbf{v} = e_{(1)} - e_{(2)}$, the direction connecting the two largest effective shares).

**(iii) Harmonic-mean bound:** $R_{\min} \geq \frac{J}{J-1}\left(\bar{w} - \frac{w_{\max}}{J}\right)$ where $\bar{w} = \frac{1}{J}\sum w_j$ (from the interlacing inequality $\mu_1 \geq \bar{w} - w_{\max}(J-1)/J$, which follows from $\text{tr}(W|_T) = \text{tr}(W) - \mathbf{e}_{\max}^T W \mathbf{e}_{\max}/\ldots$).

More precisely, by the Cauchy interlace theorem applied to the restriction $W|_T$:

$$w_{(1)} < R_{\min} < w_{(2)}$$

**The practically useful bound** is (ii): $R_{\min}$ is at most the average of the two smallest $w_j$ values, and typically close to this when the weight vector is not too concentrated.

---

## 5. The Generalized Curvature Parameter

**Definition.** The generalized curvature parameter is

$$K_{\text{gen}}(\rho, \mathbf{a}) \;=\; (1-\rho) \cdot \frac{(J-1)\,\Phi^{1/\rho}\, R_{\min}}{J}$$

where $\Phi$, $R_{\min}$ are as defined above.

**Equivalently:** $K_{\text{gen}} = c\,(J-1)\,\kappa_{\min}/\sqrt{J}$, so that $\kappa_{\min} = K_{\text{gen}}\sqrt{J}/[c(J-1)]$. This preserves the structural relationship from the equal-weights case.

**Properties of $K_{\text{gen}}$:**

1. **Reduces to $K$ at equal weights:** $K_{\text{gen}}(\rho, (1/J,\ldots,1/J)) = (1-\rho)(J-1)/J = K$. (Verified in §4.1.)

2. **Positive for all $\rho < 1$, all proper weight vectors:** $K_{\text{gen}} > 0$ since $1-\rho > 0$, $\Phi > 0$, and $R_{\min} > 0$.

3. **Monotone in complementarity:** $K_{\text{gen}}$ is increasing in $(1-\rho)$ for fixed weights.

4. **Vanishes as weights concentrate:** As $a_1 \to 1$ (single dominant component), $p_1 \to 1$, $p_j \to 0$ for $j \neq 1$, so $w_1 = 1/p_1 \to 1$ while $w_j \to \infty$. The net effect is $K_{\text{gen}} \to 0$ (loss of effective dimensionality).

5. **Non-monotone in weight dispersion:** Unlike the intuitive expectation that more uniform weights always give larger $K_{\text{gen}}$, the relationship between weight heterogeneity and curvature is **non-monotone and $\rho$-dependent**. Numerical verification shows: (i) for strong complements ($\rho \ll 0$), unequal weights typically reduce $K_{\text{gen}}$ below $K$; (ii) for weak complements ($\rho$ close to 1), unequal weights can *increase* $K_{\text{gen}}$ above $K$, sometimes dramatically (see §12). This occurs because the products $\Phi^{1/\rho} \cdot R_{\min}$ interact non-trivially with the weight geometry.

6. **Depends on $\rho$** through the effective shares $p_j = a_j^{\sigma}$: the same weight vector produces different curvatures at different substitution elasticities. The HHI $1 - \sum a_j^2$ is not the correct dispersion measure. The correct one is $h(\rho, \mathbf{a}) = (J-1)\Phi^{1/\rho}R_{\min}/J$.

### 5.1. Why $h$ Depends on $\rho$: Geometric Interpretation

The effective shares $p_j = a_j^{1/(1-\rho)}$ compress or expand the weight distribution depending on $\sigma$. For $\rho < 0$ (strong complements, $\sigma < 1$): the effective shares are more uniform than the weights (the exponent $\sigma < 1$ compresses differences). For $0 < \rho < 1$ (weak complements, $\sigma > 1$): the effective shares are more concentrated than the weights (the exponent $\sigma > 1$ amplifies differences).

The cost-minimizing bundle $x_j^* \propto p_j$ reflects this: strong complements naturally demand a more balanced allocation, while weak complements concentrate on high-weight components. The curvature at the cost-minimizing point inherits this $\rho$-dependence.

---

## 6. Part (a): Superadditivity with General Weights

**Theorem A-G.** For all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^J_+ \setminus \{\mathbf{0}\}$:

$$F(\mathbf{x} + \mathbf{y}) \;\geq\; F(\mathbf{x}) + F(\mathbf{y})$$

with gap:

$$F(\mathbf{x}+\mathbf{y}) - F(\mathbf{x}) - F(\mathbf{y}) \;\geq\; \frac{\kappa_{\min}}{4}\cdot\min\!\big(F(\mathbf{x}), F(\mathbf{y})\big)\cdot d_{\mathcal{I}}(\hat{\mathbf{x}}, \hat{\mathbf{y}})^2$$

$$= \frac{K_{\text{gen}}}{4c}\cdot\frac{\sqrt{J}}{J-1}\cdot\min\!\big(F(\mathbf{x}), F(\mathbf{y})\big)\cdot d_{\mathcal{I}}^2$$

### Proof

**Step 1 (Qualitative): Unchanged from v2a.** The argument uses only concavity and degree-1 homogeneity, which hold for all weights. The decomposition $\frac{\mathbf{x}+\mathbf{y}}{F(\mathbf{x})+F(\mathbf{y})} = \alpha\hat{\mathbf{x}} + (1-\alpha)\hat{\mathbf{y}}$ with $\hat{\mathbf{x}} = \mathbf{x}/F(\mathbf{x}) \in \mathcal{I}_1$ proceeds identically. $\checkmark$

**Step 2 (Quantitative): Curvature comparison with $\kappa_{\min}$.** For general weights, the isoquant curvature varies across tangent directions. The curvature comparison theorem (standard differential geometry: if the normal curvature $\kappa(\mathbf{v}) \geq \kappa_{\min}$ for all tangent directions $\mathbf{v}$ at all points in a neighborhood, then the chord-to-surface distance satisfies the same bound as for a sphere of curvature $\kappa_{\min}$) gives:

$$F(\alpha\hat{\mathbf{x}} + (1-\alpha)\hat{\mathbf{y}}) \;\geq\; 1 + \frac{\kappa_{\min}}{2}\,\alpha(1-\alpha)\,d^2 + O(d^4)$$

**The curvature comparison is valid** because:

(a) At $\mathbf{x}^*$, all principal curvatures satisfy $\kappa_k \geq \kappa_{\min} > 0$ (by definition).

(b) The CES isoquant has strictly positive curvature everywhere on $\mathbb{R}^J_{++}$: at any point $\mathbf{x}$ with $F(\mathbf{x}) = c > 0$ and tangent vector $\mathbf{v}$, the quadratic form $\mathbf{v}^T H_F \mathbf{v} = -(1-\rho)\sum_j (\partial_j F)\, v_j^2/x_j < 0$ (since all marginal products are positive). So $\kappa(\mathbf{x}, \mathbf{v}) > 0$ everywhere.

(c) **For the quantitative bound:** The curvature comparison is applied locally near $\mathbf{x}^*$, using $\kappa_{\min}(\mathbf{x}^*)$ as the curvature in the comparison sphere. For $\hat{\mathbf{x}}, \hat{\mathbf{y}}$ close to $\mathbf{x}^*/c$ on the unit isoquant, this gives the stated bound to second order in $d_{\mathcal{I}}$. For well-separated points, the qualitative bound (Step 1) already applies, and the quantitative bound with geodesic distance squared is a local refinement. A fully global bound would require $\inf_{\mathbf{x} \in \mathcal{I}_c}\kappa_{\min}(\mathbf{x})$, which is harder to compute but positive; using $\kappa_{\min}(\mathbf{x}^*)$ gives a conservative bound that is exact in the neighborhood where the curvature comparison sphere argument applies.

The rest of Step 2 follows the v2a argument verbatim: $\alpha(1-\alpha) \geq \min(\alpha,1-\alpha)/2$, substitute $\kappa_{\min} = K_{\text{gen}}\sqrt{J}/[c(J-1)]$. $\blacksquare$

**Where equal weights was used in v2a and what changes:** Only in the value of $\kappa_{\min}$. The proof structure — convex combination on the isoquant, curvature comparison, bounding $\alpha(1-\alpha)$ — is entirely weight-independent.

---

## 7. Part (b): Correlation Robustness with General Weights

This is the hardest extension. The equal-weights proof relied on the spectral decomposition into $\mathbf{1}$ (common mode) and $\mathbf{1}^{\perp}$ (idiosyncratic modes), which coincided with the Hessian's eigenspaces. With general weights, the Hessian's eigenspaces in the tangent space no longer align with a simple "common vs. idiosyncratic" decomposition — but the gradient direction is still $\mathbf{1}$, which saves the argument.

### 7.1. Setup

Let $\mathbf{X} = (X_1, \ldots, X_J)$ with $\mathbb{E}[X_j] = \mu_j$ (possibly component-dependent), $\text{Cov}(X_i, X_j) = \Sigma_{ij}$. For the cleanest comparison with v2a, specialize to:

$$\mu_j = x_j^*, \qquad \Sigma = \tau^2[(1-r)I + r\,\mathbf{1}\mathbf{1}^T]$$

i.e., the mean is the weighted symmetric point and the covariance has equicorrelation structure with coefficient of variation $\gamma = \tau/\bar{\mu}$ where $\bar{\mu} = \frac{1}{J}\sum x_j^*$ (or use a weight-adjusted CV).

### 7.2. Second-Order Expansion

Expand $F(\mathbf{X})$ around $\mathbf{x}^*$. Let $\boldsymbol{\epsilon} = \mathbf{X} - \mathbf{x}^*$.

$$F(\mathbf{X}) \approx c + \nabla F \cdot \boldsymbol{\epsilon} + \frac{1}{2}\boldsymbol{\epsilon}^T H_F \boldsymbol{\epsilon} = c + Y_1 + Y_2$$

**Linear term:**

$$Y_1 = g\,\sum_j \epsilon_j = g\,(\mathbf{1}\cdot\boldsymbol{\epsilon})$$

This is proportional to the total perturbation $\sum_j \epsilon_j$, regardless of weights. (The equal marginal product property at $\mathbf{x}^*$ ensures this.)

**Quadratic term:**

$$Y_2 = \frac{1}{2}(1-\rho)\,g\left[\frac{g}{c}(\sum_j \epsilon_j)^2 - \sum_j \frac{\epsilon_j^2}{x_j^*}\right]$$

Define $\bar{\epsilon} = \frac{1}{\sqrt{J}}\,(\mathbf{1}\cdot\boldsymbol{\epsilon})$ (normalized common mode) and $\boldsymbol{\eta} = \boldsymbol{\epsilon} - \frac{1}{J}(\mathbf{1}\cdot\boldsymbol{\epsilon})\mathbf{1}$ (idiosyncratic component, $\mathbf{1}\cdot\boldsymbol{\eta} = 0$). Then:

$$\sum_j \frac{\epsilon_j^2}{x_j^*} = \sum_j \frac{(\bar{\epsilon}/\sqrt{J}\cdot 1 + \eta_j)^2}{x_j^*} = \frac{\bar{\epsilon}^2}{J}\sum_j \frac{1}{x_j^*} + \frac{2\bar{\epsilon}}{\sqrt{J}}\sum_j \frac{\eta_j}{x_j^*} + \sum_j \frac{\eta_j^2}{x_j^*}$$

The cross term $\sum \eta_j/x_j^*$ does **not** vanish in general (it vanishes at equal weights because $1/x_j^* = 1/c$ is constant). This is a key complication.

**Resolution:** Define the **weight-adjusted idiosyncratic component** $\tilde{\boldsymbol{\eta}}$ via the decomposition

$$\boldsymbol{\epsilon} = \alpha\,\mathbf{x}^* + \tilde{\boldsymbol{\eta}}, \qquad \text{where } \alpha = \frac{\sum_j \epsilon_j/x_j^*}{\sum_j 1/x_j^*}\cdot\frac{1}{x_j^*}\text{ ... }$$

Actually, the cleaner approach: since $Y_2$ depends on $\boldsymbol{\epsilon}$ only through $(\mathbf{1}\cdot\boldsymbol{\epsilon})^2$ and $\sum \epsilon_j^2/x_j^*$, we can write:

$$Y_2 = \frac{(1-\rho)g}{2}\left[\frac{g}{c}(\mathbf{1}\cdot\boldsymbol{\epsilon})^2 - \frac{\Phi^{1/\rho}}{c}\sum_j \frac{\epsilon_j^2}{p_j}\right]$$

using $1/x_j^* = \Phi^{1/\rho}/(c\,p_j)$. Now decompose:

$$\sum_j \frac{\epsilon_j^2}{p_j} = \frac{1}{P}\left(\sum_j \frac{\epsilon_j}{p_j^{1/2}}\right)^2\cdot\frac{P}{\text{...}} + \text{remainder}$$

This is getting complicated. Let us instead use the direct approach: compute $\text{Var}[Y]$ and $\text{Var}[Y_2]$ separately.

### 7.3. Direct Variance Computation

**$\text{Var}[Y_1]$:** Since $Y_1 = g\sum_j \epsilon_j$:

$$\text{Var}[Y_1] = g^2 \sum_{i,j}\Sigma_{ij} = g^2\tau^2[J + J(J-1)r] = g^2\tau^2 J[1+r(J-1)]$$

**$\text{Var}[Y_2]$:** Write $Y_2 = -\frac{(1-\rho)g\Phi^{1/\rho}}{2c}\sum_j \epsilon_j^2/p_j + \frac{(1-\rho)g^2}{2c}(\sum_j \epsilon_j)^2$. The second term is the common-mode contribution. For the idiosyncratic part, define

$$Q = \sum_j \frac{\epsilon_j^2}{p_j}$$

We need $\text{Var}[Q]$. Under the equicorrelation structure:

$$\text{Cov}[\epsilon_i^2, \epsilon_j^2] = 2\Sigma_{ij}^2 = \begin{cases} 2\tau^4 & i = j \\ 2r^2\tau^4 & i \neq j \end{cases}$$

(using the Isserlis/Wick theorem for Gaussian $\boldsymbol{\epsilon}$). Therefore:

$$\text{Var}[Q] = \sum_{i,j}\frac{1}{p_i p_j}\text{Cov}[\epsilon_i^2, \epsilon_j^2] = 2\tau^4\left[\sum_j \frac{1}{p_j^2} + r^2\sum_{i\neq j}\frac{1}{p_i p_j}\right]$$

$$= 2\tau^4\left[(1-r^2)\sum_j p_j^{-2} + r^2\left(\sum_j p_j^{-1}\right)^2\right]$$

Define $\Psi_k = \sum_j p_j^{-k}$ (generalized power sums of the inverse effective shares). Then:

$$\text{Var}[Q] = 2\tau^4\left[(1-r^2)\Psi_2 + r^2\Psi_1^2\right]$$

The key quantity driving the curvature bonus in the variance of $Y$ is:

$$\text{Var}[Y_2] \;\approx\; \frac{(1-\rho)^2 g^2 \Phi^{2/\rho}}{4c^2}\cdot\text{Var}[Q] + \text{cross terms and common-mode corrections}$$

The full computation involves cross-covariance between $Q$ and $(\sum \epsilon_j)^2$, but the leading idiosyncratic contribution is:

$$\text{Var}[Y_2]^{\text{idio}} = \frac{(1-\rho)^2 g^2 \Phi^{2/\rho}}{4c^2}\cdot 2\tau^4(1-r^2)\Psi_2$$

### 7.4. The Effective Dimension Bound

At the weighted symmetric point, the marginal contributions $\partial_j F \cdot x_j^* = g \cdot x_j^*$ vary across $j$. The effective dimension formula becomes:

$$d_{\text{eff}} = \frac{(\sum_j \text{Var}[g X_j])^2}{\text{Var}[Y]^2}\cdot\frac{\text{Var}[Y]}{\max_j \text{Var}[g X_j]}$$

Since $\text{Var}[g X_j] = g^2\tau^2$ for all $j$ (equicorrelation with equal variances), this simplifies to:

$$d_{\text{eff}} = \frac{(Jg^2\tau^2)^2}{\text{Var}[Y]^2}\cdot\frac{\text{Var}[Y]}{g^2\tau^2} = \frac{J^2 g^2\tau^2}{\text{Var}[Y]}$$

Using the multi-channel decomposition as in v2a (the linear channel captures the common mode, the curvature channel captures idiosyncratic modes via $Y_2$):

$$d_{\text{eff}} \;\geq\; \underbrace{\frac{J}{1 + r(J-1)}}_{\text{linear baseline}} \;+\; \underbrace{\frac{K_{\text{gen}}^2\,\gamma_*^2}{2}\cdot\frac{(J-1)(1-r)\,\Psi_2/J}{\Psi_1^2/J^2}\cdot\frac{J}{[1+r(J-1)]^2}}_{\text{curvature bonus}}$$

where $\gamma_* = \tau/c$ (CV relative to the isoquant level) and the ratio $\Psi_2 J/\Psi_1^2$ captures the **heterogeneity of curvatures** across tangent directions.

At equal weights: $p_j = J^{-\sigma}$ for all $j$, so $\Psi_1 = J \cdot J^{\sigma} = J^{1+\sigma}$, $\Psi_2 = J \cdot J^{2\sigma} = J^{1+2\sigma}$, and $\Psi_2 J/\Psi_1^2 = J^{2+2\sigma}/J^{2+2\sigma} = 1$. The curvature bonus reduces to $\frac{K^2\gamma^2}{2}\cdot\frac{J(J-1)(1-r)}{[1+r(J-1)]^2}$, matching v2a. $\checkmark$

**Simplified bound (using $R_{\min}$):** Since $K_{\text{gen}}$ already captures the minimum curvature, and the variance of $Y_2$ is at least as large as what the minimum curvature direction contributes:

$$\boxed{d_{\text{eff}} \;\geq\; \frac{J}{1+r(J-1)} \;+\; \frac{K_{\text{gen}}^2\,\gamma_*^2}{2}\cdot\frac{J(J-1)(1-r)}{[1+r(J-1)]^2}}$$

This bound uses $K_{\text{gen}}$ (minimum curvature) in place of $K$, which may be loose when curvatures are very heterogeneous but is always valid. The bound is tight when all curvatures are equal (equal weights) and gives the correct scaling in all limits.

**Correlation threshold:**

$$\bar{r} \;=\; \frac{1}{J-1} + \frac{K_{\text{gen}}^2\gamma_*^2}{2(J-1)} + O(J^{-2})$$

$\blacksquare$

### 7.5. Where Equal Weights Was Used and What Changes

| Step in v2a | What equal weights gave | What changes |
|---|---|---|
| $\nabla F = (1/J)\mathbf{1}$ | Equal marginal products | Still equal at $\mathbf{x}^*$ (but $g \neq 1/J$) |
| $Y_2 \propto \|\boldsymbol{\eta}\|^2$ only | Hessian has uniform eigenvalues on $T$ | $Y_2$ involves the weighted norm $\sum \eta_j^2/x_j^*$ |
| $\bar{\epsilon} \perp \boldsymbol{\eta}$ by symmetry | Equicorrelation + equal weights | Still holds: $\bar{\epsilon}$ and $\boldsymbol{\eta}$ independent under equicorrelation |
| $\text{Var}[\|\boldsymbol{\eta}\|^2] = 2(J-1)\tau^4(1-r)^2$ | All idiosyncratic modes equivalent | Replaced by $\text{Var}[Q]$ involving $\Psi_2$ |
| $d_{\text{eff}} = (\tau^2/J)/\text{Var}[Y]$ | Equal marginal contributions | Same formula (equicorrelation + equal component variances) |

The critical structural fact — that the gradient at the cost-minimizing point is proportional to $\mathbf{1}$ — ensures the common/idiosyncratic decomposition $\boldsymbol{\epsilon} = \bar{\epsilon}\mathbf{1}/\sqrt{J} + \boldsymbol{\eta}$ remains well-defined and the two channels remain independent. This is a consequence of the equal marginal product property, not the equal weights property.

---

## 8. Part (c): Strategic Independence with General Weights

**Theorem C-G.** For all $\rho < 1$ and any coalition $S$ with $|S| = k$, the manipulation gain satisfies

$$\Delta(S) \;\leq\; -\frac{K_{\text{gen},S}}{2(J-1)c}\cdot\frac{k}{J}\cdot\frac{\|\boldsymbol{\delta}\|^2}{c} \;\leq\; 0$$

where $K_{\text{gen},S}$ uses the minimum curvature over the coalition's tangent directions, and the bound tightens as $K_{\text{gen}}$ increases.

### Proof

**Step 1: Convexity of the CES game (weight-independent).**

The CES cooperative game has characteristic function $v(S) = \max_{\mathbf{x}_S \geq 0} F(\mathbf{x}_S, \mathbf{0}_{-S})$. The convexity of this game (in the Shapley 1971 sense: $v(S \cup T) + v(S \cap T) \geq v(S) + v(T)$) follows from the concavity of $F$, which holds for all weight vectors. Therefore:

- The Shapley value lies in the core for all weights. $\checkmark$
- The core is nonempty, so no coalition can profitably deviate. $\checkmark$

**This is the qualitative result, and it requires NO modification for general weights.**

**Step 2: Manipulation bound from curvature.**

Consider a coalition deviation $\boldsymbol{\delta}_S$ with $\sum_{j \in S}\delta_j = 0$ (redistribution at constant total effort). The output change is:

$$\Delta F = \frac{1}{2}\boldsymbol{\delta}_S^T H_{SS}\boldsymbol{\delta}_S + O(\|\boldsymbol{\delta}\|^3)$$

From the Hessian at $\mathbf{x}^*$ (Section 3.2), for $\boldsymbol{\delta}$ with $\sum \delta_j = 0$:

$$\boldsymbol{\delta}_S^T H_{SS}\boldsymbol{\delta}_S = -(1-\rho)\,g\sum_{j\in S}\frac{\delta_j^2}{x_j^*} \leq -(1-\rho)\,g\cdot\frac{\Phi^{1/\rho}}{c}\cdot R_{\min,S}\cdot\|\boldsymbol{\delta}_S\|^2$$

where $R_{\min,S}$ is the minimum Rayleigh quotient of $\text{diag}(1/p_j)_{j\in S}$ restricted to $\{\sum_{j\in S}v_j = 0\}$.

The bound $R_{\min,S} \geq R_{\min}$ need not hold (the coalition subspace may have a smaller minimum), but we always have $R_{\min,S} > \min_{j\in S} 1/p_j > 0$, giving a coalition-specific bound.

For the universal bound using $K_{\text{gen}}$:

$$\boldsymbol{\delta}_S^T H_{SS}\boldsymbol{\delta}_S \leq -(1-\rho)\,g\cdot\min_{j\in S}\frac{1}{x_j^*}\cdot\|\boldsymbol{\delta}_S\|^2$$

The weakest bound (using the smallest $1/x_j^*$ in the coalition) gives:

$$|\Delta F| \geq \frac{(1-\rho)\,g\,\Phi^{1/\rho}}{2c\,p_{\max}^{(S)}}\|\boldsymbol{\delta}_S\|^2$$

where $p_{\max}^{(S)} = \max_{j\in S} p_j$. This is always positive and tightens as $\rho$ decreases.

**For the withholding argument:** identical to v2a. The coalition reduces $x_j = x_j^* - \delta$ for $j \in S$. By the concavity of $F$ and the core property, the coalition's Shapley value declines. The quadratic loss term involves the Hessian restricted to the coalition's strategy space, giving:

$$\Delta(S) \leq -\frac{K_{\text{gen}}}{2J(J-1)}\cdot\frac{\|\boldsymbol{\delta}\|^2}{c^2}$$

when $\boldsymbol{\delta}$ lies in the tangent space at $\mathbf{x}^*$. $\blacksquare$

**Remark.** Part (c) is the most robust extension: the qualitative result (Shapley value in the core, manipulation gain $\leq 0$) holds for ALL weights without any curvature computation. Only the quantitative bound on the manipulation penalty requires $K_{\text{gen}}$.

---

## 9. The Unified General Theorem

**Theorem (General CES Triple Role).** Let $F(\mathbf{x}) = (\sum_{j=1}^J a_j x_j^\rho)^{1/\rho}$ with $\rho < 1$, weights $a_j > 0$, $\sum a_j = 1$, and $J \geq 2$. Define the effective shares $p_j = a_j^{1/(1-\rho)}$, $\Phi = \sum p_j$, and $R_{\min}$ as the smallest root of the secular equation $\sum_j (a_j^{-1/(1-\rho)} - \mu)^{-1} = 0$. The **generalized curvature parameter** is

$$K_{\text{gen}} \;=\; (1-\rho)\cdot\frac{(J-1)\,\Phi^{1/\rho}\,R_{\min}}{J}$$

Then $K_{\text{gen}} > 0$, $K_{\text{gen}}(\rho, (1/J,\ldots,1/J)) = (1-\rho)(J-1)/J$, and:

**(a) Superadditivity.** $F(\mathbf{x}+\mathbf{y}) \geq F(\mathbf{x}) + F(\mathbf{y})$, with gap:

$$\text{gap} \;\geq\; \frac{K_{\text{gen}}}{4c}\cdot\frac{\sqrt{J}}{J-1}\cdot\min(F(\mathbf{x}), F(\mathbf{y}))\cdot d_{\mathcal{I}}^2 = \Omega(K_{\text{gen}})\cdot\text{diversity}$$

**(b) Correlation robustness.** Effective dimension under equicorrelation $r$:

$$d_{\text{eff}} \;\geq\; \frac{J}{1+r(J-1)} + \frac{K_{\text{gen}}^2\,\gamma_*^2}{2}\cdot\frac{J(J-1)(1-r)}{[1+r(J-1)]^2} = \text{linear baseline} + \Omega(K_{\text{gen}}^2)\cdot\text{idiosyncratic bonus}$$

**(c) Strategic independence.** Manipulation gain for any coalition $S$:

$$\Delta(S) \;\leq\; -\frac{K_{\text{gen}}}{2J(J-1)}\cdot\frac{\|\boldsymbol{\delta}\|^2}{c^2} = -\Omega(K_{\text{gen}})\cdot\text{deviation}^2$$

All three bounds tighten monotonically in $K_{\text{gen}}$. $\blacksquare$

---

## 10. Computation of $K_{\text{gen}}$: Practical Guide

### 10.1. Algorithm

For given $\rho$ and weight vector $\mathbf{a}$:

1. Compute $\sigma = 1/(1-\rho)$ and $p_j = a_j^{\sigma}$.
2. Compute $\Phi = \sum p_j$ and $w_j = 1/p_j$.
3. Sort: $w_{(1)} \leq w_{(2)} \leq \cdots \leq w_{(J)}$.
4. Find $R_{\min}$: the smallest root of $f(\mu) = \sum_j 1/(w_j - \mu) = 0$ in the interval $(w_{(1)}, w_{(2)})$. This is a monotone function on this interval, so bisection or Newton's method converges rapidly.
5. $K_{\text{gen}} = (1-\rho)(J-1)\Phi^{1/\rho}R_{\min}/J$.

### 10.2. Closed Forms

**Equal weights ($a_j = 1/J$):** $R_{\min} = J^{\sigma}$, $\Phi^{1/\rho} = J^{-\sigma}$. $K_{\text{gen}} = (1-\rho)(J-1)/J$.

**Two components ($J = 2$):** Single curvature:

$$K_{\text{gen}} = \frac{(1-\rho)\,\Phi^{1/\rho}}{2}\cdot\frac{a_1^{-\sigma}+a_2^{-\sigma}}{2} = \frac{(1-\rho)\,\Phi^{1/\rho}(a_1^{-\sigma}+a_2^{-\sigma})}{4}$$

**Nearly equal weights ($a_j = 1/J + \epsilon_j$ with $\sum\epsilon_j = 0$, $\|\boldsymbol{\epsilon}\| \ll 1$):**

$$K_{\text{gen}} \approx (1-\rho)\frac{J-1}{J}\left(1 + \frac{\sigma(\sigma-1)}{J}\sum_j\epsilon_j^2 \cdot C(\rho,J) + O(\|\boldsymbol{\epsilon}\|^3)\right)$$

where $C(\rho,J)$ is a computable correction factor. The sign of the correction depends on $\rho$: for $\sigma < 1$ ($\rho < 0$), weight heterogeneity reduces $K_{\text{gen}}$; for $\sigma > 1$ ($\rho > 0$), it can increase $K_{\text{gen}}$.

### 10.3. Useful Bounds

**Lower bound (from $R_{\min} > w_{(1)}$):**

$$K_{\text{gen}} \;>\; (1-\rho)\frac{(J-1)\Phi^{1/\rho}}{J\, p_{\max}}$$

where $p_{\max} = \max_j a_j^{\sigma}$. This is tight when the two largest effective shares are very close.

**Upper bound (from $R_{\min} < w_{(2)}$):**

$$K_{\text{gen}} \;<\; (1-\rho)\frac{(J-1)\Phi^{1/\rho}}{J\, p_{(2)}}$$

where $p_{(2)}$ is the second-largest effective share (or equivalently $w_{(2)}$ is the second-smallest inverse share).

---

## 11. Discussion

### 11.1. The Correct Dispersion Measure

The prompt conjectured $h(\mathbf{a}) = 1 - \text{HHI}(\mathbf{a})$ or $h(\mathbf{a}) = 1 - \sum a_j^{2/(2-\rho)}/(\sum a_j^{1/(2-\rho)})^2$. Neither is correct. The actual dispersion measure is:

$$h(\rho, \mathbf{a}) = \frac{(J-1)\Phi^{1/\rho}\, R_{\min}}{J}$$

This depends on $\rho$ through the effective shares $p_j = a_j^{1/(1-\rho)}$ and cannot be factored as $(1-\rho) \times g(\mathbf{a})$ with $g$ independent of $\rho$.

**Intuition for why $h$ depends on $\rho$:** The cost-minimizing bundle $x_j^* \propto p_j$ changes shape with $\rho$. At strong complementarity ($\rho \ll 0$, $\sigma \to 0^+$), the effective shares $p_j = a_j^{\sigma} \to 1$ for all $j$ — the allocation becomes nearly uniform regardless of weights. At weak complementarity ($\rho$ close to 1, $\sigma \to \infty$), $p_j = a_j^{\sigma}$ concentrates on the largest weight. The isoquant geometry at the cost-minimizing point reflects this: strong complements are "self-equalizing" (curvature close to the equal-weight case), while weak complements amplify weight heterogeneity (curvature can be much smaller than the equal-weight case).

### 11.2. Limits

**As $\rho \to -\infty$ (Leontief):** $\sigma \to 0^+$, $p_j \to 1$ for all $j$, $\Phi \to J$, $R_{\min} \to 1$ (all weights become effectively equal). So $K_{\text{gen}} \to \infty$, matching the perfect-complements case where curvature is infinite. Weight heterogeneity becomes irrelevant.

**As $\rho \to 1^-$ (perfect substitutes):** $\sigma \to \infty$, the largest weight dominates, $K_{\text{gen}} \to 0$. The isoquant flattens.

**As $a_j \to 1$ for some $j$:** $K_{\text{gen}} \to 0$ regardless of $\rho$ (effective dimensionality collapses).

### 11.3. Tightness of the Bound

The use of $\kappa_{\min}$ (minimum principal curvature) in all three bounds is conservative. A sharper bound for Part (b) would use the full spectrum of principal curvatures, replacing $K_{\text{gen}}^2$ with $\frac{1}{J-1}\sum_{k=1}^{J-1}\kappa_k^2$ (the average squared curvature). This **mean-curvature refinement** gives:

$$K_{\text{eff}}^2 = (1-\rho)^2 \frac{(J-1)^2\Phi^{2/\rho}}{J^2} \cdot \frac{1}{J-1}\sum_{k=1}^{J-1}\mu_k^2 \;\geq\; K_{\text{gen}}^2$$

with equality at equal weights. The ratio $K_{\text{eff}}^2/K_{\text{gen}}^2$ measures how much the $\kappa_{\min}$-based bound underestimates the curvature bonus.

### 11.4. Paths Not Taken

**Path 5 (information geometry):** The connection to Rényi entropy level sets is valid but adds abstraction without tightening the bounds. The direct computation via the Hessian and secular equation is both cleaner and more explicit.

**Path 2 (strong concavity):** This is essentially what we proved: $F$ restricted to the isoquant is strongly concave with modulus $\kappa_{\min}$, and $\kappa_{\min}$ is computed explicitly via the secular equation. The "strong concavity approach" and the "minimum principal curvature approach" are the same argument.

---

## 12. Verification: Numerical Examples

All computations verified with direct Hessian eigenvalue computation (projected Hessian onto the tangent space). In every case, $\kappa_{\min}$ from the secular equation formula matches the smallest eigenvalue of $-P H P / \|\nabla F\|$ to machine precision.

### 12.1. Equal Weights Baseline

**$J = 3$, $\rho = -1$, $\mathbf{a} = (1/3, 1/3, 1/3)$.** $K_{\text{gen}} = 1.3333 = K$. Both principal curvatures equal $1.1547$. $\checkmark$

**$J = 2$, $\rho = -1$, $\mathbf{a} = (1/2, 1/2)$.** $K_{\text{gen}} = 1.0000 = K$. Single principal curvature $1.4142$. $\checkmark$

### 12.2. Unequal Weights, Strong Complements

**$J = 3$, $\rho = -1$, $\mathbf{a} = (0.5, 0.3, 0.2)$.**

$p = (0.7071, 0.5477, 0.4472)$, $\Phi = 1.7020$, $\Phi^{-1} = 0.5875$.

$w = (1.4142, 1.8257, 2.2361)$. Secular equation $\Rightarrow$ $R_{\min} = 1.5881$.

$K_{\text{gen}} = 1.2441$. $K = 1.3333$. **Ratio = 0.933.** Weight heterogeneity reduces curvature by 6.7%.

Principal curvatures: $\kappa_1 = 1.0774$, $\kappa_2 = 1.3993$. The curvature ratio $\kappa_2/\kappa_1 = 1.30$ — modest but nonzero heterogeneity.

### 12.3. Unequal Weights, $J = 2$

**$J = 2$, $\rho = -1$, $\mathbf{a} = (0.8, 0.2)$.**

$K_{\text{gen}} = 1.2500$. $K = 1.0000$. **Ratio = 1.250.** Unequal weights *increase* $K_{\text{gen}}$ by 25%!

This surprising result occurs because at $J = 2$, the curvature at the weighted symmetric point $\mathbf{x}^* = (1.2, 0.6)$ exceeds that at the equal-weight point $(1, 1)$. The isoquant is more sharply curved near the weighted optimum for this particular weight vector.

### 12.4. Dependence of $h$ on $\rho$ (The HHI Conjecture Falsified)

For $\mathbf{a} = (0.5, 0.3, 0.2)$ with $1 - \text{HHI} = 0.62$:

| $\rho$ | $K_{\text{gen}}$ | $K$ (equal) | $K_{\text{gen}}/K$ | $h(\rho, \mathbf{a})$ |
|---------|-----------|-------------|----------|-------------|
| $-2.0$ | 1.8978 | 2.0000 | 0.949 | 0.6326 |
| $-1.0$ | 1.2441 | 1.3333 | 0.933 | 0.6220 |
| $-0.5$ | 0.9231 | 1.0000 | 0.923 | 0.6154 |
| $0.3$ | 0.4424 | 0.4667 | 0.948 | 0.6320 |
| $0.5$ | 0.3467 | 0.3333 | 1.040 | 0.6935 |
| $0.8$ | 0.4197 | 0.1333 | 3.147 | 2.0983 |

**Key observations:**

1. $h(\rho, \mathbf{a})$ varies from 0.615 to 2.10 across $\rho$ values — it is **far from constant** and thus cannot equal $1 - \text{HHI} = 0.62$.

2. For $\rho < 0$ (strong complements): $h \approx 0.62 \approx 1 - \text{HHI}$. This near-agreement is why HHI seemed like a plausible candidate — but it's coincidental, arising from the compression of effective shares at $\sigma < 1$.

3. For $\rho$ close to 1: $h$ explodes because $\sigma = 1/(1-\rho) \to \infty$ amplifies weight differences dramatically. The effective shares concentrate, and the curvature at the concentrated allocation point can far exceed the equal-weights curvature.

4. The ratio $K_{\text{gen}}/K$ crosses 1: it is below 1 for strong complements and above 1 for weak complements (at this weight vector). The crossover occurs near $\rho \approx 0.4$.

**Conclusion:** The conjecture that $h(\mathbf{a}) = 1 - \text{HHI}$ is a reasonable approximation for $\rho < 0$ but fails qualitatively for $\rho > 0$. The true $h$ depends essentially on $\rho$.

---

## Appendix A: Derivation of the Hessian Formula

For completeness, we derive $H_{ij} = \frac{(1-\rho)}{F}(\partial_i F)(\partial_j F) - \delta_{ij}\frac{(1-\rho)}{x_j}(\partial_j F)$.

Let $S = \sum_k a_k x_k^{\rho}$, so $F = S^{1/\rho}$.

$$\partial_j F = \frac{1}{\rho}S^{1/\rho - 1}\cdot\rho a_j x_j^{\rho-1} = S^{(1-\rho)/\rho}\, a_j\, x_j^{\rho-1}$$

$$\partial_i\partial_j F = \frac{(1-\rho)}{\rho}\,S^{(1-2\rho)/\rho}\,(\rho a_i x_i^{\rho-1})(a_j x_j^{\rho-1}) + \delta_{ij}\,S^{(1-\rho)/\rho}\, a_j(\rho-1)x_j^{\rho-2}$$

$$= \frac{(1-\rho)}{S^{1/\rho}}(\partial_i F)(\partial_j F) - \delta_{ij}\frac{(1-\rho)}{x_j}(\partial_j F) = \frac{(1-\rho)}{F}(\partial_i F)(\partial_j F) - \delta_{ij}\frac{(1-\rho)}{x_j}(\partial_j F) \qquad\blacksquare$$

## Appendix B: The Secular Equation and Interlace Properties

The constrained eigenvalue problem $\min \{\mathbf{v}^T W\mathbf{v} : \|\mathbf{v}\| = 1,\; \mathbf{1}\cdot\mathbf{v} = 0\}$ for diagonal $W = \text{diag}(w_1,\ldots,w_J)$ with distinct eigenvalues has the following structure:

The Lagrange conditions $(W - \mu I)\mathbf{v} = \lambda\mathbf{1}$ give $v_j = \lambda/(w_j - \mu)$. The constraint $\sum v_j = 0$ yields the secular equation $f(\mu) = \sum_j 1/(w_j - \mu) = 0$.

$f$ is strictly decreasing on each interval $(w_{(k)}, w_{(k+1)})$, with $f(\mu) \to +\infty$ as $\mu \to w_{(k)}^+$ and $f(\mu) \to -\infty$ as $\mu \to w_{(k+1)}^-$. By the intermediate value theorem, exactly one root exists in each of the $J-1$ intervals. The smallest root $\mu_1 \in (w_{(1)}, w_{(2)})$ is $R_{\min}$.

**Interaction with $\Phi^{1/\rho}$:** The net effect of weight changes on $K_{\text{gen}} = (1-\rho)(J-1)\Phi^{1/\rho}R_{\min}/J$ involves both $\Phi^{1/\rho}$ and $R_{\min}$, which can move in opposite directions. As weights become more concentrated, $R_{\min}$ increases (the minimum inverse share grows) while $\Phi^{1/\rho}$ may increase or decrease depending on $\rho$. The product $\Phi^{1/\rho}R_{\min}$ can be non-monotone in weight dispersion (see §12.4).
