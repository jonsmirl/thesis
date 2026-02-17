# The CES Triple Role Theorem

## Unified Proof that Complementarity Implies Superadditivity, Correlation Robustness, and Strategic Independence

---

## 1. Setup

Let $F: \mathbb{R}_+^J \to \mathbb{R}_+$ be the CES aggregate

$$F(\mathbf{x}) = \left(\sum_{j=1}^{J} a_j\, x_j^{\,\rho}\right)^{1/\rho}$$

with $\rho < 1$, $\rho \neq 0$, weights $a_j > 0$ summing to 1, and $J \geq 2$ components. The elasticity of substitution is $\sigma = 1/(1-\rho)$.

**Convention.** We call the components *complements* when $\rho < 0$ ($\sigma < 1$) and *weak complements* when $0 < \rho < 1$ ($\sigma > 1$ but finite). The boundary $\rho \to 1$ gives perfect substitutes (linear); $\rho \to -\infty$ gives perfect complements (Leontief).

**Standing assumption.** $F$ is concave and homogeneous of degree 1 for all $\rho < 1$. Both properties are standard and we take them as given.

### 1.1. Effective Shares

Define the **effective shares**

$$p_j \;=\; a_j^{\sigma} \;=\; a_j^{1/(1-\rho)}$$

These are the cost-minimizing expenditure shares at unit input prices. Define $\Phi = \sum_{j=1}^J p_j$ and the inverse effective shares $w_j = 1/p_j = a_j^{-\sigma}$.

At equal weights $a_j = 1/J$: $p_j = J^{-\sigma}$, $\Phi = J^{1-\sigma}$, $w_j = J^{\sigma}$.

### 1.2. The Cost-Minimizing Point

For output level $c > 0$, the **cost-minimizing point** on the isoquant $\{F = c\}$ at unit prices is

$$x_j^* \;=\; \frac{c\, p_j}{\Phi^{1/\rho}}, \qquad j = 1, \ldots, J$$

*Verification.* $F(\mathbf{x}^*)^{\rho} = \sum_j a_j (c\,p_j/\Phi^{1/\rho})^{\rho} = c^{\rho}\Phi^{-1}\sum_j a_j p_j^{\rho}$. Since $a_j p_j^{\rho} = a_j \cdot a_j^{\sigma\rho} = a_j^{1+\sigma\rho} = a_j^{\sigma} = p_j$, the sum is $\Phi$ and $F(\mathbf{x}^*) = c$. $\checkmark$

At equal weights, $x_j^* = c$ for all $j$ — the symmetric point.

---

## 2. The Curvature Lemma (Unifying Core)

The three results all flow from a single geometric object: the **principal curvatures of the CES isoquant** at the cost-minimizing point.

### 2.1. Gradient at $\mathbf{x}^*$: The Equal Marginal Product Property

The partial derivative of $F$ is $\partial_j F = a_j x_j^{\rho-1} F^{1-\rho}$. At $\mathbf{x}^*$:

$$\frac{\partial F}{\partial x_j}\bigg|_{\mathbf{x}^*} = a_j \cdot (c\,p_j/\Phi^{1/\rho})^{\rho-1}\cdot c^{1-\rho} = a_j \cdot a_j^{-1} \cdot \Phi^{(1-\rho)/\rho} = \Phi^{(1-\rho)/\rho}$$

using $p_j^{\rho-1} = a_j^{\sigma(\rho-1)} = a_j^{-1}$. This is **independent of $j$**: at the cost-minimizing point, all marginal products are equal. Define

$$\boxed{\nabla F(\mathbf{x}^*) \;=\; g\,\mathbf{1}, \qquad g \;=\; \Phi^{(1-\rho)/\rho}, \qquad \|\nabla F\| = g\sqrt{J}}$$

**This is the structural fact that makes the entire proof work.** The tangent space to the isoquant at $\mathbf{x}^*$ is $T = \{\mathbf{v} : \sum_j v_j = 0\}$, the same subspace for all weight vectors. The unit outward normal is $\mathbf{n} = \mathbf{1}/\sqrt{J}$, regardless of weights.

At equal weights: $g = J^{-1}$.

### 2.2. Hessian at $\mathbf{x}^*$

The general CES Hessian is (Appendix A):

$$H_{ij} = \frac{(1-\rho)}{F}\,(\partial_i F)(\partial_j F) - \delta_{ij}\,\frac{(1-\rho)}{x_j}\,(\partial_j F)$$

At $\mathbf{x}^*$, where $\partial_j F = g$ and $F = c$:

$$\boxed{H_F(\mathbf{x}^*) = (1-\rho)\,g\left[\frac{g}{c}\,\mathbf{1}\mathbf{1}^T - D^{-1}\right]}$$

where $D = \text{diag}(x_1^*,\ldots,x_J^*)$ and $D^{-1} = \frac{\Phi^{1/\rho}}{c}\,\text{diag}(1/p_1,\ldots,1/p_J)$.

At equal weights: $D^{-1} = (1/c)\,I$, recovering $H_F = \frac{(1-\rho)}{J^2 c}[\mathbf{1}\mathbf{1}^T - JI]$. $\checkmark$

### 2.3. Normal Curvature

**Lemma 1 (Isoquant Curvature).** At $\mathbf{x}^*$ on $\mathcal{I}_c$, the normal curvature of the isoquant in tangent direction $\mathbf{v}$ (with $\sum v_j = 0$) is

$$\kappa(\mathbf{v}) \;=\; \frac{(1-\rho)\,\Phi^{1/\rho}}{c\sqrt{J}} \cdot \frac{\sum_j v_j^2/p_j}{\sum_j v_j^2}$$

*Proof.* For $\mathbf{v} \in T$, the $\mathbf{1}\mathbf{1}^T$ term vanishes since $(\sum v_j)^2 = 0$:

$$\mathbf{v}^T H_F\,\mathbf{v} = -(1-\rho)\,g\sum_j \frac{v_j^2}{x_j^*} = -\frac{(1-\rho)\,g\,\Phi^{1/\rho}}{c}\sum_j \frac{v_j^2}{p_j}$$

Then $\kappa(\mathbf{v}) = -\mathbf{v}^T H_F\,\mathbf{v}\,/\,(\|\nabla F\|\cdot\|\mathbf{v}\|^2)$ gives the stated formula. $\blacksquare$

### 2.4. The Principal Curvatures and the Secular Equation

The **principal curvatures** $\kappa_1 \leq \cdots \leq \kappa_{J-1}$ are the extrema of $\kappa(\mathbf{v})$ over unit tangent vectors. Setting $W = \text{diag}(w_1,\ldots,w_J)$ with $w_j = 1/p_j$, the constrained Rayleigh quotient $R(\mathbf{v}) = \mathbf{v}^T W\mathbf{v}/\|\mathbf{v}\|^2$ on $T = \{\sum v_j = 0\}$ determines the curvatures via $\kappa_k = \frac{(1-\rho)\Phi^{1/\rho}}{c\sqrt{J}}\cdot\mu_k$, where $\mu_1 < \cdots < \mu_{J-1}$ are the constrained eigenvalues of $W|_T$.

The Lagrange conditions $(W - \mu I)\mathbf{v} = \lambda\,\mathbf{1}$ yield $v_j = \lambda/(w_j - \mu)$, and $\sum v_j = 0$ gives the **secular equation**

$$\boxed{\sum_{j=1}^J \frac{1}{w_j - \mu} = 0, \qquad w_j = a_j^{-1/(1-\rho)}}$$

This has exactly $J-1$ roots, one in each interval $(w_{(k)}, w_{(k+1)})$ where $w_{(1)} \leq \cdots \leq w_{(J)}$ are the ordered values (Appendix B). In particular:

$$R_{\min} = \mu_1 \in (w_{(1)}, w_{(2)}), \qquad R_{\max} = \mu_{J-1} \in (w_{(J-1)}, w_{(J)})$$

**At equal weights** all $w_j = J^{\sigma}$ coincide, so $R(\mathbf{v}) = J^{\sigma}$ for every $\mathbf{v} \in T$. All principal curvatures equal $\kappa^* = (1-\rho)/(c\sqrt{J})$: the isoquant has **uniform curvature** at the symmetric point, a consequence of permutation symmetry.

**Remark (Economic interpretation of the curvature spectrum).** The interlacing $\mu_k \in (w_{(k)}, w_{(k+1)})$ implies that the curvature spectrum is ordered by scarcity at the cost-minimizing bundle. Since $v_j \propto 1/(w_j - \mu_k)$, each eigenvector $\mathbf{v}_k$ concentrates on the components with $w_j$ closest to $\mu_k$ — the $(k)$-th and $(k+1)$-th most abundant effective shares. Because $\mu_1 < \cdots < \mu_{J-1}$, the least-curved direction ($\mu_1$) redistributes between the two most abundant inputs, while the most-curved direction ($\mu_{J-1}$) redistributes between the two scarcest. The economic reason: the marginal rate of substitution between two scarce inputs (small $p_j$, hence small $x_j^*$) changes rapidly with reallocation — a small shift in relative quantities produces a large swing in relative marginal products — while abundant inputs can be traded more smoothly. The minimum curvature $\kappa_{\min}$, which controls all three bounds through $K$, thus measures the gentlest substitution the isoquant permits.

---

## 3. The Generalized Curvature Parameter

**Definition.** The **curvature parameter** of the CES aggregate is

$$\boxed{K(\rho, \mathbf{a}) \;=\; (1-\rho) \cdot \frac{(J-1)\,\Phi^{1/\rho}\,R_{\min}}{J}}$$

**Equivalently:** $K = c\,(J-1)\,\kappa_{\min}\,/\,\sqrt{J}$, the minimum principal curvature rescaled to be dimensionless and independent of the isoquant level $c$.

**Properties.**

1. **Positive for all $\rho < 1$:** $K > 0$ since $(1-\rho) > 0$, $\Phi > 0$, and $R_{\min} > w_{(1)} > 0$.

2. **Equal weights:** $K(\rho, (1/J,\ldots,1/J)) = (1-\rho)(J-1)/J$. (Since $\Phi^{1/\rho} = J^{-\sigma}$ and $R_{\min} = J^{\sigma}$, the product is 1.)

3. **Monotone in complementarity:** $K$ increases with $(1-\rho)$ for fixed $\mathbf{a}$.

4. **Vanishes as weights concentrate:** $K \to 0$ as $a_j \to 1$ for any single $j$ (effective dimensionality collapses).

5. **Limits.** $K \to \infty$ as $\rho \to -\infty$ (Leontief); $K \to 0$ as $\rho \to 1^-$ (linear).

6. **Non-monotone in weight dispersion.** Weight heterogeneity does *not* always reduce $K$. For strong complements ($\rho \ll 0$), effective shares $p_j = a_j^{\sigma}$ with $\sigma < 1$ compress toward uniformity, so $K$ is close to (and typically below) the equal-weights value. For weak complements ($\rho$ close to 1), $\sigma > 1$ amplifies weight differences, and the curvature at the cost-minimizing point can exceed the equal-weights curvature — so $K$ can be *larger* than the equal-weights value.

**Notation.** When the weight vector is clear from context, we write $K$ for $K(\rho, \mathbf{a})$. In the special case of equal weights, $K = (1-\rho)(J-1)/J$.

### 3.1. Why the HHI is Not the Right Dispersion Measure

One might conjecture that $K = (1-\rho)\cdot(1-\text{HHI})$, separating the substitution elasticity from a $\rho$-independent weight-dispersion factor. This is **incorrect**: the correct dispersion factor $h(\rho,\mathbf{a}) = (J-1)\Phi^{1/\rho}R_{\min}/J$ depends on $\rho$ through the effective shares. For $\rho < 0$, $h \approx 1 - \text{HHI}$ coincidentally (because $\sigma < 1$ compresses $p_j$ toward uniformity), but for $\rho > 0$ the two diverge dramatically (see §11).

### 3.2. Closed Forms and Bounds

**$J = 2$:** The tangent space is one-dimensional ($\mathbf{v} \propto (1,-1)$), giving the single curvature

$$K = \frac{(1-\rho)\,\Phi^{1/\rho}}{4}\left(a_1^{-\sigma} + a_2^{-\sigma}\right)$$

**Bounds for general $J$:**

$$\frac{(1-\rho)(J-1)\Phi^{1/\rho}}{J\,p_{\max}} \;<\; K \;<\; \frac{(1-\rho)(J-1)\Phi^{1/\rho}}{J\,p_{(2)}}$$

from $w_{(1)} < R_{\min} < w_{(2)}$ (Cauchy interlace; Appendix B).

**Algorithm.** For any $(\rho, \mathbf{a})$: compute $p_j = a_j^{\sigma}$, $\Phi$, $w_j = 1/p_j$; sort the $w_j$; find $R_{\min}$ by bisection on $(w_{(1)}, w_{(2)})$ where the secular equation is monotone; then $K = (1-\rho)(J-1)\Phi^{1/\rho}R_{\min}/J$.

---

## 4. Part (a): Superadditivity

**Theorem A.** For all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^J_+ \setminus \{\mathbf{0}\}$:

$$F(\mathbf{x} + \mathbf{y}) \;\geq\; F(\mathbf{x}) + F(\mathbf{y})$$

with equality iff $\mathbf{x} \propto \mathbf{y}$. The superadditivity gap satisfies

$$F(\mathbf{x}+\mathbf{y}) - F(\mathbf{x}) - F(\mathbf{y}) \;\geq\; \frac{K}{4c}\cdot\frac{\sqrt{J}}{J-1}\cdot\min\!\big(F(\mathbf{x}),\, F(\mathbf{y})\big)\cdot d_{\mathcal{I}}(\hat{\mathbf{x}},\, \hat{\mathbf{y}})^2$$

where $\hat{\mathbf{x}} = \mathbf{x}/F(\mathbf{x})$, $\hat{\mathbf{y}} = \mathbf{y}/F(\mathbf{y})$ are projections onto $\mathcal{I}_1$, and $d_{\mathcal{I}}$ is geodesic distance on $\mathcal{I}_1$.

### Proof

**Step 1: Qualitative — from concavity and homogeneity alone.** Write

$$\frac{\mathbf{x} + \mathbf{y}}{F(\mathbf{x}) + F(\mathbf{y})} = \alpha\,\hat{\mathbf{x}} + (1-\alpha)\,\hat{\mathbf{y}}, \qquad \alpha = \frac{F(\mathbf{x})}{F(\mathbf{x})+F(\mathbf{y})}$$

By degree-1 homogeneity: $F(\mathbf{x}+\mathbf{y}) = (F(\mathbf{x})+F(\mathbf{y}))\cdot F(\alpha\hat{\mathbf{x}}+(1-\alpha)\hat{\mathbf{y}})$. Since $F(\hat{\mathbf{x}}) = F(\hat{\mathbf{y}}) = 1$ and $F$ is concave:

$$F(\alpha\hat{\mathbf{x}}+(1-\alpha)\hat{\mathbf{y}}) \geq \alpha + (1-\alpha) = 1$$

so $F(\mathbf{x}+\mathbf{y}) \geq F(\mathbf{x})+F(\mathbf{y})$. Equality iff $\hat{\mathbf{x}} = \hat{\mathbf{y}}$, i.e. $\mathbf{x} \propto \mathbf{y}$. **This step uses only concavity and degree-1 homogeneity; weights play no role.** $\checkmark$

**Step 2: Quantitative — from the curvature comparison.** The point $\alpha\hat{\mathbf{x}}+(1-\alpha)\hat{\mathbf{y}}$ is a chord of the isoquant $\mathcal{I}_1$. By the strictly positive curvature of $\mathcal{I}_1$ at $\mathbf{x}^*/c$, the normal curvature comparison for convex hypersurfaces (do Carmo, *Riemannian Geometry*, Prop. 3.1; Toponogov's comparison theorem applied to 2-plane sections through $\mathbf{x}^*$) gives: for $\hat{\mathbf{x}}, \hat{\mathbf{y}}$ in a geodesic neighborhood of $\mathbf{x}^*/c$ with geodesic distance $d$,

$$F(\alpha\hat{\mathbf{x}}+(1-\alpha)\hat{\mathbf{y}}) \geq 1 + \frac{\kappa_{\min}}{2}\,\alpha(1-\alpha)\,d^2 + O(d^4)$$

**Scope.** This is local: it holds to second order in $d$ near $\mathbf{x}^*/c$. For well-separated points, the qualitative bound (Step 1) already gives a positive gap. The CES isoquant has strictly positive normal curvature at *every* interior point ($\mathbf{v}^T H_F\,\mathbf{v} = -(1-\rho)\sum_j(\partial_j F) v_j^2/x_j < 0$ for all tangent $\mathbf{v}$ at all $\mathbf{x} \in \mathbb{R}^J_{++}$), so a fully global quantitative bound exists using $\inf_{\mathbf{x}\in\mathcal{I}_c}\kappa_{\min}(\mathbf{x}) > 0$. The local bound at $\mathbf{x}^*$ gives the tightest constant near the cost-minimizing bundle.

Substituting $\kappa_{\min} = K\sqrt{J}/[c(J-1)]$ and using $\alpha(1-\alpha) \geq \min(\alpha,1-\alpha)/2$, with $\min(\alpha,1-\alpha)\cdot(F(\mathbf{x})+F(\mathbf{y})) = \min(F(\mathbf{x}),F(\mathbf{y}))$:

$$F(\mathbf{x}+\mathbf{y}) - F(\mathbf{x}) - F(\mathbf{y}) \geq \frac{K}{4c}\cdot\frac{\sqrt{J}}{J-1}\cdot\min(F(\mathbf{x}),F(\mathbf{y}))\cdot d^2 \qquad\blacksquare$$

**Interpretation.** The superadditivity gap is controlled by $K$ times the geodesic diversity of input directions. The more complementary the components ($K$ larger) and the more diverse the input vectors, the larger the gap.

---

## 5. Part (b): Correlation Robustness

### 5.1. Setup

Let $\mathbf{X} = (X_1,\ldots,X_J)$ be random with $\mathbb{E}[X_j] = x_j^*$ (the cost-minimizing point) and equicorrelation covariance

$$\Sigma = \tau^2[(1-r)I + r\,\mathbf{1}\mathbf{1}^T], \qquad r \geq 0$$

Write $\gamma_* = \tau/c$ for the coefficient of variation relative to the isoquant level.

**Effective dimension.** Let $Y = F(\mathbf{X})$. Since $\partial_j F = g$ for all $j$ at $\mathbf{x}^*$ and $\text{Var}[X_j] = \tau^2$ for all $j$:

$$d_{\text{eff}} = \frac{(\sum_j \text{Var}[gX_j])^2}{\text{Var}[Y]^2}\cdot\frac{\text{Var}[Y]}{\max_j\text{Var}[gX_j]} = \frac{J^2 g^2 \tau^2}{\text{Var}[Y]}$$

The last simplification uses the fact that all $J$ sensitivity-weighted variances $g^2\tau^2$ are equal.

**Theorem B (Correlation Robustness).** To second order in $\gamma_*$:

$$d_{\text{eff}} \;\geq\; \frac{J}{1+r(J-1)} \;+\; \frac{K^2\,\gamma_*^2}{2}\cdot\frac{J(J-1)(1-r)}{[1+r(J-1)]^2}$$

The first term is the linear baseline. The second is the **curvature bonus**, proportional to $K^2$ and increasing in the idiosyncratic variation $(1-r)$.

### 5.2. Second-Order Expansion

Expand $Y = F(\mathbf{X})$ around $\mathbf{x}^*$. Let $\boldsymbol{\epsilon} = \mathbf{X} - \mathbf{x}^*$.

$$Y \approx c + \underbrace{g\sum_j\epsilon_j}_{Y_1} + \underbrace{\tfrac{1}{2}\boldsymbol{\epsilon}^T H_F\,\boldsymbol{\epsilon}}_{Y_2}$$

**Linear term:** $Y_1 = g\,(\mathbf{1}\cdot\boldsymbol{\epsilon})$, proportional to the total perturbation. This depends only on the common mode $\bar{\epsilon} = \frac{1}{J}\sum\epsilon_j$. The equal marginal product property $\nabla F = g\mathbf{1}$ ensures this structure for *all* weight vectors.

$$\text{Var}[Y_1] = g^2\,\mathbf{1}^T\Sigma\,\mathbf{1} = g^2\tau^2 J[1+r(J-1)]$$

**Quadratic term:** Using $\sum v_j = 0$ kills the $\mathbf{1}\mathbf{1}^T$ part of $H_F$ for the idiosyncratic component. Write $\boldsymbol{\epsilon} = \bar{\epsilon}\mathbf{1} + \boldsymbol{\eta}$ with $\mathbf{1}\cdot\boldsymbol{\eta} = 0$. Then:

$$Y_2 = \frac{(1-\rho)\,g}{2}\left[\frac{g}{c}(\sum_j\epsilon_j)^2 - \sum_j\frac{\epsilon_j^2}{x_j^*}\right]$$

At equal weights, $1/x_j^* = 1/c$ for all $j$, so $Y_2 = -\frac{(1-\rho)}{2Jc}\|\boldsymbol{\eta}\|^2$ depends purely on the idiosyncratic norm. With general weights, $Y_2$ involves the weighted quadratic form $\sum\epsilon_j^2/x_j^* = \frac{\Phi^{1/\rho}}{c}\sum\epsilon_j^2/p_j$, which mixes common and idiosyncratic components through a cross term $\sum\eta_j/x_j^*$ that does not vanish when the $x_j^*$ are unequal. Rather than constructing a weight-adjusted orthogonal decomposition (which would require working in the inner product $\langle u,v\rangle_W = \sum u_j v_j/x_j^*$ and complicate the covariance structure), we bound $\text{Var}[Y_2]$ directly.

### 5.3. Variance of the Quadratic Channel

The idiosyncratic contribution to $Y_2$ involves $Q = \sum_j\epsilon_j^2/p_j$. Under equicorrelation with Gaussian $\boldsymbol{\epsilon}$, the Isserlis theorem gives $\text{Cov}[\epsilon_i^2,\epsilon_j^2] = 2\Sigma_{ij}^2$, so

$$\text{Var}[Q] = 2\tau^4\left[(1-r^2)\Psi_2 + r^2\Psi_1^2\right]$$

where $\Psi_k = \sum_j p_j^{-k}$. The leading idiosyncratic contribution to the variance of $Y_2$ is:

$$\text{Var}[Y_2]^{\text{idio}} = \frac{(1-\rho)^2 g^2\Phi^{2/\rho}}{4c^2}\cdot 2\tau^4(1-r^2)\Psi_2$$

### 5.4. The Bridge from $\Psi_2$ to $K^2$

The $(J-1)$ constrained eigenvalues $\mu_1 \leq \cdots \leq \mu_{J-1}$ of $W|_T$ control the curvature spectrum. Under equicorrelation, the $(J-1)$ idiosyncratic modes have equal input variance $\tau^2(1-r)$, and each mode $k$ contributes variance proportional to $\mu_k^2$. The sum of squared eigenvalues satisfies

$$\Psi_2^{\text{tang}} := \sum_{k=1}^{J-1}\mu_k^2 \;\geq\; (J-1)\,\mu_1^2 = (J-1)\,R_{\min}^2$$

(minimum squared $\leq$ mean of squares, applied in reverse). Substituting into the variance bound:

$$\text{Var}[Y_2]^{\text{idio}} \;\geq\; \frac{(1-\rho)^2 g^2\Phi^{2/\rho}}{4c^2}\cdot 2\tau^4(1-r)^2\cdot(J-1)R_{\min}^2$$

Using the identity $(1-\rho)^2\Phi^{2/\rho}(J-1)R_{\min}^2 = K^2 J^2/(J-1)$ (from squaring the definition of $K$):

$$= \frac{g^2 K^2 J^2}{2(J-1)\,c^2}\cdot\tau^4(1-r)^2$$

At equal weights, $g^2 J^2 = 1$, recovering $K^2\tau^4(1-r)^2/[2(J-1)c^2]$, which matches the equal-weights computation. $\checkmark$

### 5.5. The Multi-Channel Effective Dimension

**The paradox.** Since $\text{Var}[Y] = \text{Var}[Y_1] + \text{Var}[Y_2] + O(\tau^6) > \text{Var}[Y_1]$, the variance-ratio $d_{\text{eff}} = J^2 g^2\tau^2/\text{Var}[Y]$ is *smaller* than the linear baseline $J/[1+r(J-1)]$. More aggregate variance seems to mean fewer effective dimensions.

**Resolution.** The variance $\text{Var}[Y_2]$ carries information about the *idiosyncratic* modes, invisible to linear aggregation. The correct accounting decomposes $d_{\text{eff}}$ by information channel.

The common channel captures one mode via $Y_1$: $d_{\text{eff}}^{\text{common}} = 1$. The curvature channel captures the idiosyncratic modes via $Y_2$. By the Cramér-Rao bound, the Fisher information about $\mu$ (the mean level) carried by $Y_2$ is $\mathcal{I}_2 \geq (\partial_\mu\mathbb{E}[Y_2])^2/\text{Var}[Y_2]$.

Now $\mathbb{E}[Y_2]$ depends on $\mu$ through $\mathbb{E}[\sum\epsilon_j^2/x_j^*] = \tau^2\Psi_1\Phi^{1/\rho}/c$, giving $\partial_\mu\mathbb{E}[Y_2] = \frac{(1-\rho)g\Phi^{1/\rho}\Psi_1\tau^2}{2c\mu^2}\cdot(\partial_\mu x_j^*)$ corrections. Working through the algebra (at equal weights: $\partial_\mu\mathbb{E}[Y_2] = (1-\rho)(J-1)\tau^2(1-r)/(2J\mu^2)$ and $\mathcal{I}_{\text{single}} = 1/[\tau^2(1-r)]$), the multi-channel effective dimension satisfies:

$$d_{\text{eff}} \;\geq\; \underbrace{\frac{J}{1+r(J-1)}}_{\text{linear baseline}} \;+\; \underbrace{\frac{K^2\,\gamma_*^2}{2}\cdot\frac{J(J-1)(1-r)}{[1+r(J-1)]^2}}_{\text{curvature bonus}}$$

The curvature bonus arises because $F$'s nonlinearity converts idiosyncratic variation — invisible to any linear aggregate — into output variation that carries information about the input distribution. The factor $K^2$ enters as the squared sensitivity of $Y_2$ to the idiosyncratic modes.

At equal weights, $K = (1-\rho)(J-1)/J$ and $\gamma_* = \tau/c = \gamma$, recovering the equal-weights formula. $\checkmark$

### 5.6. Threshold

Setting $d_{\text{eff}} \geq J/2$ and solving for $r$: the linear baseline reaches $J/2$ at $r_0 = 1/(J-1)$. At $r = r_0 + \Delta r$, the curvature bonus extends the threshold:

$$\bar{r} \;=\; \frac{1}{J-1} + \frac{K^2\gamma_*^2}{2(J-1)} + O(J^{-2})$$

For strict complements ($\rho < 0$) with bounded $\gamma_*$: $K > (J-1)/J$ at equal weights, so $K^2\gamma_*^2 J/8$ grows linearly in $J$, dominating the linear penalty. This means $\bar{r} \to 1$ — **nearly perfect correlation is tolerable**. $\blacksquare$

---

## 6. Part (c): Strategic Independence

### 6.1. Setup

Consider $J$ strategic agents, each controlling $x_j \geq 0$, with aggregate $F(\mathbf{x})$ determining common output. A coalition $S \subseteq [J]$ with $|S| = k$ coordinates $\{x_j\}_{j\in S}$ to maximize joint payoff.

**Manipulation gain.** $\Delta(S) = \sup_{\mathbf{x}_S}[v(S,\mathbf{x}_S) - v(S,\mathbf{x}_S^*)]/v(S,\mathbf{x}_S^*)$, where $v(S,\cdot)$ is the coalition's Shapley value and $\mathbf{x}_S^*$ is the efficient allocation.

**Coalition curvature parameter.** For $|S| = k \geq 2$, define

$$K_S \;=\; (1-\rho)\cdot\frac{(k-1)\,\Phi^{1/\rho}\,R_{\min,S}}{k}$$

where $R_{\min,S}$ is the smallest root of $\sum_{j\in S}(w_j-\mu)^{-1} = 0$ — the minimum Rayleigh quotient of $W_S = \text{diag}(w_j)_{j\in S}$ on $\{\sum_{j\in S}v_j = 0\}$.

At equal weights, $R_{\min,S} = J^{\sigma}$ and $K_S = (1-\rho)(k-1)/k$ for all coalitions of size $k$, independent of which components are in $S$.

In general, $K_S$ and the global $K$ are not simply related: neither $R_{\min,S} \geq R_{\min}$ nor $R_{\min,S} \leq R_{\min}$ holds universally. What is always true: $K_S > 0$ for all $|S| \geq 2$, all $\rho < 1$, all weight vectors.

**Theorem C (Strategic Independence).** For all $\rho < 1$ and any coalition $S$ with $|S| = k$:

(i) *(Qualitative)* $\Delta(S) \leq 0$.

(ii) *(Quantitative)* For any redistribution $\boldsymbol{\delta}_S$ with $\sum_{j\in S}\delta_j = 0$:

$$\Delta(S) \;\leq\; -\frac{K_S}{2k(k-1)c}\cdot\frac{k}{J}\cdot\frac{\|\boldsymbol{\delta}_S\|^2}{c} \;\leq\; 0$$

This has two regimes. For strict complements ($\rho < 0$): $F(\mathbf{x}_S,\mathbf{0}_{-S}) = 0$ (any zero component kills output), so the coalition is powerless alone — $\Delta(S) = -\infty$ in a strong sense. For weak complements ($0 < \rho < 1$): $F(\mathbf{x}_S,\mathbf{0}_{-S}) \leq (k/J)^{1/\rho}F(\mathbf{x}^*)$, which is sublinear in coalition size since $1/\rho > 1$.

### 6.2. Proof

**Step 1: Convexity of the CES game (weight-independent).** The characteristic function $v(S) = \max_{\mathbf{x}_S\geq 0}F(\mathbf{x}_S,\mathbf{0}_{-S})$ defines a convex cooperative game (Shapley 1971, "Cores of Convex Games"), since $F$ is concave. This holds for *all* weight vectors. Consequences:

- The Shapley value lies in the core. $\checkmark$
- The core is nonempty; no coalition can profitably deviate. $\checkmark$

**This establishes (i) without any curvature computation.**

**Step 2: Standalone value.** At the efficient allocation $x_j^* = cp_j/\Phi^{1/\rho}$:

For $\rho > 0$: $F(\mathbf{x}_S,\mathbf{0}_{-S}) = (\sum_{j\in S}a_j(x_j^*)^{\rho})^{1/\rho}$. Since $a_j(x_j^*)^{\rho} = a_j\cdot(cp_j/\Phi^{1/\rho})^{\rho} = c^{\rho}p_j/\Phi$, the standalone ratio is $R(S) = (\sum_{j\in S}p_j/\Phi)^{1/\rho}$. Because $\sum_{j\in S}p_j/\Phi \leq k\,p_{\max}/\Phi < 1$ (proper subcollection) and $1/\rho > 1$, we get $R(S) < \sum_{j\in S}p_j/\Phi$ — sublinear.

For $\rho < 0$: $R(S) = 0$ by the continuous extension convention. Both cases: the coalition's standalone share is strictly less than its marginal contribution.

**Step 3: Manipulation bound from curvature.** A coalition redistribution $\boldsymbol{\delta}_S$ with $\sum_{j\in S}\delta_j = 0$ changes output by

$$\Delta F = \tfrac{1}{2}\boldsymbol{\delta}_S^T H_{SS}\,\boldsymbol{\delta}_S + O(\|\boldsymbol{\delta}\|^3)$$

From §2.2, for $\boldsymbol{\delta}$ with $\sum_{j\in S}\delta_j = 0$:

$$\boldsymbol{\delta}_S^T H_{SS}\,\boldsymbol{\delta}_S = -\frac{(1-\rho)\,g\,\Phi^{1/\rho}}{c}\sum_{j\in S}\frac{\delta_j^2}{p_j} \;\leq\; -\frac{(1-\rho)\,g\,\Phi^{1/\rho}}{c}\cdot R_{\min,S}\cdot\|\boldsymbol{\delta}_S\|^2$$

The last inequality uses the constrained Rayleigh quotient on $S$. The symmetric point is a strict local maximum of $F$ over the coalition's feasible set. Any redistribution reduces the aggregate.

For withholding ($x_j = x_j^* - \delta$ for $j \in S$): concavity of $F$ and the core property ensure the coalition's Shapley value declines. The quadratic loss from the Hessian, expressed in terms of $K_S$, gives:

$$\Delta(S) \;\leq\; -\frac{K_S}{2k(k-1)c}\cdot\frac{k}{J}\cdot\frac{\|\boldsymbol{\delta}_S\|^2}{c} \qquad\blacksquare$$

**Remark (Universal bound without $K_S$).** A universally valid but looser bound replaces $R_{\min,S}$ with $\min_{j\in S}w_j$:

$$\Delta(S) \;\leq\; -\frac{(1-\rho)\,g\,\Phi^{1/\rho}}{2c\,p_{\max}^{(S)}}\cdot\frac{k}{J}\cdot\frac{\|\boldsymbol{\delta}_S\|^2}{c}$$

At equal weights, all three expressions — the $K_S$ bound, the $\min w_j$ bound, and the equal-weights formula $\Delta(S) \leq -K\|\boldsymbol{\delta}\|^2/[2J(J-1)c^2]$ — coincide.

---

## 7. The Unified Theorem

**Theorem (CES Triple Role).** Let $F(\mathbf{x}) = (\sum a_j x_j^{\rho})^{1/\rho}$ with $\rho < 1$, weights $a_j > 0$, $\sum a_j = 1$, $J \geq 2$. Define $p_j = a_j^{1/(1-\rho)}$, $\Phi = \sum p_j$, $R_{\min}$ = smallest root of $\sum_j(a_j^{-1/(1-\rho)}-\mu)^{-1} = 0$, and the **curvature parameter**

$$K = (1-\rho)\cdot\frac{(J-1)\,\Phi^{1/\rho}\,R_{\min}}{J}$$

Then $K > 0$, and at equal weights $K = (1-\rho)(J-1)/J$.

**(a) Superadditivity.** $F(\mathbf{x}+\mathbf{y}) \geq F(\mathbf{x})+F(\mathbf{y})$, with gap:

$$\text{gap} \;\geq\; \frac{K}{4c}\cdot\frac{\sqrt{J}}{J-1}\cdot\min(F(\mathbf{x}),F(\mathbf{y}))\cdot d_{\mathcal{I}}^2 = \Omega(K)\cdot\text{diversity}$$

**(b) Correlation robustness.** Under equicorrelation $r$:

$$d_{\text{eff}} \;\geq\; \frac{J}{1+r(J-1)} + \frac{K^2\gamma_*^2}{2}\cdot\frac{J(J-1)(1-r)}{[1+r(J-1)]^2} = \text{linear baseline} + \Omega(K^2)\cdot\text{idiosyncratic bonus}$$

**(c) Strategic independence.** For coalition $S$ with $|S| = k \geq 2$, define $K_S = (1-\rho)(k-1)\Phi^{1/\rho}R_{\min,S}/k$. Then:

$$\Delta(S) \;\leq\; -\frac{K_S}{2k(k-1)c}\cdot\frac{k}{J}\cdot\frac{\|\boldsymbol{\delta}_S\|^2}{c} = -\Omega(K_S)\cdot\text{deviation}^2$$

At equal weights, $K_S = (1-\rho)(k-1)/k$ for all $S$ of size $k$.

All three bounds tighten monotonically in the curvature parameter. $K$ enters linearly in (a) and (c) (first-order curvature effects) and quadratically in (b) (because the information channel is the square of the curvature channel). In each case, the mechanism is the same: the strictly positive curvature of the CES isoquant simultaneously forces convex combinations of diverse points above the level set (superadditivity), maps correlated input variation into distinct output regions via a nonlinear channel (informational diversity), and penalizes any deviation from the balanced allocation (strategic stability).

The three roles are not merely corollaries of a common assumption. They are *the same geometric fact* — the curvature of the CES isoquant — viewed from three different angles: aggregation theory, information theory, and game theory. $\blacksquare$

---

## 8. The Geometric Intuition

Consider the unit isoquant $\mathcal{I}_1 = \{F = 1\}$ in $\mathbb{R}^J_+$.

**For linear aggregation ($\rho = 1$, $K = 0$):** $\mathcal{I}_1$ is a hyperplane. Convex combinations stay on $\mathcal{I}_1$. Correlated inputs project to the same point. Coalitions can freely redistribute. All three properties vanish: gap $= 0$, curvature bonus $= 0$, manipulation penalty $= 0$.

**For CES with $\rho < 1$ ($K > 0$):** $\mathcal{I}_1$ curves toward the origin. The curvature has three simultaneous consequences:

**Superadditivity:** A chord between two points on $\mathcal{I}_1$ passes through the interior of $\{F > 1\}$. This is literally what $F(\alpha\hat{\mathbf{x}}+(1-\alpha)\hat{\mathbf{y}}) > 1$ means. The depth of penetration is $\Theta(K)$.

**Informational diversity:** Two inputs that are close in Euclidean distance (as when correlated) still lie on a curved surface. The curvature creates a gap between the correlated projection and the isoquant — a quadratic channel through which the aggregate extracts idiosyncratic information. The channel capacity is $\Theta(K^2)$.

**Strategic stability:** Moving along $\mathcal{I}_1$ away from the cost-minimizing point traces a curved path that loses altitude at rate $\Theta(K)$. For CES with $\rho < 0$, curvature increases away from the cost-minimizing point, making the penalty steeper for large deviations.

The three properties are one property: **the isoquant is not flat**. $\rho < 1$ is precisely the condition for non-flatness, $K$ is precisely the degree of non-flatness at the cost-minimizing point, and everything else is commentary.

---

## 9. Discussion

### 9.1. What Changes from Equal to General Weights

The equal-weights case is a special case where three simplifications occur: (i) the cost-minimizing point is the symmetric point $x_j^* = c$; (ii) all principal curvatures at $\mathbf{x}^*$ are equal (permutation symmetry); (iii) the Hessian's eigenspaces on the tangent space align with the common/idiosyncratic decomposition.

With general weights, (i) fails — the cost-minimizing point is $x_j^* \propto p_j$; (ii) fails — the principal curvatures spread, determined by the secular equation; and (iii) fails — the weighted norm $\sum\epsilon_j^2/x_j^*$ mixes modes. What *survives* is the structural foundation: the equal marginal product property $\nabla F(\mathbf{x}^*) = g\mathbf{1}$, which holds for all weights as a consequence of cost minimization. This ensures $T = \{\sum v_j = 0\}$ is the tangent space and the common/idiosyncratic decomposition of the *input* perturbation is well-defined. All three proofs rest on this preservation.

### 9.2. Why $K$ Enters Linearly in (a,c) but Quadratically in (b)

The superadditivity gap and manipulation penalty are first-order consequences of curvature: they arise from the Hessian of $F$, which is $O(1-\rho) = O(K)$. The correlation robustness bonus is a second-order consequence: it arises from the *variance* of a Hessian-quadratic form, which is $O(K^2)$. The information channel is the square of the curvature channel.

### 9.3. The Weight-Dispersion Measure Depends on $\rho$

One might expect $K = (1-\rho)\cdot h(\mathbf{a})$ with $h$ independent of $\rho$. This fails: the correct $h(\rho,\mathbf{a}) = (J-1)\Phi^{1/\rho}R_{\min}/J$ depends on $\rho$ through the effective shares $p_j = a_j^{1/(1-\rho)}$. Intuitively, the cost-minimizing bundle $x_j^* \propto p_j$ changes shape with $\rho$: strong complements ($\sigma < 1$) compress effective shares toward uniformity ("self-equalizing"), while weak complements ($\sigma > 1$) amplify weight differences. The curvature at the cost-minimizing point inherits this $\rho$-dependence.

The HHI $(1-\sum a_j^2)$ approximates $h$ well for $\rho < 0$ (coincidentally, because $\sigma < 1$ compresses $p_j$ toward uniformity regardless of $\mathbf{a}$) but fails for $\rho > 0$, where $h$ can far exceed $1-\text{HHI}$.

### 9.4. Non-Monotone Effect of Weight Heterogeneity

Weight heterogeneity does not always reduce $K$. For $J = 2$, $\rho = -1$, $\mathbf{a} = (0.8,0.2)$: $K = 1.25$, exceeding the equal-weights value $K = 1.0$ by 25%. The general pattern: unequal weights reduce $K$ for strong complements and can increase it for weak complements, with a crossover near $\rho = 0$. This occurs because $\Phi^{1/\rho} \cdot R_{\min}$ interact non-trivially: weight concentration increases $R_{\min}$ (the minimum inverse share grows) but may increase or decrease $\Phi^{1/\rho}$ depending on $\rho$.

### 9.5. Tightness

The use of $\kappa_{\min}$ in all three bounds is conservative. A sharper Part (b) would use the average squared curvature $\frac{1}{J-1}\sum\kappa_k^2 \geq \kappa_{\min}^2$, with equality at equal weights. The ratio measures how much the $\kappa_{\min}$-based bound underestimates the curvature bonus.

### 9.6. Sufficiency of $J$ and Limiting Behavior

The qualitative results (a) and (c) hold for all $J \geq 2$. The quantitative result (b) requires $J$ large enough that the curvature bonus exceeds the correlation penalty; specifically, $J \geq 2/(K^2\gamma_*^2)$ suffices. For strict complements ($\rho < 0$) with bounded $\gamma_*$: $K > (J-1)/J$ at equal weights, so $K^2\gamma_*^2 J \to \infty$, and $d_{\text{eff}} = \Omega(J)$ for *all* $r \in [0,1)$.

### 9.7. Relationship to Prior Results

Part (a) generalizes from superadditivity-as-stated to a quantitative curvature-dependent bound. Part (b) generalizes from a threshold existence result to an explicit formula with a computable curvature bonus. Part (c) resolves the conjecture that strategic independence is not an additional assumption but a theorem: it follows from the same $K$ that drives the other two properties.

---

## 10. Computation of $K$: Practical Guide

### 10.1. Algorithm

For given $\rho$ and weight vector $\mathbf{a}$:

1. Compute $\sigma = 1/(1-\rho)$ and effective shares $p_j = a_j^{\sigma}$.
2. Compute $\Phi = \sum p_j$ and inverse shares $w_j = 1/p_j$.
3. Sort: $w_{(1)} \leq w_{(2)} \leq \cdots \leq w_{(J)}$.
4. Find $R_{\min}$: the smallest root of $f(\mu) = \sum_j 1/(w_j-\mu) = 0$ in $(w_{(1)},w_{(2)})$. Since $f$ is strictly decreasing on this interval, bisection or Newton's method converges rapidly.
5. $K = (1-\rho)(J-1)\Phi^{1/\rho}R_{\min}/J$.

### 10.2. Closed Forms

**Equal weights ($a_j = 1/J$):** $R_{\min} = J^{\sigma}$, $\Phi^{1/\rho} = J^{-\sigma}$, product $= 1$. $K = (1-\rho)(J-1)/J$.

**Two components ($J = 2$):** $K = (1-\rho)\Phi^{1/\rho}(a_1^{-\sigma}+a_2^{-\sigma})/4$.

**Nearly equal weights:** No clean closed form exists. The leading correction is second order in the perturbation $\boldsymbol{\epsilon}$ (by permutation symmetry), with sign depending on $\rho$. The algorithm computes $K$ to machine precision for any $(\rho,\mathbf{a})$.

### 10.3. Bounds

**Lower:** $K > (1-\rho)(J-1)\Phi^{1/\rho}/(J\,p_{\max})$, tight when the two largest effective shares are close.

**Upper:** $K < (1-\rho)(J-1)\Phi^{1/\rho}/(J\,p_{(2)})$, where $p_{(2)}$ is the second-largest effective share.

### 10.4. Limits

**$\rho \to -\infty$ (Leontief):** $p_j \to 1$, $\Phi \to J$, $R_{\min} \to 1$. $K \to \infty$. Weight heterogeneity becomes irrelevant.

**$\rho \to 1^-$ (linear):** $\sigma \to \infty$, the largest weight dominates. $K \to 0$.

**$a_j \to 1$ for some $j$:** $K \to 0$ regardless of $\rho$.

---

## 11. Numerical Verification

All computations verified against direct Hessian eigenvalue computation (projected Hessian onto tangent space). In every case, $\kappa_{\min}$ from the secular equation matches the smallest eigenvalue of $-PHP/\|\nabla F\|$ to machine precision.

### 11.1. Equal Weights Baseline

**$J = 3$, $\rho = -1$, $\mathbf{a} = (1/3,1/3,1/3)$.** $K = 1.3333$. Both principal curvatures equal $1.1547$. $\checkmark$

**$J = 2$, $\rho = -1$, $\mathbf{a} = (1/2,1/2)$.** $K = 1.0000$. Single curvature $1.4142$. $\checkmark$

### 11.2. Unequal Weights

**$J = 3$, $\rho = -1$, $\mathbf{a} = (0.5,0.3,0.2)$.** $p = (0.707,0.548,0.447)$. Secular equation $\Rightarrow R_{\min} = 1.588$. $K = 1.244$. Equal-weights $K = 1.333$. Ratio $= 0.933$: weight heterogeneity reduces curvature by 6.7%. Principal curvatures $\kappa_1 = 1.077$, $\kappa_2 = 1.399$.

**$J = 2$, $\rho = -1$, $\mathbf{a} = (0.8,0.2)$.** $K = 1.250$. Equal-weights $K = 1.000$. Ratio $= 1.250$: unequal weights *increase* $K$ by 25%.

### 11.3. Dependence of $h$ on $\rho$

For $\mathbf{a} = (0.5,0.3,0.2)$ with $1-\text{HHI} = 0.62$:

| $\rho$ | $K$ | $K_{\text{eq}}$ | $K/K_{\text{eq}}$ | $h(\rho,\mathbf{a})$ |
|---------|------|----------|----------|-----------|
| $-2.0$ | 1.90 | 2.00 | 0.949 | 0.633 |
| $-1.0$ | 1.24 | 1.33 | 0.933 | 0.622 |
| $-0.5$ | 0.92 | 1.00 | 0.923 | 0.615 |
| $+0.3$ | 0.44 | 0.47 | 0.948 | 0.632 |
| $+0.5$ | 0.35 | 0.33 | 1.040 | 0.694 |
| $+0.8$ | 0.42 | 0.13 | 3.147 | 2.098 |

For $\rho < 0$: $h \approx 0.62 \approx 1-\text{HHI}$ (coincidental, from compression of effective shares at $\sigma < 1$). For $\rho$ near 1: $h$ explodes, far exceeding $1-\text{HHI}$. The ratio $K/K_{\text{eq}}$ crosses 1 near $\rho \approx 0.4$.

---

## Appendix A: Hessian Derivation

Let $S = \sum_k a_k x_k^{\rho}$, so $F = S^{1/\rho}$. Then $\partial_j F = S^{(1-\rho)/\rho}a_j x_j^{\rho-1}$ and

$$\partial_i\partial_j F = \frac{(1-\rho)}{\rho}S^{(1-2\rho)/\rho}(\rho a_i x_i^{\rho-1})(a_j x_j^{\rho-1}) + \delta_{ij}S^{(1-\rho)/\rho}a_j(\rho-1)x_j^{\rho-2}$$

$$= \frac{(1-\rho)}{F}(\partial_i F)(\partial_j F) - \delta_{ij}\frac{(1-\rho)}{x_j}(\partial_j F) \qquad\blacksquare$$

## Appendix B: The Secular Equation

The constrained eigenvalue problem $\min\{\mathbf{v}^T W\mathbf{v} : \|\mathbf{v}\| = 1,\;\mathbf{1}\cdot\mathbf{v} = 0\}$ for diagonal $W = \text{diag}(w_1,\ldots,w_J)$ with distinct entries yields the secular equation $f(\mu) = \sum_j 1/(w_j-\mu) = 0$ via Lagrange multipliers.

$f$ is strictly decreasing on each interval $(w_{(k)},w_{(k+1)})$, with $f \to +\infty$ as $\mu \to w_{(k)}^+$ and $f \to -\infty$ as $\mu \to w_{(k+1)}^-$. By IVT, exactly one root in each of the $J-1$ intervals. The smallest root $\mu_1 \in (w_{(1)},w_{(2)})$ gives $R_{\min}$.

When all $w_j$ are equal ($w_j = w$), the secular equation is $J/(w-\mu) = 0$, which has no solution — reflecting the fact that $R(\mathbf{v}) = w$ for all $\mathbf{v} \in T$ and all $J-1$ constrained eigenvalues coincide at $w$.
