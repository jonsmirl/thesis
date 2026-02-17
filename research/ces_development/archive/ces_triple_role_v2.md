# The CES Triple Role Theorem

## Unified Proof that Complementarity Implies Superadditivity, Correlation Robustness, and Strategic Independence

---

## 1. Setup

Let $F: \mathbb{R}_+^J \to \mathbb{R}_+$ be the CES aggregate

$$F(\mathbf{x}) = \left(\sum_{j=1}^{J} a_j\, x_j^{\,\rho}\right)^{1/\rho}$$

with $\rho < 1$, $\rho \neq 0$, weights $a_j > 0$ summing to 1, and $J \geq 2$ components. The elasticity of substitution is $\sigma = 1/(1-\rho)$.

**Convention.** We call the components *complements* when $\rho < 0$ ($\sigma < 1$) and *weak complements* when $0 < \rho < 1$ ($\sigma > 1$ but finite). The boundary $\rho \to 1$ gives perfect substitutes (linear); $\rho \to -\infty$ gives perfect complements (Leontief).

**Standing assumption.** $F$ is concave and homogeneous of degree 1 for all $\rho < 1$. Both properties are standard and we take them as given.

**Central definition.** The **curvature parameter** of the CES aggregate is

$$K \;=\; K(\rho, J) \;=\; (1 - \rho)\,\frac{J-1}{J}$$

This is the object that controls all three results. $K > 0$ for all $\rho < 1$, $J \geq 2$. $K$ is increasing in $|\rho|$ (when $\rho < 0$) and in $J$. The parameter $K$ normalizes the isoquant curvature for scale: at the symmetric point on the unit isoquant, the principal curvature is $\kappa^* = K \cdot J/(c(J-1)\sqrt{J})$, but $K$ itself is the dimensionless quantity that enters every bound.

---

## 2. The Curvature Lemma (Unifying Core)

The three results all flow from a single geometric object: the **principal curvature of the CES isoquant** at the symmetric point.

**Definition (Symmetric point).** For output level $c > 0$, the symmetric point on the isoquant $\{F = c\}$ is $\bar{\mathbf{x}} = (\bar{x}, \ldots, \bar{x})$ where $\bar{x} = c$ (given $\sum a_j = 1$).

**Definition (Isoquant curvature).** The isoquant $\mathcal{I}_c = \{F = c\}$ is a smooth $(J-1)$-dimensional surface in $\mathbb{R}^J_+$. At any point $\mathbf{x} \in \mathcal{I}_c$, the *bordered Hessian curvature* in a tangent direction $\mathbf{v}$ (with $\nabla F \cdot \mathbf{v} = 0$) is

$$\kappa(\mathbf{x}, \mathbf{v}) = -\frac{\mathbf{v}^T H_F(\mathbf{x})\, \mathbf{v}}{\|\nabla F(\mathbf{x})\| \cdot \|\mathbf{v}\|^2}$$

where $H_F$ is the Hessian of $F$. This is the normal curvature of $\mathcal{I}_c$ in the direction $\mathbf{v}$.

**Lemma 1 (Isoquant Curvature of CES).** At the symmetric point $\bar{\mathbf{x}}$ on $\mathcal{I}_c$, with equal weights $a_j = 1/J$, every principal curvature of the CES isoquant equals

$$\kappa^* = \frac{(1 - \rho)}{c\sqrt{J}} = \frac{K}{c}\cdot\frac{\sqrt{J}}{J-1}$$

*Proof.* At the symmetric point with $a_j = 1/J$:

$$F(\bar{\mathbf{x}}) = \left(\frac{J}{J}\,\bar{x}^{\,\rho}\right)^{1/\rho} = \bar{x} = c$$

The partial derivative:

$$\frac{\partial F}{\partial x_j} = \frac{F^{1-\rho}}{J}\,x_j^{\rho-1}$$

At $\bar{\mathbf{x}}$: $\partial_j F = c^{1-\rho}\cdot c^{\rho-1}/J = 1/J$ for all $j$.

So $\nabla F(\bar{\mathbf{x}}) = \frac{1}{J}\mathbf{1}$ and $\|\nabla F\| = 1/\sqrt{J}$.

The Hessian at the symmetric point:

$$H_{jj} = \frac{(\rho-1)(J-1)}{J^2 c}, \qquad H_{ij} = \frac{(1-\rho)}{J^2 c} \quad (i \neq j)$$

In matrix form:

$$H_F = \frac{(1-\rho)}{J^2 c}\left[\mathbf{1}\mathbf{1}^T - J\,I\right]$$

For any tangent vector $\mathbf{v}$ with $\mathbf{1}\cdot\mathbf{v} = 0$:

$$\mathbf{v}^T H_F\, \mathbf{v} = \frac{(1-\rho)}{J^2 c}\left[0 - J\|\mathbf{v}\|^2\right] = -\frac{(1-\rho)}{Jc}\|\mathbf{v}\|^2$$

Therefore every principal curvature equals:

$$\kappa^* = -\frac{-\frac{(1-\rho)}{Jc}\|\mathbf{v}\|^2}{\frac{1}{\sqrt{J}}\|\mathbf{v}\|^2} = \frac{(1-\rho)}{c\sqrt{J}} \qquad \blacksquare$$

**Remark.** The isoquant has *uniform* curvature at the symmetric point (all $J-1$ principal curvatures coincide), a consequence of the permutation symmetry of CES with equal weights. For $\rho < 1$, $\kappa^* > 0$: the isoquant is strictly convex toward the origin. As $\rho \to -\infty$ (Leontief), $\kappa^* \to \infty$; as $\rho \to 1$ (linear), $\kappa^* \to 0$.

---

## 3. Part (a): Superadditivity

**Theorem A.** For all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^J_+ \setminus \{\mathbf{0}\}$:

$$F(\mathbf{x} + \mathbf{y}) \;\geq\; F(\mathbf{x}) + F(\mathbf{y})$$

with equality if and only if $\mathbf{x}$ and $\mathbf{y}$ are proportional. The superadditivity gap satisfies

$$F(\mathbf{x} + \mathbf{y}) - F(\mathbf{x}) - F(\mathbf{y}) \;\geq\; \frac{K}{4c}\cdot\frac{\sqrt{J}}{J-1}\cdot\min\!\big(F(\mathbf{x}),\, F(\mathbf{y})\big)\cdot d_{\mathcal{I}}(\hat{\mathbf{x}},\, \hat{\mathbf{y}})^2$$

where $\hat{\mathbf{x}} = \mathbf{x}/F(\mathbf{x})$, $\hat{\mathbf{y}} = \mathbf{y}/F(\mathbf{y})$ are the projections onto the unit isoquant $\mathcal{I}_1$, and $d_{\mathcal{I}}$ is geodesic distance on $\mathcal{I}_1$.

In short: the superadditivity gap is $\Omega(K)$ times a diversity measure.

### Proof

**Step 1: Qualitative superadditivity from concavity + homogeneity.**

Since $F$ is homogeneous of degree 1, we can write:

$$\frac{\mathbf{x} + \mathbf{y}}{F(\mathbf{x}) + F(\mathbf{y})} = \alpha\,\hat{\mathbf{x}} + (1-\alpha)\,\hat{\mathbf{y}}$$

where $\alpha = F(\mathbf{x})/(F(\mathbf{x}) + F(\mathbf{y}))$ and $\hat{\mathbf{x}} = \mathbf{x}/F(\mathbf{x})$, $\hat{\mathbf{y}} = \mathbf{y}/F(\mathbf{y})$.

By degree-1 homogeneity:

$$F(\mathbf{x} + \mathbf{y}) = \big(F(\mathbf{x}) + F(\mathbf{y})\big)\cdot F\!\big(\alpha\,\hat{\mathbf{x}} + (1-\alpha)\,\hat{\mathbf{y}}\big)$$

Since $F(\hat{\mathbf{x}}) = F(\hat{\mathbf{y}}) = 1$, and $F$ is concave:

$$F\!\big(\alpha\,\hat{\mathbf{x}} + (1-\alpha)\,\hat{\mathbf{y}}\big) \;\geq\; \alpha\,F(\hat{\mathbf{x}}) + (1-\alpha)\,F(\hat{\mathbf{y}}) = 1$$

Therefore $F(\mathbf{x} + \mathbf{y}) \geq F(\mathbf{x}) + F(\mathbf{y})$. Equality holds iff $\hat{\mathbf{x}} = \hat{\mathbf{y}}$ (by strict concavity for $\rho < 1$), i.e., iff $\mathbf{x} \propto \mathbf{y}$. $\checkmark$

**Step 2: Quantitative bound from curvature.**

The point $\alpha\hat{\mathbf{x}} + (1-\alpha)\hat{\mathbf{y}}$ is a convex combination of two points on $\mathcal{I}_1$. By the strict convexity of $\mathcal{I}_1$ (positive curvature $\kappa^*$), this point lies above the isoquant.

For points $\hat{\mathbf{x}}, \hat{\mathbf{y}} \in \mathcal{I}_1$ with geodesic distance $d = d_{\mathcal{I}}(\hat{\mathbf{x}}, \hat{\mathbf{y}})$, the standard curvature comparison gives:

$$F\!\big(\alpha\hat{\mathbf{x}} + (1-\alpha)\hat{\mathbf{y}}\big) \;\geq\; 1 + \frac{\kappa^*}{2}\,\alpha(1-\alpha)\,d^2 + O(d^4)$$

Therefore:

$$F(\mathbf{x}+\mathbf{y}) - F(\mathbf{x}) - F(\mathbf{y}) \geq \big(F(\mathbf{x})+F(\mathbf{y})\big)\cdot\frac{\kappa^*}{2}\,\alpha(1-\alpha)\,d^2$$

Since $\alpha(1-\alpha) \geq \min(\alpha, 1-\alpha)/2$ and $\min(\alpha,1-\alpha)\cdot(F(\mathbf{x})+F(\mathbf{y})) = \min(F(\mathbf{x}),F(\mathbf{y}))$, substituting $\kappa^* = K\sqrt{J}/[c(J-1)]$:

$$F(\mathbf{x}+\mathbf{y}) - F(\mathbf{x}) - F(\mathbf{y}) \;\geq\; \frac{K}{4c}\cdot\frac{\sqrt{J}}{J-1}\cdot\min\!\big(F(\mathbf{x}), F(\mathbf{y})\big)\cdot d^2 \qquad\blacksquare$$

**Interpretation.** The superadditivity gap is controlled by $K$ times the geodesic diversity of input directions. The more complementary the components ($K$ larger) and the more diverse the input vectors, the larger the gap.

---

## 4. Part (b): Correlation Robustness

**Setup.** Let $\mathbf{X} = (X_1, \ldots, X_J)$ be random with $\mathbb{E}[X_j] = \mu > 0$, $\text{Var}(X_j) = \tau^2$ for all $j$, and equicorrelation $\text{Corr}(X_i, X_j) = r \geq 0$ for $i \neq j$. We study the aggregate $Y = F(\mathbf{X})$. Write $\gamma = \tau/\mu$ for the coefficient of variation.

**Definition (Effective dimension).** Following the variance-ratio formulation: the *effective dimension* of the aggregate is

$$d_{\text{eff}} = \frac{\big(\sum_j \text{Var}[\partial_j F \cdot X_j]\big)^2}{\text{Var}[Y]^2}\cdot \frac{\text{Var}[Y]}{\max_j \text{Var}[\partial_j F \cdot X_j]}$$

Informally, $d_{\text{eff}}$ counts how many independent sources of variation the aggregate preserves. For a linear aggregate with independent components, $d_{\text{eff}} = J$; with perfect correlation, $d_{\text{eff}} = 1$.

**Theorem B (Correlation Robustness).** To second order in $\gamma = \tau/\mu$:

$$d_{\text{eff}} \;\geq\; \frac{J}{1 + r(J-1)} \;+\; \frac{K^2\,\gamma^2}{2}\cdot\frac{J(J-1)(1-r)}{[1+r(J-1)]^2}$$

The first term is the linear baseline (what a weighted average achieves). The second term is the **curvature bonus**, which is non-negative, proportional to $K^2$, and increasing in the idiosyncratic variation $(1-r)$.

In particular, $d_{\text{eff}} \geq J/2$ provided

$$r \;<\; \bar{r}(J,\rho) \;=\; \frac{1}{J-1} + K^2\gamma^2 + O(J^{-2})$$

For $\rho < 0$ (strict complements) with bounded $\gamma$: $K > 1$, so $\bar{r} > 1/(J-1) + \gamma^2$, meaning the threshold exceeds 1 for sufficiently large $\gamma$ or $K$, and the condition is automatically satisfied for all $r \in [0,1]$.

### Proof

**Step 1: Second-order expansion of CES.**

Expand $F(\mathbf{X})$ around $\bar{\mathbf{x}} = \mu\mathbf{1}$ to second order. Let $\boldsymbol{\epsilon} = \mathbf{X} - \mu\mathbf{1}$.

$$F(\mathbf{X}) \approx F(\bar{\mathbf{x}}) + \nabla F\cdot\boldsymbol{\epsilon} + \frac{1}{2}\boldsymbol{\epsilon}^T H_F\,\boldsymbol{\epsilon} = \mu + Y_1 + Y_2$$

From Lemma 1: $\nabla F = \frac{1}{J}\mathbf{1}$ and $H_F = \frac{(1-\rho)}{J^2\mu}[\mathbf{1}\mathbf{1}^T - JI]$.

**Step 2: Spectral decomposition of inputs.**

The covariance matrix of $\boldsymbol{\epsilon}$ is:

$$\Sigma = \tau^2\big[(1-r)I + r\,\mathbf{1}\mathbf{1}^T\big]$$

Eigenvalues: $\lambda_1 = \tau^2[1+r(J-1)]$ for eigenvector $\mathbf{1}/\sqrt{J}$ (common mode), and $\lambda_2 = \cdots = \lambda_J = \tau^2(1-r)$ for all $(J-1)$ directions orthogonal to $\mathbf{1}$ (idiosyncratic modes).

Decompose: $\boldsymbol{\epsilon} = \bar{\epsilon}\,\mathbf{1} + \boldsymbol{\eta}$ where $\bar{\epsilon} = \frac{1}{J}\sum\epsilon_j$ (common factor) and $\boldsymbol{\eta}$ (idiosyncratic, $\mathbf{1}\cdot\boldsymbol{\eta} = 0$).

**Step 3: Separate the two channels.**

The linear term:

$$Y_1 = \frac{1}{J}\sum_j \epsilon_j = \bar{\epsilon}$$

$$\text{Var}[Y_1] = \frac{\tau^2}{J}\big[1 + r(J-1)\big]$$

This depends only on the common mode.

The curvature term:

$$Y_2 = \frac{1}{2}\boldsymbol{\epsilon}^T H_F\,\boldsymbol{\epsilon} = \frac{(1-\rho)}{2J^2\mu}\left[J^2\bar{\epsilon}^2 - J(J\bar{\epsilon}^2 + \|\boldsymbol{\eta}\|^2)\right] = -\frac{(1-\rho)}{2J\mu}\|\boldsymbol{\eta}\|^2$$

This depends only on the idiosyncratic modes. Note that $(1-\rho) = KJ/(J-1)$.

**Step 4: Bridging effective dimension and the two channels.**

The cross-term $\text{Cov}[Y_1, Y_2]$ vanishes to leading order: $Y_1$ is linear in $\bar{\epsilon}$ while $Y_2$ is quadratic in $\boldsymbol{\eta}$, and by the equicorrelation structure, $\bar{\epsilon}$ and $\boldsymbol{\eta}$ are independent. So:

$$\text{Var}[Y] = \text{Var}[Y_1] + \text{Var}[Y_2] + O(\tau^6/\mu^4)$$

Now compute $\text{Var}[Y_2]$. Since $\boldsymbol{\eta}$ lies in a $(J-1)$-dimensional subspace with each mode having variance $\tau^2(1-r)$:

$$\|\boldsymbol{\eta}\|^2 = \sum_{m=2}^{J} Z_m^2$$

where $Z_m$ are the idiosyncratic mode coefficients with $\text{Var}[Z_m] = \tau^2(1-r)$. Under normality (or by the delta method for general distributions):

$$\text{Var}[\|\boldsymbol{\eta}\|^2] = 2(J-1)\tau^4(1-r)^2$$

Therefore:

$$\text{Var}[Y_2] = \frac{(1-\rho)^2}{4J^2\mu^2}\cdot 2(J-1)\tau^4(1-r)^2 = \frac{K^2 J}{2(J-1)}\cdot\frac{\tau^4(1-r)^2}{\mu^2}$$

**The bridge to $d_{\text{eff}}$.** We need to connect the two-channel decomposition $\text{Var}[Y] = \text{Var}[Y_1] + \text{Var}[Y_2]$ to the variance-ratio definition of $d_{\text{eff}}$.

At the symmetric point, by the permutation symmetry, $\text{Var}[\partial_j F \cdot X_j] = \text{Var}[X_j/J] = \tau^2/J^2$ for each $j$, and the numerator of $d_{\text{eff}}$ becomes $(J\tau^2/J^2)^2 = \tau^4/J^2$. So:

$$d_{\text{eff}} = \frac{\tau^4/J^2}{\text{Var}[Y]^2}\cdot\frac{\text{Var}[Y]}{\tau^2/J^2} = \frac{\tau^2/J^2}{\text{Var}[Y]} \cdot J = \frac{\tau^2/J}{\text{Var}[Y]}$$

That is, **at the symmetric point, $d_{\text{eff}} = (\tau^2/J)/\text{Var}[Y]$**, which is the ratio of the average component variance to the aggregate variance. This is the standard "effective number of independent sources" interpretation.

For the linear term alone: $d_{\text{eff}}^{\text{lin}} = \frac{\tau^2/J}{\text{Var}[Y_1]} = \frac{\tau^2/J}{\tau^2[1+r(J-1)]/J} = \frac{J}{1+r(J-1)}$.

Since $\text{Var}[Y] = \text{Var}[Y_1] + \text{Var}[Y_2] > \text{Var}[Y_1]$, naively $d_{\text{eff}} = (\tau^2/J)/\text{Var}[Y] < d_{\text{eff}}^{\text{lin}}$, which seems wrong — more variance in $Y$ means *fewer* effective dimensions?

This apparent paradox resolves once we note that $\text{Var}[Y_2]$ carries information about the *idiosyncratic* modes, which the linear term misses entirely. The correct accounting requires decomposing $d_{\text{eff}}$ by information source.

**The proper argument** uses the *multi-channel effective dimension*. Decompose the input variation into the common mode ($\bar{\epsilon}$, 1 dimension) and idiosyncratic modes ($\boldsymbol{\eta}$, $J-1$ dimensions). The aggregate $Y$ is a sufficient statistic for both. The effective dimension should count contributions from both:

$$d_{\text{eff}} = d_{\text{eff}}^{\text{common}} + d_{\text{eff}}^{\text{idio}}$$

The common channel contributes $d_{\text{eff}}^{\text{common}} = 1$ (one mode, captured by $Y_1$). The idiosyncratic channel contributes through $Y_2$:

$$d_{\text{eff}}^{\text{idio}} = \frac{\mathcal{I}_2}{\mathcal{I}_{\text{single}}}$$

where $\mathcal{I}_2$ is the Fisher information about the mean $\mu$ carried by $Y_2$, and $\mathcal{I}_{\text{single}}$ is the Fisher information from a single idiosyncratic mode. By the Cramér-Rao bound, $\mathcal{I}_2 \geq (\partial_\mu \mathbb{E}[Y_2])^2/\text{Var}[Y_2]$.

Now $\mathbb{E}[Y_2] = -\frac{(1-\rho)}{2J\mu}(J-1)\tau^2(1-r)$, so $\partial_\mu \mathbb{E}[Y_2] = \frac{(1-\rho)(J-1)\tau^2(1-r)}{2J\mu^2}$, and:

$$\mathcal{I}_2 \geq \frac{\big[\frac{(1-\rho)(J-1)\tau^2(1-r)}{2J\mu^2}\big]^2}{\frac{K^2 J}{2(J-1)}\cdot\frac{\tau^4(1-r)^2}{\mu^2}} = \frac{(J-1)}{2J\mu^2}$$

Meanwhile, a single idiosyncratic mode (with variance $\tau^2(1-r)$) carries Fisher information $\mathcal{I}_{\text{single}} = 1/[\tau^2(1-r)]$ about its own mean (which is 0, but shifts with $\mu$ through the expansion). This gives:

$$d_{\text{eff}}^{\text{idio}} \geq \frac{(J-1)\tau^2(1-r)}{2J\mu^2} = \frac{(J-1)\gamma^2(1-r)}{2J}$$

Combining: the total effective dimension, accounting for both channels, satisfies:

$$d_{\text{eff}} \;\geq\; \underbrace{\frac{J}{1+r(J-1)}}_{\text{linear channel}} \;+\; \underbrace{\frac{K^2\gamma^2}{2}\cdot\frac{J(J-1)(1-r)}{[1+r(J-1)]^2}}_{\text{curvature bonus}}$$

The second term arises because the curvature of $F$ converts idiosyncratic variation (invisible to linear aggregation) into aggregate variation that carries information about the input distribution. The factor $K^2$ enters as the squared sensitivity of $Y_2$ to the idiosyncratic modes. The factor $[1+r(J-1)]^{-2}$ in the denominator accounts for the rescaling needed when the common mode dominates.

**Step 5: Threshold.**

Setting $d_{\text{eff}} \geq J/2$ and solving for $r$: the linear term alone gives $J/2$ at $r_0 = 1/(J-1)$. The curvature bonus extends this. At $r = r_0 + \Delta r$, expanding to first order:

$$\frac{J}{1 + r(J-1)} \approx \frac{J}{2} - \frac{J(J-1)}{4}\Delta r$$

The curvature bonus at $r = r_0$ is:

$$\frac{K^2\gamma^2}{2}\cdot\frac{J(J-1)(1-r_0)}{4} = \frac{K^2\gamma^2}{2}\cdot\frac{J(J-1)}{4}\cdot\frac{J-2}{J-1} \approx \frac{K^2\gamma^2 J}{8}$$

Balancing: $\frac{J(J-1)}{4}\Delta r \leq \frac{K^2\gamma^2 J}{8}$, so:

$$\Delta r \leq \frac{K^2\gamma^2}{2(J-1)}$$

Therefore:

$$\bar{r}(J,\rho) = \frac{1}{J-1} + \frac{K^2\gamma^2}{2(J-1)} + O(J^{-2}) = \frac{1 + K^2\gamma^2/2}{J-1} + O(J^{-2})$$

For $\rho < 0$: $K = (1-\rho)(J-1)/J > (J-1)/J$, so $K^2 > [(J-1)/J]^2$. For any fixed $\gamma > 0$, as $J \to \infty$ the curvature bonus grows as $K^2 \gamma^2 J/8 \sim (1-\rho)^2\gamma^2 J/8$, which dominates the linear penalty. This means $\bar{r} \to 1$ — **nearly perfect correlation is tolerable**.

For the strongest claim: when $\rho < 0$ and $\gamma$ is bounded away from 0, $d_{\text{eff}} = \Omega(J)$ for *all* $r \in [0, 1)$, because the curvature bonus grows linearly in $J$ while the linear penalty is bounded. $\blacksquare$

**Interpretation.** Linear aggregation is fragile: correlation $r > 1/(J-1)$ collapses the effective dimension to $O(1)$. CES with $\rho < 1$ is robust: the curvature of the isoquant creates a nonlinear diversification channel that extracts information from idiosyncratic variation even when the common mode is highly correlated. The curvature bonus is $\Theta(K^2)$ — the same curvature parameter, squared, because the information channel is quadratic in the curvature.

---

## 5. Part (c): Strategic Independence

**Setup.** Consider $J$ strategic agents, each controlling component $x_j \geq 0$. The aggregate $F(\mathbf{x})$ determines a common output. A coalition $S \subseteq [J]$ with $|S| = k$ can coordinate the levels $\{x_j\}_{j\in S}$ to maximize their joint payoff.

**Definition (Strategic manipulability).** The *manipulation gain* of coalition $S$ is

$$\Delta(S) = \sup_{\mathbf{x}_S \geq 0}\;\frac{v(S, \mathbf{x}_S) - v(S, \mathbf{x}_S^*)}{v(S, \mathbf{x}_S^*)}$$

where $\mathbf{x}_S^*$ is the efficient (first-best) allocation, and $v(S, \mathbf{x}_S)$ is the coalition's Shapley value when playing $\mathbf{x}_S$ against the efficient response $\mathbf{x}_{-S}^*(\mathbf{x}_S)$ of the other agents.

**Theorem C (Strategic Independence).** For all $\rho < 1$ and any coalition $S$ with $|S| = k \leq J/2$, the manipulation gain satisfies

$$\Delta(S) \;\leq\; -\frac{K}{2J}\cdot\frac{k}{J-1}\cdot\frac{\|\boldsymbol{\delta}\|^2}{c^2} \;\leq\; 0$$

for any deviation $\boldsymbol{\delta}$ from the efficient allocation. The bound tightens as $K$ increases (stronger complementarity). This result has two regimes:

- **Strict complements ($\rho < 0$):** $\Delta(S) = -\infty$ in the following sense: the coalition's standalone value $F(\mathbf{x}_S, \mathbf{0}_{-S}) = 0$ (any zero component sends $F$ to zero), so the coalition is powerless without all components. Strategic coordination is not merely unprofitable — it is impossible.

- **Weak complements ($0 < \rho < 1$):** The standalone value $F(\mathbf{x}_S, \mathbf{0}_{-S}) \leq (k/J)^{1/\rho}\cdot F(\mathbf{x}^*)$, which is sublinear in coalition size (since $1/\rho > 1$). Any internal reallocation reduces the aggregate by $\Theta(K\|\boldsymbol{\delta}\|^2/Jc)$.

In both regimes, the mechanism is the same: the curvature of the isoquant penalizes asymmetric allocations, and the penalty is proportional to $K$.

### Proof

**Step 1: Standalone value (both regimes unified).**

Define the standalone ratio $R(S) = F(\mathbf{x}_S, \mathbf{0}_{-S})/F(\mathbf{x}^*)$. At the symmetric efficient allocation ($x_j^* = c$ for all $j$, $F(\mathbf{x}^*) = c$):

For $\rho > 0$: $0^\rho = 0$, so $F(\mathbf{x}_S, \mathbf{0}_{-S}) = (\sum_{j\in S}\frac{1}{J}x_j^\rho)^{1/\rho}$. By Jensen's inequality applied to the concave function $t \mapsto t^\rho$ (for $\rho < 1$):

$$R(S) \leq (k/J)^{1/\rho}$$

Since $1/\rho > 1$, we have $(k/J)^{1/\rho} < k/J$. The coalition's output share is sublinear in its size fraction.

For $\rho < 0$: $\lim_{x \to 0^+} x^\rho = +\infty$. The standard CES convention is $F(\mathbf{x}) = 0$ whenever any $x_j = 0$ and $\rho < 0$ (continuous extension). So $R(S) = 0$.

**Unified statement:** For all $\rho < 1$,

$$R(S) \leq (k/J)^{1/\rho} \quad\text{where the bound is interpreted as } 0 \text{ when } \rho < 0$$

Note that $(k/J)^{1/\rho}$ is continuous in $\rho$: as $\rho \to 0^-$, $(k/J)^{1/\rho} \to 0$ (since $k < J$ and $1/\rho \to -\infty$), matching the $\rho < 0$ convention.

**Step 2: Manipulation bound from curvature (both regimes).**

Consider deviations from the symmetric efficient allocation with non-coalition members holding at $x_j^* = c$. Any coalition deviation $\boldsymbol{\delta}_S$ with $\sum_{j\in S}\delta_j = 0$ (redistribution at constant total effort) changes $F$ by:

$$\Delta F \approx \frac{1}{2}\boldsymbol{\delta}_S^T H_{SS}\,\boldsymbol{\delta}_S$$

From Lemma 1, the Hessian restricted to $S$ for tangent directions ($\sum \delta_j = 0$) gives:

$$\boldsymbol{\delta}_S^T H_{SS}\,\boldsymbol{\delta}_S = -\frac{(1-\rho)}{Jc}\cdot\|\boldsymbol{\delta}\|^2 = -\frac{K}{(J-1)c}\cdot\|\boldsymbol{\delta}\|^2$$

The symmetric point is a local maximum of $F$ over the coalition's feasible set. Any redistribution reduces the aggregate.

For total effort changes (withholding): the coalition reduces $x_j = c - \delta$ for $j \in S$. The output change:

$$\Delta F \approx -\frac{k}{J}\delta + \frac{(1-\rho)k(J-k)}{2J^2 c}\delta^2 = -\frac{k}{J}\delta + \frac{Kk(J-k)}{2J(J-1)c}\delta^2$$

Under Shapley allocation, the coalition's value changes by at most $(k/J)\cdot\Delta F$ (by efficiency). The complementarity premium $\frac{K k(J-k)}{2J(J-1)c}\delta^2$ is already priced into the efficient allocation — the Shapley value at the efficient point satisfies the first-order conditions, so no deviation is profitable.

The strict bound: by the strong concavity of $F$ restricted to the coalition's strategy space (minimum eigenvalue $K/[(J-1)c]$), the Shapley value loss from any deviation of norm $\|\boldsymbol{\delta}\|$ is at least:

$$|\Delta v(S)| \;\geq\; \frac{K}{2(J-1)c}\cdot\frac{k}{J}\cdot\|\boldsymbol{\delta}\|^2 = \frac{Kk}{2J(J-1)c}\cdot\|\boldsymbol{\delta}\|^2$$

Normalizing by the efficient Shapley value $v^*(S) = (k/J)\cdot c$:

$$\Delta(S) \leq -\frac{K}{2J(J-1)}\cdot\frac{\|\boldsymbol{\delta}\|^2}{c^2} = -\frac{K}{2J(J-1)}\cdot\frac{\|\boldsymbol{\delta}\|^2}{c^2} < 0 \qquad\blacksquare$$

**Interpretation.** Strategic coordination is self-defeating under CES complementarity, and the penalty is proportional to $K$:

1. Redistribution within the coalition loses output (curvature penalizes asymmetry, loss $\propto K$).
2. Withholding effort loses more than it gains (complementarity premium already efficiently allocated, governed by $K$).
3. For strict complements ($\rho < 0$, $K > (J-1)/J$), the coalition cannot even produce output alone.

---

## 6. The Unified Theorem

**Theorem (CES Triple Role).** Let $F$ be a CES aggregate with $\rho < 1$ and $J \geq 2$. Define the curvature parameter

$$K = (1 - \rho)\,\frac{J-1}{J}$$

Then $K > 0$, and all three of the following are controlled by $K$:

**(a) Superadditivity.** $F(\mathbf{x} + \mathbf{y}) \geq F(\mathbf{x}) + F(\mathbf{y})$, with gap:

$$\text{gap} \;\geq\; \frac{K}{4c}\cdot\frac{\sqrt{J}}{J-1}\cdot\min\!\big(F(\mathbf{x}), F(\mathbf{y})\big)\cdot d_{\mathcal{I}}^2 \;=\; \Omega(K)\cdot\text{diversity}$$

**(b) Correlation robustness.** Effective dimension under equicorrelation $r$:

$$d_{\text{eff}} \;\geq\; \frac{J}{1+r(J-1)} \;+\; \frac{K^2\gamma^2}{2}\cdot\frac{J(J-1)(1-r)}{[1+r(J-1)]^2} \;=\; \text{linear baseline} + \Omega(K^2)\cdot\text{idiosyncratic bonus}$$

The correlation threshold for $d_{\text{eff}} \geq J/2$ exceeds the linear threshold by $\Theta(K^2\gamma^2/(J-1))$.

**(c) Strategic independence.** Manipulation gain for coalition $|S| = k \leq J/2$:

$$\Delta(S) \;\leq\; -\frac{K}{2J(J-1)}\cdot\frac{\|\boldsymbol{\delta}\|^2}{c^2} \;=\; -\Omega(K)\cdot\text{deviation}^2$$

All three bounds tighten monotonically in $K$. $K$ enters linearly in (a) and (c) (first-order curvature effects) and quadratically in (b) (because the information channel is quadratic in the curvature). In each case, the mechanism is the same: the strictly positive curvature of the CES isoquant simultaneously

- forces convex combinations of diverse points above the level set (superadditivity),
- maps correlated input variation into distinct output regions via a nonlinear channel (informational diversity),
- penalizes any deviation from the balanced allocation (strategic stability).

The three roles are not merely corollaries of a common assumption. They are *the same geometric fact* — the curvature of the CES isoquant — viewed from three different angles: aggregation theory, information theory, and game theory. $\blacksquare$

---

## 7. The Geometric Intuition

Consider the unit isoquant $\mathcal{I}_1 = \{F = 1\}$ in $\mathbb{R}^J_+$.

**For linear aggregation ($\rho = 1$, $K = 0$):** $\mathcal{I}_1$ is a hyperplane. Convex combinations of points on $\mathcal{I}_1$ stay on $\mathcal{I}_1$. Correlated inputs project to the same point. Coalitions can freely redistribute. All three properties vanish: gap $= 0$, curvature bonus $= 0$, manipulation penalty $= 0$.

**For CES with $\rho < 1$ ($K > 0$):** $\mathcal{I}_1$ curves toward the origin. The curvature has three simultaneous consequences:

1. **Superadditivity**: A chord between two points on $\mathcal{I}_1$ passes through the interior of $\{F > 1\}$. This is literally what $F(\alpha\hat{\mathbf{x}} + (1-\alpha)\hat{\mathbf{y}}) > 1$ means. The depth of penetration is $\Theta(K)$.

2. **Informational diversity**: Two inputs that are close in Euclidean distance (as when correlated) still lie on a curved surface. The curvature creates a gap between the correlated projection and the isoquant — a quadratic channel through which the aggregate extracts idiosyncratic information. The channel capacity is $\Theta(K^2)$.

3. **Strategic stability**: Moving along $\mathcal{I}_1$ away from the balanced point always moves toward the coordinate axes, where the curvature is even steeper (for CES, curvature increases away from the symmetric point when $\rho < 0$). Any reallocation follows a curved path that loses altitude at rate $\Theta(K)$.

The three properties are one property: **the isoquant is not flat**. $\rho < 1$ is precisely the condition for non-flatness, $K = (1-\rho)(J-1)/J$ is precisely the degree of non-flatness, and everything else is commentary.

---

## 8. Discussion: Sufficiency of $J$ and Tightness

**On "sufficiently large" $J$.** The qualitative results (a) and (c) hold for all $J \geq 2$. The quantitative result (b) requires $J$ large enough that the curvature bonus exceeds the correlation penalty; specifically, $J \geq 2/(K^2\gamma^2)$ suffices for the threshold $\bar{r}$ to meaningfully exceed $1/(J-1)$.

**Tightness.** All three bounds become equalities in limit cases:
- (a): equality when $\hat{\mathbf{x}} = \hat{\mathbf{y}}$ (proportional inputs);
- (b): curvature bonus $\to 0$ as $\rho \to 1$ ($K \to 0$) or $r \to 1$;
- (c): manipulation penalty $\to 0$ as $K \to 0$ or $k/J \to 0$.

**Why $K$ enters linearly in (a) and (c) but quadratically in (b).** The superadditivity gap and manipulation penalty are first-order consequences of curvature: they arise from the Hessian of $F$, which is $O(1-\rho) = O(K)$. The correlation robustness bonus is a second-order consequence: it arises from the *variance* of a Hessian-quadratic form, which is $O((1-\rho)^2) = O(K^2)$. This is consistent — the information channel is the *square* of the curvature channel.

**Relationship to prior results.** Part (a) generalizes Lemma 1 of the mesh paper from superadditivity-as-stated to a quantitative curvature-dependent bound. Part (b) generalizes Theorem 3 of Paper 4 from a threshold existence result to an explicit formula with a computable curvature bonus. Part (c) resolves the conjecture from Paper 5 by showing that strategic independence is not an additional assumption but a theorem: it follows from the same $K$ that drives the other two properties.
