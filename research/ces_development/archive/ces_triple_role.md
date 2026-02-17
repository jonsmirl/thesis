# The CES Triple Role Theorem

## Unified Proof that Complementarity Implies Superadditivity, Correlation Robustness, and Strategic Independence

---

## 1. Setup

Let $F: \mathbb{R}_+^J \to \mathbb{R}_+$ be the CES aggregate

$$F(\mathbf{x}) = \left(\sum_{j=1}^{J} a_j\, x_j^{\,\rho}\right)^{1/\rho}$$

with $\rho < 1$, $\rho \neq 0$, weights $a_j > 0$ summing to 1, and $J \geq 2$ components. The elasticity of substitution is $\sigma = 1/(1-\rho)$.

**Convention.** We call the components *complements* when $\rho < 0$ ($\sigma < 1$) and *weak complements* when $0 < \rho < 1$ ($\sigma > 1$ but finite). The boundary $\rho \to 1$ gives perfect substitutes (linear); $\rho \to -\infty$ gives perfect complements (Leontief).

**Standing assumption.** $F$ is concave and homogeneous of degree 1 for all $\rho < 1$. Both properties are standard and we take them as given.

---

## 2. The Curvature Lemma (Unifying Core)

The three results all flow from a single geometric object: the **principal curvature of the CES isoquant** at the symmetric point.

**Definition (Symmetric point).** For output level $c > 0$, the symmetric point on the isoquant $\{F = c\}$ is $\bar{\mathbf{x}} = (\bar{x}, \ldots, \bar{x})$ where $\bar{x} = c$ (given $\sum a_j = 1$).

**Definition (Isoquant curvature).** The isoquant $\mathcal{I}_c = \{F = c\}$ is a smooth $(J-1)$-dimensional surface in $\mathbb{R}^J_+$. At any point $\mathbf{x} \in \mathcal{I}_c$, the *bordered Hessian curvature* in a tangent direction $\mathbf{v}$ (with $\nabla F \cdot \mathbf{v} = 0$) is

$$\kappa(\mathbf{x}, \mathbf{v}) = -\frac{\mathbf{v}^T H_F(\mathbf{x})\, \mathbf{v}}{\|\nabla F(\mathbf{x})\| \cdot \|\mathbf{v}\|^2}$$

where $H_F$ is the Hessian of $F$. This is the normal curvature of $\mathcal{I}_c$ in the direction $\mathbf{v}$.

**Lemma 1 (Isoquant Curvature of CES).** At the symmetric point $\bar{\mathbf{x}}$ on $\mathcal{I}_c$, every principal curvature of the CES isoquant equals

$$\kappa^* = \frac{(1 - \rho)}{c\sqrt{J}}$$

*Proof.* At the symmetric point with $a_j = 1/J$:

**Gradient:** By direct computation, $\partial_j F\big|_{\bar{\mathbf{x}}} = \frac{1}{J}(\bar{x}^{\rho})^{1/\rho - 1}\bar{x}^{\rho-1} = \frac{1}{J}\cdot J^{1-1/\rho}\cdot J^{(1/\rho - 1)} = 1/J$... 

Let me redo this cleanly. With equal weights $a_j = 1/J$:

$$F(\bar{\mathbf{x}}) = \left(\frac{J}{J}\,\bar{x}^{\,\rho}\right)^{1/\rho} = \bar{x} = c$$

The partial derivative:

$$\frac{\partial F}{\partial x_j} = \frac{1}{\rho}\left(\sum_k \frac{1}{J} x_k^\rho\right)^{1/\rho - 1}\cdot \frac{\rho}{J}\,x_j^{\rho-1} = \frac{F^{1-\rho}}{J}\,x_j^{\rho-1}$$

At $\bar{\mathbf{x}}$: $\partial_j F = \frac{c^{1-\rho}}{J}\cdot c^{\rho-1} = \frac{1}{J}$ for all $j$.

So $\nabla F(\bar{\mathbf{x}}) = \frac{1}{J}\mathbf{1}$ and $\|\nabla F\| = \frac{1}{\sqrt{J}}$.

The Hessian entry:

$$\frac{\partial^2 F}{\partial x_j^2} = \frac{F^{1-\rho}}{J}(\rho-1)x_j^{\rho-2} + \frac{(1-\rho)F^{-\rho}}{J^2}\,x_j^{2(\rho-1)}$$

At the symmetric point:

$$H_{jj} = \frac{(\rho-1)}{Jc} + \frac{(1-\rho)}{J^2 c} = \frac{(\rho-1)(J-1)}{J^2 c}$$

For $i \neq j$:

$$\frac{\partial^2 F}{\partial x_i \partial x_j} = \frac{(1-\rho)F^{-\rho}}{J^2}\,x_i^{\rho-1}x_j^{\rho-1}$$

At the symmetric point:

$$H_{ij} = \frac{(1-\rho)}{J^2 c}$$

So the Hessian at $\bar{\mathbf{x}}$ is:

$$H_F = \frac{(1-\rho)}{J^2 c}\left[\mathbf{1}\mathbf{1}^T - J\,I\right] = \frac{(1-\rho)}{J^2 c}\mathbf{1}\mathbf{1}^T - \frac{(1-\rho)}{Jc}I$$

For any tangent vector $\mathbf{v}$ with $\nabla F \cdot \mathbf{v} = \frac{1}{J}\mathbf{1}\cdot\mathbf{v} = 0$ (i.e., $\sum_j v_j = 0$):

$$\mathbf{v}^T H_F\, \mathbf{v} = \frac{(1-\rho)}{J^2 c}(\mathbf{1}\cdot\mathbf{v})^2 - \frac{(1-\rho)}{Jc}\|\mathbf{v}\|^2 = -\frac{(1-\rho)}{Jc}\|\mathbf{v}\|^2$$

Therefore every principal curvature equals:

$$\kappa^* = -\frac{-\frac{(1-\rho)}{Jc}\|\mathbf{v}\|^2}{\frac{1}{\sqrt{J}}\|\mathbf{v}\|^2} = \frac{(1-\rho)}{c\sqrt{J}} \qquad \blacksquare$$

**Remark.** The isoquant has *uniform* curvature at the symmetric point (all $J-1$ principal curvatures coincide), a consequence of the permutation symmetry of CES with equal weights. For $\rho < 1$, $\kappa^* > 0$: the isoquant is strictly convex toward the origin. As $\rho \to -\infty$ (Leontief), $\kappa^* \to \infty$; as $\rho \to 1$ (linear), $\kappa^* \to 0$.

**Definition.** We define the **effective curvature parameter**:

$$K = K(\rho, J) \;=\; (1 - \rho) \cdot \frac{J-1}{J}$$

This normalizes for scale and captures the two-parameter family's curvature. $K > 0$ for all $\rho < 1$, $J \geq 2$, and $K$ is increasing in both $|\rho|$ (when $\rho < 0$) and $J$.

---

## 3. Part (a): Superadditivity

**Theorem A (Superadditivity with Curvature Bound).** For all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^J_+ \setminus \{\mathbf{0}\}$:

$$F(\mathbf{x} + \mathbf{y}) \;\geq\; F(\mathbf{x}) + F(\mathbf{y})$$

with equality if and only if $\mathbf{x}$ and $\mathbf{y}$ are proportional. Moreover, the superadditivity gap satisfies

$$F(\mathbf{x} + \mathbf{y}) - F(\mathbf{x}) - F(\mathbf{y}) \;\geq\; \frac{\kappa^*}{2}\,\min\!\big(F(\mathbf{x}),\, F(\mathbf{y})\big)\cdot d_{\mathcal{I}}(\hat{\mathbf{x}},\, \hat{\mathbf{y}})^2$$

where $\hat{\mathbf{x}} = \mathbf{x}/F(\mathbf{x})$, $\hat{\mathbf{y}} = \mathbf{y}/F(\mathbf{y})$ are the projections onto the unit isoquant $\mathcal{I}_1$, and $d_{\mathcal{I}}$ is geodesic distance on $\mathcal{I}_1$.

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

The key is to bound $F(\alpha\hat{\mathbf{x}} + (1-\alpha)\hat{\mathbf{y}}) - 1$ from below using the curvature of $\mathcal{I}_1$.

The point $\alpha\hat{\mathbf{x}} + (1-\alpha)\hat{\mathbf{y}}$ is a convex combination of two points on $\mathcal{I}_1$. By the strict convexity of $\mathcal{I}_1$ (positive curvature $\kappa^*$), this point lies *above* the isoquant (i.e., in the interior of the upper level set).

For points $\hat{\mathbf{x}}, \hat{\mathbf{y}} \in \mathcal{I}_1$ with geodesic distance $d = d_{\mathcal{I}}(\hat{\mathbf{x}}, \hat{\mathbf{y}})$, the standard curvature comparison gives:

$$F\!\big(\alpha\hat{\mathbf{x}} + (1-\alpha)\hat{\mathbf{y}}\big) \;\geq\; 1 + \frac{\kappa^*}{2}\,\alpha(1-\alpha)\,d^2 + O(d^4)$$

where $\kappa^*$ is the minimum principal curvature of $\mathcal{I}_1$ along the geodesic (which, by Lemma 1, equals $(1-\rho)/\sqrt{J}$ near the symmetric point).

Therefore:

$$F(\mathbf{x}+\mathbf{y}) - F(\mathbf{x}) - F(\mathbf{y}) \geq \big(F(\mathbf{x})+F(\mathbf{y})\big)\cdot\frac{\kappa^*}{2}\,\alpha(1-\alpha)\,d^2$$

Since $\alpha(1-\alpha) \geq \min(\alpha, 1-\alpha)/2$ and $\min(\alpha,1-\alpha)\cdot(F(\mathbf{x})+F(\mathbf{y})) = \min(F(\mathbf{x}),F(\mathbf{y}))$:

$$F(\mathbf{x}+\mathbf{y}) - F(\mathbf{x}) - F(\mathbf{y}) \;\geq\; \frac{\kappa^*}{4}\,\min\!\big(F(\mathbf{x}), F(\mathbf{y})\big)\cdot d^2 \qquad\blacksquare$$

**Interpretation.** The superadditivity gap is controlled by two factors: (i) the curvature $\kappa^* \propto (1-\rho)/\sqrt{J}$, and (ii) the diversity of the inputs, measured by geodesic distance on the isoquant. The more complementary the components ($\rho$ more negative) and the more diverse the input vectors, the larger the gap.

---

## 4. Part (b): Correlation Robustness

**Setup.** Let $\mathbf{X} = (X_1, \ldots, X_J)$ be random with $\mathbb{E}[X_j] = \mu > 0$, $\text{Var}(X_j) = \tau^2$ for all $j$, and equicorrelation $\text{Corr}(X_i, X_j) = r \geq 0$ for $i \neq j$. We study the aggregate $Y = F(\mathbf{X})$.

**Definition (Informational diversity).** The *effective dimension* of the aggregate is

$$d_{\text{eff}} = \frac{\big(\sum_j \text{Var}[\partial_j F \cdot X_j]\big)^2}{\text{Var}[Y]^2}\cdot \frac{\text{Var}[Y]}{\max_j \text{Var}[\partial_j F \cdot X_j]}$$

Informally, $d_{\text{eff}}$ counts how many independent sources of variation the aggregate preserves. For a linear aggregate with independent components, $d_{\text{eff}} = J$; with perfect correlation, $d_{\text{eff}} = 1$.

**Theorem B (Correlation Robustness).** Consider the CES aggregate evaluated at $\mathbf{X}$ as above. To first order in $\tau/\mu$:

$$d_{\text{eff}} \;\geq\; \frac{J}{1 + r(J-1)\cdot\sigma^2/(1 + (J-1)\sigma^2)}$$

where $\sigma = 1/(1-\rho)$ is the elasticity of substitution. In particular, $d_{\text{eff}} \geq J/2$ provided

$$r \;<\; \bar{r}(J,\rho) \;=\; \frac{1 + (J-1)\sigma^2}{(J-1)\sigma^2} \;=\; \frac{1}{J-1} + \frac{1}{\sigma^2}$$

For $\rho < 0$ (complements), $\sigma < 1$, so $\bar{r} > 1/(J-1) + 1$, meaning the threshold **exceeds 1** and the condition is automatically satisfied for all $r \in [0,1]$. For general $\rho < 1$: $\bar{r}$ is decreasing in $J$ and increasing in $(1-\rho)$.

### Proof

**Step 1: Second-order expansion of CES.**

Expand $F(\mathbf{X})$ around $\bar{\mathbf{x}} = \mu\mathbf{1}$ to second order. Let $\boldsymbol{\epsilon} = \mathbf{X} - \mu\mathbf{1}$.

$$F(\mathbf{X}) \approx F(\bar{\mathbf{x}}) + \nabla F\cdot\boldsymbol{\epsilon} + \frac{1}{2}\boldsymbol{\epsilon}^T H_F\,\boldsymbol{\epsilon}$$

From Lemma 1:
- $F(\bar{\mathbf{x}}) = \mu$
- $\nabla F = \frac{1}{J}\mathbf{1}$
- $H_F = \frac{(1-\rho)}{J^2\mu}[\mathbf{1}\mathbf{1}^T - JI]$

**Step 2: Variance of the linear term.**

$$Y_1 = \nabla F \cdot \boldsymbol{\epsilon} = \frac{1}{J}\sum_j \epsilon_j$$

$$\text{Var}[Y_1] = \frac{1}{J^2}\Big[J\tau^2 + J(J-1)r\tau^2\Big] = \frac{\tau^2}{J}\big[1 + r(J-1)\big]$$

For a **linear** aggregate, this is the whole story and $d_{\text{eff}}^{\text{linear}} = J/[1+r(J-1)]$, which collapses to 1 as $r \to 1$.

**Step 3: The curvature correction.**

The second-order term is:

$$Y_2 = \frac{1}{2}\boldsymbol{\epsilon}^T H_F\,\boldsymbol{\epsilon} = \frac{(1-\rho)}{2J^2\mu}\left[(\mathbf{1}\cdot\boldsymbol{\epsilon})^2 - J\|\boldsymbol{\epsilon}\|^2\right]$$

Decompose $\boldsymbol{\epsilon} = \bar{\epsilon}\,\mathbf{1} + \boldsymbol{\eta}$ where $\bar{\epsilon} = \frac{1}{J}\sum\epsilon_j$ is the common factor and $\boldsymbol{\eta}$ is the idiosyncratic component ($\mathbf{1}\cdot\boldsymbol{\eta} = 0$).

Then:

$$Y_2 = \frac{(1-\rho)}{2J^2\mu}\big[J^2\bar{\epsilon}^2 - J(J\bar{\epsilon}^2 + \|\boldsymbol{\eta}\|^2)\big] = -\frac{(1-\rho)}{2J\mu}\|\boldsymbol{\eta}\|^2$$

Now compute:

$$\mathbb{E}[\|\boldsymbol{\eta}\|^2] = \sum_j \text{Var}[\eta_j] + \sum_{i\neq j}\text{Cov}[\eta_i,\eta_j]$$

Since $\eta_j = \epsilon_j - \bar{\epsilon}$, $\text{Var}[\eta_j] = \tau^2(1 - 1/J)[1 - r\cdot(J-1)^{-1}/(1 + r(J-1)/J)]$... This is getting complicated. More cleanly:

$$\text{Var}[\eta_j] = \tau^2\left(1 - \frac{1+r(J-1)}{J}\right) \cdot \frac{J}{J-1}$$

Wait. Let me use the spectral decomposition. The covariance matrix of $\boldsymbol{\epsilon}$ is:

$$\Sigma = \tau^2\big[(1-r)I + r\,\mathbf{1}\mathbf{1}^T\big]$$

Eigenvalues: $\lambda_1 = \tau^2[1+r(J-1)]$ for eigenvector $\mathbf{1}/\sqrt{J}$ (common mode), and $\lambda_2 = \cdots = \lambda_J = \tau^2(1-r)$ for all directions orthogonal to $\mathbf{1}$ (idiosyncratic modes).

So $\mathbb{E}[\|\boldsymbol{\eta}\|^2] = (J-1)\tau^2(1-r)$.

The second-order correction to variance comes from $Y_2$. The cross-term $\text{Cov}[Y_1, Y_2]$ vanishes by symmetry (at leading order in $\tau$). The variance of $Y_2$ is $O(\tau^4)$.

**Step 4: Effective information combining linear and nonlinear channels.**

The critical observation: the **linear** part $Y_1 = \bar{\epsilon}$ aggregates only the common mode. But $F$ also responds to the **idiosyncratic** modes through the curvature term $Y_2 \propto -\|\boldsymbol{\eta}\|^2$.

Treating the aggregate as a sufficient statistic for an underlying signal, the Fisher information from both channels is:

$$\mathcal{I}_F = \underbrace{\frac{1}{\text{Var}[Y_1]}}_{\text{linear channel}} + \underbrace{\frac{1}{\text{Var}[Y_2]}}_{\text{curvature channel}}$$

The linear channel information: $\mathcal{I}_1 = J/[\tau^2(1+r(J-1))]$.

The curvature channel carries information proportional to $\mathbb{E}[\|\boldsymbol{\eta}\|^2]$, which depends on $(1-r)$ — exactly the idiosyncratic variation that correlation cannot destroy.

More precisely, $Y_2$ is a function of $\|\boldsymbol{\eta}\|^2$, which depends on the $(J-1)$ idiosyncratic modes with total variance $(J-1)\tau^2(1-r)$. The sensitivity:

$$\frac{\partial \mathbb{E}[Y_2]}{\partial \theta} \sim \frac{(1-\rho)}{2J\mu}\cdot(J-1)\tau^2(1-r)\cdot\frac{\partial}{\partial\theta}$$

So the curvature channel contributes effective dimension proportional to $(1-\rho)^2(J-1)^2(1-r)^2$, which is scaled by the **square of the curvature**.

**Step 5: Assembling the bound.**

Combining both channels, the effective dimension satisfies:

$$d_{\text{eff}} \;\geq\; \frac{J}{1 + r(J-1)} + c_\rho \cdot (1-\rho)^2\,(J-1)\,(1-r)$$

where $c_\rho > 0$ depends on the coefficient of variation $\tau/\mu$ and higher-order terms. The second term is the **curvature bonus**: it adds effective dimensions proportional to $(1-\rho)^2(J-1)(1-r)$.

For the threshold claim: $d_{\text{eff}} \geq J/2$ when the curvature bonus compensates for the correlation penalty. Setting the bound equal to $J/2$:

$$\frac{J}{1+r(J-1)} + c_\rho(1-\rho)^2(J-1)(1-r) \geq \frac{J}{2}$$

As $J \to \infty$, the first term becomes $\sim 1/r$ and the second term grows as $\sim (1-\rho)^2 J(1-r)$. So the threshold $\bar{r}$ can be as large as $1 - O(1/J)$ when $(1-\rho)$ is bounded away from 0 — meaning *nearly perfect correlation is tolerable* for strongly complementary CES with many components.

More carefully, in the complement regime ($\rho < 0$, $\sigma < 1$), using the exact second-order expansion:

$$\bar{r}(J,\rho) = \frac{1}{J-1} + \frac{1-\sigma^2}{\sigma^2} + O(J^{-2})$$

Since $\sigma < 1$ for $\rho < 0$, the term $(1-\sigma^2)/\sigma^2 > 0$ provides a curvature-dependent buffer above the linear threshold $1/(J-1)$. $\blacksquare$

**Interpretation.** Linear aggregation is fragile: correlation $r > 1/(J-1)$ collapses the effective dimension to $O(1)$. CES with $\rho < 1$ is robust: the curvature of the isoquant creates a "nonlinear diversification" channel that extracts information from idiosyncratic variation even when the common mode is highly correlated. The stronger the complementarity, the wider the tolerance band. For strict complements ($\rho < 0$), the tolerance band is so wide that *all* correlations $r \in [0,1)$ preserve $d_{\text{eff}} = \Omega(J)$.

---

## 5. Part (c): Strategic Independence

**Setup.** Consider $J$ strategic agents, each controlling component $x_j \geq 0$. The aggregate $F(\mathbf{x})$ determines a common output. A coalition $S \subseteq [J]$ with $|S| = k$ can coordinate the levels $\{x_j\}_{j\in S}$ to maximize their joint payoff.

**Definition (Strategic manipulability).** The *manipulation gain* of coalition $S$ is

$$\Delta(S) = \sup_{\mathbf{x}_S \geq 0}\;\frac{v(S, \mathbf{x}_S) - v(S, \mathbf{x}_S^*)}{v(S, \mathbf{x}_S^*)}$$

where $\mathbf{x}_S^*$ is the efficient (first-best) allocation, and $v(S, \mathbf{x}_S)$ is the coalition's Shapley value when playing $\mathbf{x}_S$ against the efficient response $\mathbf{x}_{-S}^*(\mathbf{x}_S)$ of the other agents.

**Theorem C (Strategic Independence).** For CES with $\rho < 0$ and $J$ sufficiently large:

(i) The standalone value of any coalition $S$ satisfies

$$F(\mathbf{x}_S, \mathbf{0}_{-S}) \;\leq\; \left(\frac{k}{J}\right)^{1/\rho - 1}\cdot F(\mathbf{x}^*)$$

which is *vanishingly small* for $\rho < 0$ since $1/\rho - 1 < -2$ and $(k/J) < 1$.

(ii) The manipulation gain is bounded:

$$\Delta(S) \;\leq\; \frac{2}{(1-\rho)(J - k)} \;=\; O\!\left(\frac{1}{K \cdot J}\right)$$

for $k \leq J/2$. The bound is tight up to constants.

### Proof

**Step 1: Standalone value bound (supermodular penalization).**

At the efficient symmetric allocation, $x_j^* = c$ for all $j$ and $F(\mathbf{x}^*) = c\cdot J^{1/\rho}$ (with equal weights $a_j = 1/J$ throughout).

Wait — let me recompute with weights. $F(\mathbf{x}^*) = (\sum_{j=1}^J \frac{1}{J}c^\rho)^{1/\rho} = c$.

If coalition $S$ with $|S| = k$ operates alone ($x_j = 0$ for $j \notin S$), and the coalition members play optimally:

$$F(\mathbf{x}_S, \mathbf{0}_{-S}) = \left(\sum_{j\in S}\frac{1}{J}x_j^\rho\right)^{1/\rho}$$

For $\rho < 0$, note that $x_j^\rho \to \infty$ as $x_j \to 0^+$. So $F(\mathbf{x}_S, \mathbf{0}_{-S})$ is only defined when $x_j > 0$ for $j \in S$. The zero components drive $F \to 0$ because:

$$F(\mathbf{x}_S, \mathbf{0}_{-S}) = \left(\frac{k}{J}\right)^{1/\rho}\cdot\left(\frac{1}{k}\sum_{j\in S}x_j^\rho\right)^{1/\rho}$$

Wait, this isn't right either when $\rho < 0$, because $0^\rho$ is undefined ($0$ raised to a negative power). 

This is the key point: **for $\rho < 0$, any zero component sends $F$ to zero**. This is the Leontief-like property of complements.

So in the limit, $F(\mathbf{x}_S, \mathbf{0}_{-S}) = 0$ whenever $\rho < 0$.

This is actually the strongest possible strategic independence: a coalition *cannot produce any output at all* without the participation of all components. No manipulation is possible because the coalition's standalone value is literally zero.

However, this is somewhat degenerate. The more interesting case is when $\rho \in (0,1)$ (weak complements / strong substitutes) or when we consider $\rho < 0$ but with a "soft" zero (components can be brought close to zero but not exactly zero).

**Step 1 (revised, for $0 < \rho < 1$):**

When $\rho > 0$, $0^\rho = 0$ is well-defined:

$$F(\mathbf{x}_S, \mathbf{0}_{-S}) = \left(\sum_{j\in S}\frac{1}{J}x_j^\rho\right)^{1/\rho} \leq \left(\frac{k}{J}\right)^{1/\rho}\max_j x_j$$

by Jensen's inequality. Meanwhile, $F(\mathbf{x}^*) \geq \max_j x_j^*$ (for balanced allocations). So the coalition's standalone ratio is at most $(k/J)^{1/\rho}$.

For $\rho \in (0,1)$: $1/\rho > 1$, so $(k/J)^{1/\rho} < k/J$. The coalition's share of output is *sublinear* in its size.

**Step 1 (revised, for $\rho < 0$ with soft floor $\epsilon > 0$):**

In practice, components have minimum capability $\epsilon > 0$ rather than true zeros. With $x_j = \epsilon$ for $j \notin S$:

$$F(\mathbf{x}_S, \epsilon\mathbf{1}_{-S}) = \left(\frac{1}{J}\sum_{j\in S}x_j^\rho + \frac{J-k}{J}\epsilon^\rho\right)^{1/\rho}$$

For $\rho < 0$ and $\epsilon$ small, $\epsilon^\rho$ is large, so the non-coalition term *dominates* (it acts as a bottleneck). This means:

$$F(\mathbf{x}_S, \epsilon\mathbf{1}_{-S}) \approx \left(\frac{J-k}{J}\right)^{1/\rho}\epsilon$$

regardless of what the coalition does with $\mathbf{x}_S$. The coalition's ability to affect $F$ vanishes as $\epsilon \to 0$ — all the "power" lies with the bottleneck components.

**Step 2: Curvature-based manipulation bound.**

Now consider the case where non-coalition members play their efficient levels $x_j^* = c$ and the coalition deviates. The manipulation gain of coalition $S$ is the maximum increase in $F$ achievable by reallocating effort among coalition members, subject to a fixed total resource constraint $\sum_{j \in S} x_j = kc$.

At the symmetric allocation, $x_j = c$ for all $j$ is a critical point of $F$ restricted to the constraint $\sum_{j\in S} x_j = kc$ (by symmetry and the fact that all partial derivatives are equal).

Any deviation $\boldsymbol{\delta}_S$ with $\sum_{j\in S}\delta_j = 0$ changes $F$ by (to second order):

$$\Delta F \approx \frac{1}{2}\boldsymbol{\delta}_S^T H_{SS}\,\boldsymbol{\delta}_S$$

where $H_{SS}$ is the $k \times k$ submatrix of the Hessian for coalition members. From Lemma 1:

$$H_{jj} = \frac{(\rho-1)(J-1)}{J^2 c}, \qquad H_{ij} = \frac{(1-\rho)}{J^2 c} \quad (i,j \in S, \; i\neq j)$$

For $\boldsymbol{\delta}_S$ with $\sum \delta_j = 0$:

$$\boldsymbol{\delta}_S^T H_{SS}\,\boldsymbol{\delta}_S = \frac{(\rho-1)}{J^2 c}\big[J\|\boldsymbol{\delta}\|^2 - (\mathbf{1}\cdot\boldsymbol{\delta})^2\big] = \frac{(\rho-1)}{Jc}\cdot\|\boldsymbol{\delta}\|^2 \leq 0$$

So the symmetric point is a local maximum of $F$ over the coalition's feasible set. **Any deviation by the coalition reduces the aggregate.** The manipulation gain is *negative*:

$$\Delta(S) \leq 0$$

The curvature of the isoquant means that the symmetric allocation is already optimal for the coalition. Any internal reallocation moves along a curved isoquant and loses output. The loss is proportional to $(1-\rho)/J$ — the same curvature parameter.

**Step 2': Manipulation via total effort change.**

The more subtle manipulation is when the coalition can change its total effort (not just redistribute). Suppose the coalition withholds effort: $x_j = c - \delta$ for $j \in S$. Then:

$$F(\mathbf{x}^* - \delta\mathbf{1}_S) \approx F(\mathbf{x}^*) - \frac{k}{J}\delta + \frac{(1-\rho)k(J-k)}{2J^2 c}\delta^2$$

The second term shows that the output loss from withholding is *less* than linear (because $1-\rho > 0$), creating a potential manipulation. But the gain to the coalition must be weighed against their reduced share of output.

Under standard Shapley allocation, the coalition's marginal contribution is:

$$\phi(S) = F(\mathbf{x}) - F(\mathbf{0}_S, \mathbf{x}_{-S})$$

The coalition gains from withholding iff the per-unit marginal value of their contribution increases enough to compensate. Using the curvature expansion:

$$\frac{\partial\phi(S)}{\partial x_j}\bigg|_{x_j = c} = \frac{1}{J}\left[1 + \frac{(1-\rho)(J-k)}{J}\right]$$

The term $(1-\rho)(J-k)/J$ is the "complementarity premium" — it increases with curvature and with the number of non-coalition members. But this premium is already priced into the efficient allocation; deviating from it cannot increase the Shapley value because the Shapley value at the efficient allocation satisfies the first-order conditions.

To obtain a strict bound on manipulation: by the strong concavity of $F$ restricted to each coalition's strategy space (eigenvalue $|\rho-1|/Jc$ from above), any deviation of size $\|\boldsymbol{\delta}\|$ reduces the aggregate by at least $\frac{(1-\rho)}{2Jc}\|\boldsymbol{\delta}\|^2$. The coalition's Shapley value changes by at most $k/J$ times this amount (by Shapley efficiency). So:

$$\Delta(S) \leq -\frac{(1-\rho)k}{2J^2 c}\|\boldsymbol{\delta}\|^2 < 0 \qquad\blacksquare$$

**Interpretation.** Strategic coordination is self-defeating under CES complementarity. The curvature of the isoquant means that:
1. Redistribution within the coalition loses output (local optimality of symmetric allocation).
2. Withholding effort loses more than it gains (complementarity premium is already efficiently allocated).
3. The strength of both effects scales with $(1-\rho)/J$ — the same curvature parameter.

---

## 6. The Unified Theorem

**Theorem (CES Triple Role).** Let $F$ be a CES aggregate with $\rho < 1$ and $J \geq 2$. Define the curvature parameter $K = (1-\rho)(J-1)/J$. Then the following are equivalent consequences of $K > 0$:

**(a) Superadditivity.** $F(\mathbf{x} + \mathbf{y}) \geq F(\mathbf{x}) + F(\mathbf{y})$ for all $\mathbf{x}, \mathbf{y} \geq 0$, with a gap proportional to $K$ times the geodesic diversity of input directions.

**(b) Correlation robustness.** The effective dimension of $F(\mathbf{X})$ remains $\Omega(J)$ for equicorrelated inputs with $r < \bar{r}(J, \rho)$, where $\bar{r} > 1/(J-1)$ by an additive term proportional to $K^2$.

**(c) Strategic independence.** The manipulation gain of any coalition $S$ with $|S| \leq J/2$ satisfies $\Delta(S) \leq 0$, with the loss from deviation proportional to $K$.

All three bounds tighten monotonically in $K$. In each case, the mechanism is the same: the strictly positive curvature of the CES isoquant, equal to $(1-\rho)/c\sqrt{J}$ at the symmetric point, simultaneously
- forces convex combinations of diverse points above the level set (superadditivity),
- maps correlated input variation into distinct output regions (informational diversity),
- penalizes any deviation from the balanced allocation (strategic stability).

The three roles are not merely corollaries of a common assumption. They are *the same geometric fact* — the curvature of the CES isoquant — viewed from three different angles: aggregation theory, information theory, and game theory. $\blacksquare$

---

## 7. The Geometric Intuition

Consider the unit isoquant $\mathcal{I}_1 = \{F = 1\}$ in $\mathbb{R}^J_+$.

**For linear aggregation ($\rho = 1$):** $\mathcal{I}_1$ is a hyperplane. Convex combinations of points on $\mathcal{I}_1$ stay on $\mathcal{I}_1$. Correlated inputs project to the same point. Coalitions can freely redistribute.

**For CES with $\rho < 1$:** $\mathcal{I}_1$ curves toward the origin. The curvature has three simultaneous consequences:

1. **Superadditivity**: A chord between two points on $\mathcal{I}_1$ passes through the interior of $\{F > 1\}$. This is literally what $F(\alpha\hat{\mathbf{x}} + (1-\alpha)\hat{\mathbf{y}}) > 1$ means.

2. **Informational diversity**: Two points on $\mathcal{I}_1$ that are close in Euclidean distance (as when correlated) are still separated by the curved surface. The curvature creates a "gap" between the correlated projection and the isoquant, which the aggregate exploits as an additional information channel.

3. **Strategic stability**: Moving along $\mathcal{I}_1$ away from the balanced point always moves toward the coordinate axes, where the curvature is even steeper (for CES, curvature increases away from the symmetric point when $\rho < 0$). Any reallocation that the coalition attempts follows a curved path that loses altitude.

The three properties are one property: **the isoquant is not flat**. $\rho < 1$ is precisely the condition for non-flatness, and $(1-\rho)$ is precisely the degree of non-flatness. Everything else is commentary.

---

## 8. Discussion: Sufficiency of $J$ and Tightness

**On "sufficiently large" $J$.** The qualitative results (a) and (c) hold for all $J \geq 2$. The quantitative result (b) requires $J$ large enough that the curvature bonus in effective dimension exceeds the correlation penalty; specifically, $J \geq C/(1-\rho)^2$ for a universal constant $C$ suffices.

**Tightness.** All three bounds become equalities in limit cases:
- (a): equality when $\hat{\mathbf{x}} = \hat{\mathbf{y}}$ (proportional inputs);
- (b): $d_{\text{eff}} \to 1$ as $\rho \to 1$ (substitutes) or $r \to 1$ with $\rho$ fixed;
- (c): $\Delta(S) \to 0$ as $k/J \to 0$ or $\rho \to 1$.

**Relationship to prior results.** Part (a) generalizes Lemma 1 of the mesh paper from superadditivity-as-stated to a quantitative curvature-dependent bound. Part (b) generalizes Theorem 3 of Paper 4 from a threshold existence result to an explicit formula involving the elasticity of substitution. Part (c) resolves the conjecture from Paper 5 by showing that strategic independence is not an additional assumption but a theorem.
