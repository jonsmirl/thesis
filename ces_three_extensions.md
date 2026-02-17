# Three Consequences of the Bridge Equation

## Parametric Lyapunov Families, Monotone Comparative Statics, and Canard Duration

*Extension to the Unified CES Framework, Parts 7 and 9*

---

## Result 1: Parametric Lyapunov Families Indexed by Gain Elasticity

### Statement

**Proposition E.1** (Closed-form tree coefficients under power-law gains). *Suppose all gain functions are power laws: $\phi_n(z) = a_n z^{\beta_n}$ with $\beta_n > 0$. Then the tree coefficients of the Lyapunov function $V = \sum_n c_n D_{KL}(F_n \| \bar{F}_n)$ on the slow manifold are*

$$c_n = \frac{P_{\text{cycle}}\,\sigma_{n-1}}{\beta_n\,\sigma_n\,J\,\bar{F}_n}$$

*and the Lyapunov function takes the form*

$$V = \frac{P_{\text{cycle}}}{J}\sum_{n=1}^{N}\frac{\sigma_{n-1}}{\beta_n\,\sigma_n}\;g\!\left(\frac{F_n}{\bar{F}_n}\right)$$

*where $g(z) = z - 1 - \log z \geq 0$. Under uniform damping $\sigma_n = \sigma$, this simplifies to*

$$V = \frac{P_{\text{cycle}}}{\sigma J}\sum_{n=1}^{N}\frac{1}{\beta_n}\;g\!\left(\frac{F_n}{\bar{F}_n}\right)$$

*a one-parameter-per-level family indexed by the gain elasticities $\boldsymbol{\beta} = (\beta_1, \ldots, \beta_N)$.*

### Proof

From Theorem 7.2, $c_n = P_{\text{cycle}}/k_{n,n-1}$ where $k_{n,n-1} = \phi_n'(\bar{F}_{n-1})\bar{F}_{n-1}/|\sigma_{n-1}|$ is the NGM cross-level entry.

For power-law $\phi_n(z) = a_n z^{\beta_n}$:

$$\phi_n'(\bar{F}_{n-1})\cdot\bar{F}_{n-1} = a_n\beta_n\bar{F}_{n-1}^{\beta_n} = \beta_n\,\phi_n(\bar{F}_{n-1})$$

Using the equilibrium condition $\phi_n(\bar{F}_{n-1}) = \sigma_n J \bar{F}_n$ (from $\phi_n(F_{n-1})/J = \sigma_n F_n$ at steady state):

$$\phi_n'(\bar{F}_{n-1})\cdot\bar{F}_{n-1} = \beta_n\,\sigma_n\,J\,\bar{F}_n$$

Therefore:

$$k_{n,n-1} = \frac{\beta_n\,\sigma_n\,J\,\bar{F}_n}{\sigma_{n-1}}, \qquad c_n = \frac{P_{\text{cycle}}\,\sigma_{n-1}}{\beta_n\,\sigma_n\,J\,\bar{F}_n}$$

On the slow manifold, $D_{KL}(F_n \| \bar{F}_n) = \bar{F}_n\,g(F_n/\bar{F}_n)$, so:

$$c_n\,D_{KL} = \frac{P_{\text{cycle}}\,\sigma_{n-1}}{\beta_n\,\sigma_n\,J\,\bar{F}_n}\cdot\bar{F}_n\,g(F_n/\bar{F}_n) = \frac{P_{\text{cycle}}\,\sigma_{n-1}}{\beta_n\,\sigma_n\,J}\;g(F_n/\bar{F}_n) \qquad \blacksquare$$

### The Logistic Case

For logistic gain $\phi_n(z) = r_n z(1 - z/K_n)$, the elasticity at equilibrium is not constant but depends on the utilization ratio $u_n = \bar{F}_{n-1}/K_n$:

$$\varepsilon_{T_n} = \frac{\phi_n'(\bar{F}_{n-1})\bar{F}_{n-1}}{\phi_n(\bar{F}_{n-1})} = \frac{1 - 2u_n}{1 - u_n}$$

The tree coefficient becomes:

$$c_n = \frac{P_{\text{cycle}}\,\sigma_{n-1}(1 - u_n)}{\sigma_n\,J\,\bar{F}_n(1 - 2u_n)}$$

This has a pole at $u_n = 1/2$: as the upstream level approaches half its carrying capacity, $c_n \to \infty$. The Lyapunov weight at level $n$ diverges, signaling that perturbations at that level dominate the welfare loss. This is the approach to the logistic peak — the system is maximally sensitive to perturbations near the inflection point of the S-curve.

**Stability requires $u_n < 1/2$** (operating below the logistic inflection). If $u_n > 1/2$, the elasticity $\varepsilon_{T_n}$ goes negative, the tree coefficient changes sign, and $V$ ceases to be a Lyapunov function. This is the precise condition for the logistic bifurcation within the hierarchical framework.

### Policy Implication: Welfare Loss Decomposition

The Lyapunov function $V$ measures total welfare distance from equilibrium. Its decomposition $V = \sum_n c_n D_{KL}(F_n \| \bar{F}_n)$ attributes welfare loss to each level, weighted by $c_n$.

Under power-law gains with uniform damping, the contribution of level $n$ to welfare loss is proportional to $g(F_n/\bar{F}_n)/\beta_n$. Levels with **inelastic gain functions** ($\beta_n$ small) contribute more per unit of deviation. This identifies the institutional reform with the highest welfare return: **increase the gain elasticity at the level with the largest $c_n\,\bar{F}_n\,g(F_n/\bar{F}_n)$ contribution.**

Comparing two policy regimes $\boldsymbol{\beta}$ and $\boldsymbol{\beta}'$ at the same perturbation $\mathbf{F}$:

$$V(\boldsymbol{\beta}') - V(\boldsymbol{\beta}) = \frac{P_{\text{cycle}}}{\sigma J}\sum_n\left(\frac{1}{\beta_n'} - \frac{1}{\beta_n}\right)g(F_n/\bar{F}_n)$$

If $\beta_n' > \beta_n$ at level $n$ and $\beta_k' = \beta_k$ elsewhere: the welfare loss decreases by $\frac{P_{\text{cycle}}}{\sigma J}\left(\frac{1}{\beta_n} - \frac{1}{\beta_n'}\right)g(F_n/\bar{F}_n)$. This is largest for the level $n$ with the largest $g(F_n/\bar{F}_n)/\beta_n$ — the level that is both far from equilibrium and institutionally rigid.

---

## Result 2: Monotone Comparative Statics on the Bridge Matrix

### Statement

**Proposition E.2** (Monotonicity of the supply rate). *The Bridge matrix entry $W_{nn} = P_{\text{cycle}}/(|\sigma_n|\,\varepsilon_{T_n})$ satisfies:*

*(i) $\partial W_{nn}/\partial \sigma_n < 0$: increasing damping at level $n$ strictly decreases the supply rate at level $n$.*

*(ii) $\partial W_{nn}/\partial \varepsilon_{T_n} < 0$: increasing the gain elasticity at level $n$ strictly decreases the supply rate at level $n$.*

*(iii) $W_{nn}$ is independent of $\sigma_m$ and $\varepsilon_{T_m}$ for $m \neq n$ (except through $P_{\text{cycle}}$, which depends on all levels).*

*Consequently, the geometric-to-Lyapunov curvature ratio $W_{nn}^{-1} = |\sigma_n|\varepsilon_{T_n}/P_{\text{cycle}}$ is strictly increasing in both $\sigma_n$ and $\varepsilon_{T_n}$.*

### Proof

(i) $W_{nn} = P_{\text{cycle}}/(|\sigma_n|\varepsilon_{T_n})$, so $\partial W_{nn}/\partial\sigma_n = -P_{\text{cycle}}/(\sigma_n^2\varepsilon_{T_n}) < 0$ since all factors are positive.

(ii) $\partial W_{nn}/\partial\varepsilon_{T_n} = -P_{\text{cycle}}/(|\sigma_n|\varepsilon_{T_n}^2) < 0$.

(iii) The only dependence on other levels enters through $P_{\text{cycle}} = \prod_m k_{m+1,m}$, which is a product over all edges. $\blacksquare$

### What This Means for Adjustment Dynamics

The Bridge equation $\nabla^2\Phi|_{\text{slow}} = W^{-1}\nabla^2 V$ relates two curvature tensors at each level. Large $W_{nn}^{-1}$ means the free-energy curvature (geometric force toward equilibrium) is large relative to the Lyapunov curvature (effective restoring force). The monotonicity results give three robust comparative statics:

**Proposition E.3** (Damping-speed tradeoff). *For the reduced system on the slow manifold:*

*(i) The local convergence rate at level $n$ is $\sigma_n/\varepsilon_n$, strictly increasing in $\sigma_n$.*

*(ii) The equilibrium output is $\bar{F}_n = \phi_n(\bar{F}_{n-1})/(\sigma_n J)$, strictly decreasing in $\sigma_n$.*

*(iii) The Lyapunov dissipation rate at level $n$ near equilibrium is*

$$-\dot{V}_n \approx c_n\sigma_n\,\frac{(\delta F_n)^2}{\bar{F}_n} = \frac{P_{\text{cycle}}\,\sigma_{n-1}}{\beta_n\,J\,\bar{F}_n}\cdot\frac{(\delta F_n)^2}{\bar{F}_n}$$

*(under power-law gains), which is increasing in $\sigma_{n-1}$ (upstream damping) and decreasing in $\beta_n$ (gain elasticity) and $\bar{F}_n$ (equilibrium output).*

*Proof.* (i) The eigenvalue of the reduced Jacobian is $-\sigma_n/\varepsilon_n$ (Proposition 2.2 restricted to the aggregate mode). (ii) Direct from the equilibrium condition. (iii) $\dot{V}_n = -c_n\sigma_n(F_n - \bar{F}_n)^2/F_n \approx -c_n\sigma_n(\delta F_n)^2/\bar{F}_n$. Substituting $c_n$ from Proposition E.1 and noting that $c_n\sigma_n = P_{\text{cycle}}\sigma_{n-1}/(\beta_n J\bar{F}_n)$, which is independent of $\sigma_n$. $\blacksquare$

The last result is striking: **the Lyapunov dissipation rate at level $n$ does not depend on $\sigma_n$ itself** (under power-law gains). The $\sigma_n$ in $c_n$ exactly cancels the $\sigma_n$ in the dissipation formula. Instead, the dissipation depends on $\sigma_{n-1}$ — the damping of the *upstream* level.

**Policy interpretation.** Increasing damping at level $n$ does two things: (a) speeds up local convergence (higher $\sigma_n/\varepsilon_n$), but (b) lowers the equilibrium ceiling ($\bar{F}_n \propto 1/\sigma_n$). These effects exactly cancel in the Lyapunov dissipation, making the effective adjustment at level $n$ controlled by upstream conditions ($\sigma_{n-1}$) rather than local conditions ($\sigma_n$).

**The robust recommendation:** To accelerate adjustment at level $n$, increase the gain elasticity $\beta_n$ or reduce upstream damping $\sigma_{n-1}$ — not local damping $\sigma_n$.

### Partial Ordering on the Moduli Space

**Corollary E.4.** *Define the partial order $\boldsymbol{\beta} \succeq \boldsymbol{\beta}'$ iff $\beta_n \geq \beta_n'$ for all $n$. Then:*

*(i) $W_{nn}(\boldsymbol{\beta}) \leq W_{nn}(\boldsymbol{\beta}')$ for all $n$ (the Bridge tightens).*

*(ii) $V(\boldsymbol{\beta}) \leq V(\boldsymbol{\beta}')$ at every non-equilibrium state (welfare loss decreases).*

*(iii) If further $\sigma_n = \sigma$ for all $n$: $V(\boldsymbol{\beta}) = V(\boldsymbol{\beta}') \cdot (\sum_n \beta_n'^{-1}g_n)/(\sum_n \beta_n^{-1}g_n)$ where $g_n = g(F_n/\bar{F}_n)$, which is $\leq 1$.*

*Proof.* (i) Direct from Proposition E.2(ii). (ii) From $c_n \propto 1/\beta_n$, higher $\beta_n$ gives lower weight $c_n$, hence lower $V$. (iii) Ratio of two linear forms with coefficients $1/\beta_n$ vs $1/\beta_n'$, applied to the same nonneg vector $(g_1, \ldots, g_N)$. $\blacksquare$

**Translation:** More elastic gain functions are unambiguously better in the sense of the Lyapunov welfare metric. This is a robust comparative static — it holds for all perturbation states $\mathbf{F}$, not just near equilibrium.

---

## Result 3: Canard Duration at the Transcritical Bifurcation

### Setup

At the bifurcation $\rho(\mathbf{K}) = 1$, the nontrivial equilibrium collides with the trivial one in a transcritical bifurcation (Theorem 5.3). If a slow exogenous drift (e.g., secular decline in hardware costs $\gamma_c$, or gradual increase in investment efficiency $\delta_c$) pushes the system through the bifurcation, the trajectory spends finite but potentially long time near the vanishing equilibrium. This is the "crisis duration" — the low-mesh to high-mesh transition time.

The effective scalar dynamics on the slowest manifold are:

$$\dot{F}_1 = g(F_1, \mu) = \delta_c\,\Psi(F_1)^\alpha\,F_1^{\phi_c} - \gamma_c F_1$$

where $\mu$ is the slowly drifting parameter ($\mu$ could be $\delta_c$, $\gamma_c$, or any combination that moves $\rho(\mathbf{K})$ through 1).

### The Normal Form

**Proposition E.5** (Transcritical normal form). *At the bifurcation point $(\bar{F}_1, \mu^*)$ where $g = 0$ and $\partial g/\partial F_1 = 0$, the dynamics admit the local normal form*

$$\dot{y} = a\,\epsilon\,y + b\,y^2 + O(|y|^3 + |\epsilon|^2)$$

*where $y = F_1 - \bar{F}_1$, $\epsilon = \mu - \mu^*$, and*

$$a = \frac{\partial^2 g}{\partial F_1\,\partial\mu}\bigg|_{\text{bif}}, \qquad b = \frac{1}{2}\,\frac{\partial^2 g}{\partial F_1^2}\bigg|_{\text{bif}}$$

*The coefficient $b$ depends on the gain elasticities through $\Psi''(\bar{F}_1)$ and on $K$ through the CES marginal product $\partial^2 F_{\text{CES}}/\partial F_2^2$.*

*Proof.* Standard: Taylor expand $g(F_1, \mu)$ around the bifurcation point. The conditions $g = 0$ and $\partial_F g = 0$ eliminate the constant and linear terms. The nondegeneracy conditions $a \neq 0$ and $b \neq 0$ are the transversality requirements for a transcritical bifurcation. $\blacksquare$

### Passage Time Bound

**Theorem E.6** (Canard duration). *Suppose the bifurcation parameter drifts linearly: $\mu(t) = \mu^* + \varepsilon_{\text{drift}}\,t$ with $\varepsilon_{\text{drift}} > 0$ (slow drift through the bifurcation). Then the time the system spends within an $O(\delta)$ neighborhood of the ghost equilibrium satisfies*

$$\Delta t_{\text{crisis}} = \frac{\pi}{\sqrt{|a|\,\varepsilon_{\text{drift}}}} + O\!\left(\frac{\log(1/\delta)}{\sqrt{|a|\,\varepsilon_{\text{drift}}}}\right)$$

*where $a = \partial^2 g/\partial F_1\partial\mu|_{\text{bif}}$ is the sensitivity of the Jacobian to the bifurcation parameter.*

*Proof sketch.* This is the standard delayed loss of stability result for passage through a transcritical bifurcation (Neishtadt 1987, 1988; Berglund and Gentz 2006, Chapter 3). In the rescaled variable $\tau = \sqrt{|a|\varepsilon_{\text{drift}}}\,t$, the normal form becomes $dy/d\tau = \operatorname{sgn}(a)\tau y + (b/\sqrt{|a|\varepsilon_{\text{drift}}})\,y^2$, which is the Weber equation plus a quadratic perturbation. The linear part has the parabolic cylinder function as its solution, with the passage through zero eigenvalue at $\tau = 0$ creating a delay of $\pi$ time units in the rescaled variable. Converting back: $\Delta t = \pi/\sqrt{|a|\varepsilon_{\text{drift}}}$.

The logarithmic correction accounts for the entry and exit from the $O(\delta)$-neighborhood, which depends on the initial distance from the slow manifold. $\blacksquare$

### Computing $a$ for the 4-Level System

The mixed partial $a = \partial^2 g/\partial F_1\partial\mu$ evaluates the sensitivity of the growth rate to the bifurcation parameter. For the two most natural choices of $\mu$:

**Case 1: $\mu = \gamma_c$ (damping at the slowest level drifts down).**

$$g = \delta_c\Psi(F_1)^\alpha F_1^{\phi_c} - \gamma_c F_1$$

$$\frac{\partial g}{\partial\gamma_c} = -F_1, \qquad \frac{\partial^2 g}{\partial F_1\partial\gamma_c} = -1$$

So $a = -1$. The crisis duration is $\Delta t = \pi/\sqrt{\varepsilon_{\text{drift}}}$, independent of all system parameters except the drift rate. This is the simplest case: if the institutional friction $\gamma_c$ is what's slowly improving, the transition time depends only on how fast it improves.

**Case 2: $\mu = \delta_c$ (investment efficiency drifts up).**

$$\frac{\partial g}{\partial\delta_c} = \Psi(F_1)^\alpha F_1^{\phi_c}, \qquad \frac{\partial^2 g}{\partial F_1\partial\delta_c} = \left[\alpha\frac{\Psi'}{\Psi}F_1 + \phi_c\right]\Psi^\alpha F_1^{\phi_c - 1}$$

At the bifurcation point $\bar{F}_1$ where $\delta_c\Psi^\alpha\bar{F}_1^{\phi_c} = \gamma_c\bar{F}_1$:

$$a = \left[\alpha\frac{\Psi'(\bar{F}_1)}{\Psi(\bar{F}_1)}\bar{F}_1 + \phi_c\right]\frac{\gamma_c}{\delta_c}$$

The term $\Psi'(\bar{F}_1)/\Psi(\bar{F}_1)$ is the elasticity of the composite feedback, which propagates through the entire cascade. By the chain rule through $h_2, h_3, h_4$:

$$\frac{\Psi'\bar{F}_1}{\Psi} = \varepsilon_I \cdot \varepsilon_{\bar{S}} \cdot \varepsilon_{F_{\text{CES}}} \cdot \varepsilon_{h_2}$$

where each $\varepsilon$ is the elasticity of the respective function — a product of four elasticities through the cascade. The CES elasticity $\varepsilon_{F_{\text{CES}}}$ at the symmetric allocation is $1/J$ (from $\partial F_{\text{CES}}/\partial F_2 = 1/J$, Proposition 1.1), but the *second*-order behavior depends on $K$ through the Hessian.

### Where $K$ Enters the Crisis Duration

**Proposition E.7** (Curvature dependence of the canard). *The second-order coefficient $b = \frac{1}{2}\partial^2 g/\partial F_1^2|_{\text{bif}}$ satisfies*

$$b = \frac{\delta_c}{2}\left[\alpha(\alpha-1)\left(\frac{\Psi'}{\Psi}\right)^2 + \alpha\frac{\Psi''}{\Psi} + 2\alpha\phi_c\frac{\Psi'}{\Psi}\frac{1}{\bar{F}_1} + \phi_c(\phi_c-1)\frac{1}{\bar{F}_1^2}\right]\Psi^\alpha\bar{F}_1^{\phi_c}$$

*The $\Psi''/\Psi$ term contains $F_{\text{CES}}''$, which by Proposition 1.2 is proportional to $(1-\rho)/J = K/(J-1)$ at the symmetric allocation. Specifically:*

$$\frac{\partial^2 F_{\text{CES}}}{\partial F_2^2}\bigg|_{\text{sym}} = -\frac{(1-\rho)(J-1)}{J^2\bar{F}_2} = -\frac{K}{J\bar{F}_2}$$

*Therefore $b$ depends on $K$: higher curvature (stronger complementarity) increases $|b|$, which determines the sharpness of the transition. However, $b$ does not appear in the leading-order crisis duration $\pi/\sqrt{|a|\varepsilon_{\text{drift}}}$ — it enters only in the correction terms and in the amplitude of the post-bifurcation trajectory.*

*Proof.* The formula for $b$ follows from differentiating $g(F_1, \mu)$ twice in $F_1$ using the product and chain rules. The $\Psi''$ term propagates through the cascade: $\Psi'' = I''\cdot(h_4')^2\cdot(h_3')^2\cdot(h_2')^2 + I'\cdot h_4''\cdot(h_3')^2\cdot(h_2')^2 + \cdots$, and $h_3'' = (\phi_{\text{eff}}/\delta_C)\cdot F_{\text{CES}}''(h_2)\cdot(h_2')^2 + (\phi_{\text{eff}}/\delta_C)\cdot F_{\text{CES}}'(h_2)\cdot h_2''$. The CES second derivative at the symmetric point is $-(1-\rho)(J-1)/(J^2\bar{F}_2)$ by restricting the Hessian from Proposition 1.2 to the aggregate direction (the within-level Hessian applied to a perturbation that changes all components equally changes $F$ through its radial curvature, which involves $(1-\rho)$). $\blacksquare$

### The Computable Bound

Assembling the results:

**Corollary E.8** (Crisis duration estimate). *For the 4-level CES system with the bifurcation parameter $\mu$ drifting at rate $\varepsilon_{\text{drift}}$:*

*(i) If $\mu = \gamma_c$: $\Delta t_{\text{crisis}} = \pi/\sqrt{\varepsilon_{\text{drift}}}$, independent of system parameters.*

*(ii) If $\mu = \delta_c$: $\Delta t_{\text{crisis}} = \pi/\sqrt{|a(\boldsymbol{\beta}, K)|\,\varepsilon_{\text{drift}}}$, where $|a|$ increases with the product of cascade elasticities $\prod_n \beta_n$ and with $1/J$ (fewer components per level means stronger aggregate feedback).*

*(iii) In both cases, the crisis duration scales as $\varepsilon_{\text{drift}}^{-1/2}$: halving the drift rate increases the transition time by $\sqrt{2} \approx 41\%$.*

*(iv) The CES curvature $K$ does not enter the leading-order duration. It enters the correction terms and determines the post-bifurcation growth rate — higher $K$ produces a sharper transition (larger $|b|$) with faster acceleration away from the ghost equilibrium once the transition begins.*

### What This Means

The crisis duration — how long the system lingers in the low-mesh regime after conditions have become favorable for the high-mesh regime — is primarily controlled by the drift rate $\varepsilon_{\text{drift}}$ and the cascade elasticity $|a|$. The CES curvature $K$ plays a secondary role: it doesn't determine *when* the transition happens, but it determines *how sharply* the system accelerates once the transition begins.

The economically important timing question "how long does the low-mesh → high-mesh transition take?" has the answer: **$O(1/\sqrt{\varepsilon_{\text{drift}}})$, with the constant controlled by the chain of gain elasticities, not by the CES substitution parameter.** The paper can give this as a quantitative prediction calibrated to specific drift rates.

The role of $K$ is in the *amplitude* of the transition: once the system leaves the neighborhood of the ghost equilibrium, the coefficient $b$ (which contains $K$) determines the trajectory curvature and hence the overshooting/convergence behavior. Higher complementarity (larger $K$) means a faster snap to the new equilibrium, with less overshooting.

---

## Summary: What These Three Results Add

| Result | Mathematical content | Policy content |
|--------|---------------------|---------------|
| E.1 | Lyapunov function is a $1/\beta_n$-weighted family | Welfare loss decomposition identifies the most rigid institutional layer |
| E.2–E.4 | $W_{nn}$ is monotone; $c_n\sigma_n$ is independent of $\sigma_n$ | Local damping cancels in welfare dissipation; reform upstream instead |
| E.5–E.8 | Canard duration $= \pi/\sqrt{\|a\|\varepsilon_{\text{drift}}}$ | Transition timing is computable; $K$ controls sharpness, not duration |

### Addendum to Part 9 (Moduli Space Theorem)

These results refine the free-parameter characterization of Theorem 9.1. The gain elasticities $\{\beta_n\}$ are still free parameters not determined by $\rho$, but they now have concrete comparative-static content:

- $\boldsymbol{\beta}$ indexes a parametric family of Lyapunov functions (Result 1).
- The family is ordered: $\boldsymbol{\beta} \succeq \boldsymbol{\beta}'$ implies $V(\boldsymbol{\beta}) \leq V(\boldsymbol{\beta}')$ pointwise (Result 2).
- $\boldsymbol{\beta}$ enters the crisis duration through the mixed partial $a$, but $\rho$ (and hence $K$) does not, to leading order (Result 3).

The gain functions remain genuinely free — but the framework now tells you exactly what each choice costs.

---

## References (additional)

11. Berglund, N., and Gentz, B. (2006). *Noise-Induced Phenomena in Slow-Fast Dynamical Systems.* Springer.

12. Neishtadt, A. I. (1987). Persistence of stability loss for dynamical bifurcations, I. *Differential Equations* 23, 1385–1391.

13. Neishtadt, A. I. (1988). Persistence of stability loss for dynamical bifurcations, II. *Differential Equations* 24, 171–176.
