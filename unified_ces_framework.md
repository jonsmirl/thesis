# The Hierarchical CES Framework: A Unified Treatment

## Curvature, Topology, and Stability in Port-Hamiltonian Systems with Constant Elasticity of Substitution

---

## Proof Dependency Diagram

```
Part 1 (Curvature Lemma)
  ├──> Part 2 (Within-Level Eigenstructure)
  │      ├──> Part 3, Claim 1 (Aggregate Coupling, via Fenichel)
  │      └──> Part 4 (Reduced System)
  ├──> Part 8 (Triple Role of Curvature)
  │      ├── Part 8a (Superadditivity)
  │      ├── Part 8b (Correlation Robustness)
  │      └── Part 8c (Strategic Independence)
  └──> Part 3, Claim 3 (Port Direction)

Part 3 (Port Topology Theorem)
  ├── Claim 1 (Aggregate Coupling)  <── Part 2
  ├── Claim 2 (Directed Coupling)   <── Part 7 (pH structure)
  ├── Claim 3 (Port Gain)           <── Part 1
  └── Claim 4 (Nearest-Neighbor)    <── Standing Assumption (timescale separation)

Part 4 (Reduced System)  <── Parts 2, 3

Part 5 (NGM and Spectral Threshold)  <── Part 4
  └──> Part 6 (Hierarchical Ceiling)

Part 6 (Hierarchical Ceiling)  <── Parts 4, 5

Part 7 (Lyapunov Structure and Bridge)
  ├── Step 1: pH structure    <── Parts 2, 3
  ├── Step 2: Storage V       <── Li-Shuai-vdD (2010)
  └── Step 3: Bridge equation <── Parts 1, 5
       └──> Part 9 (Moduli Space Theorem)

Part 9 (Moduli Space Theorem)  <── Parts 1–8

Part 10 (Honest Limits)  <── Parts 3, 7
```

---

## Part 0: Setup and Notation

### 0.1. The CES Aggregate

For $J \geq 2$ components at level $n$ of an $N$-level hierarchy, the **CES aggregate** with equal weights is

$$F_n(x_n) = \left(\frac{1}{J}\sum_{j=1}^{J} x_{nj}^{\,\rho}\right)^{1/\rho}, \qquad x_n = (x_{n1}, \ldots, x_{nJ}) \in \mathbb{R}_{+}^J$$

where $\rho < 1$, $\rho \neq 0$, is the substitution parameter. We write $F_n^* = F_n(x_n^*)$ for the equilibrium value. The elasticity of substitution is $\sigma = 1/(1-\rho)$.

**Remark (General weights).** The equal-weight assumption simplifies the main development. The extension to weights $a_j > 0$ with $\sum a_j = 1$ replaces $1/J$ by $a_j$ in the CES definition. At the cost-minimizing point, the effective shares $p_j = a_j^{1/(1-\rho)}$ replace the symmetric allocation $x_j^* = c$. All results generalize; the curvature parameter $K$ acquires a weight-dispersion factor controlled by the secular equation of Section 1.4. The general-weight extensions are stated as remarks following the equal-weight proofs.

### 0.2. Derived Objects

| Symbol | Definition | Name |
|--------|-----------|------|
| $K$ | $(1-\rho)(J-1)/J$ | Within-level curvature parameter (equal weights) |
| $\mathbf{K}$ | Next-generation matrix (boldface) | NGM of the reduced system |
| $\rho(\mathbf{K})$ | Spectral radius of $\mathbf{K}$ | Threshold parameter |
| $\Phi$ | $-\sum_{n=1}^{N} \log F_n$ | CES free energy (Hamiltonian) |
| $V$ | $\sum_{n=1}^{N} c_n D_{KL}(x_n \| x_n^*)$ | Storage function (Lyapunov function) |
| $W$ | $\operatorname{diag}(W_{11}, \ldots, W_{NN})$ | Supply rate (Bridge) matrix |
| $P_{\text{cycle}}$ | $\prod_{n} k_{n+1,n}$ | Cycle product of the NGM |

Here $D_{KL}(x_n \| x_n^*) = \sum_{j=1}^{J}\bigl(x_{nj}/x_{nj}^* - 1 - \log(x_{nj}/x_{nj}^*)\bigr)$ is the component-wise KL divergence.

### 0.3. The Hierarchical Dynamics

The system evolves by

$$\varepsilon_n \dot{x}_{nj} = T_n(x_{n-1}) \cdot \frac{\partial F_n}{\partial x_{nj}} - \sigma_n x_{nj}, \qquad n = 1, \ldots, N, \quad j = 1, \ldots, J$$

where $T_n = \phi_n(F_{n-1}(x_{n-1}))$ is the feed-forward transmission from level $n-1$ to level $n$, $\sigma_n > 0$ is the damping rate, and $\varepsilon_n > 0$ is the characteristic timescale. The exogenous input enters at the designated source level.

### 0.4. Standing Assumptions

Throughout this document:

1. **Substitution:** $\rho < 1$, $\rho \neq 0$.
2. **Components:** $J \geq 2$ components per level, $N \geq 2$ levels.
3. **Timescale separation:** $\varepsilon_1 \gg \varepsilon_2 \gg \cdots \gg \varepsilon_N > 0$.
4. **Damping:** $\sigma_n > 0$ for all $n$.
5. **Coupling:** $\phi_n : \mathbb{R}_+ \to \mathbb{R}_+$ is $C^1$ with $\phi_n(0) = 0$ and $\phi_n' > 0$.

---

## Part 1: The Curvature Lemma

This section establishes the geometric foundation. All subsequent results depend on the eigenstructure derived here.

### 1.1. Equal Marginal Products

**Proposition 1.1** (Equal marginal products). *At the symmetric point $x_{nj} = \bar{x}$ for all $j$, where $F_n = J^{1/\rho - 1}\bar{x}$, the gradient of $F_n$ is*

$$\nabla F_n(\bar{x}\,\mathbf{1}) = \frac{1}{J}\,\mathbf{1}$$

*In particular, all marginal products are equal, and the tangent space to the isoquant $\{F_n = c\}$ at the symmetric point is $T = \{\mathbf{v} \in \mathbb{R}^J : \sum_{j} v_j = 0\} = \mathbf{1}^{\perp}$.*

*Proof.* The partial derivative of $F_n$ is $\partial F_n/\partial x_{nj} = (1/J)\, x_{nj}^{\rho-1} \, F_n^{1-\rho}$. At the symmetric point, $x_{nj} = \bar{x}$ and $F_n = (J^{-1} \cdot J \cdot \bar{x}^{\rho})^{1/\rho} = \bar{x}$, so $\partial F_n/\partial x_{nj} = (1/J)\,\bar{x}^{\rho-1}\,\bar{x}^{1-\rho} = 1/J$. $\blacksquare$

This is the structural fact that underlies the entire framework: at the symmetric allocation, the CES aggregate treats all components identically, and the tangent space to the isoquant is the hyperplane $\sum v_j = 0$ regardless of the substitution parameter.

### 1.2. Hessian at the Symmetric Point

**Proposition 1.2** (CES Hessian). *The Hessian of $F_n$ at the symmetric point $x_{nj} = \bar{x}$ is*

$$\nabla^2 F_n = \frac{(1-\rho)}{J^2 \bar{x}}\left[\mathbf{1}\mathbf{1}^T - J\,I\right]$$

*with eigenvalues:*
- *$0$ on $\mathbf{1}$ (multiplicity 1), by Euler's theorem for degree-1 homogeneous functions;*
- *$-(1-\rho)/(J\bar{x})$ on $\mathbf{1}^{\perp}$ (multiplicity $J-1$).*

*Proof.* The general CES Hessian (Appendix A of [Triple Role]) is

$$\frac{\partial^2 F}{\partial x_i \partial x_j} = \frac{(1-\rho)}{F}\,\frac{\partial F}{\partial x_i}\,\frac{\partial F}{\partial x_j} - \delta_{ij}\,\frac{(1-\rho)}{x_j}\,\frac{\partial F}{\partial x_j}$$

At the symmetric point, $\partial_j F = 1/J$, $F = \bar{x}$, and $x_j = \bar{x}$:

$$\frac{\partial^2 F}{\partial x_i \partial x_j} = \frac{(1-\rho)}{\bar{x}} \cdot \frac{1}{J^2} - \delta_{ij}\frac{(1-\rho)}{\bar{x}} \cdot \frac{1}{J} = \frac{(1-\rho)}{J^2 \bar{x}}\left(1 - J\delta_{ij}\right)$$

The matrix $\mathbf{1}\mathbf{1}^T - J\,I$ has eigenvector $\mathbf{1}$ with eigenvalue $J - J = 0$, and every $\mathbf{v} \perp \mathbf{1}$ is an eigenvector with eigenvalue $0 - J = -J$. Multiplying by $(1-\rho)/(J^2\bar{x})$ gives the stated eigenvalues. The zero eigenvalue on $\mathbf{1}$ also follows from Euler's theorem: $\nabla^2 F \cdot x = 0$ for degree-1 homogeneous $F$, and $x = \bar{x}\,\mathbf{1}$ at the symmetric point. $\blacksquare$

### 1.3. The Curvature Parameter

**Definition 1.3.** The **curvature parameter** of the equal-weight CES aggregate with $J$ components is

$$K = (1-\rho)\,\frac{J-1}{J}$$

**Properties.** (i) $K > 0$ for all $\rho < 1$. (ii) $K$ increases monotonically with $(1-\rho)$. (iii) $K \to \infty$ as $\rho \to -\infty$ (Leontief limit). (iv) $K \to 0$ as $\rho \to 1^-$ (perfect substitutes).

The curvature parameter equals $c \cdot (J-1) \cdot \kappa_{\min} / \sqrt{J}$, where $\kappa_{\min}$ is the minimum principal curvature of the CES isoquant at the symmetric point (see Proposition 1.4 below). It is dimensionless and independent of the isoquant level $c$.

### 1.4. Isoquant Curvature

**Proposition 1.4** (Isoquant curvature at the symmetric point). *At the symmetric point on $\{F_n = c\}$, all $J-1$ principal curvatures of the isoquant are equal:*

$$\kappa^* = \frac{(1-\rho)}{c\sqrt{J}} = \frac{K\sqrt{J}}{c(J-1)}$$

*Proof.* The normal curvature in tangent direction $\mathbf{v} \in \mathbf{1}^{\perp}$ is $\kappa(\mathbf{v}) = -\mathbf{v}^T \nabla^2 F\,\mathbf{v}/(\|\nabla F\|\cdot\|\mathbf{v}\|^2)$. By Proposition 1.2, $\mathbf{v}^T \nabla^2 F\,\mathbf{v} = -(1-\rho)/(J\bar{x}) \cdot \|\mathbf{v}\|^2$ for any $\mathbf{v} \in \mathbf{1}^{\perp}$. By Proposition 1.1, $\|\nabla F\| = \sqrt{J}/J$. Thus $\kappa(\mathbf{v}) = (1-\rho)/(J\bar{x}) \cdot J/\sqrt{J} = (1-\rho)/(\bar{x}\sqrt{J})$. At the isoquant $\{F = c\}$, the symmetric point has $\bar{x} = c$ (since $F = \bar{x}$ at equal weights), giving the result. The curvature is independent of $\mathbf{v}$: the isoquant has uniform curvature at the symmetric point. $\blacksquare$

**Remark 1.5** (General weights). With weights $a_j > 0$ summing to 1, the equal marginal product property persists at the cost-minimizing point $x_j^* = c\,p_j/\Phi^{1/\rho}$, where $p_j = a_j^{1/(1-\rho)}$ and $\Phi = \sum p_j$: one has $\nabla F(x^*) = g\,\mathbf{1}$ with $g = \Phi^{(1-\rho)/\rho}$. The principal curvatures are no longer degenerate. They are determined by the constrained eigenvalues $\mu_1 < \cdots < \mu_{J-1}$ of the weighted inverse-share matrix $W = \operatorname{diag}(1/p_1, \ldots, 1/p_J)$ restricted to $\mathbf{1}^{\perp}$, via the secular equation

$$\sum_{j=1}^{J} \frac{1}{w_j - \mu} = 0, \qquad w_j = a_j^{-1/(1-\rho)}$$

which has exactly $J-1$ roots, one in each interval $(w_{(k)}, w_{(k+1)})$. The generalized curvature parameter is $K(\rho, \mathbf{a}) = (1-\rho)(J-1)\Phi^{1/\rho}R_{\min}/J$, where $R_{\min} = \mu_1$ is the smallest root. At equal weights, $R_{\min} = J^{\sigma}$, $\Phi^{1/\rho} = J^{-\sigma}$, and $K$ reduces to $(1-\rho)(J-1)/J$.

---

## Part 2: Within-Level Eigenstructure

Two matrices arise at each level, with different eigenstructures. Both are correct; they measure different things.

- **Object A:** $\nabla^2 \Phi_n$ (Hessian of the free energy). This is the *geometric* object. Anisotropy ratio: $(1-\rho)$.
- **Object B:** $Df_n$ (Jacobian of the dynamics). This is the *dynamical* object. Anisotropy ratio: $(2-\rho) = 1 + (1-\rho)$. The extra $1$ comes from damping.

### 2.1. Hessian of the Free Energy (Object A)

**Proposition 2.1.** *The Hessian of $\Phi_n = -\log F_n$ at the symmetric point is*

$$\nabla^2 \Phi_n = \frac{1}{J^2 \bar{x}^2}\left[(1-\rho)\,J\,I + \rho\,\mathbf{1}\mathbf{1}^T\right]$$

*with eigenvalues:*
- *$(1-\rho)/(J\bar{x}^2)$ on $\mathbf{1}^{\perp}$ (multiplicity $J-1$);*
- *$1/(J\bar{x}^2)$ on $\mathbf{1}$ (multiplicity 1).*

*The anisotropy ratio is $\lambda_{\perp}/\lambda_{\parallel} = (1-\rho)$.*

*Proof.* By the chain rule, $\nabla^2(-\log F) = -\nabla^2 F / F + \nabla F \nabla F^T / F^2$. Substituting the results of Propositions 1.1 and 1.2 with $F = \bar{x}$:

$$\nabla^2\Phi_n = \frac{(1-\rho)}{J^2 \bar{x}^2}\left[J\,I - \mathbf{1}\mathbf{1}^T\right] + \frac{1}{J^2 \bar{x}^2}\,\mathbf{1}\mathbf{1}^T = \frac{1}{J^2\bar{x}^2}\left[(1-\rho)\,J\,I + \rho\,\mathbf{1}\mathbf{1}^T\right]$$

On $\mathbf{1}^{\perp}$: eigenvalue $(1-\rho)\,J/(J^2\bar{x}^2) = (1-\rho)/(J\bar{x}^2)$. On $\mathbf{1}$: eigenvalue $(1-\rho)\,J/(J^2\bar{x}^2) + \rho\,J/(J^2\bar{x}^2) = 1/(J\bar{x}^2)$. $\blacksquare$

### 2.2. Jacobian of the Dynamics (Object B)

**Proposition 2.2.** *At the symmetric equilibrium, where $T_n = \sigma_n J \bar{x}$ (from the equilibrium condition $T_n \cdot \partial F_n/\partial x_{nj} = \sigma_n x_{nj}$), the within-level Jacobian is*

$$Df_n = \frac{\sigma_n}{\varepsilon_n}\left[\frac{(1-\rho)}{J}\,\mathbf{1}\mathbf{1}^T - (2-\rho)\,I\right]$$

*with eigenvalues:*
- *$-\sigma_n/\varepsilon_n$ on $\mathbf{1}$ (aggregate mode, multiplicity 1);*
- *$-\sigma_n(2-\rho)/\varepsilon_n$ on $\mathbf{1}^{\perp}$ (diversity modes, multiplicity $J-1$).*

*Proof.* From the dynamics $\varepsilon_n \dot{x}_{nj} = T_n \,\partial F_n / \partial x_{nj} - \sigma_n x_{nj}$, the within-level Jacobian is $Df_n = (1/\varepsilon_n)[T_n \nabla^2 F_n - \sigma_n I]$. Substituting $T_n = \sigma_n J \bar{x}$ and the Hessian from Proposition 1.2:

$$Df_n = \frac{1}{\varepsilon_n}\left[\sigma_n J \bar{x} \cdot \frac{(1-\rho)}{J^2 \bar{x}}(\mathbf{1}\mathbf{1}^T - J\,I) - \sigma_n I\right] = \frac{\sigma_n}{\varepsilon_n}\left[\frac{(1-\rho)}{J}\mathbf{1}\mathbf{1}^T - (1-\rho)\,I - I\right]$$

$$= \frac{\sigma_n}{\varepsilon_n}\left[\frac{(1-\rho)}{J}\mathbf{1}\mathbf{1}^T - (2-\rho)\,I\right]$$

On $\mathbf{1}$: $\sigma_n/\varepsilon_n \cdot [(1-\rho) - (2-\rho)] = -\sigma_n/\varepsilon_n$. On $\mathbf{1}^{\perp}$: $\sigma_n/\varepsilon_n \cdot [0 - (2-\rho)] = -\sigma_n(2-\rho)/\varepsilon_n$. $\blacksquare$

**Corollary 2.3** (Timescale separation within levels). *The diversity modes decay faster than the aggregate mode by the factor $(2-\rho) > 1$ for all $\rho < 1$:*

$$\frac{\tau_{\text{div}}}{\tau_{\text{agg}}} = \frac{1}{2-\rho}$$

*Equivalently, the dynamical anisotropy satisfies $(2-\rho) = 1 + KJ/(J-1)$.*

*At $\rho = 0$ (Cobb-Douglas): diversity is $2\times$ faster. At $\rho = -1$: diversity is $3\times$ faster. As $\rho \to -\infty$ (Leontief): diversity is infinitely faster — components lock together instantaneously.*

---

## Part 3: Port Topology Theorem

The CES curvature constrains the network topology of the hierarchical system. The four claims below collapse what could be an arbitrary directed graph with vector-valued coupling functions into a specific nearest-neighbor chain with scalar aggregate coupling.

**Theorem 3.1** (CES-Forced Topology). *Under the standing assumptions of Section 0.4, the hierarchical CES system has the following topological properties:*

*(i) (Aggregate coupling) Each level communicates with other levels only through its aggregate $F_n$. That is, the critical manifold at level $n$ is parameterized by $F_n$ alone.*

*(ii) (Directed coupling) The between-level coupling is necessarily feed-forward (non-reciprocal). Any bidirectional coupling — whether power-preserving or merely passive — yields an unconditionally stable system incapable of bifurcation.*

*(iii) (Port alignment) The port direction is $b_n \propto \mathbf{1} = \nabla F_n / \|\nabla F_n\|^2$ at the symmetric equilibrium. The port gain functions $\phi_n$ are free parameters, not determined by $\rho$.*

*(iv) (Nearest-neighbor topology) Under the timescale separation assumption, long-range coupling is dynamically equivalent to nearest-neighbor coupling with modified exogenous inputs.*

### Proof of Claim (i): Aggregate Coupling

The argument has three steps: equilibrium uniqueness, normal hyperbolicity, and persistence.

**Step 1: Nonlinear equilibrium uniqueness.** At equilibrium, the condition $T_n \cdot \partial F_n/\partial x_{nj} = \sigma_n x_{nj}$ gives

$$\frac{T_n}{J}\,x_{nj}^{\rho-1}\,F_n^{1-\rho} = \sigma_n x_{nj}$$

Rearranging: $x_{nj}^{\rho-2} = T_n F_n^{1-\rho}/(J\sigma_n)$. The right side is independent of $j$. For $\rho \neq 2$ (which holds since $\rho < 1$), the map $x \mapsto x^{\rho-2}$ is injective on $\mathbb{R}_+$, so $x_{nj} = \bar{x}_n$ for all $j$.

**Step 2: Normal hyperbolicity.** Define the critical manifold

$$\mathcal{M}_n = \{x_n \in \mathbb{R}_+^J : x_{nj} = F_n / J^{1/\rho} \text{ for all } j\}$$

By Proposition 2.2, the transverse eigenvalues (on $\mathbf{1}^{\perp}$) are $-\sigma_n(2-\rho)/\varepsilon_n$ and the tangential eigenvalue (on $\mathbf{1}$) is $-\sigma_n/\varepsilon_n$. The spectral gap between transverse and tangential decay rates is

$$\frac{\sigma_n(2-\rho)}{\varepsilon_n} - \frac{\sigma_n}{\varepsilon_n} = \frac{\sigma_n(1-\rho)}{\varepsilon_n} > 0 \qquad \text{for all } \rho < 1$$

This verifies the normal hyperbolicity condition: the transverse contraction rate strictly exceeds the tangential rate.

**Step 3: Persistence (Fenichel's theorem).** By Fenichel's geometric singular perturbation theorem (Fenichel 1979, Theorem 9.1), the normally hyperbolic critical manifold $\mathcal{M}_n$ persists as a locally invariant slow manifold $\mathcal{M}_n^{\varepsilon}$ within $O(\varepsilon)$ of $\mathcal{M}_n$, smoothly parameterized by $F_n$. On $\mathcal{M}_n^{\varepsilon}$, the within-level state is determined by $F_n$ up to $O(\varepsilon)$ corrections. Consequently, $F_n$ is a sufficient statistic for level $n$'s state on the slow manifold: any coupling through individual components $x_{nj}$ projects onto coupling through $F_n$.

**Remark.** The $(2-\rho)$ low-pass filter (Corollary 2.3) provides the dynamical mechanism: any signal component in $\mathbf{1}^{\perp}$ injected at level $n$ is suppressed by the factor $1/(2-\rho)$ relative to the aggregate component. The suppression ranges from $1/2$ at $\rho = 0$ to $0$ as $\rho \to -\infty$. $\blacksquare$

### Proof of Claim (ii): Directed Coupling

Consider a two-level system on the slow manifold (scalar dynamics per level).

**Step 1: Power-preserving bidirectional coupling.** Suppose levels 1 and 2 are coupled bidirectionally with port powers summing to zero:

$$\varepsilon_1 \dot{F}_1 = (\beta_1 - \psi(F_2))/J - \sigma_1 F_1, \qquad \varepsilon_2 \dot{F}_2 = \phi(F_1)/J - \sigma_2 F_2$$

The power-preservation constraint $-\psi(F_2)/(JF_1) + \phi(F_1)/(JF_2) = 0$ (port powers in a port-Hamiltonian system sum to zero) forces $\phi(F_1) = cF_1$ and $\psi(F_2) = cF_2$ for some constant $c > 0$. The Jacobian is then

$$\mathcal{J}_{\text{bidir}} = \begin{pmatrix} -\sigma_1 & -c/J \\ c/J & -\sigma_2 \end{pmatrix}$$

with eigenvalues $-(\sigma_1+\sigma_2)/2 \pm \sqrt{(\sigma_1-\sigma_2)^2/4 - c^2/J^2}$. Whether the discriminant is positive (two real negative eigenvalues) or negative (complex pair with real part $-(\sigma_1+\sigma_2)/2 < 0$), both eigenvalues have strictly negative real part. The system is unconditionally stable for all $c, \sigma_1, \sigma_2 > 0$.

**Step 2: Extension to passive bidirectional coupling.** Any passive bidirectional coupling satisfies $\dot{V}_{\text{coupling}} \leq 0$ by definition (passivity: the coupling can only dissipate, not generate, energy). This contributes a negative-semidefinite term to the effective Jacobian, which can only strengthen stability. Therefore any bidirectional coupling — whether power-preserving (lossless) or merely passive (lossy) — yields an unconditionally stable system.

**Step 3: Necessity of directed coupling.** The bifurcation at $\rho(\mathbf{K}) = 1$ (Theorem 5.2 below) requires that the spectral radius of the next-generation matrix reach 1, which requires net energy injection through the hierarchy. Power-preserving bidirectional coupling contributes zero net energy; passive coupling contributes negative net energy. Neither can produce $\rho(\mathbf{K}) = 1$. Therefore the CES geometry, combined with the requirement for nontrivial dynamics (bifurcation), forces the between-level coupling to be non-reciprocal with an external energy source. $\blacksquare$

### Proof of Claim (iii): Port Alignment and Gain

**Step 1: Port direction is forced.** At a symmetric equilibrium $x_n^* = \bar{x}\,\mathbf{1}$, the equilibrium condition $u_n^* \cdot b_n = \sigma_n \bar{x}\,\mathbf{1}$ requires $b_n \propto \mathbf{1}$. Furthermore, $\nabla F_n = (1/J)\,\mathbf{1} \propto \mathbf{1}$ at the symmetric point (Proposition 1.1), so $b_n = \nabla F_n$ is the natural CES-compatible port direction.

**Step 2: Asymmetric ports are penalized.** For $b_n \not\propto \mathbf{1}$, the equilibrium $x_n^* \propto b_n$ is asymmetric. By Jensen's inequality applied to the concave CES function ($\rho < 1$), $F(x) \leq F(\bar{x}\,\mathbf{1})$ for any $x$ with $\sum x_j = J\bar{x}$. Asymmetric ports produce less aggregate output per unit input.

**Step 3: Gain function is free.** At equilibrium, $\phi_n(F_{n-1}^*) = \sigma_n J F_n^*$. For power-law gains $\phi_n(z) = a_n z^{\beta_n}$, the exponents $\beta_n$ are free parameters not determined by $\rho$. The coefficients $a_n$ adjust to satisfy the equilibrium condition but depend on $\sigma_n$, $J$, and the cascade — not on $\rho$ alone.

**Summary:** The CES geometry forces the port direction ($b_n \propto \mathbf{1}$). The port gain function $\phi_n$ is a genuine independent degree of freedom. Geometry constrains direction; physics constrains gain. $\blacksquare$

### Proof of Claim (iv): Nearest-Neighbor Topology

Consider a three-level system with long-range coupling from level 1 to level 3:

$$\varepsilon_1 \dot{F}_1 = \beta_1/J - \sigma_1 F_1, \quad \varepsilon_2 \dot{F}_2 = \phi_{21}(F_1)/J - \sigma_2 F_2, \quad \varepsilon_3 \dot{F}_3 = [\phi_{31}(F_1) + \phi_{32}(F_2)]/J - \sigma_3 F_3$$

**Step 1.** Level 1, being fastest ($\varepsilon_1 \ll \varepsilon_2$), equilibrates to $F_1^* = \beta_1/(\sigma_1 J)$, an algebraic constant on the slow manifold.

**Step 2.** The long-range coupling $\phi_{31}(F_1^*) = \phi_{31}(\beta_1/(\sigma_1 J)) \equiv \tilde{\beta}_3$ becomes a constant, absorbed into the exogenous input to level 3.

**Step 3.** Level 3's effective dynamics become $\varepsilon_3 \dot{F}_3 = [\tilde{\beta}_3 + \phi_{32}(F_2)]/J - \sigma_3 F_3$, identical to a nearest-neighbor system with modified exogenous input.

**Step 4.** The Jacobian of the reduced 2D system (levels 2, 3) is lower-triangular:

$$\mathcal{J} = \begin{pmatrix} -\sigma_2 & 0 \\ \phi_{32}'/J & -\sigma_3 \end{pmatrix}$$

This is independent of the long-range coupling strength. The long-range coupling affects the equilibrium location but not the dynamics or stability near that equilibrium.

The argument generalizes by induction to $N$ levels: after all levels faster than level $m$ equilibrate, any coupling $\phi_{nm}(F_m)$ with $m$ fast becomes $\phi_{nm}(F_m^*) = \text{const}$, absorbed into a redefined exogenous input. The effective topology is nearest-neighbor.

**Conditional:** This result requires the timescale separation of Standing Assumption 0.4(3). This condition is pre-satisfied within the framework. $\blacksquare$

---

## Part 4: The Reduced System on the Slow Manifold

After within-level equilibration (Parts 2–3), the $NJ$-dimensional system collapses to an $N$-dimensional system in the aggregates $F_1, \ldots, F_N$.

### 4.1. General Form

**Proposition 4.1** (Reduced dynamics). *On the slow manifold $\mathcal{M}^{\varepsilon} = \prod_n \mathcal{M}_n^{\varepsilon}$, the aggregate dynamics are*

$$\varepsilon_n \dot{F}_n = \phi_n(F_{n-1})/J - \sigma_n F_n, \qquad n = 1, \ldots, N$$

*with the convention that $\phi_1$ depends on $F_N$ (or on an exogenous input $\beta_1$, depending on whether the system is cyclic or open).*

*Proof.* By Claim (i) of Theorem 3.1, the slow manifold at each level is parameterized by $F_n$, with within-level state $x_{nj} = F_n/J^{1/\rho}$ for all $j$ up to $O(\varepsilon)$ corrections. Project the dynamics onto the aggregate: $\dot{F}_n = \nabla F_n \cdot \dot{x}_n = (1/J)\mathbf{1} \cdot \dot{x}_n = (1/J)\sum_j \dot{x}_{nj}$. Summing $\varepsilon_n \dot{x}_{nj} = T_n \cdot (1/J) - \sigma_n x_{nj}$ over $j$: $\varepsilon_n J \dot{F}_n = T_n - \sigma_n J F_n$, where we used $\sum_j \partial F_n/\partial x_{nj} = 1$ (Euler's theorem for degree-1 $F_n$) and $\sum_j x_{nj} = J F_n$ (at the symmetric allocation). Thus $\varepsilon_n \dot{F}_n = T_n/J - \sigma_n F_n = \phi_n(F_{n-1})/J - \sigma_n F_n$. $\blacksquare$

### 4.2. The 4-Level Cyclic System

For the specific application with Wright's Law learning, logistic growth, CES capability aggregation, and settlement dynamics, the reduced system is:

$$\dot{F}_1 = \delta_c \cdot I(F_4)^{\alpha} \cdot F_1^{\phi_c} - \gamma_c F_1$$
$$\dot{F}_2 = \beta(F_1) \cdot F_2 \cdot (1 - F_2/N^*(F_1)) - \mu F_2$$
$$\dot{F}_3 = \phi_{\text{eff}} \cdot F_{\text{CES}}(F_2) - \delta_C F_3$$
$$\dot{F}_4 = \eta(F_3, F_2) \cdot F_4 \cdot (1 - F_4/\bar{S}(F_3)) - \nu F_4$$

Here $I(F_4)$ is the investment function, $N^*(F_1)$ is the carrying capacity for mesh agents, $F_{\text{CES}}(F_2)$ is the CES capability aggregate, $\bar{S}(F_3)$ is the safe asset capacity, and $\phi_{\text{eff}} = \phi_0/(1 - \beta_{\text{auto}}\phi_0)$ is the effective production multiplier including autocatalytic feedback.

---

## Part 5: NGM and Spectral Threshold

### 5.1. The Next-Generation Matrix

At the nontrivial equilibrium $\bar{F} = (\bar{F}_1, \ldots, \bar{F}_N)$, the Jacobian of the reduced $N$-dimensional system decomposes as $J_{\text{agg}} = T + \Sigma$, where $\Sigma = \operatorname{diag}(\sigma_1, \ldots, \sigma_N)$ with $\sigma_n < 0$ encodes damping (transitions) and $T$ encodes amplification (transmission). The **next-generation matrix** is

$$\mathbf{K} = -T\Sigma^{-1}$$

with entries $K_{nn} = d_n = T_{nn}/|\sigma_n|$ (within-level reproduction) and $K_{n,n-1} = k_{n,n-1} = T_{n,n-1}/|\sigma_{n-1}|$ (cross-level amplification).

### 5.2. Characteristic Polynomial

**Theorem 5.1** (NGM characteristic polynomial). *For a cyclic $N$-level system with diagonal entries $d_1, \ldots, d_N$, nearest-neighbor coupling $k_{n+1,n}$, and cycle closure $k_{1N}$, the characteristic polynomial of the NGM is*

$$p(\lambda) = \prod_{i=1}^{N}(d_i - \lambda) - P_{\text{cycle}}$$

*where $P_{\text{cycle}} = \prod_n k_{n+1,n}$ is the cycle product.*

*Proof.* In the Leibniz expansion $\det(\mathbf{K} - \lambda I) = \sum_{\pi \in S_N} \operatorname{sgn}(\pi) \prod_i (\mathbf{K} - \lambda I)_{i,\pi(i)}$, a permutation $\pi$ contributes a nonzero product only if every factor $(\mathbf{K} - \lambda I)_{i,\pi(i)}$ is nonzero. The matrix $\mathbf{K} - \lambda I$ has diagonal entries $(d_i - \lambda)$, subdiagonal entries $k_{n+1,n}$, corner entry $k_{1N}$, and all others zero.

Only two permutations have all nonzero factors:

(a) The identity permutation $\pi = \operatorname{id}$, contributing $\prod_i (d_i - \lambda)$.

(b) The full $N$-cycle $\pi = (1\;2\;\cdots\;N)$, which maps $i \mapsto i-1 \pmod{N}$. This permutation selects entries $(\mathbf{K}-\lambda I)_{i, i-1}$ for $i = 2, \ldots, N$ (the subdiagonal entries $k_{i,i-1}$) and $(\mathbf{K}-\lambda I)_{1,N} = k_{1N}$ (the corner entry). Its contribution is $\operatorname{sgn}(\pi) \cdot \prod k_{i,i-1} \cdot k_{1N}$. The sign of an $N$-cycle is $(-1)^{N-1}$. The product of off-diagonal entries is $P_{\text{cycle}}$. Thus the contribution is $(-1)^{N-1} P_{\text{cycle}}$.

Any other permutation (transpositions, 3-cycles, products of disjoint cycles, etc.) must include at least one factor $(\mathbf{K} - \lambda I)_{i, \pi(i)}$ where $\pi(i) \neq i$, $\pi(i) \neq i-1 \pmod{N}$, and $(i, \pi(i)) \neq (1, N)$. Such entries are zero by the sparsity of the cyclic-plus-diagonal structure.

The characteristic polynomial is therefore

$$\det(\mathbf{K} - \lambda I) = \prod_i(d_i - \lambda) + (-1)^{N-1} P_{\text{cycle}}$$

Equating to zero and noting that for $N = 4$: $(-1)^{N-1} = (-1)^3 = -1$, giving $p(\lambda) = \prod(d_i - \lambda) - P_{\text{cycle}} = 0$. $\blacksquare$

### 5.3. Spectral Radius

**Corollary 5.2** (Spectral radius). *The spectral radius $\rho(\mathbf{K})$ satisfies:*

*(i) Equal diagonals ($d_n = d$ for all $n$): $\rho(\mathbf{K}) = d + P_{\text{cycle}}^{1/N}$.*

*(ii) Zero diagonals ($d_n = 0$): $\rho(\mathbf{K}) = P_{\text{cycle}}^{1/N}$.*

*(iii) General: $\rho(\mathbf{K}) > \max_i d_i$ (Perron-Frobenius, since $\mathbf{K}$ is nonnegative irreducible with positive cycle), and $\rho(\mathbf{K}) \leq \max_i \sum_j K_{ij}$ (row-sum bound).*

*Proof.* (i) Setting $d_n = d$: $(d-\lambda)^N = P_{\text{cycle}}$, so $\lambda_k = d + P_{\text{cycle}}^{1/N} \cdot e^{2\pi i k/N}$ for $k = 0, \ldots, N-1$. The largest real eigenvalue is $\lambda_0 = d + P_{\text{cycle}}^{1/N}$. (ii) Set $d = 0$ in (i). (iii) Standard Perron-Frobenius bounds for nonnegative irreducible matrices. $\blacksquare$

**Theorem 5.3** (Bifurcation threshold). *The nontrivial equilibrium of the reduced system exists and is locally asymptotically stable if and only if $\rho(\mathbf{K}) > 1$. The bifurcation at $\rho(\mathbf{K}) = 1$ is a transcritical bifurcation exchanging stability between the trivial and nontrivial equilibria.*

This is the standard next-generation matrix threshold result (Diekmann, Heesterbeek, and Metz 1990). The system can be globally super-threshold ($\rho(\mathbf{K}) > 1$) from individually sub-threshold levels ($d_n < 1$ for all $n$) when $P_{\text{cycle}}^{1/N} > 1 - \max_i d_i$: the cycle amplification compensates for sub-threshold individual levels.

### 5.4. Where Curvature Enters the NGM

The CES curvature parameter $K$ enters the NGM through the cross-level coupling strengths $k_{n,n-1}$. In particular, the coupling from mesh density to CES capability involves the CES marginal product:

$$k_{32} = \frac{\phi_{\text{eff}}}{|\sigma_2|} \cdot \frac{\partial F_{\text{CES}}}{\partial F_2}\bigg|_{\bar{F}}$$

At the symmetric allocation, $\partial F_{\text{CES}}/\partial F_2 = 1/J$ (Proposition 1.1). Away from the symmetric point, this marginal product depends on $\rho$ through $F^{1-\rho}x_j^{\rho-1}$, and the curvature $K$ controls its sensitivity to reallocation.

---

## Part 6: Hierarchical Ceiling

### 6.1. Slow Manifold Cascade

**Proposition 6.1** (Ceiling functions). *Under the timescale ordering $\varepsilon_N \ll \cdots \ll \varepsilon_1$, successive equilibration of levels from fastest to slowest yields the cascade:*

*Level $N$ (fastest):* $h_N(F_{N-2}, F_{N-1})$ *from setting $\dot{F}_N = 0$.*

*Level $N-1$:* $h_{N-1}(F_{N-2})$ *from setting $\dot{F}_{N-1} = 0$ and substituting $F_N = h_N$.*

*Continuing until only Level 1 (slowest) has free dynamics.*

For the 4-level system:

**Level 4** (fastest): Setting $\dot{F}_4 = 0$ and factoring:

$$h_4(F_2, F_3) = \bar{S}(F_3)\left(1 - \frac{\nu}{\eta(F_3, F_2)}\right)$$

Existence requires $\eta(F_3, F_2) > \nu$, the local threshold condition. **Ceiling:** $F_4 \leq \bar{S}(F_3)$.

**Level 3**: Setting $\dot{F}_3 = 0$:

$$h_3(F_2) = \frac{\phi_{\text{eff}}}{\delta_C} \cdot F_{\text{CES}}(F_2)$$

**Ceiling:** $F_3 \leq (\phi_{\text{eff}}/\delta_C) \cdot F_{\text{CES}}(N^*(F_1))$. The CES diversity premium (controlled by $K$ through Theorem 8.1(a)) enters here: higher $K$ yields a larger diversity bonus in the capability aggregate.

**Level 2**: Setting $\dot{F}_2 = 0$:

$$h_2(F_1) = N^*(F_1)\left(1 - \frac{\mu}{\beta(F_1)}\right)$$

Existence requires $\beta(F_1) > \mu$. **Ceiling:** $F_2 \leq N^*(F_1)$.

### 6.2. Effective Dynamics and the Baumol Bottleneck

Substituting the cascade into the Level 1 dynamics:

$$\dot{F}_1 = \delta_c \cdot \Psi(F_1)^{\alpha} \cdot F_1^{\phi_c} - \gamma_c F_1$$

where $\Psi(F_1) = I(h_4(h_2(F_1), h_3(h_2(F_1))))$ is the composite feedback function encoding the entire cascade. The long-run growth rate is determined entirely by Level 1 — the slowest-adapting sector. The cascade of ceilings $F_1 \to h_2(F_1) \leq N^* \to h_3 \leq (\phi_{\text{eff}}/\delta_C)F_{\text{CES}}(N^*) \to h_4 \leq \bar{S}(h_3)$ bounds each level by a function of the level below it in the timescale hierarchy.

---

## Part 7: Lyapunov Structure and the Eigenstructure Bridge

### 7.1. Port-Hamiltonian Structure

**Proposition 7.1.** *The hierarchical CES system admits a port-Hamiltonian representation with dissipation:*

$$\dot{x} = [\mathcal{J}(x) - \mathcal{R}(x)]\nabla H(x) + \mathcal{G}(x)u$$

*where:*
- *$H = \Phi = -\sum_n \log F_n$ is the Hamiltonian (CES free energy);*
- *$\mathcal{R} = \operatorname{diag}(\sigma_n I_J)$ is the dissipation matrix (positive semidefinite);*
- *$\mathcal{J}$ encodes the directed coupling: it has lower-triangular block structure with rank-1 cross-level blocks;*
- *$\mathcal{G}$ encodes the exogenous input.*

Three structural consequences:

(a) $\mathcal{J}$ is lower-triangular with nonzero sub-diagonal blocks, so the system is **not** a gradient flow ($\mathcal{J} \neq 0$). This is a topological obstruction: no coordinate transformation can symmetrize a lower-triangular Jacobian.

(b) $\mathcal{J}$ has rank-1 cross-level blocks (of the form $c \cdot \mathbf{1}\mathbf{1}^T$), confirming that coupling passes through $F_n$ (Claim (i) of Theorem 3.1).

(c) $\mathcal{J}\nabla H \neq 0$, so the GENERIC degeneracy condition $L\nabla\Phi = 0$ fails. The directed coupling injects free energy from lower to higher levels. This is not a deficiency: it is the mechanism by which the bifurcation at $\rho(\mathbf{K}) = 1$ becomes possible.

**Remark.** The symmetric part $\mathcal{M} = \frac{1}{2}(Df + Df^T)$ is negative definite at the symmetric equilibrium (verified numerically for all tested parameter values). The system satisfies conditions (a) and (b) of the GENERIC framework but fails condition (c). It is "almost GENERIC" — the obstruction is specifically that the feed-forward coupling does not conserve the free energy.

### 7.2. The Storage Function

**Theorem 7.2** (Graph-theoretic Lyapunov function). *Define*

$$V(x) = \sum_{n=1}^{N} c_n \sum_{j=1}^{J} \left(\frac{x_{nj}}{x_{nj}^*} - 1 - \log\frac{x_{nj}}{x_{nj}^*}\right) = \sum_{n=1}^{N} c_n\, D_{KL}(x_n \| x_n^*)$$

*with tree coefficients $c_n = P_{\text{cycle}} / k_{n,n-1}$ (the cycle product divided by the incoming edge weight at node $n$). Then $V$ is a Lyapunov function for the hierarchical CES system near the nontrivial equilibrium: $V \geq 0$ with $V = 0$ iff $x = x^*$, and $\dot{V} \leq 0$.*

*Proof.* Nonnegativity follows from $g(z) = z - 1 - \log z \geq 0$ with equality iff $z = 1$.

Along trajectories:

$$\dot{V} = \sum_n c_n \sum_j \left(1 - \frac{x_{nj}^*}{x_{nj}}\right) f_{nj}(x)$$

The within-level contributions are

$$\dot{V}_{\text{within}} = -\sum_n c_n \sigma_n \sum_j \frac{(x_{nj} - x_{nj}^*)^2}{x_{nj}} \leq 0$$

The cross-level contributions cancel by the tree condition on $c_n$. This is the Li-Shuai-van den Driessche (2010) construction applied to the cycle-graph topology, using the Volterra-Lyapunov identity $a - b\log(a/b) \leq a - b + b\log(b/a)$ for each coupling term. The specific coefficients $c_n = P_{\text{cycle}}/k_{n,n-1}$ are precisely those required for the cancellation on the cycle graph.

For the 4-level cycle, there is exactly one spanning in-tree per root. The in-tree rooted at node $n$ uses all cycle edges except the one entering $n$, giving weight $P_{\text{cycle}}/k_{n,n-1}$. $\blacksquare$

### 7.3. The Eigenstructure Bridge

The free energy $\Phi$ and the storage function $V$ are different objects — $\Phi$ encodes geometry, $V$ controls dynamics — but they share a precise algebraic relationship.

**Theorem 7.3** (Eigenstructure Bridge). *On the slow manifold (scalar dynamics per level), the Hessians of $\Phi$ and $V$ at the nontrivial equilibrium satisfy*

$$\nabla^2\Phi\big|_{\text{slow}} = W^{-1} \cdot \nabla^2 V$$

*where $W = \operatorname{diag}(W_{11}, \ldots, W_{NN})$ is the **Bridge matrix** with entries*

$$W_{nn} = \frac{P_{\text{cycle}}}{|\sigma_n|\,\varepsilon_{T_n}}$$

*Here $\varepsilon_{T_n} = T_n'(\bar{F}_{n-1})\bar{F}_{n-1}/T_n(\bar{F}_{n-1})$ is the elasticity of the coupling at level $n$.*

*Proof.* On the slow manifold, $\Phi|_{\text{slow}} = -\sum_n \log F_n$ and $V = \sum_n c_n \bar{F}_n\,g(F_n/\bar{F}_n)$, where $g(z) = z - 1 - \log z$. Their Hessians at the equilibrium $F_n = \bar{F}_n$ are diagonal:

$$(\nabla^2\Phi|_{\text{slow}})_{nn} = \frac{1}{\bar{F}_n^2}, \qquad (\nabla^2 V)_{nn} = \frac{c_n}{\bar{F}_n}$$

The ratio is $(\nabla^2\Phi)_{nn}/(\nabla^2 V)_{nn} = 1/(c_n \bar{F}_n) = W_{nn}^{-1}$. Expressing $c_n = P_{\text{cycle}}/k_{n,n-1}$ and $k_{n,n-1} = T_n'(\bar{F}_{n-1})\bar{F}_{n-1}/|\sigma_{n-1}|$, together with the equilibrium relation $T_n(\bar{F}_{n-1}) = |\sigma_n|\bar{F}_n$:

$$c_n\bar{F}_n = \frac{P_{\text{cycle}}}{k_{n,n-1}}\cdot\bar{F}_n = P_{\text{cycle}}\cdot\frac{|\sigma_{n-1}|}{T_n'\bar{F}_{n-1}}\cdot\bar{F}_n = \frac{P_{\text{cycle}}}{|\sigma_n|}\cdot\frac{T_n}{T_n'\bar{F}_{n-1}} = \frac{P_{\text{cycle}}}{|\sigma_n|\,\varepsilon_{T_n}}$$

This gives $W_{nn} = c_n\bar{F}_n = P_{\text{cycle}}/(|\sigma_n|\,\varepsilon_{T_n})$. $\blacksquare$

### 7.4. Special Cases and Interpretation

**Power-law coupling** ($\phi_n(z) = a_n z^{\beta_n}$): The elasticity is constant, $\varepsilon_{T_n} = \beta_n$, giving $W_{nn} = P_{\text{cycle}}/(\beta_n|\sigma_n|)$.

**Linear coupling** ($\beta_n = 1$) with uniform damping ($\sigma_n = \sigma$): $W = (P_{\text{cycle}}/\sigma)\,I$, so $\nabla^2\Phi|_{\text{slow}} = (\sigma/P_{\text{cycle}})\,\nabla^2 V$. The free energy is proportional to the storage function — they are the same object up to scale.

**Physical interpretation.** The Bridge matrix $W$ encodes how three independent aspects of the system — graph topology (through $P_{\text{cycle}}$), dissipation (through $|\sigma_n|$), and coupling nonlinearity (through the elasticity $\varepsilon_{T_n}$) — together deform the natural free-energy geometry $\nabla^2\Phi$ into the Lyapunov geometry $\nabla^2 V$. When all three aspects are "uniform" (linear coupling, equal damping), the deformation is trivial and the two geometries coincide.

### 7.5. The Correct Mathematical Narrative

The CES free energy $\Phi$ is not the system's potential — the system is not a gradient flow (Proposition 7.1(a)). But $\Phi$ determines the system's geometry: it controls the curvature structure, eigenspace decomposition, and spectral properties at each level. The storage function $V$ is the dynamically admissible Lyapunov function, constructed from the graph topology via the Li-Shuai-van den Driessche method. The Bridge equation (Theorem 7.3) is the precise statement of how $V$ inherits $\Phi$'s geometry through the supply rate $W$.

The relationship is analogous to two well-known situations in mathematical physics:

(a) In statistical mechanics, the free energy determines equilibrium properties while the relative entropy (KL divergence) controls convergence to equilibrium. They are different objects encoding the same physics.

(b) In evolutionary game theory, the fitness landscape determines equilibria while the Shahshahani gradient structure controls dynamics. Not every evolutionary game is a potential game, but the eigenstructure of the fitness matrix still determines qualitative behavior.


---

## Part 8: The Triple Role of Curvature

The curvature parameter $K$ from Part 1 simultaneously controls three properties of the CES aggregate: superadditivity, correlation robustness, and strategic independence. These are not separate consequences of a common assumption; they are the same geometric fact — the curvature of the CES isoquant — viewed from three angles.

### 8.1. Part (a): Superadditivity

**Theorem 8.1** (Superadditivity). *For all $\mathbf{x}, \mathbf{y} \in \mathbb{R}_+^J \setminus \{\mathbf{0}\}$:*

$$F(\mathbf{x} + \mathbf{y}) \geq F(\mathbf{x}) + F(\mathbf{y})$$

*with equality if and only if $\mathbf{x} \propto \mathbf{y}$. The superadditivity gap satisfies*

$$F(\mathbf{x}+\mathbf{y}) - F(\mathbf{x}) - F(\mathbf{y}) \geq \frac{K}{4c}\cdot\frac{\sqrt{J}}{J-1}\cdot\min\!\bigl(F(\mathbf{x}),\, F(\mathbf{y})\bigr)\cdot d_{\mathcal{I}}(\hat{\mathbf{x}},\, \hat{\mathbf{y}})^2$$

*where $\hat{\mathbf{x}} = \mathbf{x}/F(\mathbf{x})$, $\hat{\mathbf{y}} = \mathbf{y}/F(\mathbf{y})$ are projections onto $\mathcal{I}_1$, $d_{\mathcal{I}}$ is geodesic distance on $\mathcal{I}_1$, and the bound holds locally near the symmetric point.*

*Proof.* **Step 1 (Qualitative — from concavity and homogeneity alone).** Write

$$\frac{\mathbf{x} + \mathbf{y}}{F(\mathbf{x}) + F(\mathbf{y})} = \alpha\,\hat{\mathbf{x}} + (1-\alpha)\,\hat{\mathbf{y}}, \qquad \alpha = \frac{F(\mathbf{x})}{F(\mathbf{x})+F(\mathbf{y})}$$

By degree-1 homogeneity, $F(\mathbf{x}+\mathbf{y}) = (F(\mathbf{x})+F(\mathbf{y}))\cdot F(\alpha\hat{\mathbf{x}}+(1-\alpha)\hat{\mathbf{y}})$. Since $F(\hat{\mathbf{x}}) = F(\hat{\mathbf{y}}) = 1$ and $F$ is concave:

$$F(\alpha\hat{\mathbf{x}}+(1-\alpha)\hat{\mathbf{y}}) \geq \alpha F(\hat{\mathbf{x}}) + (1-\alpha) F(\hat{\mathbf{y}}) = 1$$

So $F(\mathbf{x}+\mathbf{y}) \geq F(\mathbf{x})+F(\mathbf{y})$, with equality iff $\hat{\mathbf{x}} = \hat{\mathbf{y}}$, i.e., $\mathbf{x} \propto \mathbf{y}$. This uses only concavity and degree-1 homogeneity.

**Step 2 (Quantitative — from curvature comparison).** The point $\alpha\hat{\mathbf{x}}+(1-\alpha)\hat{\mathbf{y}}$ lies on the chord of the isoquant $\mathcal{I}_1$. By Proposition 1.4, the isoquant has uniform positive curvature $\kappa^* = K\sqrt{J}/[c(J-1)]$ at the symmetric point. The curvature comparison theorem for convex hypersurfaces (Toponogov comparison applied to 2-plane sections through $\mathbf{x}^*/c$; see do Carmo, *Riemannian Geometry*, Prop. 3.1) gives, for $\hat{\mathbf{x}}, \hat{\mathbf{y}}$ in a geodesic neighborhood of $\mathbf{x}^*/c$ with geodesic distance $d$:

$$F(\alpha\hat{\mathbf{x}}+(1-\alpha)\hat{\mathbf{y}}) \geq 1 + \frac{\kappa_{\min}}{2}\,\alpha(1-\alpha)\,d^2 + O(d^4)$$

Substituting $\kappa_{\min} = K\sqrt{J}/[c(J-1)]$ and using $\alpha(1-\alpha) \geq \min(\alpha,1-\alpha)/2$, with $\min(\alpha,1-\alpha)\cdot(F(\mathbf{x})+F(\mathbf{y})) = \min(F(\mathbf{x}),F(\mathbf{y}))$, yields the quantitative bound. $\blacksquare$

**Remark.** The CES isoquant has strictly positive normal curvature at every interior point, so a global quantitative bound exists using $\inf_{\mathbf{x}\in\mathcal{I}_c}\kappa_{\min}(\mathbf{x}) > 0$. The local bound at $\mathbf{x}^*$ gives the tightest constant near the cost-minimizing bundle.

### 8.2. Part (b): Correlation Robustness

**Theorem 8.2** (Correlation robustness). *Let $\mathbf{X} = (X_1, \ldots, X_J)$ be random with $\mathbb{E}[X_j] = x_j^*$ (the symmetric allocation) and equicorrelation covariance $\Sigma = \tau^2[(1-r)I + r\,\mathbf{1}\mathbf{1}^T]$ with $r \geq 0$. Let $\gamma_* = \tau/c$ be the coefficient of variation. Then the effective dimension $d_{\text{eff}} = J^2 g^2 \tau^2 / \operatorname{Var}[F(\mathbf{X})]$ satisfies, to second order in $\gamma_*$:*

$$d_{\text{eff}} \geq \frac{J}{1+r(J-1)} + \frac{K^2\,\gamma_*^2}{2}\cdot\frac{J(J-1)(1-r)}{[1+r(J-1)]^2}$$

*The first term is the linear baseline (achievable by any linear aggregate). The second is the **curvature bonus**, proportional to $K^2$ and increasing in idiosyncratic variation $(1-r)$.*

*Proof.* Expand $Y = F(\mathbf{X})$ around $x^*$. Let $\boldsymbol{\epsilon} = \mathbf{X} - x^*$.

The **linear term** $Y_1 = g\,\mathbf{1}\cdot\boldsymbol{\epsilon}$ (where $g = 1/J$ at equal weights) depends only on the common mode $\bar{\epsilon}$. Its variance is $\operatorname{Var}[Y_1] = g^2\tau^2 J[1+r(J-1)]$.

The **quadratic term** $Y_2 = \frac{1}{2}\boldsymbol{\epsilon}^T \nabla^2 F\,\boldsymbol{\epsilon}$ captures idiosyncratic variation through the CES Hessian. Decomposing $\boldsymbol{\epsilon} = \bar{\epsilon}\,\mathbf{1} + \boldsymbol{\eta}$ with $\mathbf{1}\cdot\boldsymbol{\eta} = 0$: $Y_2 = -\frac{(1-\rho)}{2Jc}\|\boldsymbol{\eta}\|^2$ at equal weights, depending purely on the idiosyncratic norm.

The variance of $Y_2$ under equicorrelation, via the Isserlis theorem for Gaussian $\boldsymbol{\epsilon}$, satisfies

$$\operatorname{Var}[Y_2]^{\text{idio}} \geq \frac{g^2 K^2 J^2}{2(J-1)c^2}\cdot\tau^4(1-r)^2$$

where the bound uses $(J-1)R_{\min}^2 \leq \sum_{k=1}^{J-1}\mu_k^2$ (minimum squared $\leq$ mean of squares) and the identity $(1-\rho)^2\Phi^{2/\rho}(J-1)R_{\min}^2 = K^2 J^2/(J-1)$.

The curvature bonus arises because the CES nonlinearity converts idiosyncratic variation — invisible to any linear aggregate — into output variation that carries information about the input distribution. By the Cramér-Rao bound, the Fisher information about the mean level $\mu$ carried by $Y_2$ is $\mathcal{I}_2 \geq (\partial_\mu\mathbb{E}[Y_2])^2/\operatorname{Var}[Y_2]$. Combining the linear and curvature information channels yields the stated bound. $\blacksquare$

**Remark.** For strict complements ($\rho < 0$) with bounded $\gamma_*$: $K > (J-1)/J$ at equal weights, so $K^2\gamma_*^2 J$ grows linearly in $J$, and $\bar{r} \to 1$ — nearly perfect correlation becomes tolerable.

### 8.3. Part (c): Strategic Independence

**Theorem 8.3** (Strategic independence). *For $J$ strategic agents controlling $x_j \geq 0$, any coalition $S \subseteq [J]$ with $|S| = k \geq 2$ has manipulation gain $\Delta(S) \leq 0$. For any redistribution $\boldsymbol{\delta}_S$ with $\sum_{j \in S}\delta_j = 0$:*

$$\Delta(S) \leq -\frac{K_S}{2k(k-1)c}\cdot\frac{k}{J}\cdot\frac{\|\boldsymbol{\delta}_S\|^2}{c} \leq 0$$

*where $K_S = (1-\rho)(k-1)\Phi^{1/\rho}R_{\min,S}/k$ is the coalition curvature parameter.*

*Proof.* **Step 1 (Qualitative — from convexity of the game).** The characteristic function $v(S) = \max_{\mathbf{x}_S \geq 0} F(\mathbf{x}_S, \mathbf{0}_{-S})$ defines a convex cooperative game (Shapley 1971, "Cores of Convex Games"), since $F$ is concave. The Shapley value lies in the core, and no coalition can profitably deviate. This holds for all weight vectors, without any curvature computation.

**Step 2 (Quantitative — from the constrained Rayleigh quotient).** A coalition redistribution $\boldsymbol{\delta}_S$ with $\sum_{j\in S}\delta_j = 0$ changes output by $\Delta F = \frac{1}{2}\boldsymbol{\delta}_S^T H_{SS}\,\boldsymbol{\delta}_S + O(\|\boldsymbol{\delta}\|^3)$. For $\boldsymbol{\delta}$ with $\sum_{j\in S}\delta_j = 0$, the Hessian quadratic form satisfies

$$\boldsymbol{\delta}_S^T H_{SS}\,\boldsymbol{\delta}_S = -\frac{(1-\rho)\,g\,\Phi^{1/\rho}}{c}\sum_{j\in S}\frac{\delta_j^2}{p_j} \leq -\frac{(1-\rho)\,g\,\Phi^{1/\rho}}{c}\cdot R_{\min,S}\cdot\|\boldsymbol{\delta}_S\|^2$$

using the constrained Rayleigh quotient on $S$. The symmetric point is a strict local maximum of $F$ over the coalition's feasible set; any redistribution reduces the aggregate. The quadratic loss, expressed through $K_S$, gives the stated bound. $\blacksquare$

**Remark (Equal weights).** At equal weights, $K_S = (1-\rho)(k-1)/k$ for all coalitions of size $k$, independent of which components are in $S$.

**Remark (General weights).** With general weights, the coalition curvature $K_S$ is computed via the secular equation restricted to $S$: $\sum_{j \in S}(w_j - \mu)^{-1} = 0$. The smallest root $R_{\min,S}$ determines $K_S$. The interlacing property of the secular equation ensures $K_S > 0$ for all coalitions of size $k \geq 2$, all $\rho < 1$, all weight vectors.

### 8.4. The Unified Perspective

$K$ enters linearly in Theorems 8.1 and 8.3 (first-order curvature effects from the Hessian) and quadratically in Theorem 8.2 (second-order effect: the variance of a Hessian quadratic form). The three roles are the same geometric fact — the strictly positive curvature of the CES isoquant at the cost-minimizing point — viewed from three perspectives:

- **Aggregation theory** (8.1): Curvature forces convex combinations of diverse points above the level set.
- **Information theory** (8.2): Curvature creates a nonlinear channel through which correlated inputs map to distinct output regions, extracting idiosyncratic information.
- **Game theory** (8.3): Curvature penalizes deviations from the balanced allocation, making the cost-minimizing bundle a Nash equilibrium.

For $\rho = 1$ (linear aggregation, $K = 0$): the isoquant is flat. Gap $= 0$, curvature bonus $= 0$, manipulation penalty $= 0$. All three properties vanish simultaneously.

---

## Part 9: Moduli Space Theorem

**Theorem 9.1** (Structural Determination). *Fix the CES parameter $\rho < 1$ and the structural integers $(J, N)$. The hierarchical CES system is determined as follows.*

*Qualitative invariants determined by $\rho$ (and $J$, $N$):*

1. *Within-level eigenstructure and curvature $K = (1-\rho)(J-1)/J$ (Propositions 1.2–2.2).*
2. *Coupling topology: aggregate, directed, nearest-neighbor, and port-aligned (Theorem 3.1).*
3. *Existence of a bifurcation threshold at $\rho(\mathbf{K}) = 1$ (Theorem 5.3).*
4. *Superadditivity, correlation robustness, and strategic independence, with bounds controlled by $K$ (Theorems 8.1–8.3).*
5. *The Eigenstructure Bridge relating the free-energy geometry to the Lyapunov geometry (Theorem 7.3).*

*Free parameters (quantitative degrees of freedom):*

1. *Timescales $(\varepsilon_1, \ldots, \varepsilon_N) \in \mathbb{R}_{++}^N$ with the ordering $\varepsilon_1 \gg \cdots \gg \varepsilon_N$.*
2. *Damping rates $(\sigma_1, \ldots, \sigma_N) \in \mathbb{R}_{++}^N$.*
3. *Gain functions $\phi_n: \mathbb{R}_+ \to \mathbb{R}_+$, $C^1$, $\phi_n(0) = 0$, $\phi_n' > 0$.*

*The free parameters determine: the equilibrium cascade $\{F_n^*\}$, the Lyapunov weights $\{c_n\}$, the Bridge matrix $W$, and the convergence rates. The qualitative dynamics — topology, threshold structure, stability mechanism, and the triple role of curvature — are invariant across the moduli space.*

*Proof.* Parts 1–8 collectively establish each qualitative invariant. The free parameters are identified by Theorem 3.1(iii) (gain functions free), Corollary 2.3 (timescales enter only through ratios), and the equilibrium conditions $\phi_n(F_{n-1}^*) = \sigma_n J F_n^*$ (which relate gains, damping, and equilibria without constraining $\phi_n$'s functional form). The Bridge matrix $W_{nn} = P_{\text{cycle}}/(|\sigma_n|\varepsilon_{T_n})$ depends on $\sigma_n$ and the elasticity of $\phi_n$, both of which are free. $\blacksquare$

---

## Part 10: Honest Limits

The framework has four structural limitations that cannot be resolved within its current scope.

**Limitation 1: Gain functions are not determined by $\rho$.** Theorem 3.1(iii) establishes this as a proved impossibility. The exponents and coefficients of the gain functions $\phi_n$ are genuine free parameters. Since the gain functions determine the equilibrium cascade $\{F_n^*\}$, which in turn sets the Lyapunov weights $\{c_n\}$ and the Bridge matrix $W$, this freedom propagates through the quantitative predictions.

**Limitation 2: Timescale separation cannot be eliminated.** The Fenichel persistence theorem (used in Theorem 3.1(i)) requires the spectral gap between transverse and tangential eigenvalues to be uniformly bounded away from zero. This is the condition $\varepsilon_n \ll \varepsilon_{n-1}$ for each $n$. Without timescale separation, the slow manifold need not exist, and the reduction from $NJ$ to $N$ dimensions is not justified. The nearest-neighbor topology result (Theorem 3.1(iv)) also requires timescale separation.

**Limitation 3: The system is not a gradient flow.** The lower-triangular Jacobian $\mathcal{J}$ is a topological obstruction (Proposition 7.1(a)). No coordinate transformation, metric choice, or potential modification can make the hierarchical CES system a gradient flow while preserving the directed coupling structure. The GENERIC framework fails at the degeneracy condition $\mathcal{J}\nabla\Phi \neq 0$ (Proposition 7.1(c)). The Lyapunov function $V$ is a storage function in the port-Hamiltonian sense, not a potential.

**Limitation 4: Global stability is not established.** Theorem 7.2 proves $\dot{V} \leq 0$, establishing local asymptotic stability of the nontrivial equilibrium. Global asymptotic stability would require additional boundary analysis: showing that trajectories starting at the boundary of $\mathbb{R}_+^{NJ}$ enter the basin of attraction of the interior equilibrium. This depends on the specific gain functions and is not a consequence of the CES geometry alone.

---

## References

1. Akin, E. (1979). *The Geometry of Population Genetics.* Lecture Notes in Biomathematics 31. Springer.

2. Diekmann, O., Heesterbeek, J. A. P., and Metz, J. A. J. (1990). On the definition and the computation of the basic reproduction ratio $R_0$ in models for infectious diseases in heterogeneous populations. *J. Math. Biol.* 28, 365–382.

3. do Carmo, M. P. (1992). *Riemannian Geometry.* Birkhäuser.

4. Fenichel, N. (1979). Geometric singular perturbation theory for ordinary differential equations. *J. Differential Equations* 31, 53–98.

5. Hirsch, M. W. (1985). Systems of differential equations which are competitive or cooperative. II: Convergence almost everywhere. *SIAM J. Math. Anal.* 16, 423–439.

6. Li, M. Y., Shuai, Z., and van den Driessche, P. (2010). Global-stability problem for coupled systems of differential equations on networks. *J. Differential Equations* 248, 1–20.

7. Shahshahani, S. (1979). A new mathematical framework for the study of linkage and selection. *Mem. Amer. Math. Soc.* 211.

8. Shapley, L. S. (1971). Cores of convex games. *Int. J. Game Theory* 1, 11–26.

9. Smith, H. L. (1995). *Monotone Dynamical Systems: An Introduction to the Theory of Competitive and Cooperative Systems.* AMS Mathematical Surveys and Monographs 41.

10. van der Schaft, A., and Jeltsema, D. (2014). Port-Hamiltonian systems theory: An introductory overview. *Foundations and Trends in Systems and Control* 1(2–3), 173–378.

---

*End of document.*
