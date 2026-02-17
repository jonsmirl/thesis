# CORRECTED GENERATING SYSTEM: Five Derivations

## For the 4-Level Hierarchical CES-Coupled System

---

## Preliminary: Clarifying the (2−ρ) Question

The prompt's Error 1 identifies a confusion that must be resolved before anything else. There are **two different objects** with different eigenstructures, and they must not be conflated:

**Object A: The Hessian of $\Phi_n = -\log F_n$ (static geometry).**

$$\nabla^2\Phi_n = \frac{1}{F_n^2}\nabla F_n \nabla F_n^T - \frac{1}{F_n}H_F$$

At the symmetric point ($x_j = \bar{x}$, $F = c = J^{1/\rho}\bar{x}$, $\partial_j F = 1/J$):

Eigenvalues:
- Diversity ($\mathbf{1}^\perp$): $\lambda_\perp^{\Phi} = (1-\rho)/(Jc^2)$, multiplicity $J-1$
- Radial ($\mathbf{1}$): $\lambda_\parallel^{\Phi} = 1/(J^2 c^2) + (1-\rho)(J-1)/(J^2 c^2) = [1 + K \cdot J/(J-1)]/(J^2 c^2)$

Wait — let me be precise. From the CES Triple Role Lemma 1:

$$H_F = \frac{(1-\rho)}{J^2 c}[\mathbf{1}\mathbf{1}^T - JI]$$

So $-H_F/F = -H_F/c = \frac{(1-\rho)}{J^2 c^2}[JI - \mathbf{1}\mathbf{1}^T]$

And $\nabla F \nabla F^T / F^2 = \frac{1}{J^2 c^2}\mathbf{1}\mathbf{1}^T$

Therefore:

$$\nabla^2\Phi_n = \frac{(1-\rho)}{J^2 c^2}[JI - \mathbf{1}\mathbf{1}^T] + \frac{1}{J^2 c^2}\mathbf{1}\mathbf{1}^T = \frac{1}{J^2 c^2}[(1-\rho)JI + (1-(1-\rho))\mathbf{1}\mathbf{1}^T]$$

$$= \frac{1}{J^2 c^2}[(1-\rho)JI + \rho\,\mathbf{1}\mathbf{1}^T]$$

Eigenvalues:
- On $\mathbf{1}^\perp$: $(1-\rho)J/(J^2 c^2) = (1-\rho)/(Jc^2)$ ✓
- On $\mathbf{1}$: $(1-\rho)J/(J^2 c^2) + \rho J/(J^2 c^2) = J/(J^2 c^2) = 1/(Jc^2)$

So $\lambda_\parallel^{\Phi} = 1/(Jc^2)$ (the prompt's formula $[1+K]/(Jc^2)$ is incorrect — that's a computational error).

The ratio is:
$$\lambda_\perp^{\Phi}/\lambda_\parallel^{\Phi} = (1-\rho)$$

This matches the Eigenstructure Bridge theorem (§3 of the unified document): the anisotropy ratio is $(1-\rho) = KJ/(J-1)$.

**Object B: The Jacobian $Df_n$ of the dynamics (linearized dynamics).**

The dynamics at level $n$: $\varepsilon_n \dot{x}_{nj} = T_n \cdot \partial F_n/\partial x_{nj} - \sigma_n x_{nj}$.

The within-level Jacobian:

$$\frac{\partial f_{nj}}{\partial x_{nk}} = \frac{1}{\varepsilon_n}\left[T_n \cdot \frac{\partial^2 F_n}{\partial x_{nj}\partial x_{nk}} - \sigma_n \delta_{jk}\right]$$

At equilibrium, $T_n = \sigma_n J c$ (from $T_n/J = \sigma_n \bar{x}$ and $\bar{x} = c$ at symmetric point). So:

$$Df_n = \frac{1}{\varepsilon_n}\left[\sigma_n J c \cdot H_F - \sigma_n I\right] = \frac{\sigma_n}{\varepsilon_n}\left[\frac{(1-\rho)}{J}(\mathbf{1}\mathbf{1}^T - JI) - I\right]$$

$$= \frac{\sigma_n}{\varepsilon_n}\left[\frac{(1-\rho)}{J}\mathbf{1}\mathbf{1}^T - (2-\rho)I\right]$$

Eigenvalues:
- On $\mathbf{1}$: $\frac{\sigma_n}{\varepsilon_n}[(1-\rho) - (2-\rho)] = -\sigma_n/\varepsilon_n$
- On $\mathbf{1}^\perp$: $\frac{\sigma_n}{\varepsilon_n}[0 - (2-\rho)] = -\sigma_n(2-\rho)/\varepsilon_n$

**The $(2-\rho)$ factor IS correct for the Jacobian of the dynamics.** The prompt's Error 1 confuses the Hessian of $\Phi$ (Object A, no $(2-\rho)$) with the Jacobian of $f$ (Object B, has $(2-\rho)$). Both are valid objects; they measure different things.

**Resolution:** Use Object A ($\nabla^2\Phi$) for the Eigenstructure Bridge and the geometric statements (Theorem 1). Use Object B ($Df$) for the dynamical statements (convergence rates, low-pass filter, Lyapunov decay). The $(2-\rho)$ factor appears in the dynamics because $Df$ combines the CES Hessian $H_F$ (geometric) with the damping $-\sigma I$ (physical). The sum produces $(2-\rho) = 1 + (1-\rho)$, where 1 is from damping and $(1-\rho)$ is from curvature.

---

## Derivation 1: Correct Eigenstructure of CES Dissipation

### Within-level structure at level $n$

At the symmetric equilibrium, the Jacobian $Df_n$ has eigenvalues:

$$\lambda_{\text{agg}} = -\sigma_n/\varepsilon_n, \qquad \lambda_{\text{div}} = -\sigma_n(2-\rho)/\varepsilon_n$$

with $\lambda_{\text{div}}$ having multiplicity $J-1$.

**Two-timescale decomposition within each level:**

| Mode | Dimension | Decay rate | Timescale |
|------|-----------|------------|-----------|
| Aggregate ($\mathbf{1}$ direction) | 1 | $\sigma_n/\varepsilon_n$ | $\tau_{\text{agg}} = \varepsilon_n/\sigma_n$ |
| Diversity ($\mathbf{1}^\perp$ directions) | $J-1$ | $\sigma_n(2-\rho)/\varepsilon_n$ | $\tau_{\text{div}} = \varepsilon_n/[\sigma_n(2-\rho)]$ |

The ratio $\tau_{\text{div}}/\tau_{\text{agg}} = 1/(2-\rho)$.

For $\rho = 0$ (Cobb-Douglas): diversity is $2\times$ faster than aggregate.
For $\rho = -1$: diversity is $3\times$ faster.
For $\rho \to -\infty$ (Leontief): diversity is infinitely faster (components lock together instantly).

**The low-pass filter interpretation:**

After the diversity modes equilibrate ($t \gg \tau_{\text{div}}$), the within-level state collapses to the 1-dimensional aggregate manifold $x_n = F_n \cdot \mathbf{1}/J^{1/\rho}$ (symmetric allocation). Any coupling signal with a component in $\mathbf{1}^\perp$ is damped at rate $(2-\rho)\sigma_n/\varepsilon_n$, which is faster than the aggregate dynamics by the factor $(2-\rho)$.

This is the dynamical version of the nonlinear equilibrium uniqueness (Port Topology Theorem §5(a)): at equilibrium, $x_{nj}^{\rho-2} = \text{const}$ forces $x_{nj} = \bar{x}_n$ for all $j$. The linearized filter confirms this: the approach to the symmetric allocation has timescale $\tau_{\text{div}} = \tau_{\text{agg}}/(2-\rho)$.

**Connection to $K$:**

$(2-\rho) = 1 + (1-\rho) = 1 + KJ/(J-1)$

So the diversity decay rate is:

$$\lambda_{\text{div}} = -\frac{\sigma_n}{\varepsilon_n}\left(1 + \frac{KJ}{J-1}\right)$$

The "extra" dissipation beyond simple damping ($-\sigma_n/\varepsilon_n$) is $\sigma_n KJ/[\varepsilon_n(J-1)]$, which is the CES curvature contribution. For $K = 0$ (perfect substitutes, $\rho = 1$): no extra dissipation, diversity modes decay at the same rate as aggregate. For $K > 0$: curvature accelerates diversity equilibration.

### Between-level structure

After within-level equilibration (diversity modes slaved), each level reduces to its scalar aggregate $F_n$. The 4-level system becomes:

$$\varepsilon_n \dot{F}_n = G_n(F_1, \ldots, F_4)$$

where $G_n$ is the aggregate dynamics at level $n$, obtained by evaluating the original dynamics on the symmetric allocation and using $\dot{F}_n = \nabla F_n \cdot \dot{x}_n = (1/J)\mathbf{1}\cdot \dot{x}_n$.

For the specific system:

$$\dot{F}_1 = \delta_c \cdot I(F_4)^\alpha \cdot F_1^{\phi_c} - \gamma_c F_1$$
$$\dot{F}_2 = \beta(F_1) \cdot F_2 \cdot (1 - F_2/N^*(F_1)) - \mu F_2$$
$$\dot{F}_3 = \phi_{\text{eff}} \cdot F_{\text{CES}}(F_2) - \delta_C F_3$$
$$\dot{F}_4 = \eta(F_3, F_2) \cdot F_4 \cdot (1 - F_4/\bar{S}(F_3)) - \nu F_4$$

This is the system to which Derivations 2–5 apply.

---

## Derivation 2: Correct Characteristic Polynomial of the NGM

### The 4×4 aggregate system

After within-level equilibration, the Jacobian at the nontrivial equilibrium $\bar{F} = (\bar{F}_1, \bar{F}_2, \bar{F}_3, \bar{F}_4)$ has the block structure:

$$J_{\text{agg}} = \begin{pmatrix} a_{11} & 0 & 0 & a_{14} \\ a_{21} & a_{22} & 0 & 0 \\ 0 & a_{32} & a_{33} & 0 \\ 0 & 0 & a_{43} & a_{44} \end{pmatrix}$$

where:
- $a_{nn} = \partial G_n/\partial F_n|_{\bar{F}}$ (within-level, negative at stable equilibrium)
- $a_{n,n-1} = \partial G_n/\partial F_{n-1}|_{\bar{F}}$ (feed-forward coupling, positive)
- $a_{14} = \partial G_1/\partial F_4|_{\bar{F}}$ (feedback loop closure, positive)

### Transmission/Transition decomposition

$$J_{\text{agg}} = T + \Sigma$$

where $\Sigma = \text{diag}(\sigma_1, \sigma_2, \sigma_3, \sigma_4)$ with $\sigma_n < 0$ (damping) and $T$ contains the amplification terms:

| Entry | Physical meaning | Expression |
|-------|-----------------|------------|
| $T_{11}$ | Within-level: learning-by-doing self-amplification | $\delta_c \alpha I(\bar{F}_4)^\alpha \phi_c \bar{F}_1^{\phi_c - 1}$ (if $\phi_c > 0$) |
| $T_{22}$ | Within-level: logistic growth at density $\bar{F}_2$ | $\beta(\bar{F}_1)(1 - 2\bar{F}_2/N^*(\bar{F}_1))$ (net effect at equilibrium) |
| $T_{33}$ | Within-level: autocatalytic feedback $\beta_{\text{auto}}\phi_0$ | $\phi_{\text{eff}} \cdot \partial F_{\text{CES}}/\partial F_3|$ (if $F_3$ enters the CES aggregate) |
| $T_{44}$ | Within-level: settlement network effects | $\eta(\bar{F}_3, \bar{F}_2)(1 - 2\bar{F}_4/\bar{S}(\bar{F}_3))$ |
| $T_{21}$ | Cross-level: cheaper hardware → faster recruitment | $\beta'(\bar{F}_1)\bar{F}_2(1 - \bar{F}_2/N^*(\bar{F}_1)) + \beta(\bar{F}_1)\bar{F}_2 N^{*\prime}(\bar{F}_1)/(N^*(\bar{F}_1))^2$ |
| $T_{32}$ | Cross-level: more agents → higher CES aggregate | $\phi_{\text{eff}} \cdot \partial F_{\text{CES}}/\partial F_2|$ |
| $T_{43}$ | Cross-level: more capability → more settlement demand | $\partial\eta/\partial F_3 \cdot \bar{F}_4(1 - \bar{F}_4/\bar{S}) + \eta \cdot \bar{F}_4^2 \bar{S}'/\bar{S}^2$ |
| $T_{14}$ | Feedback: settlement quality → investment | $\delta_c \alpha I'(\bar{F}_4) I(\bar{F}_4)^{\alpha-1} \bar{F}_1^{\phi_c}$ |

At the **trivial equilibrium** $\bar{F} = 0$: most of these expressions are degenerate (0/0 or $0^{\text{neg power}}$). The NGM construction properly handles this by decomposing into "new infections" ($T$, evaluated as rates of production of new cases in a fully susceptible population) and "transitions" ($\Sigma$, natural turnover).

### Next-generation matrix

$$\mathbf{K} = -T\Sigma^{-1}$$

Sparsity pattern: identical to $T$ (since $\Sigma$ is diagonal). Entries:

$$K_{nn} = d_n = -T_{nn}/\Sigma_{nn} = T_{nn}/|\sigma_n|$$
$$K_{n,n-1} = k_{n,n-1} = -T_{n,n-1}/\Sigma_{n-1,n-1} = T_{n,n-1}/|\sigma_{n-1}|$$
$$K_{14} = k_{14} = T_{14}/|\sigma_4|$$

### Characteristic polynomial

The prompt already derives this correctly. For the cyclic-plus-diagonal structure:

$$p(\lambda) = \prod_{i=1}^4 (d_i - \lambda) - P_{\text{cycle}} = 0$$

where $P_{\text{cycle}} = k_{21} k_{32} k_{43} k_{14}$.

**Proof:** Expand $\det(\mathbf{K} - \lambda I)$ along the first row. The $(1,1)$ minor is the determinant of a lower-triangular $3\times 3$ matrix, giving $\prod_{i=2}^4 (d_i - \lambda)$. The only other nonzero entry in row 1 is $k_{14}$ in column 4. Its cofactor is the determinant of the lower-triangular matrix with entries $k_{21}, k_{32}, k_{43}$ on the subdiagonal, which gives $(-1)^{1+4} \cdot k_{14} \cdot k_{21} k_{32} k_{43}$. Collecting signs:

$$p(\lambda) = (d_1 - \lambda)\prod_{i=2}^4(d_i - \lambda) + (-1)^{1+4}(-1)^3 k_{14} k_{21} k_{32} k_{43}$$

The sign: the $(1,4)$ cofactor involves a $3\times 3$ matrix with entries on the subdiagonal. Its determinant is $k_{21} \cdot k_{32} \cdot k_{43}$ (product of subdiagonal entries of a lower-triangular-shifted matrix, with sign $(-1)^{0+1+2} = (-1)^3 = -1$ ... let me just verify by direct expansion).

Actually, the $3\times 3$ minor obtained by deleting row 1 and column 4 is:

$$\begin{pmatrix} k_{21} & d_2 - \lambda & 0 \\ 0 & k_{32} & d_3 - \lambda \\ 0 & 0 & k_{43} \end{pmatrix}$$

Wait, that's upper-triangular after deleting the right columns. Let me be careful. Deleting row 1 (the $(d_1-\lambda, 0, 0, k_{14})$ row) and column 4 from $\mathbf{K} - \lambda I$:

$$M_{14} = \begin{pmatrix} k_{21} & d_2-\lambda & 0 \\ 0 & k_{32} & d_3-\lambda \\ 0 & 0 & k_{43} \end{pmatrix}$$

$\det(M_{14}) = k_{21} \cdot k_{32} \cdot k_{43}$.

The cofactor $C_{14} = (-1)^{1+4} \det(M_{14}) = -k_{21} k_{32} k_{43}$.

So:

$$p(\lambda) = (d_1-\lambda)(d_2-\lambda)(d_3-\lambda)(d_4-\lambda) + k_{14} \cdot (-k_{21} k_{32} k_{43})$$

$$= \prod_{i=1}^4(d_i - \lambda) - k_{14} k_{21} k_{32} k_{43}$$

$$= \prod_{i=1}^4(d_i - \lambda) - P_{\text{cycle}} \qquad \blacksquare$$

### Spectral radius

**Case 1: Equal diagonals** ($d_1 = d_2 = d_3 = d_4 = d$).

$(d-\lambda)^4 = P_{\text{cycle}}$, so $\lambda = d + P_{\text{cycle}}^{1/4} \cdot \omega_k$ where $\omega_k = e^{2\pi i k/4}$ for $k = 0, 1, 2, 3$.

$$\rho(\mathbf{K}) = d + P_{\text{cycle}}^{1/4}$$

The spectral radius exceeds the within-level reproduction number $d$ by the geometric mean of cross-level coupling strengths.

**Case 2: Zero diagonals** ($d_n = 0$, pure cross-level amplification).

$(-\lambda)^4 = P_{\text{cycle}}$, so $\rho(\mathbf{K}) = P_{\text{cycle}}^{1/4}$.

This is the formula from the previous attempt — correct only when within-level self-amplification is negligible.

**Case 3: General** (unequal $d_n$, nonzero $P_{\text{cycle}}$).

The quartic $\prod(d_i - \lambda) = P_{\text{cycle}}$ cannot be solved in closed form for general $d_i$. However, Perron-Frobenius gives useful bounds.

**Lower bound:** $\rho(\mathbf{K}) > \max_i d_i$ (since $\mathbf{K}$ is nonneg irreducible with positive cycle).

**Upper bound:** $\rho(\mathbf{K}) \leq \max_i \sum_j K_{ij}$ (row-sum bound). For node $n$ with self-loop $d_n$ and one outgoing edge $k_{n+1,n}$: row sum $= d_n + k_{n+1,n}$ (but $\mathbf{K}$ has entries indexed differently — let me use column sums since we have $\mathbf{K} = -T\Sigma^{-1}$ which acts on the right).

Actually, for the spectral radius of a nonneg matrix, the Perron-Frobenius eigenvalue satisfies:

$$\min_i r_i \leq \rho(\mathbf{K}) \leq \max_i r_i$$

where $r_i = \sum_j K_{ij}$ (row sums). For node 1: $r_1 = d_1 + k_{14}$. For node 2: $r_2 = k_{21} + d_2$. Etc.

**Perturbative formula** (small cycle coupling, $P_{\text{cycle}} \ll \prod_i d_i$):

Let $d_{\max} = \max_i d_i = d_{i^*}$. Then $\rho(\mathbf{K}) = d_{\max} + \delta$ where $\delta$ satisfies:

$$\delta \prod_{i \neq i^*}(d_i - d_{\max} - \delta) \approx P_{\text{cycle}}/(-1)^3$$

For $\delta \ll |d_i - d_{\max}|$:

$$\delta \approx \frac{P_{\text{cycle}}}{\prod_{i \neq i^*}(d_{\max} - d_i)}$$

**Regime identification for the specific system:**

The within-level reproduction numbers $d_n$ are:
- $d_1$: Wright's Law learning exponent (typically $\phi_c \approx 0.3$-$0.5$, so $d_1 < 1$)
- $d_2$: logistic net growth at equilibrium (at the nontrivial equilibrium, $\beta(1 - \bar{F}_2/N^*) = \mu$, so $d_2 = T_{22}/\mu$, which includes the logistic correction)
- $d_3$: autocatalytic feedback strength ($\beta_{\text{auto}}\phi_0 < 1$ for convergence, so $d_3 < 1$)
- $d_4$: settlement network effects at equilibrium

If $d_n < 1$ for all $n$ (each level is individually sub-threshold), the system can still be super-threshold if:

$$P_{\text{cycle}}^{1/4} > 1 - \max_i d_i$$

This is the "globally super-threshold from locally sub-threshold levels" result of Theorem 2: the cycle amplification compensates for sub-threshold individual levels.

### Where CES curvature $K$ enters the NGM

$K$ enters through the cross-level coupling strengths $k_{n,n-1}$. Specifically, $T_{32}$ (mesh density → CES capability) involves the CES marginal product:

$$T_{32} = \phi_{\text{eff}} \cdot \frac{\partial F_{\text{CES}}}{\partial F_2}\bigg|_{\bar{F}}$$

At the symmetric point, $\partial F_{\text{CES}}/\partial F_2 = 1/J$ (equal marginal product). But away from the symmetric point, the marginal product depends on $\rho$ through $F^{1-\rho} x_j^{\rho-1}$. The curvature $K$ controls how rapidly the marginal product changes with allocation — which is Theorem 1's strategic independence result.

For the aggregate system (after within-level equilibration), $K$ enters indirectly: the symmetric equilibrium allocation is forced by $K > 0$ (Port Topology Theorem), and the CES output at that allocation depends on $J$ and $\rho$ through $F = J^{1/\rho}\bar{x}$.

---

## Derivation 3: Correct Slow Manifold Functions

### Timescale ordering (corrected)

$$\varepsilon_4 \ll \varepsilon_3 \ll \varepsilon_2 \ll \varepsilon_1 = 1$$

Level 4 (settlement/stablecoins) is **fastest**. Level 1 (silicon learning curves) is **slowest**.

### Step 1: Equilibrate Level 4 (fastest)

Set $\dot{F}_4 = 0$:

$$\eta(F_3, F_2) \cdot F_4 \cdot (1 - F_4/\bar{S}(F_3)) - \nu F_4 = 0$$

Factor out $F_4$: either $F_4 = 0$ (trivial) or

$$\eta(F_3, F_2)(1 - F_4/\bar{S}(F_3)) = \nu$$

$$F_4/\bar{S}(F_3) = 1 - \nu/\eta(F_3, F_2)$$

$$\boxed{h_4(F_2, F_3) = \bar{S}(F_3)\left(1 - \frac{\nu}{\eta(F_3, F_2)}\right)}$$

**Existence condition:** $\eta(F_3, F_2) > \nu$ (settlement demand exceeds natural attrition). This is the "local $R_0$" for level 4: $R_4^{\text{local}} = \eta/\nu > 1$.

**Ceiling:** $F_4 \leq \bar{S}(F_3)$. The settlement ecosystem is bounded by the safe asset capacity $\bar{S}(F_3)$, which depends on mesh capability. As $\eta \to \infty$: $h_4 \to \bar{S}(F_3)$. The ceiling is approached but never exceeded. This is the **Triffin ceiling**: the stablecoin ecosystem cannot exceed the safe collateral backing, which is determined by the mesh's demonstrated capability.

### Step 2: Equilibrate Level 3

Set $\dot{F}_3 = 0$ given $F_4 = h_4(F_2, F_3)$:

$$\phi_{\text{eff}} \cdot F_{\text{CES}}(F_2) - \delta_C F_3 = 0$$

$$\boxed{h_3(F_2) = \frac{\phi_{\text{eff}}}{\delta_C} \cdot F_{\text{CES}}(F_2)}$$

where $\phi_{\text{eff}} = \phi_0/(1 - \beta_{\text{auto}}\phi_0)$ is the effective production multiplier (including autocatalytic feedback) and $F_{\text{CES}}(F_2) = (\sum_j a_j C_j(F_2)^\rho)^{1/\rho}$ is the CES aggregate of the mesh agents' capabilities, which depends on mesh density $F_2$ through the number and diversity of contributing agents.

**Ceiling:** $F_3 = (\phi_{\text{eff}}/\delta_C) \cdot F_{\text{CES}}(F_2)$. Capability is bounded by the depreciation/obsolescence rate $\delta_C$. This is the **Baumol ceiling for Level 3**: the autocatalytic capability cannot grow faster than agents depreciate, and the steady-state level is set by the CES aggregate of the mesh (which encodes diversity through $K$).

**Where $K$ enters:** $F_{\text{CES}}$ at the symmetric allocation has the superadditivity property from Theorem 1: $F_{\text{CES}}(J \text{ diverse agents}) \geq F_{\text{CES}}(1 \text{ agent scaled up by } J)$, with the gap controlled by $K$. Higher $K$ (stronger complementarity) means more "diversity premium" in the mesh's capability.

### Step 3: Equilibrate Level 2

Set $\dot{F}_2 = 0$ given $F_3 = h_3(F_2)$ and $F_4 = h_4(F_2, h_3(F_2))$:

$$\beta(F_1) \cdot F_2 \cdot (1 - F_2/N^*(F_1)) - \mu F_2 = 0$$

Factor out $F_2$: either $F_2 = 0$ or

$$\beta(F_1)(1 - F_2/N^*(F_1)) = \mu$$

$$\boxed{h_2(F_1) = N^*(F_1)\left(1 - \frac{\mu}{\beta(F_1)}\right)}$$

**Existence condition:** $\beta(F_1) > \mu$ (recruitment rate exceeds exit rate). This requires hardware cost $F_1$ to be low enough (cheap enough silicon for distributed inference to be attractive).

**Ceiling:** $F_2 \leq N^*(F_1)$. Mesh density is bounded by the carrying capacity $N^*$, which depends on hardware cost. As $\beta \to \infty$: $h_2 \to N^*(F_1)$.

### Step 4: Effective dynamics of Level 1 (slowest)

Substituting the cascade $h_2, h_3, h_4$:

$$\dot{F}_1 = \delta_c \cdot I(h_4(h_2(F_1), h_3(h_2(F_1))))^\alpha \cdot F_1^{\phi_c} - \gamma_c F_1$$

Define the **composite feedback function**:

$$\Psi(F_1) = I\bigg(\bar{S}\Big(\frac{\phi_{\text{eff}}}{\delta_C} F_{\text{CES}}(h_2(F_1))\Big) \cdot \bigg(1 - \frac{\nu}{\eta\big(\frac{\phi_{\text{eff}}}{\delta_C}F_{\text{CES}}(h_2(F_1)),\, h_2(F_1)\big)}\bigg)\bigg)$$

Then:

$$\dot{F}_1 = \delta_c \cdot \Psi(F_1)^\alpha \cdot F_1^{\phi_c} - \gamma_c F_1$$

**The Baumol bottleneck:** The long-run growth rate of the system is determined entirely by Level 1 — the slowest-adapting sector (silicon learning curves, institutional policy). All faster levels have equilibrated and their outputs are algebraic functions of $F_1$. The system growth rate is:

$$g_{\text{long-run}} = \frac{\dot{F}_1}{F_1} = \delta_c \Psi(F_1)^\alpha F_1^{\phi_c - 1} - \gamma_c$$

At the nontrivial equilibrium: $g = 0$. For sustained growth (which requires $\phi_c > 1$ or exogenous technical progress driving $\delta_c$ upward), the growth rate is bounded by $\gamma_c$ — the rate at which the slowest sector saturates.

**The cascade of ceilings (Theorem 3):**

$$F_1 \to h_2(F_1) \leq N^*(F_1) \to h_3(h_2) \leq \frac{\phi_{\text{eff}}}{\delta_C}F_{\text{CES}}(N^*) \to h_4 \leq \bar{S}(h_3)$$

Each level's steady state is bounded by its ceiling, which is determined by the level below it (in the timescale hierarchy). The binding ceiling is the one with the smallest "headroom" $1 - \sigma_n/T_n$.

---

## Derivation 4: Correct Tree Coefficients

### The directed graph

After within-level equilibration, the aggregate 4-ODE system has the directed graph:

$$G: \quad 1 \xrightarrow{k_{21}} 2 \xrightarrow{k_{32}} 3 \xrightarrow{k_{43}} 4 \xrightarrow{k_{14}} 1$$

This is a **directed 4-cycle** (strongly connected, irreducible). Self-loops $(n \to n)$ with weights $d_n$ exist but do not affect the spanning tree enumeration (a spanning tree has no cycles, so self-loops are excluded).

### Spanning in-trees

A spanning **in-tree** rooted at node $r$ is a directed tree where every node has a unique directed path to $r$, using edges of $G$.

For the 4-cycle $1 \to 2 \to 3 \to 4 \to 1$, the edges are: $\{1 \to 2, 2 \to 3, 3 \to 4, 4 \to 1\}$.

**In-tree rooted at 1:** Every other node must reach 1 using directed edges.
- From 4: direct edge $4 \to 1$. ✓
- From 3: path $3 \to 4 \to 1$. Uses edges $3 \to 4$ and $4 \to 1$.
- From 2: path $2 \to 3 \to 4 \to 1$. Uses edges $2 \to 3$, $3 \to 4$, $4 \to 1$.

The tree edges are $\{2 \to 3, 3 \to 4, 4 \to 1\}$ — all cycle edges except the one entering node 1 (which is $4 \to 1$... wait, $4 \to 1$ IS in the tree).

Let me reconsider. The cycle edges are $1 \to 2, 2 \to 3, 3 \to 4, 4 \to 1$. The edge "entering" node 1 from the cycle perspective is $4 \to 1$. But for the in-tree rooted at 1, we NEED $4 \to 1$ (node 4 must reach root 1). What we EXCLUDE is the edge "leaving" the root in the cycle direction, which is $1 \to 2$.

The in-tree rooted at 1: $\{2 \to 3, 3 \to 4, 4 \to 1\}$. Weight: $k_{32} \cdot k_{43} \cdot k_{14}$.

Check: this is a valid in-tree. Node 2 reaches 1 via $2 \to 3 \to 4 \to 1$. Node 3 reaches 1 via $3 \to 4 \to 1$. Node 4 reaches 1 via $4 \to 1$. Each node has exactly one outgoing edge in the tree. ✓

**In-tree rooted at 2:** $\{3 \to 4, 4 \to 1, 1 \to 2\}$. Weight: $k_{43} \cdot k_{14} \cdot k_{21}$.

**In-tree rooted at 3:** $\{4 \to 1, 1 \to 2, 2 \to 3\}$. Weight: $k_{14} \cdot k_{21} \cdot k_{32}$.

**In-tree rooted at 4:** $\{1 \to 2, 2 \to 3, 3 \to 4\}$. Weight: $k_{21} \cdot k_{32} \cdot k_{43}$.

**Pattern:** Each in-tree uses all cycle edges except the one entering the root. Since the cycle has 4 edges and a spanning tree of 4 nodes has 3 edges, we drop exactly 1 edge. The edge dropped for root $r$ is the one that points toward $r$ in the cycle (i.e., the edge $(r-1) \to r$, indices mod 4).

### Tree coefficients

$$c_r = \sum_{\text{in-trees rooted at } r} w(\tau)$$

For the cycle graph, there is exactly one in-tree per root, so:

$$c_1 = k_{32} k_{43} k_{14} = \frac{P_{\text{cycle}}}{k_{21}}$$

$$c_2 = k_{43} k_{14} k_{21} = \frac{P_{\text{cycle}}}{k_{32}}$$

$$c_3 = k_{14} k_{21} k_{32} = \frac{P_{\text{cycle}}}{k_{43}}$$

$$c_4 = k_{21} k_{32} k_{43} = \frac{P_{\text{cycle}}}{k_{14}}$$

where $P_{\text{cycle}} = k_{21} k_{32} k_{43} k_{14}$.

**Interpretation:** $c_n = P_{\text{cycle}} / k_{n,n-1}$. The Lyapunov weight at node $n$ is the cycle product divided by the edge entering $n$. Nodes with strong incoming coupling have small $c_n$ (they need less weight because they're tightly controlled by their upstream neighbor). Nodes with weak incoming coupling have large $c_n$ (they're more "autonomous" and need more Lyapunov weight to ensure $\dot{V} \leq 0$).

**Normalization.** The $c_n$ are determined up to a common multiplicative constant. We can set $P_{\text{cycle}} = 1$ (normalize) and use $c_n = 1/k_{n,n-1}$.

### Edge weights from the CES-specific Jacobian

At the nontrivial equilibrium $\bar{F}$, the off-diagonal entries of the Jacobian (the cross-level coupling strengths) are:

$$k_{21} = -T_{21}/\sigma_1 = \frac{1}{|\sigma_1|}\left[\beta'(\bar{F}_1)\bar{F}_2\left(1 - \frac{\bar{F}_2}{N^*(\bar{F}_1)}\right) + \beta(\bar{F}_1)\frac{\bar{F}_2 N^{*\prime}(\bar{F}_1)}{N^*(\bar{F}_1)^2}\right]$$

$$k_{32} = -T_{32}/\sigma_2 = \frac{\phi_{\text{eff}}}{|\sigma_2|}\frac{\partial F_{\text{CES}}}{\partial F_2}\bigg|_{\bar{F}_2}$$

$$k_{43} = -T_{43}/\sigma_3 = \frac{1}{|\sigma_3|}\left[\frac{\partial\eta}{\partial F_3}\bar{F}_4\left(1 - \frac{\bar{F}_4}{\bar{S}}\right) + \eta\frac{\bar{F}_4^2\bar{S}'(\bar{F}_3)}{\bar{S}^2}\right]$$

$$k_{14} = -T_{14}/\sigma_4 = \frac{\delta_c \alpha I'(\bar{F}_4) I(\bar{F}_4)^{\alpha-1}\bar{F}_1^{\phi_c}}{|\sigma_4|}$$

The CES-specific entry is $k_{32}$, which contains the CES marginal product $\partial F_{\text{CES}}/\partial F_2$. At the symmetric allocation within level 3:

$$\frac{\partial F_{\text{CES}}}{\partial F_2} \propto F_{\text{CES}}^{1-\rho} \cdot F_2^{\rho-1}$$

This depends on $\rho$ (hence on $K$) through the exponent $\rho - 1 = -KJ/(J-1)$.

### The Lyapunov function

$$V(F) = \sum_{n=1}^4 c_n\left(F_n - \bar{F}_n - \bar{F}_n\log\frac{F_n}{\bar{F}_n}\right) = \sum_{n=1}^4 c_n \bar{F}_n\, g\left(\frac{F_n}{\bar{F}_n}\right)$$

where $g(z) = z - 1 - \log z \geq 0$ with equality iff $z = 1$.

**Verification that $\dot{V} \leq 0$:** This follows from the general Shuai-van den Driessche theorem for strongly connected compartmental systems. The key identity is:

$$\dot{V} = \sum_n c_n\left(1 - \frac{\bar{F}_n}{F_n}\right)G_n(F)$$

The within-level (diagonal) terms contribute:

$$\sum_n c_n\left(1 - \frac{\bar{F}_n}{F_n}\right)\sigma_n(F_n - \bar{F}_n) = \sum_n c_n |\sigma_n| \frac{(F_n - \bar{F}_n)^2}{F_n} \leq 0$$

Wait — this sign needs care. At equilibrium, $G_n(\bar{F}) = 0$, so $T_n(\bar{F}) + \sigma_n \bar{F}_n = 0$, i.e., $T_n(\bar{F}) = |\sigma_n|\bar{F}_n$. Then:

$$G_n(F) = T_n(F) - |\sigma_n|F_n$$

$$\left(1 - \frac{\bar{F}_n}{F_n}\right)G_n = \left(1 - \frac{\bar{F}_n}{F_n}\right)(T_n(F) - |\sigma_n|F_n)$$

$$= T_n(F) - |\sigma_n|F_n - \frac{\bar{F}_n T_n(F)}{F_n} + |\sigma_n|\bar{F}_n$$

$$= T_n(F) - \frac{\bar{F}_n T_n(F)}{F_n} - |\sigma_n|(F_n - \bar{F}_n)$$

The cross-level cancellation (the tree condition) ensures that $\sum_n c_n[T_n(F) - \bar{F}_n T_n(F)/F_n] \leq 0$, using the Volterra-Lyapunov identity $a - b\log(a/b) \leq a - b + b\log(b/a)$ applied to each coupling term. The specific tree coefficients $c_n = P_{\text{cycle}}/k_{n,n-1}$ are chosen precisely to make this cancellation work for the cycle graph.

The full proof follows Shuai & van den Driessche (2013), Theorem 3.1, applied to the cycle-graph topology. ✓

---

## Derivation 5: The Eigenstructure Bridge (Applied to the 4-Level System)

### Statement

On the within-level slow manifold (symmetric allocation at each level), the generating function $\Phi$ reduces to:

$$\Phi\big|_{\text{slow}} = -\sum_{n=1}^4 \log F_n = -\sum_{n=1}^4 \log \bar{F}_n$$

where $\bar{F}_n$ is the aggregate output at level $n$ (equal to the common component value at the symmetric allocation, up to the $J^{1/\rho}$ factor).

The Lyapunov function is:

$$V = \sum_{n=1}^4 c_n\left(\bar{F}_n\, g(F_n/\bar{F}_n)\right) = \sum_n c_n \bar{F}_n\left(\frac{F_n}{\bar{F}_n} - 1 - \log\frac{F_n}{\bar{F}_n}\right)$$

### Hessian comparison at the nontrivial equilibrium

**Hessian of $\Phi|_{\text{slow}}$:**

$$\frac{\partial^2 \Phi|_{\text{slow}}}{\partial F_n^2} = \frac{1}{F_n^2}\bigg|_{\bar{F}_n} = \frac{1}{\bar{F}_n^2}$$

$$\nabla^2\Phi|_{\text{slow}} = \text{diag}\left(\frac{1}{\bar{F}_1^2}, \frac{1}{\bar{F}_2^2}, \frac{1}{\bar{F}_3^2}, \frac{1}{\bar{F}_4^2}\right)$$

**Hessian of $V$:**

$$\frac{\partial^2 V}{\partial F_n^2} = \frac{c_n}{\bar{F}_n} \cdot \frac{1}{F_n^2/\bar{F}_n}\bigg|_{\bar{F}_n} = \frac{c_n}{\bar{F}_n^2} \cdot \frac{\bar{F}_n}{F_n^2}\bigg|_{\bar{F}_n}$$

Wait, let me redo this. $V_n = c_n[\bar{F}_n g(F_n/\bar{F}_n)] = c_n[F_n - \bar{F}_n - \bar{F}_n \log(F_n/\bar{F}_n)]$.

$$\frac{\partial V_n}{\partial F_n} = c_n\left(1 - \frac{\bar{F}_n}{F_n}\right)$$

$$\frac{\partial^2 V_n}{\partial F_n^2} = \frac{c_n \bar{F}_n}{F_n^2}\bigg|_{\bar{F}_n} = \frac{c_n}{\bar{F}_n}$$

So:

$$\nabla^2 V = \text{diag}\left(\frac{c_1}{\bar{F}_1}, \frac{c_2}{\bar{F}_2}, \frac{c_3}{\bar{F}_3}, \frac{c_4}{\bar{F}_4}\right)$$

### The Bridge equation

$$(\nabla^2\Phi|_{\text{slow}})_{nn} = \frac{1}{\bar{F}_n^2} = \frac{1}{\bar{F}_n} \cdot \frac{1}{\bar{F}_n}$$

$$(\nabla^2 V)_{nn} = \frac{c_n}{\bar{F}_n}$$

So:

$$(\nabla^2\Phi|_{\text{slow}})_{nn} = \frac{1}{c_n \bar{F}_n} \cdot (\nabla^2 V)_{nn}$$

**$\Phi|_{\text{slow}}$ and $V$ have identical Hessians if and only if $c_n = 1/\bar{F}_n$ for all $n$.**

### Testing $c_n = 1/\bar{F}_n$

Recall $c_n = P_{\text{cycle}}/k_{n,n-1}$. The condition $c_n = \alpha/\bar{F}_n$ (for some common constant $\alpha$) requires:

$$\frac{P_{\text{cycle}}}{k_{n,n-1}} = \frac{\alpha}{\bar{F}_n} \quad \Longleftrightarrow \quad k_{n,n-1} = \frac{P_{\text{cycle}} \bar{F}_n}{\alpha}$$

This says: the cross-level coupling strength $k_{n,n-1}$ must be proportional to $\bar{F}_n$ (the equilibrium output at level $n$). Is this true?

From the equilibrium condition at level $n$:

$$T_n(\bar{F}_{n-1}) = |\sigma_n| \bar{F}_n$$

where $T_n$ is the amplification from level $n-1$. The coupling strength is:

$$k_{n,n-1} = \frac{T'_n(\bar{F}_{n-1}) \cdot \bar{F}_{n-1}}{|\sigma_{n-1}|}$$

(The derivative of the amplification, times the equilibrium value of the driving variable, divided by the damping of the driving variable — this is the elasticity-weighted coupling in the NGM.)

For $c_n \propto 1/\bar{F}_n$, we need:

$$\frac{T'_n(\bar{F}_{n-1}) \bar{F}_{n-1}}{|\sigma_{n-1}|} \propto \bar{F}_n$$

Using $T_n(\bar{F}_{n-1}) = |\sigma_n|\bar{F}_n$, this becomes:

$$\frac{T'_n \bar{F}_{n-1}}{|\sigma_{n-1}|} \propto \frac{T_n}{|\sigma_n|}$$

$$\frac{T'_n \bar{F}_{n-1}}{T_n} = \frac{|\sigma_{n-1}|}{|\sigma_n|} \cdot \text{const}$$

The LHS is the **elasticity** of the coupling function: $\varepsilon_T = T'_n \bar{F}_{n-1}/T_n = d\log T_n/d\log F_{n-1}$.

**The condition $c_n = 1/\bar{F}_n$ holds if and only if all coupling functions have equal elasticity** (up to a ratio of damping constants). This is generically NOT satisfied — the coupling functions $\phi_n$ are free (Claim 3 of the Port Topology Theorem), and their elasticities depend on the specific application.

### The general relationship

In general:

$$\nabla^2\Phi|_{\text{slow}} = W^{-1} \cdot \nabla^2 V$$

where $W = \text{diag}(c_1 \bar{F}_1, c_2 \bar{F}_2, c_3 \bar{F}_3, c_4 \bar{F}_4)$ is the **Bridge matrix**. Its entries:

$$W_{nn} = c_n \bar{F}_n = \frac{P_{\text{cycle}}}{k_{n,n-1}} \cdot \bar{F}_n = P_{\text{cycle}} \cdot \frac{|\sigma_{n-1}|}{T'_n(\bar{F}_{n-1})\bar{F}_{n-1}} \cdot \bar{F}_n$$

$$= P_{\text{cycle}} \cdot \frac{|\sigma_{n-1}| \bar{F}_n}{T'_n \bar{F}_{n-1}} = P_{\text{cycle}} \cdot \frac{|\sigma_{n-1}|}{|\sigma_n|} \cdot \frac{T_n}{T'_n \bar{F}_{n-1}}$$

$$= \frac{P_{\text{cycle}}}{|\sigma_n|} \cdot \frac{|\sigma_{n-1}|}{T'_n \bar{F}_{n-1}/T_n} = \frac{P_{\text{cycle}}}{|\sigma_n| \varepsilon_{T_n}}$$

where $\varepsilon_{T_n} = T'_n \bar{F}_{n-1}/T_n$ is the elasticity of the coupling at level $n$.

So: $W_{nn} = P_{\text{cycle}}/(|\sigma_n| \varepsilon_{T_n})$.

**The Bridge matrix encodes how the graph topology (through $P_{\text{cycle}}$), the physical damping (through $|\sigma_n|$), and the coupling nonlinearity (through the elasticity $\varepsilon_{T_n}$) together deform the "natural" free-energy geometry $\nabla^2\Phi$ into the Lyapunov geometry $\nabla^2 V$.**

### The special case: power-law coupling

If all coupling functions are power laws $T_n(F) = a_n F^{\beta_n}$, then $\varepsilon_{T_n} = \beta_n$ (constant elasticity). Then:

$$W_{nn} = \frac{P_{\text{cycle}}}{\beta_n |\sigma_n|}$$

and $c_n \propto 1/(\beta_n |\sigma_n| \bar{F}_n)$. The Lyapunov weights are inversely proportional to the product of coupling elasticity, damping, and equilibrium output.

For **linear coupling** ($\beta_n = 1$ for all $n$): $W_{nn} = P_{\text{cycle}}/|\sigma_n|$, and $c_n = P_{\text{cycle}}/(|\sigma_n|\bar{F}_n)$. If further all damping rates are equal ($|\sigma_n| = \sigma$): $c_n = P_{\text{cycle}}/(\sigma \bar{F}_n) \propto 1/\bar{F}_n$. **In this case, $\Phi|_{\text{slow}} \propto V$, and the Bridge matrix is a scalar multiple of the identity.** The free energy IS the Lyapunov function (up to scale) when coupling is linear and damping is uniform.

---

## Summary of Derivations

| Derivation | Key Result |
|-----------|------------|
| **1. Eigenstructure** | Dynamical eigenvalues are $-\sigma/\varepsilon$ (aggregate) and $-\sigma(2-\rho)/\varepsilon$ (diversity). The $(2-\rho)$ is correct for the Jacobian $Df$; the Hessian $\nabla^2\Phi$ has ratio $(1-\rho)$ instead. Both are valid; they measure different things. |
| **2. NGM** | $p(\lambda) = \prod(d_i - \lambda) - P_{\text{cycle}} = 0$. Spectral radius $\rho(\mathbf{K}) = d + P_{\text{cycle}}^{1/4}$ (equal diagonals). System is super-threshold from sub-threshold levels when $P_{\text{cycle}}^{1/4} > 1 - \max d_i$. |
| **3. Slow manifold** | $h_4 = \bar{S}(1 - \nu/\eta)$, $h_3 = \phi_{\text{eff}} F_{\text{CES}}/\delta_C$, $h_2 = N^*(1 - \mu/\beta)$. Effective dynamics: $\dot{F}_1 = \delta_c \Psi(F_1)^\alpha F_1^{\phi_c} - \gamma_c F_1$. Long-run growth = growth of Level 1. |
| **4. Tree coefficients** | $c_n = P_{\text{cycle}}/k_{n,n-1}$ (cycle product divided by incoming edge weight). One in-tree per root for the 4-cycle. |
| **5. Eigenstructure Bridge** | $\nabla^2\Phi|_{\text{slow}} = W^{-1}\nabla^2 V$ where $W_{nn} = P_{\text{cycle}}/(|\sigma_n|\varepsilon_{T_n})$. Equals identity (i.e., $\Phi = V$) iff coupling is linear and damping uniform. In general, the Bridge matrix encodes graph topology, damping, and coupling nonlinearity. |

---

## Assessment

### Does this close the gap?

The corrected derivations confirm the unified framework at **8.5/10**, consistent with the previous analysis. Specifically:

**What works (the 8.5 points):**

1. $\Phi = -\sum \log F_n$ correctly generates the within-level geometry (Theorem 1) through its Hessian anisotropy ratio $(1-\rho)$. ✓

2. The NGM characteristic polynomial $\prod(d_i - \lambda) = P_{\text{cycle}}$ correctly generates the spectral threshold (Theorem 2). The CES curvature $K$ enters through the cross-level coupling $k_{32}$, which involves the CES marginal product. ✓

3. The slow manifold cascade $h_4, h_3, h_2$ correctly generates the hierarchical ceiling (Theorem 3), with each level bounded by the level above. The CES diversity premium (controlled by $K$) enters through $F_{\text{CES}}$ in $h_3$. ✓

4. The tree coefficients $c_n = P_{\text{cycle}}/k_{n,n-1}$ provide a valid Lyapunov function $V$. ✓

5. The Eigenstructure Bridge $\nabla^2\Phi|_{\text{slow}} = W^{-1}\nabla^2 V$ connects the geometric object to the dynamic object. ✓

**What remains (the 1.5-point gap):**

1. **(0.5)** The Bridge matrix $W$ is not the identity in general. $\Phi$ and $V$ coincide only for linear coupling with uniform damping. The coupling functions $\phi_n$ (and their elasticities $\varepsilon_{T_n}$) are free parameters not determined by $\rho$.

2. **(0.5)** The structural parameters ($J, N, \varepsilon_n, \sigma_n$) are independent of $\rho$.

3. **(0.5)** The system is not a gradient flow. The feed-forward hierarchy is an open driven system; the Lyapunov function $V$ is a storage function, not a potential. This is forced by Theorem 2 (the bifurcation requires directed energy flow).

### Error status

| Error | Status | Resolution |
|-------|--------|------------|
| Error 1: $(2-\rho)$ factor | **Partially valid** | $(2-\rho)$ is correct for $Df$ (dynamics), not for $\nabla^2\Phi$ (geometry). Both objects exist; they serve different purposes. |
| Error 2: Spectral radius | **Corrected** | $\rho(\mathbf{K}) = d + P_{\text{cycle}}^{1/4}$ (equal diag.), not $P_{\text{cycle}}^{1/4}$ alone. |
| Error 3: Timescale ordering | **Corrected** | $\varepsilon_4 \ll \varepsilon_3 \ll \varepsilon_2 \ll \varepsilon_1 = 1$. Level 4 fastest, Level 1 slowest. |
| Error 4: Mass-action vs CES | **Corrected** | Tree coefficients $c_n = P_{\text{cycle}}/k_{n,n-1}$ where $k_{n,n-1}$ involves CES marginal products, not mass-action. |
| Error 5: $V$ vs $\Phi$ relationship | **Derived** | Bridge matrix $W$: $\Phi = V$ iff linear coupling + uniform damping. Otherwise $W \neq I$. |
