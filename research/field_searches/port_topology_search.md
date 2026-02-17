# PROMPT: IS THE PORT TOPOLOGY FORCED BY THE CES GEOMETRY?

## Closing the Gap from 7.5 to 9/10

---

## WHAT THIS IS

A series of computations has established that the hierarchical CES system is best described as a **port-Hamiltonian network**: each level is a dissipative system with Hamiltonian $H_n = -\log F_n$ and dissipation controlled by $K = (1-\rho)(J-1)/J$, coupled through directed ports where the output of level $n$ feeds the input of level $n+1$.

The current score is **7.5/10**. The three theorems all emerge from this structure:

- **Theorem 1:** The dissipation matrix $R_n$ has eigenvalue gap controlled by $K$.
- **Theorem 2:** The power balance $\dot{H} = -\nabla H^T R \nabla H + \nabla H^T B u$ determines the activation threshold $\rho(K) = 1$.
- **Theorem 3:** Port-Hamiltonian singular perturbation produces the slow manifold.

**The 1.5-point gap** comes from the fact that the port interconnection structure — which level drives which, with what gain — appears to be **additional input beyond $\rho$**. If the port topology is a free choice (you could wire level 1 to level 3 instead of level 2, or make the coupling bidirectional, or change the gain function $\phi$), then $\rho$ is not the single control parameter of the entire system. It controls the within-level dissipation, but the between-level architecture is specified separately.

**The question:** Is there a sense in which the CES geometry **forces** the port topology? Can we derive from the properties of the CES function that the coupling must be:
1. Directed (feed-forward, not bidirectional)
2. Nearest-neighbor (level $n$ drives level $n+1$, not level $n+2$)
3. Through the CES output $F_n$ (not through individual components $x_{nj}$)
4. With gain determined by $\rho$ (not by an independent parameter)

If yes, then $\rho$ determines everything and the score reaches 9/10. If no, then the port topology is genuinely independent of the CES geometry, and 7.5 is the correct ceiling.

---

## WHAT HAS BEEN ESTABLISHED

### The Port-Hamiltonian Structure (proven)

Each level $n$ is a dissipative port-Hamiltonian system:

$$\varepsilon_n \dot{x}_n = -R_n \nabla H_n(x_n) + B_n u_n$$

where:
- $H_n(x_n) = -\log F_n(x_n)$ is the Hamiltonian (negative log of CES aggregate)
- $R_n$ is the dissipation matrix, symmetric positive definite, with eigenvalues:
  - $\sigma_n / \varepsilon_n$ on the mean-field direction $\mathbf{1}$
  - $\sigma_n(2-\rho) / \varepsilon_n$ on the tangent space $\mathbf{1}^\perp$ ($(J-1)$-fold degenerate)
  - The gap $(1-\rho)\sigma_n / \varepsilon_n = K \cdot J\sigma_n / [(J-1)\varepsilon_n]$ is controlled by $K$
- $B_n u_n$ is the port input from the level below: $u_n = \phi_n(F_{n-1}(x_{n-1}))$

The port output of level $n$ is $y_n = F_n(x_n)$ (the CES aggregate output).

### The GENERIC Failure (proven)

The feed-forward hierarchy is not a closed thermodynamic system. The degeneracy condition $L\nabla\Phi = 0$ fails because level $n-1$ injects energy into level $n$ without receiving anything back. This is not a technical obstacle but a structural property of open driven systems.

### The Lyapunov Function (proven)

$V(x) = \sum_{n=1}^{N} \alpha_n [F_n/F_n^* - \log(F_n/F_n^*) - 1]$ is a Lyapunov function for the system, with $\dot{V} \leq 0$ globally (for linear coupling $\phi$). The weights $\alpha_n$ depend on the coupling topology and the equilibrium values, not solely on $\rho$.

### The Residual Gap

The port-Hamiltonian structure says: "each node is a CES dissipative system; the wiring between nodes is specified separately." The question is whether "the wiring between nodes" follows from the CES structure or is genuinely independent.

---

## THE FOUR CLAIMS TO INVESTIGATE

### Claim 1: The coupling must pass through $F_n$ (the CES output), not through individual components $x_{nj}$

**Statement:** In a port-Hamiltonian network of CES dissipative systems, the only coupling that is consistent with the Hamiltonian structure $H_n = -\log F_n$ is one that depends on $x_{n-1}$ only through $F_{n-1}(x_{n-1})$.

**Why this might be true:** The port-Hamiltonian framework requires that the port output $y_n$ be the "co-energy variable" conjugate to the port input $u_n$. For $H_n = -\log F_n$, the natural output is:

$$y_n = B_n^T \nabla H_n = B_n^T \left(-\frac{\nabla F_n}{F_n}\right)$$

If $B_n = \nabla F_n / \|\nabla F_n\|$ (port aligned with the gradient of $F_n$), then:

$$y_n = -\frac{\|\nabla F_n\|^2}{F_n}$$

At the symmetric point, $\|\nabla F_n\|^2 = 1/J$, so $y_n = -1/(JF_n)$. This is a function of $F_n$ only.

**The argument:** The CES function is homogeneous of degree 1, so $\sum_j x_{nj} \partial F_n / \partial x_{nj} = F_n$ (Euler's theorem). At the symmetric point, the gradient $\nabla F_n = (1/J)\mathbf{1}$ points in the direction $\mathbf{1}$. Any port input that is NOT aligned with $\mathbf{1}$ would have a component in $\mathbf{1}^\perp$, which the dissipation matrix $R_n$ would damp at the enhanced rate $\sigma_n(2-\rho)/\varepsilon_n$. So the "surviving" port signal — the one that persists on the slow manifold — is the projection onto $\mathbf{1}$, which depends only on $F_n$.

**More precisely:** Decompose any port input $u_n \in \mathbb{R}^J$ into $u_n = u_n^{\parallel}\mathbf{1}/J + u_n^{\perp}$ where $\mathbf{1} \cdot u_n^{\perp} = 0$. The response to $u_n^{\parallel}$ decays at rate $\sigma_n/\varepsilon_n$; the response to $u_n^{\perp}$ decays at rate $\sigma_n(2-\rho)/\varepsilon_n$. For $\rho < 1$, the perpendicular decay is faster by a factor $(2-\rho)$. On the slow manifold (after within-level equilibration), only the $\mathbf{1}$ component survives. The $\mathbf{1}$ component of $\nabla F_n$ is $\|\nabla F_n \cdot \mathbf{1}\|/J = F_n^{1-\rho} \cdot \bar{x}_n^{\rho-1}/J = 1/J$. So the effective port output is $(1/J)\sum_j \dot{x}_{nj}$, which equals $\dot{F}_n / J$ at the symmetric point.

**Conclusion:** The CES dissipation structure (specifically, the eigenvalue gap in $R_n$ controlled by $K$) acts as a **low-pass filter** that strips out all information about the internal allocation $x_{nj}$ and passes only the aggregate $F_n$. The coupling must pass through $F_n$ because any coupling through individual components is damped away by the CES curvature.

**Computation needed:** Verify that for the nonlinear system (not just the linearization), the slow manifold condition $\nabla_{x_n^{\perp}} H_n = 0$ implies $x_n \propto \mathbf{1}$ (all components equal), so that $F_n$ is a sufficient statistic for level $n$'s state on the slow manifold.

### Claim 2: The coupling must be directed (feed-forward)

**Statement:** If each level's Hamiltonian is $H_n = -\log F_n$ and the CES parameter $\rho < 1$ is the same at all levels, then the port-Hamiltonian interconnection must be **causal** (directed acyclic graph) rather than bidirectional, as a consequence of the passivity properties of the CES dissipative systems.

**Why this might be true:** A port-Hamiltonian system is **passive** if $\dot{H} \leq y^T u$ (the stored energy decreases except for energy supplied through the port). For a CES dissipative system:

$$\dot{H}_n = \nabla H_n^T \dot{x}_n = -\nabla H_n^T R_n \nabla H_n + \nabla H_n^T B_n u_n$$

The first term is $\leq 0$ (dissipation). The second term is the port power. The system is passive with supply rate $y_n^T u_n$.

**If the coupling were bidirectional** ($u_n$ depends on $x_{n-1}$ AND $u_{n-1}$ depends on $x_n$), then the interconnection creates a feedback loop. For the closed-loop system to be stable, the interconnection must satisfy a **passivity theorem**: the composition of two passive systems is passive if the interconnection is "power-preserving" (the power extracted from one system equals the power injected into the other).

But the CES dissipative systems are **strictly passive** (the dissipation term is strictly negative). A bidirectional interconnection of strictly passive systems is always stable (by the passivity theorem), regardless of the coupling strength. This means $\rho(K) = 1$ would never be reached — the system would always be sub-threshold.

**The argument (sketch):** In a bidirectional coupling, the total stored energy $H_{\text{total}} = \sum H_n$ satisfies:

$$\dot{H}_{\text{total}} = -\sum_n \nabla H_n^T R_n \nabla H_n + \sum_n y_n^T u_n$$

If the interconnection is power-preserving ($\sum_n y_n^T u_n = 0$), then $\dot{H}_{\text{total}} \leq 0$ always, and the trivial equilibrium is globally asymptotically stable for all coupling strengths. No bifurcation at $\rho(K) = 1$.

For the bifurcation to exist, the interconnection must be **non-power-preserving**: the port must inject net energy into the system. This requires an **external energy source**, which in the hierarchical CES system is the exogenous input at the lowest level ($T_1 = \beta_1$). The energy flows unidirectionally from the source through the hierarchy: level 1 → level 2 → ... → level $N$.

**Conclusion:** The existence of the bifurcation (Theorem 2) requires that the interconnection be **non-power-preserving**, which requires directed (non-reciprocal) coupling with an external source. Bidirectional power-preserving coupling would make the system unconditionally stable, eliminating Theorem 2. So **Theorem 2 forces the coupling to be directed.**

**Computation needed:**
1. For the 2-level system with bidirectional coupling ($u_2 = \phi(F_1)$ AND $u_1 = \psi(F_2)$), compute $\dot{H}_{\text{total}}$ and show it is always $\leq 0$ when the interconnection is power-preserving.
2. For the unidirectional coupling ($u_2 = \phi(F_1)$, $u_1 = \beta_1$), show that $\dot{H}_{\text{total}}$ can be positive when $\beta_1$ is large enough, and that the threshold is $\rho(K) = 1$.

### Claim 3: The port gain is determined by $\rho$

**Statement:** The gain of the port coupling — how strongly $F_{n-1}$ drives level $n$ — is not an independent parameter but is determined by the CES parameter $\rho$ (and the component count $J$) through the requirement that the coupled system have a nontrivial equilibrium.

**Why this might be true:** At the symmetric equilibrium, the balance equation is:

$$T_n / J = \sigma_n \bar{x}_n \quad \Rightarrow \quad \phi_n(F_{n-1}^*) = \sigma_n J F_n^*$$

The CES output at level $n$ is $F_n^* = \bar{x}_n$. If $\phi_n$ is a power law $\phi_n(z) = a_n z^{\beta_n}$, then:

$$a_n (F_{n-1}^*)^{\beta_n} = \sigma_n J F_n^*$$

The equilibrium cascade from level 1 to level $N$ determines $F_n^*$ as a function of $F_1^*$ (or equivalently, of $\beta_1$). The exponents $\beta_n$ are free parameters — they are NOT determined by $\rho$.

**Why this might be false:** The gain function $\phi_n$ describes how one level's output becomes the next level's input. This is a property of the **application** (how AI agent output feeds stablecoin dynamics, how stablecoin dynamics feeds monetary policy, etc.), not of the CES aggregation. Two systems could have the same CES parameter $\rho$ but completely different port gains $\phi_n$.

**However:** The CES structure may constrain the *class* of allowable $\phi_n$. Specifically, for the coupled system to have a symmetric equilibrium (all $J$ components equal at each level), the port input must be compatible with the CES symmetry: $B_n u_n$ must be proportional to $\mathbf{1}$ (equal input to all components). If $u_n$ is a scalar (function of $F_{n-1}$ only), then $B_n u_n = u_n \nabla F_n$ (port input aligned with CES gradient), which is indeed proportional to $\mathbf{1}$ at the symmetric point. Any other port structure would break the within-level symmetry.

**Computation needed:**
1. For a general port matrix $B_n$ (not aligned with $\nabla F_n$), show that the symmetric equilibrium does not exist (or is unstable).
2. For $B_n = \nabla F_n$ (the natural CES-aligned port), compute the gain function $\phi_n$ that is consistent with the power balance at threshold $\rho(K) = 1$. Does $\phi_n$ depend on $\rho$?

### Claim 4: The nearest-neighbor topology is forced by timescale separation

**Statement:** In a port-Hamiltonian network with timescale separation $\varepsilon_1 \ll \varepsilon_2 \ll \ldots \ll \varepsilon_N$, the effective coupling on the slow manifold is nearest-neighbor: level $n$ effectively couples only to levels $n-1$ and $n+1$, even if the original coupling includes longer-range connections.

**Why this might be true:** On the slow manifold, fast levels have equilibrated. Level 1 (fastest) reaches equilibrium first; its output $F_1^*$ becomes an algebraic function of the other states. Level 2 (next fastest) then equilibrates given $F_1^*$ and the slower states. The cascade of equilibrations proceeds from fast to slow.

If level 1 is coupled to level 3 (skipping level 2), the effect of this coupling on the slow manifold is: level 1 equilibrates → its output $F_1^*$ is determined by the slow states → the coupling to level 3 becomes $\phi_{31}(F_1^*)$, which is now an algebraic function of the slow states. But $F_1^*$ is itself determined by the coupling from level 0 (or exogenous input). So the level 1 → level 3 coupling is **mediated** by the equilibrium at level 1, which depends on level 1's own inputs. On the slow manifold, this is indistinguishable from a nearest-neighbor chain: level 0 → level 1 (equilibrium) → level 2 (equilibrium) → level 3.

**The argument:** Define the reduced system on the slow manifold. After the $k$ fastest levels have equilibrated, the state is $(F_1^*, \ldots, F_k^*, x_{k+1}, \ldots, x_N)$ where $F_m^*$ solves $\nabla_{x_m} H_m = 0$ given the slower states. Any coupling from level $m \leq k$ to level $n > k$ is a function of $F_m^*$, which is itself determined by $(F_{m-1}^*, \sigma_m, \varepsilon_m)$. The chain of dependencies is:

$$F_1^* = F_1^*(\beta_1, \sigma_1) \to F_2^* = F_2^*(\phi_2(F_1^*), \sigma_2) \to \ldots$$

This is a **nearest-neighbor chain** regardless of the original coupling topology, because each level's equilibrium output is determined by the level immediately below it.

**Computation needed:** For a 3-level system with coupling from level 1 to both level 2 and level 3, show that on the slow manifold (level 1 equilibrated), the effective dynamics of levels 2 and 3 depend on level 1 only through $F_1^*$, and that the coupling from level 1 to level 3 can be absorbed into the coupling from level 2 to level 3 (since $F_2^*$ is itself a function of $F_1^*$).

---

## WHAT WOULD CONSTITUTE AN ANSWER

### Full Success (9/10): All Four Claims Proven

All four claims hold, and therefore:
- The CES dissipation structure forces coupling through $F_n$ (Claim 1)
- The bifurcation theorem forces directed coupling (Claim 2)
- The CES symmetry constrains the port structure to $B_n = \nabla F_n$ (Claim 3, partial)
- Timescale separation forces nearest-neighbor topology (Claim 4)

Together, these mean: given $\rho$, $J$, $N$, the timescales $\varepsilon_n$, and the damping rates $\sigma_n$, the port-Hamiltonian network topology is determined. The "additional input" beyond $\rho$ is reduced to the physically necessary parameters ($J$, $N$, $\varepsilon_n$, $\sigma_n$) which are not geometric but structural (how many levels, how many components, how fast, how much damping). The CES geometry (controlled by $\rho$) determines the qualitative topology; the structural parameters determine the quantitative gains.

**This is 9/10** because $\rho$ determines all the qualitative features (directed, nearest-neighbor, aggregate-coupled, with dissipation gap $K$), while the remaining parameters ($J$, $N$, $\varepsilon_n$, $\sigma_n$) are genuinely independent structural choices. The missing 1 point is that $J$ and $N$ are not determined by $\rho$ — you must still choose how many levels and how many components per level.

### Partial Success (8.5/10): Claims 1 and 2 Proven, Claims 3 and 4 Partial

The coupling passes through $F_n$ (Claim 1) and must be directed (Claim 2), but the gain function $\phi_n$ is genuinely free (Claim 3 fails) and the nearest-neighbor structure requires timescale separation (Claim 4 is conditional, not forced by CES alone).

**This is 8.5** because the CES geometry forces the essential qualitative features (aggregate coupling, directedness) but leaves quantitative features (gain, topology) to the application.

### Honest Negative (7.5 stands): One or More Claims Fail

Possible failure modes:
- **Claim 1 fails:** There exist stable equilibria where coupling passes through individual components $x_{nj}$, not just $F_n$. The CES curvature does not act as a low-pass filter for the port signal.
- **Claim 2 fails:** Bidirectional coupling CAN produce a bifurcation (the passivity argument is wrong because the systems are not strictly passive at the nontrivial equilibrium).
- **Claim 4 fails:** Long-range coupling (level 1 → level 3) is NOT absorbed on the slow manifold; it produces qualitatively different behavior (e.g., different Perron-Frobenius eigenvector) that cannot be replicated by nearest-neighbor coupling.

Any of these would confirm that the port topology is genuinely independent of the CES geometry and that 7.5 is the correct score.

---

## SPECIFIC COMPUTATIONS TO PERFORM

### Computation A: CES Curvature as Low-Pass Filter (Claim 1)

**For a single CES level** with $J$ components, Hamiltonian $H = -\log F$, dissipation matrix $R$, and port input $u \in \mathbb{R}^J$:

1. Decompose $u = u^{\parallel}(\mathbf{1}/\sqrt{J}) + u^{\perp}$ where $\mathbf{1} \cdot u^{\perp} = 0$.
2. Compute the steady-state response $\delta x_{\text{ss}}$ to a constant port input $u$ by solving $R \nabla^2 H \cdot \delta x = -u$ (linearized).
3. Show that $\|\delta x_{\text{ss}}^{\perp}\| / \|\delta x_{\text{ss}}^{\parallel}\| = O((2-\rho)^{-1}) = O(1/(1 + K \cdot J/(J-1)))$.
4. Conclude: for $\rho \ll 1$ (strong complementarity, large $K$), the perpendicular response is suppressed by factor $K$. The effective port output is $F_n$ (the aggregate), not the individual $x_{nj}$.

### Computation B: Passivity and Bidirectional Coupling (Claim 2)

**For the 2-level system** with bidirectional coupling:

$$\varepsilon_1 \dot{x}_1 = T_1 \nabla F_1 - \sigma_1 x_1 + \psi(F_2) \nabla F_1$$
$$\varepsilon_2 \dot{x}_2 = \phi(F_1) \nabla F_2 - \sigma_2 x_2$$

1. Compute $\dot{H}_{\text{total}} = \dot{H}_1 + \dot{H}_2$ where $H_n = -\log F_n$.
2. Show that if the interconnection is power-preserving ($\phi(F_1) \cdot (-\nabla H_2^T \nabla F_2) + \psi(F_2) \cdot (-\nabla H_1^T \nabla F_1) = 0$), then $\dot{H}_{\text{total}} \leq 0$ always. No bifurcation possible.
3. Show that for the unidirectional coupling ($\psi = 0$), the power balance $\dot{H}_{\text{total}}$ can change sign, with the threshold at $\rho(K) = 1$.
4. Conclude: the bifurcation requires non-power-preserving (directed) coupling.

### Computation C: Port Alignment with CES Gradient (Claim 3)

**For a single CES level** receiving port input through a general matrix $B_n$:

$$\varepsilon_n \dot{x}_n = u_n B_n \mathbf{1} - \sigma_n x_n$$

where $u_n$ is a scalar input and $B_n \in \mathbb{R}^{J \times J}$ is the port matrix.

1. Show that the symmetric equilibrium $x_n = \bar{x}_n \mathbf{1}$ exists if and only if $B_n \mathbf{1} \propto \mathbf{1}$ (port input is equal across components).
2. For $B_n = \nabla F_n$ (CES gradient), verify $B_n \mathbf{1} = (1/J)\mathbf{1}$ at symmetric point. ✓
3. For a generic $B_n$ with $B_n \mathbf{1} \not\propto \mathbf{1}$, show that the equilibrium is asymmetric and that the CES curvature penalty (eigenvalue gap in $R_n$) makes this equilibrium less stable than the symmetric one.
4. Conclude: the CES geometry (via the Hessian eigenvalue gap) selects $B_n \propto \nabla F_n$ as the "natural" port structure — the one compatible with the symmetric equilibrium that maximizes dissipation.

### Computation D: Slow-Manifold Reduction of Long-Range Coupling (Claim 4)

**For a 3-level system** with timescales $\varepsilon_1 \ll \varepsilon_2 \ll \varepsilon_3$ and couplings:

- Level 1 → Level 2: $T_2 = \phi_{21}(F_1)$ (nearest-neighbor)
- Level 1 → Level 3: $T_3^{(1)} = \phi_{31}(F_1)$ (long-range)
- Level 2 → Level 3: $T_3^{(2)} = \phi_{32}(F_2)$ (nearest-neighbor)

Total input to level 3: $T_3 = \phi_{31}(F_1) + \phi_{32}(F_2)$.

1. Compute the slow manifold after level 1 equilibrates: $F_1^* = F_1^*(\beta_1, \sigma_1)$ (function of exogenous input and damping only).
2. On this manifold, level 3's input becomes $T_3 = \phi_{31}(F_1^*) + \phi_{32}(F_2)$. Since $F_1^*$ is a constant (on the slow manifold), $\phi_{31}(F_1^*)$ is an additive constant to level 3's amplification.
3. Show that this additive constant can be absorbed into a redefined exogenous input for level 3: $\tilde{\beta}_3 = \phi_{31}(F_1^*)$. The effective dynamics on the slow manifold are then nearest-neighbor: level 2 → level 3 with input $\phi_{32}(F_2) + \tilde{\beta}_3$.
4. Compute the NGM of the reduced (slow-manifold) system and compare to the NGM of a nearest-neighbor system. Are the spectral radii equal? Are the Perron-Frobenius eigenvectors the same?
5. If yes: long-range coupling is absorbed on the slow manifold. Nearest-neighbor is the effective topology. If no: characterize what long-range coupling adds that nearest-neighbor cannot replicate.

---

## CONTEXT: THE PORT-HAMILTONIAN GENERATING SYSTEM

For reference, here is the full port-Hamiltonian structure as established by previous computations.

### The generating object

A **directed port-Hamiltonian network** $\mathcal{N} = (G, \{H_n, R_n\}_{n=1}^N, \{B_n, \phi_n\}_{n=1}^N)$ where:

- $G = (V, E)$ is a directed acyclic graph (the hierarchy): $V = \{1, \ldots, N\}$, $E = \{(n-1, n) : n = 2, \ldots, N\}$
- $H_n(x_n) = -\log F_n(x_n)$ is the Hamiltonian at level $n$
- $R_n$ is the dissipation matrix at level $n$, with CES-controlled eigenvalue gap
- $B_n$ is the port input matrix, $\phi_n$ is the port gain function

### How the three theorems emerge

| Theorem | Port-Hamiltonian origin | Role of $\rho$ |
|---------|------------------------|----------------|
| 1 (Triple Role) | Eigenvalue gap of $R_n$: $\sigma_n(2-\rho)/\varepsilon_n$ vs $\sigma_n/\varepsilon_n$ | $K = (1-\rho)(J-1)/J$ controls the gap |
| 2 (Master $R_0$) | Power balance: port input vs dissipation at trivial equilibrium | $K$ enters the dissipation; $\rho(K) = 1$ when input = dissipation |
| 3 (Hierarchical Ceiling) | Singular perturbation: fast levels equilibrate at port-Hamiltonian equilibrium | $K$ controls equilibration rate (via eigenvalues of $R_n$) |

### What $\rho$ does NOT currently determine

- The graph $G$ (which levels are connected)
- The port gain functions $\phi_n$ (how strongly level $n-1$ drives level $n$)
- The structural parameters $J$, $N$, $\varepsilon_n$, $\sigma_n$

### What this prompt aims to show

That $\rho$ + CES properties force:
- $G$ to be a directed nearest-neighbor chain (Claims 2, 4)
- The port input to pass through $F_n$ only (Claim 1)
- The port structure to be $B_n = \nabla F_n$ (Claim 3)

Leaving only $J$, $N$, $\varepsilon_n$, $\sigma_n$, and the scalar gain functions $\phi_n$ as independent inputs.

---

## OUTPUT FORMAT

### For Each Claim (1-4)

**Setup:** State the claim precisely and define all objects.
**Computation:** Show the explicit calculation (matrices, eigenvalues, inequalities).
**Verdict:** Proven / Partially proven / Failed. If failed, explain precisely where.

### Summary Table

| Claim | Status | Consequence for Score |
|-------|--------|----------------------|
| 1: Coupling through $F_n$ | ? | If yes: $\rho$ controls port signal |
| 2: Directed coupling | ? | If yes: $\rho$ forces hierarchy direction |
| 3: Port gain constrained | ? | If yes: $\rho$ constrains gain |
| 4: Nearest-neighbor topology | ? | If yes: $\rho$ + timescale forces topology |

### Updated Score

Based on the results, update the unification score with justification.

---

## CONSTRAINTS

- **Every claim must be checked by computation, not by narrative.** The previous round showed that conceptual arguments ("the spectral radius corresponds to...") fail when you actually compute. Write out the matrices. Check the eigenvalues. Verify the inequalities.
- **The honest negative is valuable.** If any claim fails, that's information. It means the port topology carries genuine independent degrees of freedom, and the CES geometry does not determine them. This would confirm 7.5 as the ceiling and identify exactly what the independent degrees of freedom are.
- **Work at the 2-level, $J$-component level.** The 4-level system is too complex for closed-form computation. If the claims hold for 2 levels, they likely generalize (by induction on $N$). If they fail for 2 levels, they fail generally.
- **The economic application is irrelevant.** This is about the mathematical structure of port-Hamiltonian networks with CES dissipation. The fact that the levels represent AI agents, stablecoins, etc. is irrelevant to whether the port topology is forced by the CES function.
