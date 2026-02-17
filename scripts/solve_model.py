"""
Endogenous Decentralization: Numerical Solution
=================================================
Solves the continuous-time differential game from Section 3.

State variable: x(t) = Q̄ - Q(t) ∈ [0, x₀]
    x = remaining cumulative production until crossing threshold
    x = 0 means crossing has occurred

Nash MPE:  rV_N = (a - V_N')(a - N²V_N') / (b(N+1)²)
Cooperative: rV_P = (a - NV_P')² / (4bN)

Both solved via backward shooting from boundary x=0, V=S.

Outputs:
    Figure 1: Value functions V_N(x) vs V_P(x)
    Figure 2: Total output rates Q_N(x) vs Q_C(x)  
    Figure 3: Shadow costs showing internalization gap
    Figure 4: Crossing time ratio T*_Nash/T*_Coop vs N
    Figure 5: Crossing time sensitivity to displacement rate δ
    Table:    Summary statistics for baseline calibration

Connor Smirl, Tufts University, February 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# ============================================================
# MODEL PARAMETERS
# ============================================================

class Parameters:
    """Calibrated parameters for the endogenous decentralization model."""
    
    def __init__(self, N=5, a=10.0, b=0.5, r=0.05, delta=0.30, x0=10.0):
        self.N = N          # Number of symmetric firms
        self.a = a          # Demand intercept
        self.b = b          # Demand slope
        self.r = r          # Discount rate
        self.delta = delta  # Post-crossing displacement rate (IBM calibration: ~0.30)
        self.x0 = x0        # Initial distance to crossing (normalized)
        
        # Derived quantities
        # Pre-crossing per-firm Cournot flow profit
        self.pi_bar = (a / (N + 1))**2 / b
        # Post-crossing continuation value per firm
        self.S = self.pi_bar / (r + delta)
        
    def __repr__(self):
        return (f"Parameters(N={self.N}, a={self.a}, b={self.b}, "
                f"r={self.r}, δ={self.delta}, x₀={self.x0}, "
                f"π̄={self.pi_bar:.3f}, S={self.S:.3f})")


# ============================================================
# ODE SOLUTIONS
# ============================================================

def nash_Vprime(V, p):
    """
    Compute V'(x) for the Nash equilibrium value function.
    
    From the HJB in symmetric MPE:
        rV = (a - V')(a - N²V') / (b(N+1)²)
    
    This is quadratic in V':
        N²(V')² - a(1+N²)V' + (a² - b(N+1)²rV) = 0
    
    We take the economically relevant (minus) root.
    """
    N, a, b, r = p.N, p.a, p.b, p.r
    
    A_coeff = N**2
    B_coeff = -a * (1 + N**2)
    C_coeff = a**2 - b * (N + 1)**2 * r * V
    
    discriminant = B_coeff**2 - 4 * A_coeff * C_coeff
    # D = a²(1-N²)² + 4N²b(N+1)²rV > 0 always for V > 0
    
    if discriminant < 0:
        return np.nan
    
    # Minus root gives the interior equilibrium (V' < a, output > 0)
    Vp = (-B_coeff - np.sqrt(discriminant)) / (2 * A_coeff)
    return Vp


def coop_Vprime(V, p):
    """
    Compute V'(x) for the cooperative per-firm value function.
    
    From the planner's HJB:
        rV_P = (a - NV_P')² / (4bN)
    
    Solving: V_P' = (a - sqrt(4bNrV_P)) / N
    """
    N, a, b, r = p.N, p.a, p.b, p.r
    
    inner = 4 * b * N * r * V
    if inner < 0:
        return np.nan
    
    sqrt_term = np.sqrt(inner)
    if sqrt_term > a:  # Would give negative output
        return np.nan
    
    Vp = (a - sqrt_term) / N
    return Vp


def nash_output(Vp, p):
    """Total Nash equilibrium output rate Q_N = N(a - V') / (b(N+1))"""
    return p.N * (p.a - Vp) / (p.b * (p.N + 1))


def coop_output(Vp, p):
    """Total cooperative output rate Q_C = (a - NV') / (2b)"""
    return (p.a - p.N * Vp) / (2 * p.b)


def solve_value_functions(p, num_points=2000):
    """
    Solve both Nash and Cooperative value function ODEs via forward Euler
    from boundary x=0 (V=S) outward to x=x₀.
    
    Returns arrays of (x, V_N, V_P, Vp_N, Vp_P, Q_N, Q_C)
    """
    dx = p.x0 / num_points
    x = np.zeros(num_points + 1)
    V_N = np.zeros(num_points + 1)
    V_P = np.zeros(num_points + 1)
    Vp_N = np.zeros(num_points + 1)
    Vp_P = np.zeros(num_points + 1)
    Q_N = np.zeros(num_points + 1)
    Q_C = np.zeros(num_points + 1)
    
    # Boundary conditions at x = 0
    V_N[0] = p.S
    V_P[0] = p.S
    
    # Compute initial derivatives
    Vp_N[0] = nash_Vprime(p.S, p)
    Vp_P[0] = coop_Vprime(p.S, p)
    Q_N[0] = nash_output(Vp_N[0], p)
    Q_C[0] = coop_output(Vp_P[0], p)
    
    # Forward Euler integration (4th-order Runge-Kutta for accuracy)
    for i in range(num_points):
        x[i + 1] = x[i] + dx
        
        # RK4 for Nash
        k1_N = nash_Vprime(V_N[i], p)
        k2_N = nash_Vprime(V_N[i] + 0.5 * dx * k1_N, p)
        k3_N = nash_Vprime(V_N[i] + 0.5 * dx * k2_N, p)
        k4_N = nash_Vprime(V_N[i] + dx * k3_N, p)
        
        if any(np.isnan([k1_N, k2_N, k3_N, k4_N])):
            # Truncate at divergence
            x = x[:i+1]; V_N = V_N[:i+1]; V_P = V_P[:i+1]
            Vp_N = Vp_N[:i+1]; Vp_P = Vp_P[:i+1]
            Q_N = Q_N[:i+1]; Q_C = Q_C[:i+1]
            break
        
        V_N[i + 1] = V_N[i] + (dx / 6) * (k1_N + 2*k2_N + 2*k3_N + k4_N)
        Vp_N[i + 1] = nash_Vprime(V_N[i + 1], p)
        Q_N[i + 1] = nash_output(Vp_N[i + 1], p)
        
        # RK4 for Cooperative
        k1_C = coop_Vprime(V_P[i], p)
        k2_C = coop_Vprime(V_P[i] + 0.5 * dx * k1_C, p)
        k3_C = coop_Vprime(V_P[i] + 0.5 * dx * k2_C, p)
        k4_C = coop_Vprime(V_P[i] + dx * k3_C, p)
        
        if any(np.isnan([k1_C, k2_C, k3_C, k4_C])):
            x = x[:i+1]; V_N = V_N[:i+1]; V_P = V_P[:i+1]
            Vp_N = Vp_N[:i+1]; Vp_P = Vp_P[:i+1]
            Q_N = Q_N[:i+1]; Q_C = Q_C[:i+1]
            break
        
        V_P[i + 1] = V_P[i] + (dx / 6) * (k1_C + 2*k2_C + 2*k3_C + k4_C)
        Vp_P[i + 1] = coop_Vprime(V_P[i + 1], p)
        Q_C[i + 1] = coop_output(Vp_P[i + 1], p)
    
    return x, V_N, V_P, Vp_N, Vp_P, Q_N, Q_C


def compute_crossing_time(x, Q):
    """
    Compute crossing time T* = ∫₀^x₀ dx/Q(x)
    
    Since dx/dt = -Q (state depletes at output rate), 
    time to go from x₀ to 0 is ∫₀^x₀ (1/Q(x)) dx
    """
    # Trapezoidal integration of 1/Q over x
    valid = Q > 1e-10
    integrand = np.zeros_like(Q)
    integrand[valid] = 1.0 / Q[valid]
    integrand[~valid] = np.nan
    
    mask = ~np.isnan(integrand)
    T = np.trapezoid(integrand[mask], x[mask])
    return T


# ============================================================
# FIGURE GENERATION
# ============================================================

def setup_style():
    """Set publication-quality matplotlib style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.0,
    })


def fig1_value_functions(x, V_N, V_P, p, outdir):
    """Figure 1: Value functions V_N(x) vs V_P(x)"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(x, V_N, 'b-', label=f'Nash MPE, $V_N(x)$', linewidth=2.5)
    ax.plot(x, V_P, 'r--', label=f'Cooperative, $V_P(x)$', linewidth=2.5)
    ax.axhline(y=p.S, color='gray', linestyle=':', alpha=0.5, label=f'$S = {p.S:.2f}$ (boundary)')
    
    ax.set_xlabel('Distance to crossing, $x$')
    ax.set_ylabel('Per-firm value, $V(x)$')
    ax.set_title(f'Value Functions: Nash vs. Cooperative (N={p.N})')
    ax.legend(loc='lower right')
    
    # Annotate the gap
    mid = len(x) // 2
    gap = V_P[mid] - V_N[mid]
    if gap > 0:
        ax.annotate(f'Cooperative premium\n= {gap:.2f}',
                    xy=(x[mid], (V_N[mid] + V_P[mid]) / 2),
                    fontsize=9, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig1_value_functions.png'))
    plt.close(fig)
    print("  Figure 1: Value functions")


def fig2_output_rates(x, Q_N, Q_C, p, outdir):
    """Figure 2: Total output rates Q_N(x) vs Q_C(x)"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(x, Q_N, 'b-', label=f'Nash total output, $Q_N(x)$', linewidth=2.5)
    ax.plot(x, Q_C, 'r--', label=f'Cooperative total output, $Q_C(x)$', linewidth=2.5)
    
    # Shade the overinvestment gap
    ax.fill_between(x, Q_C, Q_N, alpha=0.15, color='blue', label='Overinvestment gap')
    
    ax.set_xlabel('Distance to crossing, $x$')
    ax.set_ylabel('Total output rate, $Q(x)$')
    ax.set_title(f'Aggregate Output: Nash Overinvestment (N={p.N})')
    ax.legend(loc='upper right')
    
    # Annotate percentage gap at midpoint
    mid = len(x) // 2
    if Q_C[mid] > 0:
        pct = (Q_N[mid] - Q_C[mid]) / Q_C[mid] * 100
        ax.annotate(f'Nash overproduces\nby {pct:.1f}% at $x={x[mid]:.1f}$',
                    xy=(x[mid], (Q_N[mid] + Q_C[mid]) / 2),
                    fontsize=9, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig2_output_rates.png'))
    plt.close(fig)
    print("  Figure 2: Output rates")


def fig3_shadow_costs(x, Vp_N, Vp_P, p, outdir):
    """Figure 3: Shadow costs showing internalization gap"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    social_shadow = p.N * Vp_P  # Total social shadow cost
    
    ax.plot(x, Vp_N, 'b-', label="Private shadow cost, $V_N'(x)$", linewidth=2.5)
    ax.plot(x, social_shadow, 'r--', label="Social shadow cost, $NV_P'(x)$", linewidth=2.5)
    
    # Shade the externality gap
    ax.fill_between(x, Vp_N, social_shadow, alpha=0.15, color='red',
                    label='Uninternalized externality')
    
    ax.set_xlabel('Distance to crossing, $x$')
    ax.set_ylabel('Shadow cost of state depletion')
    ax.set_title(f'The Learning Externality: Private vs. Social Shadow Cost (N={p.N})')
    ax.legend(loc='lower right')
    
    # Annotate
    mid = len(x) // 2
    gap = social_shadow[mid] - Vp_N[mid]
    if gap > 0:
        ax.annotate(
            f'Each firm ignores\n{(1-1/p.N)*100:.0f}% of social cost',
            xy=(x[mid], (Vp_N[mid] + social_shadow[mid]) / 2),
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig3_shadow_costs.png'))
    plt.close(fig)
    print("  Figure 3: Shadow costs")


def fig4_crossing_time_vs_N(p_base, outdir):
    """Figure 4: Crossing time ratio as function of N"""
    N_range = range(2, 16)
    T_nash = []
    T_coop = []
    ratios = []
    
    for N in N_range:
        p = Parameters(N=N, a=p_base.a, b=p_base.b, r=p_base.r, 
                       delta=p_base.delta, x0=p_base.x0)
        x, V_N, V_P, Vp_N, Vp_P, Q_N, Q_C = solve_value_functions(p)
        
        tn = compute_crossing_time(x, Q_N)
        tc = compute_crossing_time(x, Q_C)
        
        T_nash.append(tn)
        T_coop.append(tc)
        ratios.append(tn / tc if tc > 0 else np.nan)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel: absolute crossing times
    ax1.plot(list(N_range), T_nash, 'bo-', label='$T^*_{Nash}$', linewidth=2, markersize=6)
    ax1.plot(list(N_range), T_coop, 'rs--', label='$T^*_{Coop}$', linewidth=2, markersize=6)
    ax1.fill_between(list(N_range), T_nash, T_coop, alpha=0.15, color='green',
                     label='Acceleration gap')
    ax1.set_xlabel('Number of firms, $N$')
    ax1.set_ylabel('Crossing time, $T^*$')
    ax1.set_title('Crossing Time vs. Number of Firms')
    ax1.legend()
    
    # Right panel: ratio
    ax2.plot(list(N_range), ratios, 'ko-', linewidth=2, markersize=6)
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Number of firms, $N$')
    ax2.set_ylabel('$T^*_{Nash} / T^*_{Coop}$')
    ax2.set_title('Nash Crossing as Fraction of Cooperative')
    ax2.set_ylim(0, 1.05)
    
    # Annotate key points
    for N_mark in [5, 10]:
        if N_mark - 2 < len(ratios):
            idx = N_mark - 2
            ax2.annotate(f'N={N_mark}: {ratios[idx]:.3f}',
                        xy=(N_mark, ratios[idx]),
                        xytext=(N_mark + 1, ratios[idx] + 0.05),
                        arrowprops=dict(arrowstyle='->', color='black'),
                        fontsize=9)
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig4_crossing_time_vs_N.png'))
    plt.close(fig)
    print("  Figure 4: Crossing time vs N")
    
    return list(N_range), T_nash, T_coop, ratios


def fig5_sensitivity_delta(p_base, outdir):
    """Figure 5: Sensitivity of crossing time to displacement rate δ"""
    deltas = np.linspace(0.05, 0.80, 20)
    T_nash = []
    T_coop = []
    
    for d in deltas:
        p = Parameters(N=p_base.N, a=p_base.a, b=p_base.b, r=p_base.r,
                       delta=d, x0=p_base.x0)
        x, V_N, V_P, Vp_N, Vp_P, Q_N, Q_C = solve_value_functions(p)
        T_nash.append(compute_crossing_time(x, Q_N))
        T_coop.append(compute_crossing_time(x, Q_C))
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(deltas, T_nash, 'b-', label='$T^*_{Nash}$', linewidth=2.5)
    ax.plot(deltas, T_coop, 'r--', label='$T^*_{Coop}$', linewidth=2.5)
    ax.fill_between(deltas, T_nash, T_coop, alpha=0.15, color='green')
    
    # Mark IBM calibration
    ax.axvline(x=0.30, color='gray', linestyle=':', alpha=0.7)
    ax.annotate('IBM calibration\n$\\delta \\approx 0.30$',
                xy=(0.30, max(T_coop) * 0.5),
                fontsize=9, ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Displacement rate, $\\delta$')
    ax.set_ylabel('Crossing time, $T^*$')
    ax.set_title(f'Crossing Time Sensitivity to Displacement Rate (N={p_base.N})')
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig5_sensitivity_delta.png'))
    plt.close(fig)
    print("  Figure 5: Sensitivity to δ")


def fig6_decomposition(x, Vp_N, Q_N, Q_C, p, outdir):
    """
    Figure 6: Decomposition of overinvestment into Cournot and learning channels.
    
    Total overinvestment = Q_N - Q_C
    Cournot channel: output gap that would exist even without learning externality (δ=0)
    Learning channel: additional gap from uninternalized shadow cost
    """
    # Solve the δ=0 case (pure Cournot, no crossing concern)
    p_nocross = Parameters(N=p.N, a=p.a, b=p.b, r=p.r, delta=0.001, x0=p.x0)
    x2, V_N2, V_P2, Vp_N2, Vp_P2, Q_N2, Q_C2 = solve_value_functions(p_nocross)
    
    # Static Cournot output (no dynamic concerns at all)
    Q_static_nash = p.N * p.a / (p.b * (p.N + 1))
    Q_static_coop = p.a / (2 * p.b)  # Monopoly output (planner maximizes joint profit)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Use minimum length
    n = min(len(x), len(x2))
    x_plot = x[:n]
    
    ax.plot(x_plot, Q_N[:n], 'b-', label='Nash (with learning externality)', linewidth=2.5)
    ax.plot(x_plot, Q_C[:n], 'r--', label='Cooperative (with learning externality)', linewidth=2.5)
    ax.axhline(y=Q_static_nash, color='blue', linestyle=':', alpha=0.4, 
               label=f'Static Cournot Nash = {Q_static_nash:.2f}')
    ax.axhline(y=Q_static_coop, color='red', linestyle=':', alpha=0.4,
               label=f'Static cooperative = {Q_static_coop:.2f}')
    
    ax.fill_between(x_plot, Q_C[:n], Q_N[:n], alpha=0.15, color='blue')
    
    ax.set_xlabel('Distance to crossing, $x$')
    ax.set_ylabel('Total output rate, $Q(x)$')
    ax.set_title(f'Output Decomposition: Dynamic vs. Static (N={p.N})')
    ax.legend(loc='best', fontsize=9)
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig6_decomposition.png'))
    plt.close(fig)
    print("  Figure 6: Output decomposition")


# ============================================================
# CLOSED-FORM SOLUTIONS (Wolfram-verified)
# ============================================================

def cooperative_x_of_V(V, p):
    """
    Exact closed-form: cooperative implicit solution x(V).
    
    From separable ODE V' = (a - sqrt(4bnrV))/n:
        x(V) = [a ln((a - 2sqrt(bnrS))/(a - 2sqrt(bnrV))) + 2(sqrt(bnrS) - sqrt(bnrV))] / (2br)
    
    Wolfram Alpha confirmed: ∫ n/(a - sqrt(4bnrv)) dv = -[a ln(a - 2sqrt(bnrv)) + 2sqrt(bnrv)] / (2br)
    """
    a, b, n, r, S = p.a, p.b, p.N, p.r, p.S
    
    sqrtS = np.sqrt(b * n * r * S)
    sqrtV = np.sqrt(b * n * r * V)
    
    # Boundary term at V=S gives x=0
    x = (a * np.log((a - 2*sqrtS) / (a - 2*sqrtV)) + 2*(sqrtS - sqrtV)) / (2 * b * r)
    return x


def nash_x_of_V(V, p):
    """
    Exact closed-form: Nash MPE implicit solution x(V).
    
    From separable ODE V' = [a(1+N²) - sqrt(D + EV)] / (2N²), where
        D = a²(N²-1)²
        E = 4N²rb(N+1)²
    
    Substituting u = sqrt(D + EV):
        x(V) = (2N²/E) [(u₀ - u) + a(1+N²) ln((a(1+N²) - u₀)/(a(1+N²) - u))]
    
    Wolfram Alpha confirmed: ∫ 2u/(A - u) du = -2u - 2A ln(u - A)
    """
    a, b, N, r, S = p.a, p.b, p.N, p.r, p.S
    
    D = a**2 * (N**2 - 1)**2
    E = 4 * N**2 * r * b * (N + 1)**2
    A = a * (1 + N**2)
    
    u0 = np.sqrt(D + E * S)    # u at boundary V=S, x=0
    u  = np.sqrt(D + E * V)    # u at general V
    
    # Factor is 4N²/E from substitution: dV = 2u/E du, times 2N² from ODE
    x = (4 * N**2 / E) * ((u0 - u) + A * np.log((A - u0) / (A - u)))
    return x


def verify_closed_forms(x_num, V_N, V_P, p, outdir):
    """
    Compare closed-form x(V) against numerical RK4 solution.
    Produces verification figure and reports max absolute error.
    """
    # Cooperative verification
    valid_C = V_P > p.S * 1.001  # avoid boundary singularity
    x_coop_exact = cooperative_x_of_V(V_P[valid_C], p)
    x_coop_numeric = x_num[valid_C]
    err_coop = np.abs(x_coop_exact - x_coop_numeric)
    
    # Nash verification
    valid_N = V_N > p.S * 1.001
    x_nash_exact = nash_x_of_V(V_N[valid_N], p)
    x_nash_numeric = x_num[valid_N]
    err_nash = np.abs(x_nash_exact - x_nash_numeric)
    
    print(f"\n{'CLOSED-FORM VERIFICATION':=^60}")
    print(f"  Cooperative:")
    print(f"    Max |x_exact - x_numeric| = {np.max(err_coop):.2e}")
    print(f"    Mean error                 = {np.mean(err_coop):.2e}")
    print(f"    Points compared            = {np.sum(valid_C)}")
    print(f"  Nash:")
    print(f"    Max |x_exact - x_numeric|  = {np.max(err_nash):.2e}")
    print(f"    Mean error                 = {np.mean(err_nash):.2e}")
    print(f"    Points compared            = {np.sum(valid_N)}")
    
    # Verification figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: overlay exact vs numeric
    ax = axes[0]
    ax.plot(x_coop_numeric, V_P[valid_C], 'r-', label='Cooperative (RK4)', linewidth=2.5)
    ax.plot(x_coop_exact, V_P[valid_C], 'r:', label='Cooperative (exact)', linewidth=2)
    ax.plot(x_nash_numeric, V_N[valid_N], 'b-', label='Nash (RK4)', linewidth=2.5)
    ax.plot(x_nash_exact, V_N[valid_N], 'b:', label='Nash (exact)', linewidth=2)
    ax.set_xlabel('$x$ (distance to crossing)')
    ax.set_ylabel('$V(x)$')
    ax.set_title('Closed-Form vs. Numerical Solution')
    ax.legend(fontsize=9)
    
    # Right: error magnitudes
    ax = axes[1]
    ax.semilogy(x_coop_numeric, err_coop, 'r-', label='Cooperative error', linewidth=2)
    ax.semilogy(x_nash_numeric, err_nash, 'b-', label='Nash error', linewidth=2)
    ax.set_xlabel('$x$ (distance to crossing)')
    ax.set_ylabel('$|x_{exact} - x_{numeric}|$')
    ax.set_title('Verification Error (log scale)')
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig7_closed_form_verification.png'))
    plt.close(fig)
    print(f"  Figure 7: Closed-form verification")
    
    return err_coop, err_nash


def compute_crossing_time_exact(p):
    """
    Exact crossing time from closed-form solution.
    
    T* = ∫₀^x₀ dx/Q(x). But since x(V) is known exactly, we can compute
    T* by integrating dt = dx/Q = (dx/dV)(dV/Q) over V from S to V(x₀).
    
    Alternatively, since x(V) is monotone, we evaluate x₀ = x(V_∞) where
    V_∞ is the value at x = x₀. We invert numerically.
    """
    from scipy.optimize import brentq
    
    # Nash: find V* such that nash_x_of_V(V*, p) = x₀
    def nash_residual(V):
        try:
            return nash_x_of_V(V, p) - p.x0
        except:
            return -1e10
    
    # Cooperative: find V* such that coop_x_of_V(V*, p) = x₀
    def coop_residual(V):
        try:
            return cooperative_x_of_V(V, p) - p.x0
        except:
            return -1e10
    
    # Search for upper bound on V
    V_upper = p.S * 100
    for mult in [2, 5, 10, 50, 100, 500, 1000]:
        V_test = p.S * mult
        try:
            if nash_x_of_V(V_test, p) > p.x0:
                V_upper = V_test
                break
        except:
            continue
    
    try:
        V_nash_x0 = brentq(nash_residual, p.S * 1.0001, V_upper)
        V_coop_x0 = brentq(coop_residual, p.S * 1.0001, V_upper)
        return V_nash_x0, V_coop_x0
    except Exception as e:
        print(f"  Warning: exact inversion failed ({e})")
        return None, None


# ============================================================
# MAIN
# ============================================================

def main():
    setup_style()
    outdir = '/home/claude/model_output'
    os.makedirs(outdir, exist_ok=True)
    
    print("=" * 60)
    print("ENDOGENOUS DECENTRALIZATION: NUMERICAL SOLUTION")
    print("=" * 60)
    
    # Baseline parameters
    p = Parameters(N=5, a=10.0, b=0.5, r=0.05, delta=0.30, x0=10.0)
    print(f"\nBaseline: {p}")
    
    # Solve
    print("\nSolving value function ODEs...")
    x, V_N, V_P, Vp_N, Vp_P, Q_N, Q_C = solve_value_functions(p)
    print(f"  Solved over x ∈ [0, {x[-1]:.2f}], {len(x)} points")
    
    # Crossing times
    T_nash = compute_crossing_time(x, Q_N)
    T_coop = compute_crossing_time(x, Q_C)
    
    print(f"\n{'BASELINE RESULTS (N=5)':=^60}")
    print(f"  T*_Nash  = {T_nash:.4f}")
    print(f"  T*_Coop  = {T_coop:.4f}")
    print(f"  Ratio    = {T_nash/T_coop:.4f}")
    print(f"  Acceleration = {(1 - T_nash/T_coop)*100:.1f}%")
    print(f"")
    print(f"  At x = x₀ = {p.x0}:")
    print(f"    Q_N(x₀) = {Q_N[-1]:.4f}  |  Q_C(x₀) = {Q_C[-1]:.4f}")
    print(f"    Overproduction = {(Q_N[-1]/Q_C[-1] - 1)*100:.1f}%")
    print(f"    V_N'(x₀) = {Vp_N[-1]:.4f}  |  NV_P'(x₀) = {p.N*Vp_P[-1]:.4f}")
    print(f"    Shadow cost ratio = {p.N*Vp_P[-1]/Vp_N[-1]:.2f}x")
    print(f"")
    print(f"  At x = 0 (boundary):")
    print(f"    V_N'(0) = {Vp_N[0]:.4f}  |  NV_P'(0) = {p.N*Vp_P[0]:.4f}")
    print(f"    Q_N(0)  = {Q_N[0]:.4f}  |  Q_C(0)  = {Q_C[0]:.4f}")
    
    # Verify Proposition 1
    overinvestment_holds = np.all(Q_N[1:] > Q_C[1:])
    shadow_ordering = np.all(p.N * Vp_P[1:] > Vp_N[1:])
    print(f"\n{'PROPOSITION 1 VERIFICATION':=^60}")
    print(f"  Q_N(x) > Q_C(x) for all x > 0: {overinvestment_holds}")
    print(f"  NV_P'(x) > V_N'(x) for all x > 0: {shadow_ordering}")
    print(f"  T*_Nash < T*_Coop: {T_nash < T_coop}")
    
    # Generate figures
    print(f"\n{'GENERATING FIGURES':=^60}")
    fig1_value_functions(x, V_N, V_P, p, outdir)
    fig2_output_rates(x, Q_N, Q_C, p, outdir)
    fig3_shadow_costs(x, Vp_N, Vp_P, p, outdir)
    N_range, T_n_list, T_c_list, ratios = fig4_crossing_time_vs_N(p, outdir)
    fig5_sensitivity_delta(p, outdir)
    fig6_decomposition(x, Vp_N, Q_N, Q_C, p, outdir)
    
    # Closed-form verification
    err_coop, err_nash = verify_closed_forms(x, V_N, V_P, p, outdir)
    
    # Exact crossing time via closed-form inversion
    V_nash_x0, V_coop_x0 = compute_crossing_time_exact(p)
    if V_nash_x0 is not None:
        print(f"\n{'EXACT VALUES AT x = x₀':=^60}")
        print(f"  V_N(x₀) exact  = {V_nash_x0:.6f}  |  numeric = {V_N[-1]:.6f}")
        print(f"  V_P(x₀) exact  = {V_coop_x0:.6f}  |  numeric = {V_P[-1]:.6f}")
        print(f"  Nash error     = {abs(V_nash_x0 - V_N[-1]):.2e}")
        print(f"  Coop error     = {abs(V_coop_x0 - V_P[-1]):.2e}")
    
    # Corollary 1 verification table
    print(f"\n{'COROLLARY 1: ∂Q_N/∂N > 0':=^60}")
    print(f"  {'N':>4}  {'T*_Nash':>10}  {'T*_Coop':>10}  {'Ratio':>8}  {'Accel%':>8}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for i, N in enumerate(N_range):
        if T_c_list[i] > 0:
            accel = (1 - T_n_list[i]/T_c_list[i]) * 100
            print(f"  {N:4d}  {T_n_list[i]:10.4f}  {T_c_list[i]:10.4f}  {ratios[i]:8.4f}  {accel:7.1f}%")
    
    # Sensitivity summary
    print(f"\n{'SENSITIVITY TO δ (displacement rate)':=^60}")
    for d_test in [0.10, 0.20, 0.30, 0.50, 0.70]:
        p_test = Parameters(N=5, a=10.0, b=0.5, r=0.05, delta=d_test, x0=10.0)
        x_t, V_N_t, V_P_t, _, _, Q_N_t, Q_C_t = solve_value_functions(p_test)
        tn = compute_crossing_time(x_t, Q_N_t)
        tc = compute_crossing_time(x_t, Q_C_t)
        print(f"  δ={d_test:.2f}: T*_N={tn:.4f}, T*_C={tc:.4f}, ratio={tn/tc:.4f}")
    
    # AI-specific calibration note
    print(f"\n{'AI INFRASTRUCTURE CALIBRATION':=^60}")
    print(f"  The normalized model maps to the AI case as follows:")
    print(f"  - x₀ represents the gap between current cumulative HBM")
    print(f"    production and the ~50GB consumer threshold")
    print(f"  - N=5 baseline matches the five hyperscalers (2023)")
    print(f"  - N=10+ matches current investor set (2025-26)")
    print(f"  - δ=0.30 from IBM trajectory (60% profit erosion in 3 years)")
    print(f"  - α=0.22 enters through the mapping from investment I(t)")
    print(f"    to cumulative production Q(t) via the learning curve")
    print(f"")
    print(f"  Key quantitative prediction:")
    print(f"  Moving from N=5 to N=10 compresses T*_Nash by", end=" ")
    p5 = Parameters(N=5); p10 = Parameters(N=10)
    x5, _, _, _, _, Q5, _ = solve_value_functions(p5)
    x10, _, _, _, _, Q10, _ = solve_value_functions(p10)
    t5 = compute_crossing_time(x5, Q5)
    t10 = compute_crossing_time(x10, Q10)
    print(f"{(1-t10/t5)*100:.1f}%")
    print(f"  (from T*={t5:.3f} to T*={t10:.3f} in normalized units)")
    
    print(f"\nFigures saved to {outdir}/")
    print("Done.")


if __name__ == "__main__":
    main()
