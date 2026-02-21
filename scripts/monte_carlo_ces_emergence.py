#!/usr/bin/env python3
"""
monte_carlo_ces_emergence.py
============================
Two Monte Carlo tests of CES emergence under hierarchical aggregation.

SIMULATION 1 — Houthakker Aggregation:
    DGP: N firms with DRS Cobb-Douglas, heterogeneous capital shares alpha_j.
    Firms profit-maximize at common factor prices (r_t, w_t).
    Aggregate Y=sum(y_j), K=sum(k_j), L=sum(l_j).
    Houthakker (1955) predicts: aggregate is approximately CES.

SIMULATION 2 — Translog Interaction Decay:
    DGP: N firms with heterogeneous translog production functions.
    Inputs driven by common factors + idiosyncratic shocks.
    Aggregate by summation at hierarchy levels.
    Prediction: translog interaction terms shrink with aggregation.

Output: thesis_data/monte_carlo_emergence.csv
        thesis_data/monte_carlo_emergence_table.tex
"""

import numpy as np
from scipy.optimize import minimize
from scipy import stats
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# CONFIGURATION
# =========================================================
T_PERIODS = 500
N_MC = 30
SEED_BASE = 42

# Houthakker simulation
N_FIRMS_H = 200   # more firms = closer to CES limit
NU = 0.85          # DRS parameter (< 1 needed for finite firm size)
HIERARCHY_H = [5, 5, 4, 2]  # 200 -> 40 -> 8 -> 2 -> 1

# Translog decay simulation
N_FIRMS_T = 120
INTERACTION_SCALE = 0.15
HIERARCHY_T = [5, 4, 3, 2]  # 120 -> 24 -> 6 -> 2 -> 1


# =========================================================
# ESTIMATION: TRANSLOG
# =========================================================

def fit_translog(ln_y, ln_k, ln_l):
    """Fit translog to time series. Returns (R2_tl, R2_cd, interaction_ratio, F_stat, p_value)."""
    n = len(ln_y)
    X = np.column_stack([
        np.ones(n), ln_k, ln_l,
        0.5 * ln_k**2, 0.5 * ln_l**2, ln_k * ln_l
    ])
    b = np.linalg.lstsq(X, ln_y, rcond=None)[0]
    ss_res = np.sum((ln_y - X @ b)**2)
    ss_tot = np.sum((ln_y - ln_y.mean())**2)
    R2_tl = 1 - ss_res / ss_tot if ss_tot > 1e-20 else 1.0

    X_cd = X[:, :3]
    b_cd = np.linalg.lstsq(X_cd, ln_y, rcond=None)[0]
    ss_res_cd = np.sum((ln_y - X_cd @ b_cd)**2)
    R2_cd = 1 - ss_res_cd / ss_tot if ss_tot > 1e-20 else 1.0

    alpha_norm = np.sqrt(b[1]**2 + b[2]**2)
    beta_norm = np.sqrt(b[3]**2 + b[4]**2 + b[5]**2)
    ir = beta_norm / alpha_norm if alpha_norm > 1e-10 else np.nan

    df2 = n - 6
    if df2 > 0 and ss_res > 1e-20:
        F = ((ss_res_cd - ss_res) / 3) / (ss_res / df2)
        p = 1 - stats.f.cdf(max(F, 0), 3, df2)
    else:
        F, p = np.nan, np.nan

    return R2_tl, R2_cd, ir, F, p


# =========================================================
# ESTIMATION: CES
# =========================================================

def fit_ces(ln_y, ln_k, ln_l):
    """Fit CES: Y = A*(w*K^rho + (1-w)*L^rho)^(1/rho). Returns (R2, rho)."""
    def sse(params):
        ln_A, w_logit, rho = params
        w = 1.0 / (1.0 + np.exp(-np.clip(w_logit, -10, 10)))
        rho_c = np.clip(rho, -5, 0.99)
        if abs(rho_c) < 1e-6:
            pred = ln_A + w * ln_k + (1 - w) * ln_l
        else:
            inner = w * np.exp(rho_c * ln_k) + (1 - w) * np.exp(rho_c * ln_l)
            inner = np.maximum(inner, 1e-30)
            pred = ln_A + (1.0 / rho_c) * np.log(inner)
        return np.sum((ln_y - pred)**2)

    best = None
    m = np.mean(ln_y)
    for rho0 in [-2, -1, -0.5, -0.1, 0.01, 0.3, 0.5, 0.8]:
        for w0 in [-0.5, 0.0, 0.5]:
            try:
                res = minimize(sse, [m, w0, rho0], method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-12})
                if best is None or res.fun < best.fun:
                    best = res
            except Exception:
                pass

    if best is None:
        return np.nan, np.nan

    ss_tot = np.sum((ln_y - ln_y.mean())**2)
    R2 = 1 - best.fun / ss_tot if ss_tot > 1e-20 else 1.0
    rho = np.clip(best.x[2], -5, 0.99)
    return R2, rho


# =========================================================
# AGGREGATION
# =========================================================

def aggregate(Y, K, L, group_size):
    """Aggregate units into groups by summation."""
    n_units = Y.shape[0]
    n_groups = n_units // group_size
    Y_a = np.zeros((n_groups, Y.shape[1]))
    K_a = np.zeros_like(Y_a)
    L_a = np.zeros_like(Y_a)
    for g in range(n_groups):
        s = slice(g * group_size, (g + 1) * group_size)
        Y_a[g] = Y[s].sum(axis=0)
        K_a[g] = K[s].sum(axis=0)
        L_a[g] = L[s].sum(axis=0)
    return Y_a, K_a, L_a


def estimate_level(Y, K, L):
    """Estimate CES and translog for all units, return mean stats."""
    n = Y.shape[0]
    stats_list = []
    for i in range(n):
        ln_y, ln_k, ln_l = np.log(Y[i]), np.log(K[i]), np.log(L[i])
        R2_tl, R2_cd, ir, F, p = fit_translog(ln_y, ln_k, ln_l)
        R2_ces, rho = fit_ces(ln_y, ln_k, ln_l)
        stats_list.append({
            'R2_tl': R2_tl, 'R2_cd': R2_cd, 'R2_ces': R2_ces,
            'ir': ir, 'F': F, 'p': p, 'rho': rho
        })
    df = pd.DataFrame(stats_list)
    return {
        'R2_translog': df['R2_tl'].mean(),
        'R2_cd': df['R2_cd'].mean(),
        'R2_ces': df['R2_ces'].mean(),
        'interaction_ratio': df['ir'].mean(),
        'pct_F_reject': (df['p'] < 0.05).mean(),
        'rho_mean': df['rho'].mean(),
        'rho_std': df['rho'].std(),
        'n_units': n,
    }


# ==========================================================
# SIMULATION 1: HOUTHAKKER AGGREGATION
# ==========================================================

def houthakker_panel(n_firms, T, nu, seed):
    """Generate panel from heterogeneous DRS Cobb-Douglas firms at common prices.

    Firm j: y_j = A_j * k_j^(a_j*nu) * l_j^((1-a_j)*nu)
    Profit max: k_j = a_j*nu*y_j/r, l_j = (1-a_j)*nu*y_j/w
    Closed form for y_j as function of (r, w).
    """
    rng = np.random.default_rng(seed)

    alpha = rng.beta(2, 3, n_firms)  # skewed toward labor-intensive
    alpha = np.clip(alpha, 0.05, 0.95)
    A = np.exp(rng.normal(1.0, 0.5, n_firms))

    # Factor prices: AR(1)
    ln_r = np.zeros(T)
    ln_w = np.zeros(T)
    for t in range(1, T):
        ln_r[t] = 0.85 * ln_r[t-1] + rng.normal(0, 0.12)
        ln_w[t] = 0.85 * ln_w[t-1] + rng.normal(0, 0.12)
    r = np.exp(ln_r)
    w = np.exp(ln_w)

    Y = np.zeros((n_firms, T))
    K = np.zeros((n_firms, T))
    L = np.zeros((n_firms, T))

    for j in range(n_firms):
        a = alpha[j]
        a_nu = a * nu
        b_nu = (1 - a) * nu

        # y_j = C_j * r^(-a_nu/(1-nu)) * w^(-b_nu/(1-nu))
        # C_j = [A_j * (a_nu)^a_nu * (b_nu)^b_nu]^(1/(1-nu))
        ln_C = (np.log(A[j]) + a_nu * np.log(a_nu) + b_nu * np.log(b_nu)) / (1 - nu)
        C = np.exp(ln_C)

        y_j = C * r**(-a_nu / (1 - nu)) * w**(-b_nu / (1 - nu))
        k_j = a_nu * y_j / r
        l_j = b_nu * y_j / w

        # Small measurement noise
        y_j *= np.exp(rng.normal(0, 0.005, T))
        k_j *= np.exp(rng.normal(0, 0.005, T))
        l_j *= np.exp(rng.normal(0, 0.005, T))

        Y[j] = y_j
        K[j] = k_j
        L[j] = l_j

    return Y, K, L


def run_houthakker(mc_id):
    """One MC replication of Houthakker simulation."""
    seed = SEED_BASE + mc_id * 1000
    Y, K, L = houthakker_panel(N_FIRMS_H, T_PERIODS, NU, seed)

    results = []

    # Aggregate levels
    Y_cur, K_cur, L_cur = Y, K, L
    for lev, gs in enumerate(HIERARCHY_H):
        Y_cur, K_cur, L_cur = aggregate(Y_cur, K_cur, L_cur, gs)
        n = Y_cur.shape[0]
        res = estimate_level(Y_cur, K_cur, L_cur)
        res['level'] = lev + 1
        res['label'] = f'n={n}'
        results.append(res)

    return results


# ==========================================================
# SIMULATION 2: TRANSLOG INTERACTION DECAY
# ==========================================================

def translog_panel(n_firms, T, interaction_scale, seed):
    """Generate heterogeneous translog firm panel."""
    rng = np.random.default_rng(seed)

    a0 = rng.normal(2.0, 0.3, n_firms)
    aK = np.clip(rng.normal(0.35, 0.05, n_firms), 0.15, 0.55)
    aL = 1.0 - aK
    bKK = rng.normal(0, interaction_scale, n_firms)
    bLL = rng.normal(0, interaction_scale, n_firms)
    bKL = rng.normal(0, interaction_scale, n_firms)

    # Common factors
    zK = np.zeros(T)
    zL = np.zeros(T)
    for t in range(1, T):
        zK[t] = 0.7 * zK[t-1] + rng.normal(0, 0.3)
        zL[t] = 0.7 * zL[t-1] + rng.normal(0, 0.3)

    muK = rng.normal(3.0, 0.5, n_firms)
    muL = rng.normal(4.0, 0.5, n_firms)
    lamK = rng.uniform(0.5, 1.5, n_firms)
    lamL = rng.uniform(0.5, 1.5, n_firms)

    ln_K = np.zeros((n_firms, T))
    ln_L = np.zeros((n_firms, T))
    ln_Y = np.zeros((n_firms, T))

    for j in range(n_firms):
        ln_K[j] = muK[j] + lamK[j] * zK + rng.normal(0, 0.3, T)
        ln_L[j] = muL[j] + lamL[j] * zL + rng.normal(0, 0.3, T)
        ln_Y[j] = (a0[j] + aK[j]*ln_K[j] + aL[j]*ln_L[j]
                    + 0.5*bKK[j]*ln_K[j]**2 + 0.5*bLL[j]*ln_L[j]**2
                    + bKL[j]*ln_K[j]*ln_L[j]
                    + rng.normal(0, 0.02, T))

    return np.exp(ln_Y), np.exp(ln_K), np.exp(ln_L)


def run_translog_decay(mc_id):
    """One MC replication of translog decay simulation."""
    seed = SEED_BASE + mc_id * 2000 + 500
    Y, K, L = translog_panel(N_FIRMS_T, T_PERIODS, INTERACTION_SCALE, seed)

    results = []
    Y_cur, K_cur, L_cur = Y, K, L
    for lev, gs in enumerate(HIERARCHY_T):
        Y_cur, K_cur, L_cur = aggregate(Y_cur, K_cur, L_cur, gs)
        n = Y_cur.shape[0]
        res = estimate_level(Y_cur, K_cur, L_cur)
        res['level'] = lev + 1
        res['label'] = f'n={n}'
        results.append(res)

    return results


# ==========================================================
# MAIN
# ==========================================================

def summarize_mc(all_results, n_levels):
    """Summarize MC replications across levels."""
    keys = ['R2_translog', 'R2_cd', 'R2_ces', 'interaction_ratio',
            'pct_F_reject', 'rho_mean', 'rho_std']
    rows = []
    for lev in range(1, n_levels + 1):
        level_runs = [r for run in all_results for r in run if r['level'] == lev]
        if not level_runs:
            continue
        row = {'level': lev, 'label': level_runs[0]['label'],
               'n_units': level_runs[0]['n_units']}
        for k in keys:
            vals = [r[k] for r in level_runs if not np.isnan(r.get(k, np.nan))]
            row[f'{k}_mean'] = np.mean(vals) if vals else np.nan
            row[f'{k}_se'] = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
        rows.append(row)
    return rows


def main():
    t0 = time.time()

    # ---- SIMULATION 1: HOUTHAKKER ----
    print("=" * 70)
    print("  SIMULATION 1: HOUTHAKKER AGGREGATION")
    print(f"  {N_FIRMS_H} DRS Cobb-Douglas firms, nu={NU}, T={T_PERIODS}")
    n = N_FIRMS_H
    print(f"  Hierarchy: {n}", end="")
    for g in HIERARCHY_H:
        n //= g
        print(f" -> {n}", end="")
    print()
    print("=" * 70)

    h_results = []
    for mc in range(N_MC):
        if mc % 5 == 0:
            print(f"  MC {mc+1}/{N_MC}  ({time.time()-t0:.0f}s)")
        h_results.append(run_houthakker(mc))

    h_summary = summarize_mc(h_results, len(HIERARCHY_H))

    print("\n  HOUTHAKKER RESULTS:")
    for row in h_summary:
        print(f"    Level {row['level']} ({row['label']}): "
              f"R²_CES={row['R2_ces_mean']:.4f}  "
              f"||β||/||α||={row['interaction_ratio_mean']:.4f}  "
              f"F-reject={row['pct_F_reject_mean']*100:.0f}%  "
              f"ρ={row['rho_mean_mean']:.3f}±{row['rho_std_mean']:.3f}")

    # ---- SIMULATION 2: TRANSLOG DECAY ----
    print("\n" + "=" * 70)
    print("  SIMULATION 2: TRANSLOG INTERACTION DECAY")
    print(f"  {N_FIRMS_T} translog firms, interaction_scale={INTERACTION_SCALE}")
    n = N_FIRMS_T
    print(f"  Hierarchy: {n}", end="")
    for g in HIERARCHY_T:
        n //= g
        print(f" -> {n}", end="")
    print()
    print("=" * 70)

    t_results = []
    for mc in range(N_MC):
        if mc % 5 == 0:
            print(f"  MC {mc+1}/{N_MC}  ({time.time()-t0:.0f}s)")
        t_results.append(run_translog_decay(mc))

    t_summary = summarize_mc(t_results, len(HIERARCHY_T))

    print("\n  TRANSLOG DECAY RESULTS:")
    for row in t_summary:
        print(f"    Level {row['level']} ({row['label']}): "
              f"R²_CES={row['R2_ces_mean']:.4f}  "
              f"||β||/||α||={row['interaction_ratio_mean']:.4f}  "
              f"F-reject={row['pct_F_reject_mean']*100:.0f}%  "
              f"ρ={row['rho_mean_mean']:.3f}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")

    # ---- KEY TESTS ----
    print("\n" + "=" * 70)
    print("  KEY PREDICTIONS")
    print("=" * 70)

    # Houthakker: CES R² should be high at all aggregate levels
    h_ces = [r['R2_ces_mean'] for r in h_summary]
    h_ir = [r['interaction_ratio_mean'] for r in h_summary]
    print(f"\n  Houthakker — CES R² across levels: {['%.4f' % v for v in h_ces]}")
    print(f"  Houthakker — Interaction ratio:     {['%.4f' % v for v in h_ir]}")
    if h_ces:
        print(f"  CES R² at coarsest level: {h_ces[-1]:.4f}")
        print(f"  Interaction ratio at coarsest: {h_ir[-1]:.4f}")

    # Translog decay: interaction ratio should decrease
    t_ir = [r['interaction_ratio_mean'] for r in t_summary]
    print(f"\n  Translog decay — Interaction ratio: {['%.4f' % v for v in t_ir]}")
    if len(t_ir) >= 2:
        print(f"  Decay ratio (first to last): {t_ir[-1]/t_ir[0]:.3f}")
        monotone = all(t_ir[i] >= t_ir[i+1] for i in range(len(t_ir)-1))
        print(f"  Monotone decay: {'YES' if monotone else 'NO'}")

    # ---- SAVE RESULTS ----
    all_rows = []
    for row in h_summary:
        row['simulation'] = 'Houthakker'
        all_rows.append(row)
    for row in t_summary:
        row['simulation'] = 'Translog'
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    csv_path = '/home/jonsmirl/thesis/thesis_data/monte_carlo_emergence.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ---- LATEX TABLE ----
    tex_path = '/home/jonsmirl/thesis/thesis_data/monte_carlo_emergence_table.tex'
    with open(tex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Monte Carlo: CES emergence under hierarchical aggregation}\n")
        f.write("\\label{tab:monte_carlo}\n\\small\n")
        f.write("\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\n")
        f.write("Simulation & Level & $n$ & $\\|\\beta\\|/\\|\\alpha\\|$ & "
                "$R^2_{\\text{CES}}$ & $R^2_{\\text{TL}}$ & "
                "\\% $F$-rej. & $\\hat{\\rho}$ \\\\\n")
        f.write("\\midrule\n")

        for sim_name, summary in [('Houthakker', h_summary), ('Translog decay', t_summary)]:
            for i, row in enumerate(summary):
                label = sim_name if i == 0 else ''
                f.write(f"{label} & {row['level']} & {row['n_units']} & "
                        f"{row['interaction_ratio_mean']:.3f} & "
                        f"{row['R2_ces_mean']:.4f} & "
                        f"{row['R2_translog_mean']:.4f} & "
                        f"{row['pct_F_reject_mean']*100:.0f}\\% & "
                        f"{row['rho_mean_mean']:.2f} \\\\\n")
            f.write("\\addlinespace\n")

        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\begin{minipage}{0.95\\textwidth}\n\\vspace{0.5em}\n")
        f.write(f"\\footnotesize\\textit{{Notes:}} {N_MC} Monte Carlo replications. "
                f"Houthakker: {N_FIRMS_H} DRS Cobb-Douglas firms ($\\nu={NU}$) with "
                "heterogeneous capital shares $\\alpha_j \\sim \\text{Beta}(2,3)$, "
                "profit-maximizing at common factor prices. "
                f"Translog decay: {N_FIRMS_T} firms with heterogeneous translog production "
                f"(interaction scale $= {INTERACTION_SCALE}$), common-factor inputs. "
                f"$T = {T_PERIODS}$. Aggregation by summation. "
                "$\\|\\beta\\|/\\|\\alpha\\|$: translog interaction ratio. "
                "$R^2_{\\text{CES}}$: one-parameter CES. "
                "$R^2_{\\text{TL}}$: full translog. "
                "\\% $F$-rej.: CES restriction rejected at 5\\%.\n")
        f.write("\\end{minipage}\n\\end{table}\n")

    print(f"  Saved: {tex_path}")
    print("=" * 70)

    return df


if __name__ == '__main__':
    main()
