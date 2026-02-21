#!/usr/bin/env python3
"""
test_nber_ces.py
================
Test CES emergence using NBER-CES Manufacturing Industry Database.

Uses actual production data (value added, capital, labor) at 6-digit NAICS,
aggregated to 5, 4, 3, and 2-digit levels. At each level, estimates CES
and translog, testing whether:
    1. CES R² is high at all levels
    2. Translog interaction ratio decreases with aggregation
    3. CES ρ is estimated consistently

Data: NBER-CES Manufacturing Industry Database v1 (1958-2018, 2012 NAICS).
Variables: vadd (value added), cap (capital stock), prodh (production hours).

Output: thesis_data/nber_ces_emergence.csv
        thesis_data/nber_ces_emergence_table.tex
"""

import numpy as np
from scipy.optimize import minimize
from scipy import stats
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/home/jonsmirl/thesis/data/nberces5818v1_n2012.csv'
OUT_CSV = '/home/jonsmirl/thesis/thesis_data/nber_ces_emergence.csv'
OUT_TEX = '/home/jonsmirl/thesis/thesis_data/nber_ces_emergence_table.tex'

# Use a stable recent window for estimation
YEAR_MIN = 1977
YEAR_MAX = 2018

print("=" * 70)
print("  NBER-CES MANUFACTURING: CES EMERGENCE UNDER NAICS AGGREGATION")
print(f"  Data: {DATA_PATH}")
print(f"  Sample: {YEAR_MIN}-{YEAR_MAX}")
print("=" * 70)


# =========================================================
# LOAD AND PREPARE DATA
# =========================================================

df = pd.read_csv(DATA_PATH)
df = df[(df['year'] >= YEAR_MIN) & (df['year'] <= YEAR_MAX)].copy()
df['naics'] = df['naics'].astype(str).str.strip()

# Real variables: use value added, capital stock, production hours
# Filter out rows with missing/zero values
for v in ['vadd', 'cap', 'prodh']:
    df = df[df[v] > 0]

print(f"  Loaded {len(df)} observations, {df['naics'].nunique()} industries, "
      f"{df['year'].nunique()} years")

# Create NAICS hierarchy codes
df['n6'] = df['naics'].str[:6]
df['n5'] = df['naics'].str[:5]
df['n4'] = df['naics'].str[:4]
df['n3'] = df['naics'].str[:3]
df['n2'] = df['naics'].str[:2]

# Count industries at each level
for d, col in [(6, 'n6'), (5, 'n5'), (4, 'n4'), (3, 'n3'), (2, 'n2')]:
    print(f"  {d}-digit: {df[col].nunique()} unique codes")


# =========================================================
# AGGREGATION FUNCTIONS
# =========================================================

def aggregate_to_level(df, level_col, year_col='year'):
    """Aggregate vadd, cap, prodh to a given NAICS level by summation within year."""
    agg = df.groupby([level_col, year_col]).agg(
        vadd=('vadd', 'sum'),
        cap=('cap', 'sum'),
        prodh=('prodh', 'sum'),
        n_sub=('n6', 'nunique')  # number of sub-industries
    ).reset_index()
    agg.rename(columns={level_col: 'code'}, inplace=True)
    return agg


# =========================================================
# ESTIMATION: TRANSLOG
# =========================================================

def fit_translog_ts(ln_y, ln_k, ln_l):
    """Fit translog with time trend to time series. Returns dict of results."""
    n = len(ln_y)
    t = np.arange(n, dtype=float)
    t = (t - t.mean()) / t.std()  # standardize

    # Full translog + time trend
    X = np.column_stack([
        np.ones(n), t, ln_k, ln_l,
        0.5 * ln_k**2, 0.5 * ln_l**2, ln_k * ln_l
    ])
    b = np.linalg.lstsq(X, ln_y, rcond=None)[0]
    ss_res = np.sum((ln_y - X @ b)**2)
    ss_tot = np.sum((ln_y - ln_y.mean())**2)
    R2_tl = 1 - ss_res / ss_tot if ss_tot > 1e-20 else np.nan

    # Cobb-Douglas + time trend (restricted: no interaction terms)
    X_cd = X[:, :4]
    b_cd = np.linalg.lstsq(X_cd, ln_y, rcond=None)[0]
    ss_res_cd = np.sum((ln_y - X_cd @ b_cd)**2)
    R2_cd = 1 - ss_res_cd / ss_tot if ss_tot > 1e-20 else np.nan

    # Interaction ratio: ||beta_interactions|| / ||alpha_firstorder||
    # b indices: [0]=const, [1]=trend, [2]=ln_k, [3]=ln_l, [4]=0.5*lnk^2, [5]=0.5*lnl^2, [6]=lnk*lnl
    alpha_norm = np.sqrt(b[2]**2 + b[3]**2)
    beta_norm = np.sqrt(b[4]**2 + b[5]**2 + b[6]**2)
    ir = beta_norm / alpha_norm if alpha_norm > 1e-10 else np.nan

    df2 = n - 7
    if df2 > 0 and ss_res > 1e-20:
        F = ((ss_res_cd - ss_res) / 3) / (ss_res / df2)
        p = 1 - stats.f.cdf(max(F, 0), 3, df2)
    else:
        F, p = np.nan, np.nan

    return {'R2_tl': R2_tl, 'R2_cd': R2_cd, 'ir': ir, 'F': F, 'p': p,
            'alpha_K': b[2], 'alpha_L': b[3],
            'beta_KK': b[4], 'beta_LL': b[5], 'beta_KL': b[6]}


# =========================================================
# ESTIMATION: CES
# =========================================================

def fit_ces_ts(ln_y, ln_k, ln_l):
    """Fit CES with time trend: Y = A*exp(gamma*t)*(w*K^rho + (1-w)*L^rho)^(1/rho)."""
    n = len(ln_y)
    t = np.arange(n, dtype=float)
    t = (t - t.mean()) / t.std()

    def sse(params):
        ln_A, gamma, w_logit, rho = params
        w = 1.0 / (1.0 + np.exp(-np.clip(w_logit, -10, 10)))
        rho_c = np.clip(rho, -5, 0.99)
        if abs(rho_c) < 1e-6:
            pred = ln_A + gamma * t + w * ln_k + (1 - w) * ln_l
        else:
            inner = w * np.exp(rho_c * ln_k) + (1 - w) * np.exp(rho_c * ln_l)
            inner = np.maximum(inner, 1e-30)
            pred = ln_A + gamma * t + (1.0 / rho_c) * np.log(inner)
        return np.sum((ln_y - pred)**2)

    best = None
    m = np.mean(ln_y)
    for rho0 in [-3, -1.5, -0.5, -0.1, 0.01, 0.3, 0.6, 0.9]:
        for w0 in [-1, 0, 1]:
            for g0 in [0.0, 0.5]:
                try:
                    res = minimize(sse, [m, g0, w0, rho0], method='Nelder-Mead',
                                 options={'maxiter': 8000, 'xatol': 1e-9, 'fatol': 1e-12})
                    if best is None or res.fun < best.fun:
                        best = res
                except Exception:
                    pass

    if best is None:
        return {'R2_ces': np.nan, 'rho': np.nan, 'w': np.nan}

    ss_tot = np.sum((ln_y - ln_y.mean())**2)
    R2 = 1 - best.fun / ss_tot if ss_tot > 1e-20 else np.nan
    rho = np.clip(best.x[3], -5, 0.99)
    w = 1.0 / (1.0 + np.exp(-np.clip(best.x[2], -10, 10)))

    return {'R2_ces': R2, 'rho': rho, 'w': w}


# =========================================================
# CROSS-SECTIONAL ESTIMATION (pooled across industries)
# =========================================================

def estimate_cross_section(agg_df, label):
    """Estimate CES and translog on pooled cross-section of industries at a level."""
    codes = agg_df['code'].unique()
    n_codes = len(codes)

    # Per-industry time-series estimation
    rho_list, R2_ces_list, R2_tl_list, R2_cd_list, ir_list, p_list = [], [], [], [], [], []

    for code in codes:
        sub = agg_df[agg_df['code'] == code].sort_values('year')
        if len(sub) < 15:  # need enough years
            continue
        ln_y = np.log(sub['vadd'].values)
        ln_k = np.log(sub['cap'].values)
        ln_l = np.log(sub['prodh'].values)

        tl = fit_translog_ts(ln_y, ln_k, ln_l)
        ces = fit_ces_ts(ln_y, ln_k, ln_l)

        R2_tl_list.append(tl['R2_tl'])
        R2_cd_list.append(tl['R2_cd'])
        ir_list.append(tl['ir'])
        p_list.append(tl['p'])
        R2_ces_list.append(ces['R2_ces'])
        rho_list.append(ces['rho'])

    valid = [r for r in rho_list if not np.isnan(r)]

    return {
        'label': label,
        'n_industries': n_codes,
        'n_estimated': len(valid),
        'R2_translog': np.nanmedian(R2_tl_list),
        'R2_cd': np.nanmedian(R2_cd_list),
        'R2_ces': np.nanmedian(R2_ces_list),
        'interaction_ratio': np.nanmedian(ir_list),
        'pct_F_reject': np.nanmean([1 if p < 0.05 else 0 for p in p_list if not np.isnan(p)]),
        'rho_median': np.nanmedian(valid),
        'rho_iqr': np.nanpercentile(valid, 75) - np.nanpercentile(valid, 25) if len(valid) > 2 else np.nan,
        'rho_mean': np.nanmean(valid),
    }


# =========================================================
# MAIN ANALYSIS
# =========================================================

def main():
    results = []

    # --- Level 1: Individual 6-digit industries (finest level) ---
    print("\n  Estimating at 6-digit level (individual industries)...")
    # For 6-digit, use pooled panel estimation across industries within each year
    # Each industry gets a time-series estimate
    agg6 = df.groupby(['n6', 'year']).agg(
        vadd=('vadd', 'sum'), cap=('cap', 'sum'), prodh=('prodh', 'sum')
    ).reset_index().rename(columns={'n6': 'code'})
    res = estimate_cross_section(agg6, '6-digit')
    res['digits'] = 6
    results.append(res)
    print(f"    {res['n_estimated']}/{res['n_industries']} industries estimated")
    print(f"    Median R²_CES = {res['R2_ces']:.4f}, "
          f"median ||β||/||α|| = {res['interaction_ratio']:.4f}, "
          f"median ρ = {res['rho_median']:.3f}")

    # --- Levels 2-5: Aggregated NAICS levels ---
    for digits, col in [(5, 'n5'), (4, 'n4'), (3, 'n3'), (2, 'n2')]:
        print(f"\n  Estimating at {digits}-digit level...")
        agg = aggregate_to_level(df, col)
        res = estimate_cross_section(agg, f'{digits}-digit')
        res['digits'] = digits
        results.append(res)
        print(f"    {res['n_estimated']}/{res['n_industries']} industries estimated")
        print(f"    Median R²_CES = {res['R2_ces']:.4f}, "
              f"median ||β||/||α|| = {res['interaction_ratio']:.4f}, "
              f"median ρ = {res['rho_median']:.3f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  {'Level':<10} {'n':>5} {'R²_CES':>8} {'R²_TL':>7} "
          f"{'||β||/||α||':>12} {'%F-rej':>7} {'ρ_med':>7} {'ρ_IQR':>7}")
    print("  " + "-" * 68)
    for res in results:
        print(f"  {res['label']:<10} {res['n_estimated']:>5} "
              f"{res['R2_ces']:>8.4f} {res['R2_translog']:>7.4f} "
              f"{res['interaction_ratio']:>12.4f} "
              f"{res['pct_F_reject']*100:>6.0f}% "
              f"{res['rho_median']:>7.3f} {res['rho_iqr']:>7.3f}")

    # --- Key tests ---
    ir_vals = [r['interaction_ratio'] for r in results]
    ces_vals = [r['R2_ces'] for r in results]

    print(f"\n  Interaction ratio (6→2 digit): {ir_vals[0]:.4f} → {ir_vals[-1]:.4f}")
    if ir_vals[0] > 0:
        print(f"  Decay ratio: {ir_vals[-1]/ir_vals[0]:.3f}")
    mono = all(ir_vals[i] >= ir_vals[i+1] for i in range(len(ir_vals)-1))
    print(f"  Monotone decay: {'YES' if mono else 'NO'}")

    print(f"\n  CES R² (6→2 digit): {ces_vals[0]:.4f} → {ces_vals[-1]:.4f}")

    # --- Save CSV ---
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved: {OUT_CSV}")

    # --- Save LaTeX ---
    with open(OUT_TEX, 'w') as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{CES emergence in NBER-CES Manufacturing data "
                "under NAICS aggregation}\n")
        f.write("\\label{tab:nber_ces}\n\\small\n")
        f.write("\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\n")
        f.write("NAICS & $n$ & $R^2_{\\text{CES}}$ & $R^2_{\\text{TL}}$ & "
                "$\\|\\beta\\|/\\|\\alpha\\|$ & \\% $F$-rej. & "
                "$\\tilde{\\rho}$ & IQR($\\rho$) \\\\\n")
        f.write("\\midrule\n")
        for res in results:
            f.write(f"{res['label']} & {res['n_estimated']} & "
                    f"{res['R2_ces']:.3f} & {res['R2_translog']:.3f} & "
                    f"{res['interaction_ratio']:.3f} & "
                    f"{res['pct_F_reject']*100:.0f}\\% & "
                    f"{res['rho_median']:.2f} & "
                    f"{res['rho_iqr']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\begin{minipage}{0.95\\textwidth}\n\\vspace{0.5em}\n")
        f.write("\\footnotesize\\textit{Notes:} NBER-CES Manufacturing "
                "Industry Database (1977--2018, 2012 NAICS). "
                "Value added regressed on capital stock and production hours. "
                "Each industry estimated by time series; medians reported across "
                "industries at each aggregation level. "
                "$\\|\\beta\\|/\\|\\alpha\\|$: translog interaction ratio. "
                "$\\tilde{\\rho}$: median CES curvature. "
                "IQR: interquartile range of $\\hat{\\rho}$ across industries.\n")
        f.write("\\end{minipage}\n\\end{table}\n")
    print(f"  Saved: {OUT_TEX}")
    print("=" * 70)

    return df_out


if __name__ == '__main__':
    main()
