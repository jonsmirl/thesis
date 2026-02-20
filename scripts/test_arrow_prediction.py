#!/usr/bin/env python3
"""
test_arrow_prediction.py — Empirical test of the Free Energy Framework

Arrow Prediction (Proposition 9 & 10):
    Majoritarian systems (high effective ρ) tolerate more informational
    noise than consensus systems (low effective ρ).

Test specification:
    GovEff_{it} = α_i + γ_t + β·(Majoritarian_i × Internet_{it}) + δ·X_{it} + ε_{it}

    where α_i = country FE, γ_t = year FE, Internet = users per capita,
    Majoritarian = Lijphart (2012) executives-parties dimension score.

Prediction: β > 0 — majoritarian systems maintain governance quality
better as informational noise (internet penetration) increases.

Data sources:
    1. WGI (World Governance Indicators) — outcome variables — World Bank API
    2. Internet users per capita — World Bank WDI API
    3. GDP per capita, population — World Bank WDI API (controls)
    4. Majoritarian score — Lijphart (2012), extended (embedded)

Usage:
    source .venv/bin/activate && source env.sh
    python scripts/test_arrow_prediction.py
"""

import requests
import pandas as pd
import numpy as np
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Check dependencies
# ---------------------------------------------------------------------------
try:
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS
except ImportError:
    print("ERROR: statsmodels required.  pip install statsmodels")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not installed — figures will not be generated.")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'thesis_data')
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================================
# SECTION 1: Majoritarian Score — Lijphart (2012) Executives-Parties Dimension
# ============================================================================
# Scale: higher = more majoritarian (Westminster, FPTP, concentrated executive)
#         lower  = more consensus (PR, coalitions, dispersed power)
# Source: Lijphart, "Patterns of Democracy" (2012), Table 14.2, extended
#         with additional democracies classified by electoral system type.

MAJORITARIAN_SCORE = {
    # --- Lijphart 36 democracies (direct from Table 14.2) ---
    'GBR':  1.21,  'NZL':  1.19,  'FRA':  0.94,  'CAN':  1.11,
    'AUS':  0.86,  'GRC':  1.24,  'USA':  0.95,  'IND':  0.34,
    'JPN':  0.33,  'KOR':  0.25,  'ITA':  0.12,  'IRL': -0.01,
    'PRT':  0.03,  'ESP': -0.16,  'DEU': -0.63,  'NLD': -0.81,
    'BEL': -0.69,  'CHE': -1.31,  'DNK': -0.68,  'SWE': -0.67,
    'NOR': -0.60,  'FIN': -0.51,  'AUT': -0.34,  'ISR': -0.65,
    'LUX': -0.27,  'ISL': -0.60,  'CRI': -0.30,  'COL':  0.50,
    'URY': -0.20,  'BWA':  0.90,  'MUS':  0.70,  'TTO':  1.50,
    'BRB':  1.73,  'BHS':  1.39,  'JAM':  1.65,

    # --- Extended: FPTP systems (majoritarian, +0.7 to +1.0) ---
    'BGD':  0.80,  'GHA':  0.85,  'KEN':  0.75,  'MWI':  0.80,
    'ZMB':  0.80,  'TZA':  0.75,  'UGA':  0.70,  'SLE':  0.70,

    # --- Extended: Presidential + mixed/PR ---
    'BRA':  0.40,  'MEX':  0.45,  'CHL': -0.10,  'ARG':  0.30,
    'PER':  0.35,  'PHL':  0.50,  'IDN': -0.20,  'TUR':  0.20,
    'TWN':  0.20,  'THA':  0.30,

    # --- Extended: Parliamentary PR (consensus, -0.3 to -0.8) ---
    'POL': -0.35,  'CZE': -0.45,  'HUN': -0.10,  'SVK': -0.40,
    'SVN': -0.50,  'EST': -0.45,  'LVA': -0.40,  'LTU': -0.30,
    'ZAF': -0.50,  'ROU': -0.25,  'BGR': -0.30,  'HRV': -0.35,
}


# ============================================================================
# SECTION 2: World Bank API
# ============================================================================

def download_wb(indicator, start=1996, end=2023):
    """Download a single WB indicator for all countries, with retry."""
    records = []
    page = 1
    while True:
        url = (f"http://api.worldbank.org/v2/country/all/indicator/{indicator}"
               f"?format=json&per_page=10000&date={start}:{end}&page={page}")
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  FAILED after 3 attempts: {indicator} page {page}: {e}")
                    return pd.DataFrame(records)
                time.sleep(2 ** attempt)

        if len(data) < 2 or data[1] is None:
            break
        for item in data[1]:
            if item['value'] is not None and item['countryiso3code']:
                records.append({
                    'iso3': item['countryiso3code'],
                    'year': int(item['date']),
                    'value': float(item['value']),
                })
        if page >= data[0].get('pages', 1):
            break
        page += 1
        time.sleep(0.3)

    return pd.DataFrame(records)


# ============================================================================
# SECTION 3: Build Panel
# ============================================================================

INDICATORS = {
    # WGI outcomes (scale ≈ -2.5 to +2.5)
    'GE.EST':           'gov_effectiveness',
    'RQ.EST':           'regulatory_quality',
    'RL.EST':           'rule_of_law',
    'VA.EST':           'voice_accountability',
    'PV.EST':           'political_stability',
    'CC.EST':           'control_corruption',
    # Controls & treatment
    'IT.NET.USER.ZS':   'internet_users_pct',
    'NY.GDP.PCAP.PP.CD':'gdp_pc_ppp',
    'SP.POP.TOTL':      'population',
}


def build_panel(force=False):
    """Download all indicators and build the panel."""
    cache = os.path.join(OUTPUT_DIR, 'arrow_test_panel.csv')
    if os.path.exists(cache) and not force:
        print(f"Loading cached panel from {cache}")
        return pd.read_csv(cache)

    frames = {}
    for code, name in INDICATORS.items():
        print(f"  Downloading {name} ({code}) ...")
        df = download_wb(code)
        if len(df) == 0:
            print(f"    WARNING: no data for {code}")
            continue
        df = df.rename(columns={'value': name})
        frames[name] = df[['iso3', 'year', name]]
        print(f"    {len(df):,} observations")

    # Merge
    panel = None
    for name, df in frames.items():
        panel = df if panel is None else panel.merge(df, on=['iso3', 'year'], how='outer')

    # Restrict to classified democracies
    panel = panel[panel['iso3'].isin(MAJORITARIAN_SCORE)].copy()
    panel['majoritarian'] = panel['iso3'].map(MAJORITARIAN_SCORE)

    # Derived variables
    panel['internet_01'] = panel['internet_users_pct'] / 100.0  # rescale to [0,1]
    panel['log_gdp_pc']  = np.log(panel['gdp_pc_ppp'].clip(lower=100))
    panel['log_pop']     = np.log(panel['population'].clip(lower=1000))
    panel['maj_x_inet']  = panel['majoritarian'] * panel['internet_01']

    panel.to_csv(cache, index=False)
    nc = panel['iso3'].nunique()
    print(f"\nPanel: {len(panel):,} obs, {nc} countries, "
          f"{panel['year'].min()}-{panel['year'].max()}")
    return panel


# ============================================================================
# SECTION 4: Regression
# ============================================================================

def run_panel_ols(panel, yvar, label):
    """OLS with country + year FE, clustered SE by country."""
    cols = [yvar, 'internet_01', 'maj_x_inet', 'log_gdp_pc', 'iso3', 'year']
    df = panel.dropna(subset=cols).copy()
    if len(df) < 60:
        print(f"\n  Skipping {label}: only {len(df)} obs")
        return None

    # Dummies
    c_dum = pd.get_dummies(df['iso3'], prefix='c', drop_first=True, dtype=float)
    y_dum = pd.get_dummies(df['year'], prefix='y', drop_first=True, dtype=float)

    X = pd.concat([df[['internet_01', 'maj_x_inet', 'log_gdp_pc']].reset_index(drop=True),
                    c_dum.reset_index(drop=True),
                    y_dum.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = df[yvar].reset_index(drop=True)

    model = OLS(y.astype(float), X.astype(float))
    try:
        res = model.fit(cov_type='cluster',
                        cov_kwds={'groups': df['iso3'].reset_index(drop=True)})
    except Exception:
        res = model.fit(cov_type='HC1')

    # Report
    header = f"\n{'='*72}\n  {label}\n{'='*72}"
    print(header)
    print(f"  N = {int(res.nobs)},  countries = {df['iso3'].nunique()},  "
          f"years = {df['year'].min()}-{df['year'].max()}")
    print(f"  R² = {res.rsquared:.4f}")
    print(f"\n  {'Variable':<28} {'β':>9} {'SE':>9} {'t':>7} {'p':>8}")
    print(f"  {'-'*62}")
    for v in ['internet_01', 'maj_x_inet', 'log_gdp_pc']:
        b = res.params[v]
        s = res.bse[v]
        t = res.tvalues[v]
        p = res.pvalues[v]
        star = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '†' if p < .1 else ''
        print(f"  {v:<28} {b:>9.4f} {s:>9.4f} {t:>7.2f} {p:>8.4f} {star}")

    b_int = res.params['maj_x_inet']
    p_int = res.pvalues['maj_x_inet']
    print(f"\n  PREDICTION:  β(maj_x_inet) > 0")
    if b_int > 0 and p_int < 0.05:
        print(f"  VERDICT:     CONFIRMED  (β = {b_int:.4f}, p = {p_int:.4f})")
    elif b_int > 0 and p_int < 0.10:
        print(f"  VERDICT:     WEAK SUPPORT  (β = {b_int:.4f}, p = {p_int:.4f})")
    elif b_int > 0:
        print(f"  VERDICT:     CORRECT SIGN, not significant  (β = {b_int:.4f}, p = {p_int:.4f})")
    else:
        print(f"  VERDICT:     WRONG SIGN  (β = {b_int:.4f}, p = {p_int:.4f})")

    return {'outcome': label, 'beta': b_int, 'se': res.bse['maj_x_inet'],
            'p': p_int, 'n': int(res.nobs), 'r2': res.rsquared,
            'beta_inet': res.params['internet_01'],
            'p_inet': res.pvalues['internet_01']}


# ============================================================================
# SECTION 5: Figures
# ============================================================================

def make_figures(panel):
    """Produce the main scatter/interaction figure."""
    if not HAS_MPL:
        return

    yvar = 'gov_effectiveness'
    df = panel.dropna(subset=[yvar, 'internet_01', 'majoritarian']).copy()
    med = df['majoritarian'].median()
    df['system'] = np.where(df['majoritarian'] > med, 'Majoritarian (high ρ)',
                            'Consensus (low ρ)')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel A: level scatter ---
    ax = axes[0]
    for sys, col in [('Majoritarian (high ρ)', '#2166ac'),
                     ('Consensus (low ρ)', '#b2182b')]:
        sub = df[df['system'] == sys]
        ax.scatter(sub['internet_01'] * 100, sub[yvar], s=8, alpha=0.25, color=col,
                   label=sys)
        # Binned means (deciles of internet)
        sub = sub.copy()
        sub['inet_bin'] = pd.qcut(sub['internet_01'], 10, duplicates='drop')
        means = sub.groupby('inet_bin', observed=True).agg(
            x=('internet_01', 'mean'), y=(yvar, 'mean')).reset_index()
        ax.plot(means['x'] * 100, means['y'], '-o', color=col, lw=2, ms=5)

    ax.set_xlabel('Internet users (% of population)', fontsize=11)
    ax.set_ylabel('Government Effectiveness (WGI)', fontsize=11)
    ax.set_title('A.  Governance vs. Internet Penetration', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    # --- Panel B: long-difference scatter ---
    ax = axes[1]
    changes = []
    for cc in df['iso3'].unique():
        c = df[df['iso3'] == cc].sort_values('year')
        early = c[c['year'] <= 2005]
        late  = c[c['year'] >= 2015]
        if len(early) >= 2 and len(late) >= 2:
            changes.append({
                'iso3': cc,
                'maj': c['majoritarian'].iloc[0],
                'dy': late[yvar].mean() - early[yvar].mean(),
                'di': late['internet_01'].mean() - early['internet_01'].mean(),
            })
    ch = pd.DataFrame(changes)

    if len(ch) > 10:
        colors = ['#2166ac' if m > med else '#b2182b' for m in ch['maj']]
        ax.scatter(ch['maj'], ch['dy'], c=colors, s=50, alpha=0.75, edgecolors='k',
                   linewidths=0.5, zorder=3)
        for _, r in ch.iterrows():
            ax.annotate(r['iso3'], (r['maj'], r['dy']), fontsize=6.5, alpha=0.7,
                        xytext=(3, 3), textcoords='offset points')
        # Trend line
        z = np.polyfit(ch['maj'], ch['dy'], 1)
        xs = np.linspace(ch['maj'].min() - 0.1, ch['maj'].max() + 0.1, 100)
        ax.plot(xs, np.polyval(z, xs), 'k--', lw=1.5, label=f'slope = {z[0]:.3f}')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xlabel('Majoritarian score (Lijphart)', fontsize=11)
        ax.set_ylabel('Δ Gov. Effectiveness  (post-2015 − pre-2005)', fontsize=11)
        ax.set_title('B.  Long-Difference: Who Held Up Better?', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path_png = os.path.join(FIGURE_DIR, 'arrow_prediction_test.png')
    path_pdf = os.path.join(FIGURE_DIR, 'arrow_prediction_test.pdf')
    plt.savefig(path_png, dpi=200, bbox_inches='tight')
    plt.savefig(path_pdf, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to {path_png}")


# ============================================================================
# SECTION 6: LaTeX Table
# ============================================================================

def latex_table(rows):
    """Write a publication-ready LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Arrow Prediction Test: Majoritarian Systems and Informational Noise}",
        r"\label{tab:arrow_test}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"WGI Outcome & $\hat\beta_{\text{Maj}\times\text{Inet}}$ & SE"
        r" & $p$ & $N$ & $R^2$ \\",
        r"\midrule",
    ]
    for r in rows:
        s = '***' if r['p'] < .001 else '**' if r['p'] < .01 else '*' if r['p'] < .05 else ''
        lines.append(
            f"  {r['outcome']} & {r['beta']:.4f}{s} & ({r['se']:.4f})"
            f" & {r['p']:.3f} & {r['n']} & {r['r2']:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{4pt}",
        r"\parbox{0.9\textwidth}{\footnotesize",
        r"  Panel regression with country and year fixed effects;",
        r"  standard errors clustered by country.",
        r"  Internet = users as fraction of population (0--1).",
        r"  Majoritarian score from Lijphart (2012) executives-parties dimension,",
        r"  extended to additional democracies.",
        r"  Theory predicts $\hat\beta > 0$: majoritarian systems (high $\rho$)",
        r"  tolerate more informational noise.",
        r"  $^{***}p<0.001$; $^{**}p<0.01$; $^{*}p<0.05$.",
        r"}",
        r"\end{table}",
    ]
    path = os.path.join(OUTPUT_DIR, 'arrow_test_table.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"LaTeX table saved to {path}")


# ============================================================================
# SECTION 7: Robustness — Binary Split
# ============================================================================

def robustness_binary(panel, yvar='gov_effectiveness'):
    """Robustness: split sample into majoritarian vs consensus, compare slopes."""
    cols = [yvar, 'internet_01', 'log_gdp_pc', 'iso3', 'year', 'majoritarian']
    df = panel.dropna(subset=cols).copy()
    med = df['majoritarian'].median()

    print(f"\n{'='*72}")
    print("  ROBUSTNESS: Split-sample comparison")
    print(f"{'='*72}")
    print(f"  Median majoritarian score: {med:.2f}")

    for label, mask in [("MAJORITARIAN (above median)", df['majoritarian'] > med),
                        ("CONSENSUS (below median)",    df['majoritarian'] <= med)]:
        sub = df[mask].copy()
        c_dum = pd.get_dummies(sub['iso3'], prefix='c', drop_first=True, dtype=float)
        y_dum = pd.get_dummies(sub['year'], prefix='y', drop_first=True, dtype=float)
        X = pd.concat([sub[['internet_01', 'log_gdp_pc']].reset_index(drop=True),
                        c_dum.reset_index(drop=True),
                        y_dum.reset_index(drop=True)], axis=1)
        X = sm.add_constant(X)
        y = sub[yvar].reset_index(drop=True)
        try:
            res = OLS(y.astype(float), X.astype(float)).fit(
                cov_type='cluster',
                cov_kwds={'groups': sub['iso3'].reset_index(drop=True)})
        except Exception:
            res = OLS(y.astype(float), X.astype(float)).fit(cov_type='HC1')

        b = res.params['internet_01']
        s = res.bse['internet_01']
        p = res.pvalues['internet_01']
        star = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        print(f"\n  {label}")
        print(f"    Countries: {sub['iso3'].nunique()},  N = {int(res.nobs)}")
        print(f"    β(internet) = {b:.4f} (SE {s:.4f}, p = {p:.4f}) {star}")

    print(f"\n  PREDICTION: β(internet) in majoritarian subsample should be")
    print(f"  MORE POSITIVE (or less negative) than in consensus subsample.")


# ============================================================================
# SECTION 8: Main
# ============================================================================

def main():
    print("=" * 72)
    print("  FREE ENERGY FRAMEWORK — EMPIRICAL TEST")
    print("  Arrow Prediction: Democratic Robustness vs. ρ")
    print("=" * 72)
    print()
    print("  HYPOTHESIS")
    print("  Majoritarian systems (high effective ρ) tolerate more informational")
    print("  noise than consensus systems (low effective ρ).")
    print()
    print("  TEST")
    print("  In a panel of democracies (1996-2023), the interaction")
    print("  Majoritarian × Internet → governance should be positive (β > 0).")
    print()
    print("  INTERPRETATION")
    print("  If β > 0, then as internet penetration (proxy for informational")
    print("  noise T) increases, majoritarian countries maintain governance")
    print("  better than consensus countries — confirming Arrow Proposition 10.")
    print()

    # Build data
    print("─" * 72)
    print("  DOWNLOADING DATA")
    print("─" * 72)
    panel = build_panel()

    # Summary stats
    print(f"\n  Countries in sample: {panel['iso3'].nunique()}")
    for cc in sorted(panel['iso3'].unique()):
        score = MAJORITARIAN_SCORE.get(cc, 0)
        n = panel[panel['iso3'] == cc].dropna(subset=['gov_effectiveness']).shape[0]
        if n > 0:
            lbl = "MAJ" if score > 0 else "CON"
            print(f"    {cc} ({lbl:>3}, score={score:+.2f}): {n} years of GovEff data")

    # Main regressions
    print(f"\n{'─'*72}")
    print("  MAIN RESULTS: Panel Regressions (Country + Year FE)")
    print(f"{'─'*72}")

    outcomes = [
        ('gov_effectiveness',   'Government Effectiveness'),
        ('regulatory_quality',  'Regulatory Quality'),
        ('rule_of_law',         'Rule of Law'),
        ('control_corruption',  'Control of Corruption'),
        ('voice_accountability','Voice & Accountability'),
        ('political_stability', 'Political Stability'),
    ]

    results = []
    for var, label in outcomes:
        if var in panel.columns and panel[var].notna().sum() > 50:
            r = run_panel_ols(panel, var, label)
            if r:
                results.append(r)

    # Summary
    if results:
        print(f"\n\n{'='*72}")
        print("  SUMMARY TABLE")
        print(f"{'='*72}")
        print(f"  {'Outcome':<28} {'β(Maj×Inet)':>12} {'p':>8} {'N':>6}  Verdict")
        print(f"  {'-'*66}")
        confirmed = 0
        for r in results:
            star = '***' if r['p'] < .001 else '**' if r['p'] < .01 else '*' if r['p'] < .05 else ''
            v = 'CONFIRMED' if r['beta'] > 0 and r['p'] < .05 else \
                'WEAK'      if r['beta'] > 0 and r['p'] < .10 else \
                'right sign' if r['beta'] > 0 else 'wrong sign'
            if r['beta'] > 0 and r['p'] < .05:
                confirmed += 1
            print(f"  {r['outcome']:<28} {r['beta']:>12.4f} {r['p']:>8.4f}"
                  f" {r['n']:>6}  {v} {star}")

        print(f"\n  Confirmed at 5%: {confirmed}/{len(results)} outcomes")

        # LaTeX table
        latex_table(results)

    # Robustness
    robustness_binary(panel)

    # Figures
    if HAS_MPL:
        print(f"\n{'─'*72}")
        print("  GENERATING FIGURES")
        print(f"{'─'*72}")
        make_figures(panel)

    # Interpretation
    print(f"\n\n{'='*72}")
    print("  INTERPRETATION NOTES")
    print(f"{'='*72}")
    print("""
  The Free Energy Framework (Proposition 10) predicts:

    T*_democracy(ρ) is increasing in ρ

  where ρ maps to the "effective elasticity of substitution" of the
  democratic aggregation rule. Majoritarian/Westminster systems
  (plurality voting, concentrated executive power) correspond to
  high ρ ≈ 1 (utilitarian SWF). Consensus systems (PR, coalitions,
  dispersed veto points) correspond to low ρ (Rawlsian SWF).

  The mechanism: at high ρ, the CES social welfare function
  approximates the arithmetic mean of utilities. The mean is
  estimated consistently by majority voting (law of large numbers),
  so noise washes out — high T* tolerance.

  At low ρ (consensus), the CES aggregate approaches the minimum
  utility. Estimating the minimum requires precise information
  about distributional tails, which is highly sensitive to noise —
  low T* tolerance.

  Internet penetration proxies for informational noise T because
  it increases both information and misinformation simultaneously.

  CAUTION: This is a cross-country panel test and subject to
  standard identification concerns (omitted variables, reverse
  causality). Country FE absorb time-invariant confounds; year FE
  absorb global trends. The remaining variation exploited is
  differential internet adoption speed × institutional type.

  ALTERNATIVE TESTS (not implemented here):
  1. Akerlof: Cross-industry bid-ask spreads vs. trade elasticity σ
     (Data: Ahmad-Riker USITC + Yahoo Finance)
  2. Myerson: Insurance market adverse selection DWL vs. plan
     differentiation (Data: MEPS + CMS)
  3. Akerlof: FCC spectrum auction rents vs. license complementarity
     (Data: CAPCP Penn State)
""")


if __name__ == '__main__':
    main()
