#!/usr/bin/env python3
"""
test_akerlof_banking.py — Akerlof prediction in banking regulation

The CES Free Energy Framework predicts:
    Markets with higher σ (more substitutable products) have lower
    critical temperature T* — they are more vulnerable to adverse
    selection when information quality deteriorates.

Mapping to banking:
    Activity Restrictions Index (ARI) → σ
        High ARI = banks restricted to commodity lending = high σ
        Low ARI  = banks can diversify (securities, insurance, RE) = low σ

    Private Monitoring Index (PMI) → inverse T (information quality)
        High PMI = strong disclosure, auditing, market discipline = low T
        Low PMI  = poor information environment = high T

    Financial Development (domestic credit / GDP) → market functioning

Prediction:
    β₃(ARI × PMI) > 0
    Activity restrictions hurt financial development MORE when
    information quality is LOW.  Better private monitoring MITIGATES
    the harm from regulatory homogenization.

    This is because high-σ commodity banking has low T*: it collapses
    under adverse selection at lower noise thresholds.  Strong private
    monitoring lowers T below T*, allowing the market to function.

Data:
    1. BCL regulatory indices — bcl_indices_panel.csv (already built)
       - ARI: Activity Restrictions Index → σ proxy
       - PMI: Private Monitoring Index → 1/T proxy
    2. Financial development (FD.AST.PRVT.GD.ZS) — World Bank API
    3. Controls: GDP per capita — World Bank API

Novelty:
    Barth-Caprio-Levine (2004, 2006, 2008) test ARI and PMI as SEPARATE
    regressors.  Houston et al. (2010) test creditor rights × info sharing
    but not ARI × PMI.  This specific interaction has not been published.

Usage:
    source .venv/bin/activate && source env.sh
    python scripts/test_akerlof_banking.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import requests
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
except ImportError:
    print("ERROR: pip install statsmodels"); sys.exit(1)

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, 'thesis_data')
FIGDIR = os.path.join(BASE, 'figures')
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

# ISO2 → ISO3 mapping for merging BCL (ISO2) with WDI (ISO3)
ISO2_TO_ISO3 = {}
ISO3_TO_ISO2 = {}


def build_iso_map():
    """Download ISO2 ↔ ISO3 mapping from World Bank country metadata."""
    global ISO2_TO_ISO3, ISO3_TO_ISO2
    cache = os.path.join(OUTPUT, 'iso_mapping.csv')
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        ISO2_TO_ISO3 = dict(zip(df['iso2'], df['iso3']))
        ISO3_TO_ISO2 = dict(zip(df['iso3'], df['iso2']))
        return

    print("  Downloading country code mapping from World Bank...")
    url = "http://api.worldbank.org/v2/country?format=json&per_page=400"
    try:
        r = requests.get(url, timeout=30)
        data = r.json()
        records = []
        for c in data[1]:
            if c.get('iso2Code') and c.get('id') and len(c['id']) == 3:
                records.append({'iso2': c['iso2Code'], 'iso3': c['id']})
        df = pd.DataFrame(records)
        df.to_csv(cache, index=False)
        ISO2_TO_ISO3 = dict(zip(df['iso2'], df['iso3']))
        ISO3_TO_ISO2 = dict(zip(df['iso3'], df['iso2']))
        print(f"    {len(df)} country codes mapped")
    except Exception as e:
        print(f"    Warning: could not download ISO mapping: {e}")


def download_wb(indicator, start=1999, end=2023):
    """Download a World Bank indicator."""
    records = []
    page = 1
    while True:
        url = (f"http://api.worldbank.org/v2/country/all/indicator/{indicator}"
               f"?format=json&per_page=10000&date={start}:{end}&page={page}")
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    FAILED: {indicator}: {e}")
                    return pd.DataFrame()
                import time; time.sleep(2 ** attempt)

        if len(data) < 2 or data[1] is None:
            break
        for item in data[1]:
            if item['value'] is not None and item.get('countryiso3code'):
                records.append({
                    'iso3': item['countryiso3code'],
                    'year': int(item['date']),
                    'value': float(item['value']),
                })
        if page >= data[0].get('pages', 1):
            break
        page += 1
        import time; time.sleep(0.3)

    return pd.DataFrame(records)


# ============================================================================
# SECTION 1: Load BCL Panel
# ============================================================================

def load_bcl():
    """Load BCL indices panel and add ISO3 codes."""
    path = os.path.join(BASE, 'bcl_indices_panel.csv')
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found. Run build_bcl_indices.py first.")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"  BCL panel: {len(df)} obs, {df['country_code'].nunique()} countries, "
          f"waves {df['wave'].min()}-{df['wave'].max()}")

    # Add ISO3
    df['iso3'] = df['country_code'].map(ISO2_TO_ISO3)
    missing = df[df['iso3'].isna()]['country_code'].unique()
    if len(missing) > 0:
        print(f"    Warning: {len(missing)} ISO2 codes unmapped: {list(missing)[:10]}...")
    df = df.dropna(subset=['iso3'])

    return df


# ============================================================================
# SECTION 2: Download WDI Variables
# ============================================================================

def download_wdi_panel():
    """Download financial development and GDP per capita."""
    cache = os.path.join(OUTPUT, 'akerlof_banking_wdi_v2.csv')
    if os.path.exists(cache):
        print(f"  Loading cached WDI data from {cache}")
        return pd.read_csv(cache)

    indicators = {
        'FD.AST.PRVT.GD.ZS':  'fin_dev',        # Domestic credit to private sector (% GDP)
        'NY.GDP.PCAP.PP.CD':  'gdp_pc_ppp',      # GDP per capita PPP
        'FB.AST.NPER.ZS':     'npl_ratio',        # Bank NPL ratio (secondary info proxy)
    }

    frames = {}
    for code, name in indicators.items():
        print(f"  Downloading {name} ({code})...")
        df = download_wb(code)
        if len(df) > 0:
            df = df.rename(columns={'value': name})
            frames[name] = df[['iso3', 'year', name]]
            print(f"    {len(df):,} observations")
        else:
            print(f"    WARNING: no data")

    panel = None
    for name, df in frames.items():
        panel = df if panel is None else panel.merge(df, on=['iso3', 'year'], how='outer')

    if panel is not None:
        panel.to_csv(cache, index=False)
    return panel


# ============================================================================
# SECTION 3: Merge and Build Analysis Dataset
# ============================================================================

def build_analysis_panel(bcl, wdi):
    """Merge BCL wave data with WDI indicators at closest year."""
    wave_years = {1: 2001, 2: 2003, 3: 2007, 4: 2012, 5: 2019}

    records = []
    for _, row in bcl.iterrows():
        iso3 = row['iso3']
        wave = row['wave']
        bcl_year = wave_years.get(wave, row['year'])

        # Match WDI data: try exact year, then ±1, ±2
        wdi_sub = wdi[wdi['iso3'] == iso3]
        best_match = None
        for offset in [0, -1, 1, -2, 2]:
            y = bcl_year + offset
            match = wdi_sub[wdi_sub['year'] == y]
            if len(match) > 0:
                best_match = match.iloc[0]
                break

        if best_match is None:
            continue

        rec = {
            'iso3': iso3,
            'iso2': row['country_code'],
            'country': row.get('country_name', ''),
            'wave': wave,
            'year': bcl_year,
            # BCL indices (both σ proxy and 1/T proxy come from same dataset)
            'ari': row.get('activity_restrictions_idx', np.nan),
            'pmi': row.get('private_monitoring_idx', np.nan),
            'csi': row.get('capital_stringency_idx', np.nan),
            'spi': row.get('supervisory_power_idx', np.nan),
            'ebi': row.get('entry_barriers_idx', np.nan),
            # WDI variables
            'fin_dev': best_match.get('fin_dev', np.nan),
            'gdp_pc_ppp': best_match.get('gdp_pc_ppp', np.nan),
            'npl_ratio': best_match.get('npl_ratio', np.nan),
        }
        records.append(rec)

    df = pd.DataFrame(records)

    # Derived variables
    df['log_gdp_pc'] = np.log(df['gdp_pc_ppp'].clip(lower=100))
    df['log_fin_dev'] = np.log(df['fin_dev'].clip(lower=0.1))

    # Interaction terms (raw)
    df['ari_x_pmi'] = df['ari'] * df['pmi']

    # Standardized versions for interpretation
    for col in ['ari', 'pmi', 'csi', 'spi', 'ebi']:
        s = df[col].dropna()
        if len(s) > 0 and s.std() > 0:
            df[f'{col}_z'] = (df[col] - s.mean()) / s.std()

    df['ari_z_x_pmi_z'] = df['ari_z'] * df['pmi_z']

    # Placebo interactions (other BCL indices × PMI)
    df['csi_x_pmi'] = df['csi'] * df['pmi']
    df['spi_x_pmi'] = df['spi'] * df['pmi']
    df['ebi_x_pmi'] = df['ebi'] * df['pmi']

    print(f"\n  Analysis panel: {len(df)} obs, {df['iso3'].nunique()} countries")
    return df


# ============================================================================
# SECTION 4: Regressions
# ============================================================================

def run_regression(df, model_label, depvar, depvar_label, indvars, indvar_labels,
                   fe_var=None, cluster_var=None):
    """Run OLS with optional FE and clustering."""
    cols = [depvar] + indvars + ([fe_var] if fe_var else [])
    sub = df.dropna(subset=cols).copy().reset_index(drop=True)
    if len(sub) < 30:
        print(f"\n  {model_label}: Too few observations ({len(sub)})")
        return None

    X = sub[indvars].copy()
    if fe_var:
        dummies = pd.get_dummies(sub[fe_var], prefix='fe', drop_first=True, dtype=float)
        X = pd.concat([X, dummies], axis=1)
    X = sm.add_constant(X)
    y = sub[depvar]

    model = sm.OLS(y.astype(float), X.astype(float))
    if cluster_var and cluster_var in sub.columns:
        try:
            res = model.fit(cov_type='cluster',
                            cov_kwds={'groups': sub[cluster_var]})
        except Exception:
            res = model.fit(cov_type='HC1')
    else:
        res = model.fit(cov_type='HC1')

    # Print
    print(f"\n  {'─'*68}")
    print(f"  {model_label}")
    print(f"  {'─'*68}")
    print(f"  Dep. var: {depvar_label}")
    print(f"  N = {int(res.nobs)},  countries = {sub['iso3'].nunique()},  "
          f"R² = {res.rsquared:.4f},  Adj R² = {res.rsquared_adj:.4f}")
    if fe_var:
        print(f"  Fixed effects: {fe_var}")
    print(f"\n  {'Variable':<30} {'β':>9} {'SE':>9} {'t':>7} {'p':>8}")
    print(f"  {'-'*64}")

    result_info = {'label': model_label, 'n': int(res.nobs),
                   'r2': res.rsquared, 'r2_adj': res.rsquared_adj,
                   'countries': sub['iso3'].nunique()}

    for var, vlabel in zip(indvars, indvar_labels):
        if var in res.params:
            b = res.params[var]
            s = res.bse[var]
            t = res.tvalues[var]
            p = res.pvalues[var]
            star = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else '†' if p<.1 else ''
            print(f"  {vlabel:<30} {b:>9.4f} {s:>9.4f} {t:>7.2f} {p:>8.4f} {star}")
            result_info[var] = {'beta': b, 'se': s, 'p': p}

    return result_info


# ============================================================================
# SECTION 5: Figures
# ============================================================================

def make_figures(df):
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel A: ARI vs Financial Development, colored by PMI ---
    ax = axes[0]
    sub = df.dropna(subset=['ari', 'fin_dev', 'pmi'])
    sc = ax.scatter(sub['ari'], sub['fin_dev'], s=25, alpha=0.5,
                    c=sub['pmi'], cmap='RdYlGn', edgecolors='k', lw=0.3)
    z = np.polyfit(sub['ari'], sub['fin_dev'], 1)
    xs = np.linspace(sub['ari'].min(), sub['ari'].max(), 100)
    ax.plot(xs, np.polyval(z, xs), 'k--', lw=1.5)
    ax.set_xlabel('Activity Restrictions Index (higher = more restricted = higher σ)', fontsize=9)
    ax.set_ylabel('Financial Development (domestic credit / GDP %)', fontsize=9)
    ax.set_title('A.  ARI vs. Financial Development', fontsize=11)
    ax.grid(True, alpha=0.25)
    plt.colorbar(sc, ax=ax, label='Private Monitoring Index (1/T proxy)', shrink=0.8)

    # --- Panel B: Interaction effect — split by PMI median ---
    ax = axes[1]
    sub = df.dropna(subset=['ari', 'fin_dev', 'pmi'])
    med_pmi = sub['pmi'].median()
    high_pmi = sub[sub['pmi'] >= med_pmi]
    low_pmi = sub[sub['pmi'] < med_pmi]

    for data, label, color in [(high_pmi, f'Good info (PMI ≥ {med_pmi:.0f})', '#2166ac'),
                                (low_pmi, f'Poor info (PMI < {med_pmi:.0f})', '#b2182b')]:
        ax.scatter(data['ari'], data['fin_dev'], s=20, alpha=0.4, color=color, label=label)
        if len(data) > 5:
            z2 = np.polyfit(data['ari'], data['fin_dev'], 1)
            ax.plot(xs, np.polyval(z2, xs), color=color, lw=2, ls='--',
                    label=f'  slope = {z2[0]:.1f}')

    ax.set_xlabel('Activity Restrictions Index', fontsize=9)
    ax.set_ylabel('Financial Development (%GDP)', fontsize=9)
    ax.set_title('B.  Interaction: ARI Effect × Private Monitoring', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # --- Panel C: ARI × PMI space colored by FinDev ---
    ax = axes[2]
    sub = df.dropna(subset=['ari', 'pmi', 'fin_dev'])
    sc = ax.scatter(sub['ari'], sub['pmi'], c=sub['fin_dev'],
                    s=40, cmap='viridis', alpha=0.7, edgecolors='k', lw=0.3,
                    vmin=0, vmax=150)
    ax.set_xlabel('Activity Restrictions Index (σ proxy)', fontsize=9)
    ax.set_ylabel('Private Monitoring Index (1/T proxy)', fontsize=9)
    ax.set_title('C.  Financial Development in (σ, T) Space', fontsize=11)
    plt.colorbar(sc, ax=ax, label='Domestic Credit / GDP (%)', shrink=0.8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(FIGDIR, 'akerlof_banking_test.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved to {path}")


# ============================================================================
# SECTION 6: LaTeX Table
# ============================================================================

def save_latex(results):
    """Write LaTeX regression table."""
    n_cols = len(results)
    col_spec = 'l' + 'c' * n_cols
    headers = ' & '.join([f'({i+1})' for i in range(n_cols)])
    model_names = ' & '.join([r.get('short_label', '') for r in results if r])

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Akerlof Prediction in Banking: Activity Restrictions $\times$ Private Monitoring}",
        r"\label{tab:akerlof_banking}",
        r"\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f" & {headers} \\\\",
        f" & {model_names} \\\\",
        r"\midrule",
    ]

    vars_to_show = [
        ('ari', 'ARI (Activity Restrictions)'),
        ('pmi', 'PMI (Private Monitoring)'),
        ('ari_x_pmi', r'ARI $\times$ PMI'),
        ('log_gdp_pc', r'$\log$ GDP per capita'),
        ('csi_x_pmi', r'CSI $\times$ PMI (placebo)'),
        ('spi_x_pmi', r'SPI $\times$ PMI (placebo)'),
        ('ebi_x_pmi', r'EBI $\times$ PMI (placebo)'),
    ]

    for var, label in vars_to_show:
        cells = [f"  {label}"]
        any_present = False
        for r in results:
            if r and var in r:
                any_present = True
                b = r[var]['beta']
                s = r[var]['se']
                p = r[var]['p']
                star = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else ''
                cells.append(f"\\makecell{{{b:.3f}{star} \\\\ ({s:.3f})}}")
            else:
                cells.append("")
        if any_present:
            lines.append(" & ".join(cells) + r" \\[4pt]")

    # N and R2
    lines.append(r"\midrule")
    cells_n = ["  $N$"]
    cells_r2 = [r"  $R^2$"]
    cells_c = ["  Countries"]
    cells_fe = ["  Wave FE"]
    cells_cl = ["  Clustering"]
    for r in results:
        if r:
            cells_n.append(str(r['n']))
            cells_r2.append(f"{r['r2']:.3f}")
            cells_c.append(str(r['countries']))
            cells_fe.append('Yes' if r.get('has_fe') else 'No')
            cells_cl.append('Country' if r.get('has_cluster') else 'HC1')
        else:
            for c in [cells_n, cells_r2, cells_c, cells_fe, cells_cl]:
                c.append("")
    lines.append(" & ".join(cells_n) + r" \\")
    lines.append(" & ".join(cells_c) + r" \\")
    lines.append(" & ".join(cells_r2) + r" \\")
    lines.append(" & ".join(cells_fe) + r" \\")
    lines.append(" & ".join(cells_cl) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{4pt}",
        r"\parbox{0.92\textwidth}{\footnotesize",
        r"  Dependent variable: domestic credit to private sector (\% of GDP).",
        r"  ARI = Barth-Caprio-Levine Activity Restrictions Index (higher = banks",
        r"  restricted to commodity lending = higher $\sigma$).",
        r"  PMI = Barth-Caprio-Levine Private Monitoring Index (higher = stronger",
        r"  information disclosure and market discipline = lower $T$).",
        r"  CSI = Capital Stringency, SPI = Supervisory Power, EBI = Entry Barriers.",
        r"  Robust (HC1) standard errors in columns (1)--(3); clustered by country in (4)--(5).",
        r"  Theory predicts only the ARI $\times$ PMI interaction is positive:",
        r"  activity restrictions (high $\sigma$, low $T^*$) hurt financial development",
        r"  less when private monitoring (low $T$) is strong.",
        r"  $^{***}p<0.001$; $^{**}p<0.01$; $^{*}p<0.05$.",
        r"}",
        r"\end{table}",
    ]
    path = os.path.join(OUTPUT, 'akerlof_banking_table.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  LaTeX table saved to {path}")


# ============================================================================
# SECTION 7: Main
# ============================================================================

def main():
    print("=" * 72)
    print("  FREE ENERGY FRAMEWORK — EMPIRICAL TEST")
    print("  Akerlof Prediction in Banking Regulation")
    print("=" * 72)
    print()
    print("  HYPOTHESIS")
    print("  Activity restrictions compress banking toward commodity lending")
    print("  (high σ), lowering the critical temperature T*.  This makes")
    print("  financial development more sensitive to information quality.")
    print()
    print("  TEST")
    print("  FinDev = α + β₁·ARI + β₂·PMI + β₃·(ARI × PMI) + γ·X + ε")
    print("  Prediction: β₃ > 0")
    print()
    print("  MAPPING")
    print("  ARI → σ  (activity restrictions → commodity lending → substitutability)")
    print("  PMI → 1/T  (private monitoring → disclosure → information quality)")
    print()

    # Build ISO mapping
    build_iso_map()

    # Load BCL
    print("─" * 72)
    print("  LOADING DATA")
    print("─" * 72)
    bcl = load_bcl()

    # Download WDI (only fin_dev and GDP — credit info now from BCL PMI)
    wdi = download_wdi_panel()

    # Merge
    print("\n  Merging BCL and WDI panels...")
    df = build_analysis_panel(bcl, wdi)

    # Summary stats
    vars_of_interest = ['ari', 'pmi', 'fin_dev', 'gdp_pc_ppp', 'csi', 'spi', 'ebi']
    labels = ['Activity Restrictions (σ proxy)', 'Private Monitoring (1/T proxy)',
              'Financial Development (%GDP)', 'GDP per capita PPP ($)',
              'Capital Stringency', 'Supervisory Power', 'Entry Barriers']
    print(f"\n  {'Variable':<35} {'N':>5} {'Mean':>8} {'SD':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*73}")
    for var, lab in zip(vars_of_interest, labels):
        if var in df.columns:
            s = df[var].dropna()
            if len(s) > 0:
                print(f"  {lab:<35} {len(s):>5} {s.mean():>8.1f} {s.std():>8.1f} "
                      f"{s.min():>8.1f} {s.max():>8.1f}")

    # Correlation matrix of BCL indices
    print(f"\n  BCL Index Correlations:")
    bcl_cols = ['ari', 'pmi', 'csi', 'spi', 'ebi']
    corr = df[bcl_cols].corr()
    print(f"  {'':>5}", end='')
    for c in bcl_cols:
        print(f"  {c:>5}", end='')
    print()
    for r in bcl_cols:
        print(f"  {r:>5}", end='')
        for c in bcl_cols:
            print(f"  {corr.loc[r,c]:>5.2f}", end='')
        print()

    # Save
    df.to_csv(os.path.join(OUTPUT, 'akerlof_banking_panel.csv'), index=False)

    # ── Regressions ──
    print(f"\n{'─'*72}")
    print("  REGRESSIONS")
    print(f"{'─'*72}")

    results = []

    # Model 1: Baseline (no interaction)
    r1 = run_regression(
        df, "Model 1: Baseline (no interaction)",
        'fin_dev', 'Financial Development (% GDP)',
        ['ari', 'pmi', 'log_gdp_pc'],
        ['ARI', 'PMI', 'log GDP per capita'],
    )
    if r1: r1['short_label'] = 'Baseline'; r1['has_fe'] = False; r1['has_cluster'] = False
    results.append(r1)

    # Model 2: With ARI × PMI interaction
    r2 = run_regression(
        df, "Model 2: ARI × PMI interaction",
        'fin_dev', 'Financial Development (% GDP)',
        ['ari', 'pmi', 'ari_x_pmi', 'log_gdp_pc'],
        ['ARI', 'PMI', 'ARI × PMI', 'log GDP per capita'],
    )
    if r2: r2['short_label'] = 'Interaction'; r2['has_fe'] = False; r2['has_cluster'] = False
    results.append(r2)

    # Model 3: With wave FE
    r3 = run_regression(
        df, "Model 3: Interaction + wave FE",
        'fin_dev', 'Financial Development (% GDP)',
        ['ari', 'pmi', 'ari_x_pmi', 'log_gdp_pc'],
        ['ARI', 'PMI', 'ARI × PMI', 'log GDP per capita'],
        fe_var='wave',
    )
    if r3: r3['short_label'] = 'Wave FE'; r3['has_fe'] = True; r3['has_cluster'] = False
    results.append(r3)

    # Model 4: Clustered SE by country
    r4 = run_regression(
        df, "Model 4: Clustered by country",
        'fin_dev', 'Financial Development (% GDP)',
        ['ari', 'pmi', 'ari_x_pmi', 'log_gdp_pc'],
        ['ARI', 'PMI', 'ARI × PMI', 'log GDP per capita'],
        fe_var='wave', cluster_var='iso3',
    )
    if r4: r4['short_label'] = 'Clustered'; r4['has_fe'] = True; r4['has_cluster'] = True
    results.append(r4)

    # Model 5: Placebo — other BCL indices × PMI
    r5 = run_regression(
        df, "Model 5: Placebo (all BCL × PMI)",
        'fin_dev', 'Financial Development (% GDP)',
        ['ari', 'pmi', 'ari_x_pmi', 'csi_x_pmi', 'spi_x_pmi', 'ebi_x_pmi', 'log_gdp_pc'],
        ['ARI', 'PMI', 'ARI × PMI', 'CSI × PMI (placebo)',
         'SPI × PMI (placebo)', 'EBI × PMI (placebo)', 'log GDP per capita'],
        fe_var='wave', cluster_var='iso3',
    )
    if r5: r5['short_label'] = 'Placebo'; r5['has_fe'] = True; r5['has_cluster'] = True
    results.append(r5)

    # ── Summary ──
    print(f"\n\n{'='*72}")
    print("  SUMMARY: Interaction Term β₃(ARI × PMI)")
    print(f"{'='*72}")
    for r in results:
        if r is None:
            continue
        for key in ['ari_x_pmi']:
            if key in r:
                b = r[key]['beta']
                p = r[key]['p']
                star = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else '†' if p<.1 else ''
                verdict = ('CONFIRMED' if b > 0 and p < .05 else
                          'WEAK' if b > 0 and p < .10 else
                          'right sign' if b > 0 else 'wrong sign')
                print(f"  {r['label']:<50}")
                print(f"    β₃ = {b:+.4f}  (p = {p:.4f})  {verdict} {star}")

    # Placebo summary
    if r5:
        print(f"\n  Placebo check (Model 5):")
        for key, label in [('ari_x_pmi', 'ARI × PMI'),
                           ('csi_x_pmi', 'CSI × PMI'),
                           ('spi_x_pmi', 'SPI × PMI'),
                           ('ebi_x_pmi', 'EBI × PMI')]:
            if key in r5:
                b = r5[key]['beta']
                p = r5[key]['p']
                star = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else '†' if p<.1 else ''
                print(f"    {label:<20} β = {b:+.4f}  (p = {p:.4f})  {star}")

    # LaTeX
    save_latex(results)

    # Figures
    if HAS_MPL:
        print(f"\n{'─'*72}")
        print("  FIGURES")
        print(f"{'─'*72}")
        make_figures(df)

    print(f"\n\n{'='*72}")
    print("  INTERPRETATION")
    print(f"{'='*72}")
    print("""
  The Free Energy Framework (Proposition 8) maps to banking as follows:

    Activity Restrictions Index → σ (substitutability of banking products)
        Restricted banks offer homogeneous commodity loans (high σ).
        Unrestricted banks diversify into securities, insurance, real
        estate — differentiated, complementary services (low σ).

    Private Monitoring Index → 1/T (inverse information temperature)
        Strong private monitoring → mandated disclosure, auditing,
        market discipline → low T → agents can distinguish quality.
        Weak private monitoring → opaque lending → high T.

    T*(σ): critical temperature above which market collapses
        From Proposition 8: T* ∝ K = (1-ρ)(J-1)/J, decreasing in σ.
        High-σ (restricted) banking has LOW T* → collapses at lower
        information noise → more sensitive to monitoring quality.

  The interaction β₃ > 0 means: better private monitoring MITIGATES
  the harm from activity restrictions.  This is exactly what the
  framework predicts — high-σ markets (restricted banking) need better
  information (lower T < T*) to function.

  PLACEBO: the interaction should be specific to ARI (the σ proxy).
  Other BCL indices (capital stringency, supervisory power, entry
  barriers) do not map to σ, so their PMI interactions should be
  weaker or zero.  The placebo test verifies this specificity.

  This test is novel: Barth-Caprio-Levine (2004, 2006, 2008) test ARI
  and PMI as separate, additive regressors.  This specific interaction
  has not been published.
""")


if __name__ == '__main__':
    main()
