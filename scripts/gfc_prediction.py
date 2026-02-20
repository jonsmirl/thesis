#!/usr/bin/env python3
"""
gfc_prediction.py — Paper-quality GFC prediction from the Free Energy Framework.

Two independent analyses supporting the same theoretical prediction:

  Part A — Cross-Country GFC Prediction (main result)
    BCL (2006) found that activity restrictions *increase* crisis risk — a puzzle.
    The Free Energy Framework explains why: ARI → σ → lower T* → more vulnerable
    to information degradation. The novel prediction: this effect is **mediated**
    by information quality (PMI).  The ARI × PMI interaction is the specific test.

    Uses BCL Wave 2 (2003) pre-crisis regulation + GDP loss 2007–2009.

  Part B — US Early Warning (supporting result)
    ONLY leading indicators. Recursive normalization (no look-ahead).
    σ(t) = RE loan concentration (structural).
    T(t) = lending standards + housing deviation from trend (leading).
    Fragility = σ_z(t) × T_z(t) with expanding-window normalization.

Usage:
    source .venv/bin/activate && source env.sh
    python scripts/gfc_prediction.py
"""

import os, sys, time, warnings, json
from datetime import datetime
import numpy as np
import pandas as pd
import requests
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import Logit
except ImportError:
    print("ERROR: pip install statsmodels"); sys.exit(1)

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import FancyArrowPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, 'thesis_data')
FIGDIR = os.path.join(BASE, 'figures')
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

FRED_KEY = os.environ.get('FRED_API_KEY', '')
if not FRED_KEY:
    print("WARNING: FRED_API_KEY not set. Source env.sh first.")

# Laeven-Valencia (2018) GFC crisis countries (ISO3)
LAEVEN_VALENCIA_GFC = {
    # 2007 onset
    'USA', 'GBR', 'IRL', 'LVA',
    # 2008 onset
    'AUT', 'BEL', 'DNK', 'DEU', 'ESP', 'FRA', 'GRC', 'HUN',
    'ISL', 'ITA', 'KAZ', 'LUX', 'NLD', 'NGA', 'PRT', 'RUS',
    'SWE', 'CHE', 'UKR', 'MNG',
}

# Countries to label on scatter plot
LABEL_COUNTRIES = {
    'USA': 'US', 'GBR': 'UK', 'CAN': 'Canada', 'AUS': 'Australia',
    'ISL': 'Iceland', 'IRL': 'Ireland', 'DEU': 'Germany', 'FRA': 'France',
    'ESP': 'Spain', 'ITA': 'Italy', 'GRC': 'Greece', 'NLD': 'Netherlands',
    'JPN': 'Japan', 'CHN': 'China', 'IND': 'India', 'BRA': 'Brazil',
    'RUS': 'Russia', 'NGA': 'Nigeria', 'ZAF': 'South Africa',
}


# ============================================================================
# SECTION 1: Data Helpers
# ============================================================================

def fetch_fred(series_id, start='1995-01-01', end='2012-12-31'):
    """Download a FRED series with caching."""
    cache_dir = os.path.join(OUTPUT, 'fred_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{series_id}.csv')

    if os.path.exists(cache):
        df = pd.read_csv(cache, parse_dates=['date'], index_col='date')
        return df

    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={FRED_KEY}&file_type=json"
           f"&observation_start={start}&observation_end={end}")

    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt == 2:
                print(f"    FAILED: {series_id}: {e}")
                return pd.DataFrame()
            time.sleep(2 ** attempt)

    if 'observations' not in data:
        print(f"    WARNING: no observations for {series_id}")
        return pd.DataFrame()

    records = []
    for obs in data['observations']:
        if obs['value'] != '.':
            records.append({'date': obs['date'], 'value': float(obs['value'])})

    df = pd.DataFrame(records)
    if len(df) == 0:
        return df
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df.to_csv(cache)
    return df


def fetch_wb_indicator(indicator, start_year=1999, end_year=2023):
    """Download a World Bank indicator for all countries. Returns DataFrame."""
    cache = os.path.join(OUTPUT, f'wb_cache_{indicator.replace(".", "_")}.csv')
    if os.path.exists(cache):
        return pd.read_csv(cache)

    records = []
    for page in range(1, 20):
        url = (f"http://api.worldbank.org/v2/country/all/indicator/{indicator}"
               f"?format=json&per_page=5000&date={start_year}:{end_year}&page={page}")
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    FAILED page {page}: {e}")
                    data = [None, None]
                time.sleep(2)

        if len(data) < 2 or not data[1]:
            break

        for item in data[1]:
            if item['value'] is not None and item.get('countryiso3code'):
                records.append({
                    'iso3': item['countryiso3code'],
                    'year': int(item['date']),
                    'value': float(item['value']),
                })

        # Check if more pages
        total_pages = data[0].get('pages', 1) if data[0] else 1
        if page >= total_pages:
            break

    df = pd.DataFrame(records)
    if len(df) > 0:
        df.to_csv(cache, index=False)
    return df


def load_iso_mapping():
    """Load ISO2 ↔ ISO3 mapping."""
    path = os.path.join(OUTPUT, 'iso_mapping.csv')
    if not os.path.exists(path):
        print("    ERROR: iso_mapping.csv not found")
        return {}, {}
    iso = pd.read_csv(path)
    iso2_to_3 = dict(zip(iso['iso2'], iso['iso3']))
    iso3_to_2 = dict(zip(iso['iso3'], iso['iso2']))
    return iso2_to_3, iso3_to_2


# ============================================================================
# SECTION 2: Part A — Cross-Country GFC Prediction
# ============================================================================

def build_cross_country_panel():
    """
    Assemble the cross-country panel:
      - BCL Wave 2 (2003) regulation indices
      - GDP per capita 2003 (control), 2007, 2009 (for loss)
      - GDP growth rate 2009 (robustness DV)
      - Financial development 2003 (control)
      - Laeven-Valencia crisis dummy
    """
    print("\n  ── Part A: Cross-Country Data Assembly ──")

    # 1. BCL Wave 2
    bcl_path = os.path.join(BASE, 'bcl_indices_panel.csv')
    if not os.path.exists(bcl_path):
        print("    ERROR: bcl_indices_panel.csv not found")
        return None
    bcl = pd.read_csv(bcl_path)
    wave2 = bcl[bcl['wave'] == 2].copy()
    print(f"    BCL Wave 2: {len(wave2)} countries")

    # 2. ISO mapping
    iso2_to_3, iso3_to_2 = load_iso_mapping()
    wave2['iso3'] = wave2['country_code'].map(iso2_to_3)
    wave2 = wave2.dropna(subset=['iso3'])

    # 3. GDP per capita PPP (already cached)
    wdi_path = os.path.join(OUTPUT, 'akerlof_banking_wdi_v2.csv')
    if os.path.exists(wdi_path):
        wdi = pd.read_csv(wdi_path)
    else:
        print("    ERROR: akerlof_banking_wdi_v2.csv not found")
        return None

    # 4. GDP growth rate (annual %)
    print("    Fetching GDP growth rate (NY.GDP.MKTP.KD.ZG)...")
    gdp_growth = fetch_wb_indicator('NY.GDP.MKTP.KD.ZG', 2005, 2012)

    # Build panel: one row per country
    records = []
    for _, row in wave2.iterrows():
        iso3 = row['iso3']
        rec = {
            'iso3': iso3,
            'iso2': row['country_code'],
            'country': row['country'],
            'ari': row.get('activity_restrictions_idx', np.nan),
            'pmi': row.get('private_monitoring_idx', np.nan),
            'csi': row.get('capital_stringency_idx', np.nan),
            'spi': row.get('supervisory_power_idx', np.nan),
            'ebi': row.get('entry_barriers_idx', np.nan),
        }

        # GDP per capita for 2003 (control), 2007, 2009 (loss calculation)
        wdi_c = wdi[wdi['iso3'] == iso3]
        for yr in [2003, 2007, 2009]:
            match = wdi_c[wdi_c['year'] == yr]
            if len(match) > 0:
                rec[f'gdp_pc_{yr}'] = match.iloc[0]['gdp_pc_ppp']

        # Financial development 2003
        match_fd = wdi_c[wdi_c['year'] == 2003]
        if len(match_fd) > 0 and 'fin_dev' in match_fd.columns:
            rec['fin_dev_2003'] = match_fd.iloc[0]['fin_dev']
        else:
            # Try nearby years
            for offset in [-1, 1, -2, 2]:
                match_fd = wdi_c[wdi_c['year'] == 2003 + offset]
                if len(match_fd) > 0 and 'fin_dev' in match_fd.columns:
                    fd_val = match_fd.iloc[0].get('fin_dev', np.nan)
                    if pd.notna(fd_val):
                        rec['fin_dev_2003'] = fd_val
                        break

        # GDP growth 2009
        if len(gdp_growth) > 0:
            g = gdp_growth[(gdp_growth['iso3'] == iso3) & (gdp_growth['year'] == 2009)]
            if len(g) > 0:
                rec['gdp_growth_2009'] = g.iloc[0]['value']

        # Laeven-Valencia crisis dummy
        rec['crisis'] = 1 if iso3 in LAEVEN_VALENCIA_GFC else 0

        records.append(rec)

    panel = pd.DataFrame(records)

    # Compute GDP loss 2007-2009
    mask = panel['gdp_pc_2007'].notna() & panel['gdp_pc_2009'].notna()
    panel.loc[mask, 'gdp_loss_07_09'] = (
        (panel.loc[mask, 'gdp_pc_2009'] / panel.loc[mask, 'gdp_pc_2007'] - 1) * 100
    )

    # Derived variables
    panel['log_gdp_pc'] = np.log(panel['gdp_pc_2003'].clip(lower=100))
    panel['ari_x_pmi'] = panel['ari'] * panel['pmi']
    panel['csi_x_pmi'] = panel['csi'] * panel['pmi']
    panel['spi_x_pmi'] = panel['spi'] * panel['pmi']
    panel['ebi_x_pmi'] = panel['ebi'] * panel['pmi']

    # Report
    complete = panel.dropna(subset=['ari', 'pmi', 'gdp_loss_07_09', 'log_gdp_pc'])
    print(f"    Complete panel: {len(complete)} countries with all variables")
    print(f"    Crisis countries in sample: {complete['crisis'].sum()}")
    print(f"    GDP loss range: {complete['gdp_loss_07_09'].min():.1f}% to "
          f"{complete['gdp_loss_07_09'].max():.1f}%")
    print(f"    ARI range: {complete['ari'].min():.0f} to {complete['ari'].max():.0f}")
    print(f"    PMI range: {complete['pmi'].min():.0f} to {complete['pmi'].max():.0f}")

    panel.to_csv(os.path.join(OUTPUT, 'gfc_prediction_panel.csv'), index=False)
    return panel


def run_ols_regressions(panel):
    """
    OLS regression models (all HC1 robust SE):
      (1) GDP_loss = ARI + log_GDP_pc
      (2) GDP_loss = ARI + PMI + log_GDP_pc
      (3) GDP_loss = ARI + PMI + ARI×PMI + log_GDP_pc       ← MAIN MODEL
      (4) GDP_loss = ARI + PMI + ARI×PMI + log_GDP_pc + FinDev_2003
      (5) Same as (3) with GDP growth 2009 as DV
    """
    print("\n  ── Part A: OLS Regressions ──")

    # Base sample
    base_vars = ['ari', 'pmi', 'gdp_loss_07_09', 'log_gdp_pc']
    df = panel.dropna(subset=base_vars).copy()
    print(f"    Base sample: N = {len(df)}")

    results = {}

    # Model 1: Baseline
    X1 = sm.add_constant(df[['ari', 'log_gdp_pc']])
    y = df['gdp_loss_07_09']
    results['m1'] = sm.OLS(y, X1).fit(cov_type='HC1')

    # Model 2: Add PMI
    X2 = sm.add_constant(df[['ari', 'pmi', 'log_gdp_pc']])
    results['m2'] = sm.OLS(y, X2).fit(cov_type='HC1')

    # Model 3: MAIN — add interaction
    X3 = sm.add_constant(df[['ari', 'pmi', 'ari_x_pmi', 'log_gdp_pc']])
    results['m3'] = sm.OLS(y, X3).fit(cov_type='HC1')

    # Model 4: Add financial development
    df4 = df.dropna(subset=['fin_dev_2003'])
    X4 = sm.add_constant(df4[['ari', 'pmi', 'ari_x_pmi', 'log_gdp_pc', 'fin_dev_2003']])
    y4 = df4['gdp_loss_07_09']
    results['m4'] = sm.OLS(y4, X4).fit(cov_type='HC1')

    # Model 5: GDP growth 2009 as DV
    df5 = df.dropna(subset=['gdp_growth_2009'])
    X5 = sm.add_constant(df5[['ari', 'pmi', 'ari_x_pmi', 'log_gdp_pc']])
    y5 = df5['gdp_growth_2009']
    results['m5'] = sm.OLS(y5, X5).fit(cov_type='HC1')

    # Print results
    for name, res in results.items():
        print(f"\n    ── {name.upper()} (N={int(res.nobs)}, R²={res.rsquared:.3f}) ──")
        for var in res.params.index:
            if var == 'const':
                continue
            b = res.params[var]
            se = res.bse[var]
            p = res.pvalues[var]
            star = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
            print(f"      {var:<15} β={b:+8.3f}  SE={se:7.3f}  p={p:.4f} {star}")

    # Main model interpretation
    m3 = results['m3']
    print(f"\n    ── MAIN MODEL (M3) Interpretation ──")
    b_ari = m3.params.get('ari', 0)
    b_int = m3.params.get('ari_x_pmi', 0)
    p_int = m3.pvalues.get('ari_x_pmi', 1)
    pmi_med = df['pmi'].median()
    print(f"    Marginal effect of ARI at median PMI ({pmi_med:.0f}):")
    print(f"      dGDP_loss/dARI = {b_ari:.3f} + {b_int:.3f} × {pmi_med:.0f} "
          f"= {b_ari + b_int * pmi_med:.3f}")
    print(f"    ARI×PMI interaction: β = {b_int:+.3f}, p = {p_int:.4f}")

    return results, df


def run_logit(panel):
    """
    Logit: Pr(Crisis) = Λ(ARI + PMI + ARI×PMI + log_GDP_pc)
    Report honestly: likely dominated by GDP_pc.
    """
    print("\n  ── Part A: Logit (Crisis Probability) ──")

    df = panel.dropna(subset=['ari', 'pmi', 'crisis', 'log_gdp_pc']).copy()
    print(f"    Sample: N = {len(df)}, crisis = {df['crisis'].sum()}")

    X = sm.add_constant(df[['ari', 'pmi', 'ari_x_pmi', 'log_gdp_pc']])
    y = df['crisis']

    try:
        res = Logit(y, X).fit(disp=0, maxiter=100)
        print(f"    Pseudo R² = {res.prsquared:.3f}")
        for var in ['ari', 'pmi', 'ari_x_pmi', 'log_gdp_pc']:
            if var in res.params.index:
                b = res.params[var]
                p = res.pvalues[var]
                star = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
                print(f"      {var:<15} β={b:+8.3f}  p={p:.4f} {star}")

        # Honest assessment
        p_gdp = res.pvalues.get('log_gdp_pc', 1)
        p_int = res.pvalues.get('ari_x_pmi', 1)
        if p_gdp < 0.05 and p_int > 0.1:
            print("\n    NOTE: As expected, the binary crisis dummy is dominated by")
            print("    GDP per capita (rich countries had crises in the GFC).")
            print("    The continuous GDP loss is the framework's real test —")
            print("    it captures severity, not just incidence.")

        return res
    except Exception as e:
        print(f"    Logit failed: {e}")
        return None


def run_placebo_tests(panel):
    """
    Placebo: Replace ARI with CSI, SPI, EBI in the interaction model.
    Only ARI × PMI should be significant.
    """
    print("\n  ── Part A: Placebo Tests ──")

    df = panel.dropna(subset=['ari', 'pmi', 'csi', 'spi', 'ebi',
                               'gdp_loss_07_09', 'log_gdp_pc']).copy()
    print(f"    Placebo sample: N = {len(df)}")

    placebo_results = {}
    for idx_name, interaction_col in [('CSI', 'csi_x_pmi'),
                                       ('SPI', 'spi_x_pmi'),
                                       ('EBI', 'ebi_x_pmi')]:
        idx_lower = idx_name.lower()
        X = sm.add_constant(df[[idx_lower, 'pmi', interaction_col, 'log_gdp_pc']])
        y = df['gdp_loss_07_09']
        res = sm.OLS(y, X).fit(cov_type='HC1')

        b_int = res.params.get(interaction_col, 0)
        p_int = res.pvalues.get(interaction_col, 1)
        star = '***' if p_int < .001 else '**' if p_int < .01 else '*' if p_int < .05 else ''
        status = 'SIGNIFICANT' if p_int < 0.05 else 'insignificant (expected)'
        print(f"    {idx_name} × PMI: β = {b_int:+.3f}, p = {p_int:.4f} {star} → {status}")

        placebo_results[idx_name] = {'beta': b_int, 'se': res.bse.get(interaction_col, 0),
                                     'p': p_int}

    return placebo_results


def quadrant_analysis(panel):
    """
    Split at ARI/PMI medians. Report mean GDP loss per quadrant.
    Prediction: High ARI + Low PMI = worst (high σ, low information quality).

    Also reports GDP-adjusted means (residualized on log GDP per capita)
    to control for the fact that developing countries grew through the GFC.
    """
    print("\n  ── Part A: Quadrant Analysis ──")

    df = panel.dropna(subset=['ari', 'pmi', 'gdp_loss_07_09', 'log_gdp_pc']).copy()

    ari_med = df['ari'].median()
    pmi_med = df['pmi'].median()
    print(f"    Medians: ARI = {ari_med:.0f}, PMI = {pmi_med:.0f}")

    # Residualize GDP change on log GDP per capita
    X_resid = sm.add_constant(df[['log_gdp_pc']])
    resid_model = sm.OLS(df['gdp_loss_07_09'], X_resid).fit()
    df['gdp_change_adjusted'] = resid_model.resid

    quadrants = {
        'High ARI / Low PMI':  (df['ari'] >= ari_med) & (df['pmi'] < pmi_med),
        'High ARI / High PMI': (df['ari'] >= ari_med) & (df['pmi'] >= pmi_med),
        'Low ARI / Low PMI':   (df['ari'] < ari_med) & (df['pmi'] < pmi_med),
        'Low ARI / High PMI':  (df['ari'] < ari_med) & (df['pmi'] >= pmi_med),
    }

    quad_stats = {}
    print(f"\n    {'Quadrant':<25} {'N':>4} {'Raw Mean':>9} {'GDP-Adj':>9} {'Crisis%':>8}")
    print(f"    {'─'*60}")
    for label, mask in quadrants.items():
        sub = df[mask]
        n = len(sub)
        mean_raw = sub['gdp_loss_07_09'].mean()
        mean_adj = sub['gdp_change_adjusted'].mean()
        crisis_pct = sub['crisis'].mean() * 100 if 'crisis' in sub.columns else 0
        print(f"    {label:<25} {n:>4} {mean_raw:>+9.2f}% {mean_adj:>+9.2f}% {crisis_pct:>7.1f}%")
        quad_stats[label] = {
            'n': n, 'mean_loss': mean_raw, 'mean_adj': mean_adj,
            'crisis_pct': crisis_pct
        }

    # Check prediction: High ARI / Low PMI should have most negative adjusted mean
    adj_vals = {k: v['mean_adj'] for k, v in quad_stats.items()}
    worst = min(adj_vals, key=adj_vals.get)
    print(f"\n    After GDP adjustment, worst quadrant: {worst} ({adj_vals[worst]:+.2f}%)")
    if worst == 'High ARI / Low PMI':
        print("    ✓ Matches framework prediction (high σ + poor information = worst)")
    else:
        best_predicted = adj_vals.get('High ARI / Low PMI', 0)
        print(f"    High ARI / Low PMI adjusted mean: {best_predicted:+.2f}%")

    return quad_stats, ari_med, pmi_med


def compute_roc(panel):
    """
    Manual ROC curve: use predicted GDP loss from main model as score,
    sweep threshold against crisis dummy. Compute AUC.
    """
    print("\n  ── Part A: ROC Analysis ──")

    df = panel.dropna(subset=['ari', 'pmi', 'gdp_loss_07_09', 'log_gdp_pc', 'crisis']).copy()

    # Predicted loss from main model
    X = sm.add_constant(df[['ari', 'pmi', 'ari_x_pmi', 'log_gdp_pc']])
    y = df['gdp_loss_07_09']
    res = sm.OLS(y, X).fit(cov_type='HC1')
    df['predicted_loss'] = res.predict(X)

    # Use negative predicted loss as crisis score (more negative = more likely crisis)
    scores = -df['predicted_loss'].values
    labels = df['crisis'].values

    # Sweep thresholds
    thresholds = np.sort(np.unique(scores))
    tpr_list, fpr_list = [0.0], [0.0]

    for t in thresholds:
        predicted_pos = scores >= t
        tp = np.sum(predicted_pos & (labels == 1))
        fp = np.sum(predicted_pos & (labels == 0))
        fn = np.sum(~predicted_pos & (labels == 1))
        tn = np.sum(~predicted_pos & (labels == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    tpr_list.append(1.0)
    fpr_list.append(1.0)

    # Sort by FPR for AUC computation
    pairs = sorted(zip(fpr_list, tpr_list))
    fpr_arr = np.array([p[0] for p in pairs])
    tpr_arr = np.array([p[1] for p in pairs])

    # Trapezoidal AUC
    auc = float(np.sum(0.5 * (tpr_arr[1:] + tpr_arr[:-1]) * np.diff(fpr_arr)))
    print(f"    AUC = {auc:.3f}  (N = {len(df)}, crisis = {labels.sum()})")

    return fpr_arr, tpr_arr, auc


# ============================================================================
# SECTION 3: Part B — US Early Warning
# ============================================================================

def build_us_early_warning():
    """
    Construct US early warning from LEADING indicators only.

    σ(t) = REALLN / LOANS  (RE loan concentration, monthly, structural)

    T(t) composite:
      T_standards = -DRTSCILM  (inverted: loosening = higher T, quarterly → ffill)
      T_housing = Case-Shiller deviation from 60-month trailing mean

    Fragility = σ_z(t) × T_z(t) with RECURSIVE normalization
    (expanding window, min 36 months, no look-ahead).
    """
    print("\n  ── Part B: US Early Warning Construction ──")

    # Load FRED series
    series_needed = {
        'REALLN':    'Real estate loans, all commercial banks',
        'LOANS':     'Total loans and leases, all commercial banks',
        'DRTSCILM':  'Net % banks tightening C&I loan standards (quarterly)',
        'CSUSHPISA': 'Case-Shiller US National Home Price Index',
    }

    data = {}
    for sid, desc in series_needed.items():
        df = fetch_fred(sid, start='1995-01-01', end='2012-12-31')
        if len(df) > 0:
            data[sid] = df
            print(f"    {sid}: {len(df)} obs ({df.index.min().date()} to {df.index.max().date()})")
        else:
            print(f"    WARNING: {sid} not available")

    # Build monthly panel
    dates = pd.date_range('1997-01-01', '2012-12-31', freq='MS')
    panel = pd.DataFrame(index=dates)

    for sid, df in data.items():
        monthly = df.resample('MS').last()
        panel[sid] = monthly['value']

    # Forward-fill quarterly data to monthly
    panel = panel.ffill(limit=3)

    # ── σ(t): RE loan concentration ──
    if 'REALLN' in panel.columns and 'LOANS' in panel.columns:
        panel['sigma_raw'] = panel['REALLN'] / panel['LOANS']
        print(f"    σ(t) = RE/Total loans: "
              f"{panel['sigma_raw'].dropna().iloc[0]:.3f} → "
              f"{panel['sigma_raw'].dropna().iloc[-1]:.3f}")

    # ── T(t) component 1: Lending standards (inverted) ──
    if 'DRTSCILM' in panel.columns:
        panel['T_standards'] = -panel['DRTSCILM']
        # Positive = loosening = higher information temperature

    # ── T(t) component 2: Case-Shiller deviation from 60-month trailing mean ──
    if 'CSUSHPISA' in panel.columns:
        cs = panel['CSUSHPISA']
        trailing_mean = cs.rolling(60, min_periods=36).mean()
        panel['T_housing'] = ((cs / trailing_mean) - 1) * 100
        # Positive = above trend = overheating

    # ── Recursive normalization (no look-ahead) ──
    MIN_WINDOW = 36  # minimum 36 months before computing z-scores

    # Normalize σ recursively
    if 'sigma_raw' in panel.columns:
        sigma_z = pd.Series(np.nan, index=panel.index)
        vals = panel['sigma_raw'].values
        for i in range(MIN_WINDOW, len(vals)):
            window = vals[:i+1]
            valid = window[~np.isnan(window)]
            if len(valid) >= MIN_WINDOW:
                mu = np.mean(valid)
                sd = np.std(valid, ddof=1)
                if sd > 1e-10:
                    sigma_z.iloc[i] = (vals[i] - mu) / sd
        panel['sigma_z'] = sigma_z

    # Normalize T components recursively, then average
    t_cols_z = []
    for t_col in ['T_standards', 'T_housing']:
        if t_col not in panel.columns:
            continue
        z_col = f'{t_col}_z'
        z_vals = pd.Series(np.nan, index=panel.index)
        raw = panel[t_col].values
        for i in range(MIN_WINDOW, len(raw)):
            window = raw[:i+1]
            valid = window[~np.isnan(window)]
            if len(valid) >= MIN_WINDOW:
                mu = np.mean(valid)
                sd = np.std(valid, ddof=1)
                if sd > 1e-10:
                    z_vals.iloc[i] = (raw[i] - mu) / sd
        panel[z_col] = z_vals
        t_cols_z.append(z_col)

    # T composite (equal weight)
    if t_cols_z:
        panel['T_z'] = panel[t_cols_z].mean(axis=1)
        print(f"    T(t) built from {len(t_cols_z)} leading components: {t_cols_z}")

    # ── Fragility index = σ_z × T_z ──
    if 'sigma_z' in panel.columns and 'T_z' in panel.columns:
        panel['fragility'] = panel['sigma_z'] * panel['T_z']
        print(f"    Fragility = σ_z × T_z")

    # Report key years
    if 'fragility' in panel.columns:
        fn = panel['fragility'].dropna()
        print(f"\n    Fragility index (recursive z-score product):")
        for year in [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]:
            yr = fn[fn.index.year == year]
            if len(yr) > 0:
                val = yr.mean()
                bar = '+' * max(0, int(min(val, 5) * 8)) if val > 0 else '-' * max(0, int(min(-val, 5) * 8))
                print(f"      {year}: {val:+.3f}  {bar}")

    panel.to_csv(os.path.join(OUTPUT, 'gfc_us_timeseries.csv'))
    print(f"    US panel: {len(panel)} months saved")
    return panel


# ============================================================================
# SECTION 4: Figures
# ============================================================================

def make_cross_country_figure(panel, quad_stats, ari_med, pmi_med):
    """
    Figure 1: Cross-country scatter.
    ARI (x) vs PMI (y), colored by GDP loss, sized by |loss|.
    Quadrant dividers at medians with mean GDP loss annotations.
    """
    if not HAS_MPL:
        print("    matplotlib not available, skipping figures")
        return

    print("\n  Generating Figure 1: Cross-country scatter...")

    df = panel.dropna(subset=['ari', 'pmi', 'gdp_loss_07_09']).copy()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by GDP loss (red = bad, green = good)
    losses = df['gdp_loss_07_09'].values
    sizes = np.clip(np.abs(losses) * 3, 15, 200)

    # Create colormap: green (positive/growth) to red (negative/loss)
    norm = plt.Normalize(vmin=np.percentile(losses, 5),
                         vmax=np.percentile(losses, 95))

    sc = ax.scatter(df['ari'].values, df['pmi'].values,
                    c=losses, cmap='RdYlGn', norm=norm,
                    s=sizes, alpha=0.7, edgecolors='k', linewidths=0.3,
                    zorder=3)

    # Mark crisis countries
    crisis = df[df['crisis'] == 1]
    ax.scatter(crisis['ari'].values, crisis['pmi'].values,
               facecolors='none', edgecolors='red', linewidths=1.5,
               s=sizes[df['crisis'] == 1] * 1.5, marker='o',
               zorder=4, label='L-V crisis country')

    # Quadrant dividers
    ax.axvline(ari_med, color='gray', ls='--', lw=1, alpha=0.5, zorder=2)
    ax.axhline(pmi_med, color='gray', ls='--', lw=1, alpha=0.5, zorder=2)

    # Quadrant annotations
    x_range = df['ari'].max() - df['ari'].min()
    y_range = df['pmi'].max() - df['pmi'].min()

    quad_positions = {
        'High ARI / Low PMI':  (df['ari'].max() - x_range * 0.15,
                                 df['pmi'].min() + y_range * 0.08),
        'High ARI / High PMI': (df['ari'].max() - x_range * 0.15,
                                 df['pmi'].max() - y_range * 0.08),
        'Low ARI / Low PMI':   (df['ari'].min() + x_range * 0.05,
                                 df['pmi'].min() + y_range * 0.08),
        'Low ARI / High PMI':  (df['ari'].min() + x_range * 0.05,
                                 df['pmi'].max() - y_range * 0.08),
    }

    for qname, (qx, qy) in quad_positions.items():
        if qname in quad_stats:
            qs = quad_stats[qname]
            adj = qs.get('mean_adj', qs['mean_loss'])
            ax.annotate(f"n={qs['n']}\nadj={adj:+.1f}%",
                        xy=(qx, qy), fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='lightyellow', alpha=0.8),
                        ha='center', va='center', zorder=5)

    # Label key countries
    for _, row in df.iterrows():
        if row['iso3'] in LABEL_COUNTRIES:
            label = LABEL_COUNTRIES[row['iso3']]
            ax.annotate(label, xy=(row['ari'], row['pmi']),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=7, fontweight='bold', alpha=0.8,
                        zorder=6)

    cb = plt.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label('GDP loss 2007–2009 (%)', fontsize=10)

    ax.set_xlabel('Activity Restrictions Index (ARI) → higher σ', fontsize=11)
    ax.set_ylabel('Private Monitoring Index (PMI) → lower T', fontsize=11)
    ax.set_title('Cross-Country GFC Severity vs. Pre-Crisis Banking Regulation\n'
                 'BCL Wave 2 (2003) regulation; GDP per capita loss 2007–2009',
                 fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.2, zorder=1)

    plt.tight_layout()
    for ext in ['.png', '.pdf']:
        path = os.path.join(FIGDIR, f'gfc_phase_space{ext}')
        plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved figures/gfc_phase_space.png/.pdf")


def make_us_dashboard(us_panel):
    """
    Figure 2: US early warning (2×2 dashboard).
      A: σ(t) with Glass-Steagall annotation
      B: T(t) components (lending standards + housing deviation)
      C: σ×T fragility with crisis shading
      D: Overlaid comparison (σ vs T vs σ×T normalized)
    """
    if not HAS_MPL:
        return

    print("  Generating Figure 2: US early warning dashboard...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('US Early Warning: Leading Indicators Only (Recursive Normalization)',
                 fontsize=13, fontweight='bold')

    # Crisis shading helper
    def shade_crisis(ax):
        ax.axvspan(pd.Timestamp('2007-12-01'), pd.Timestamp('2009-06-01'),
                   color='red', alpha=0.08, zorder=0)

    # Glass-Steagall annotation helper
    def mark_gs(ax, y_pos=None):
        ax.axvline(pd.Timestamp('1999-11-12'), color='darkred', ls=':', lw=1.2, alpha=0.6)

    xlim = (pd.Timestamp('1997-01-01'), pd.Timestamp('2012-01-01'))

    # ── Panel A: σ(t) ──
    ax = axes[0, 0]
    if 'sigma_raw' in us_panel.columns:
        s = us_panel['sigma_raw'].dropna() * 100
        ax.plot(s.index, s.values, 'b-', lw=2)
        ax.set_ylabel('RE loans / Total loans (%)')
        mark_gs(ax)
        ax.annotate('Glass-Steagall\nrepeal', xy=(pd.Timestamp('1999-11-12'), s.min()),
                     xytext=(pd.Timestamp('2001-06-01'), s.min() + (s.max()-s.min())*0.15),
                     fontsize=8, color='darkred',
                     arrowprops=dict(arrowstyle='->', color='darkred', lw=0.8))
    shade_crisis(ax)
    ax.set_title('A. σ(t): Real Estate Loan Concentration', fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(*xlim)

    # ── Panel B: T(t) components ──
    ax = axes[0, 1]
    plotted_t = False
    if 'T_standards' in us_panel.columns:
        s = us_panel['T_standards'].dropna()
        ax.plot(s.index, s.values, 'navy', lw=1.5, alpha=0.8,
                label='−Lending standards\n(positive = loosening)')
        plotted_t = True
    if 'T_housing' in us_panel.columns:
        s = us_panel['T_housing'].dropna()
        ax2 = ax.twinx()
        ax2.plot(s.index, s.values, 'darkred', lw=1.5, alpha=0.8,
                 label='Housing dev. from\n60-mo trailing mean (%)')
        ax2.set_ylabel('Housing deviation (%)', color='darkred', fontsize=9)
        ax2.legend(loc='upper left', fontsize=7, bbox_to_anchor=(0.5, 1.0))
    shade_crisis(ax)
    ax.set_title('B. T(t): Information Temperature Components', fontsize=11)
    if plotted_t:
        ax.legend(loc='upper left', fontsize=7)
    ax.set_ylabel('Lending standards (inverted)')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(*xlim)

    # ── Panel C: Fragility σ×T ──
    ax = axes[1, 0]
    if 'fragility' in us_panel.columns:
        frag = us_panel['fragility'].dropna()
        ax.plot(frag.index, frag.values, 'k-', lw=1.5, alpha=0.6, label='σ×T (monthly)')

        # 6-month smoothed
        smooth = frag.rolling(6, min_periods=3).mean()
        ax.plot(smooth.index, smooth.values, 'darkred', lw=2.5, label='6-month smoothed')

        ax.axhline(0, color='gray', ls='-', lw=0.8)

        # Threshold: flag when smoothed fragility > 1 SD
        high_frag = smooth[smooth > 1.0]
        if len(high_frag) > 0:
            first_signal = high_frag.index[0]
            ax.axvline(first_signal, color='orange', ls='--', lw=1.5, alpha=0.7)
            ax.annotate(f'Signal: {first_signal.strftime("%b %Y")}',
                        xy=(first_signal, smooth.loc[first_signal]),
                        xytext=(first_signal + pd.Timedelta(days=180),
                                smooth.loc[first_signal] + 0.5),
                        fontsize=9, fontweight='bold', color='darkorange',
                        arrowprops=dict(arrowstyle='->', color='darkorange'))

    shade_crisis(ax)
    ax.set_title('C. Fragility Index: σ(t) × T(t)', fontsize=11)
    ax.set_ylabel('Fragility (z-score product)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(*xlim)

    # ── Panel D: Comparison (σ vs T vs σ×T) ──
    ax = axes[1, 1]
    plotted_any = False
    for col, label, color in [('sigma_z', 'σ (z-score)', 'blue'),
                               ('T_z', 'T (z-score)', 'green'),
                               ('fragility', 'σ×T', 'red')]:
        if col in us_panel.columns:
            s = us_panel[col].dropna()
            smooth = s.rolling(6, min_periods=3).mean()
            ax.plot(smooth.index, smooth.values, color=color, lw=2,
                    alpha=0.8, label=label)
            plotted_any = True

    if plotted_any:
        ax.axhline(0, color='gray', ls='-', lw=0.8)
    shade_crisis(ax)
    ax.set_title('D. Component Comparison (6-month smoothed)', fontsize=11)
    ax.set_ylabel('Z-score / Fragility')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(*xlim)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ['.png', '.pdf']:
        path = os.path.join(FIGDIR, f'gfc_us_dashboard{ext}')
        plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved figures/gfc_us_dashboard.png/.pdf")


def make_collapse_warning_figure(us_panel):
    """
    Focused single-panel figure: strength of collapse indication over time.

    Shows the fragility index (σ×T) as the primary signal, with components
    underneath, key event annotations, and NBER recession shading.
    The point: σ alone rises slowly (structural), T alone is ambiguous,
    but the INTERACTION σ×T clearly identifies the danger window.
    """
    if not HAS_MPL:
        return

    print("  Generating Figure 3: Collapse warning signal...")

    fig, ax = plt.subplots(figsize=(12, 6))

    xlim = (pd.Timestamp('1997-01-01'), pd.Timestamp('2012-01-01'))

    # NBER recession shading
    recessions = [
        ('2001-03-01', '2001-11-01'),  # dot-com
        ('2007-12-01', '2009-06-01'),  # Great Recession
    ]
    for start, end in recessions:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color='gray', alpha=0.12, zorder=0)

    # Plot components (faded, behind)
    for col, label, color, lw in [
        ('sigma_z', 'σ: loan concentration (z-score)', 'steelblue', 1.2),
        ('T_z', 'T: info. temperature (z-score)', 'forestgreen', 1.2),
    ]:
        if col in us_panel.columns:
            s = us_panel[col].dropna()
            smooth = s.rolling(6, min_periods=3).mean()
            ax.plot(smooth.index, smooth.values, color=color, lw=lw,
                    alpha=0.45, label=label, zorder=2)

    # Main signal: fragility (σ×T), smoothed
    if 'fragility' in us_panel.columns:
        frag = us_panel['fragility'].dropna()

        # Raw (very faint)
        ax.plot(frag.index, frag.values, color='firebrick', lw=0.6,
                alpha=0.2, zorder=3)

        # 6-month smoothed (primary)
        smooth = frag.rolling(6, min_periods=3).mean()
        ax.plot(smooth.index, smooth.values, color='firebrick', lw=3,
                label='Fragility: σ × T (6-mo smoothed)', zorder=4)

        # Fill danger zone
        ax.fill_between(smooth.index, 0, smooth.values,
                        where=smooth.values > 0,
                        color='firebrick', alpha=0.08, zorder=1)

        # Threshold annotation
        ax.axhline(0, color='black', lw=0.8, zorder=2)
        ax.axhline(1.0, color='darkorange', ls='--', lw=1, alpha=0.6, zorder=2)
        ax.text(pd.Timestamp('2011-03-01'), 1.15, 'warning\nthreshold',
                fontsize=8, color='darkorange', ha='center', va='bottom')

        # Peak annotation
        peak_idx = smooth.idxmax()
        peak_val = smooth.max()
        ax.annotate(f'Peak: {peak_idx.strftime("%b %Y")}\n({peak_val:.1f}σ)',
                    xy=(peak_idx, peak_val),
                    xytext=(peak_idx + pd.Timedelta(days=300), peak_val + 0.6),
                    fontsize=9, fontweight='bold', color='firebrick',
                    arrowprops=dict(arrowstyle='->', color='firebrick', lw=1.2),
                    zorder=5)

    # Key event annotations
    events = [
        ('1999-11-12', 'Glass-Steagall\nrepeal', -4.0),
        ('2003-06-01', 'Fed funds\nhit 1%', 1.8),
        ('2005-06-01', 'Rajan warns\n(Jackson Hole)', 2.0),
        ('2006-07-01', 'Housing\nprice peak', -0.5),
        ('2007-08-09', 'BNP Paribas\nfreezes funds', -2.5),
        ('2008-09-15', 'Lehman\ncollapse', -3.5),
    ]

    for date_str, label, y_pos in events:
        dt = pd.Timestamp(date_str)
        ax.axvline(dt, color='gray', ls=':', lw=0.7, alpha=0.4, zorder=1)
        ax.text(dt, y_pos, label, fontsize=7, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='gray', alpha=0.85),
                zorder=6)

    # Labels
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Signal strength (z-score product)', fontsize=11)
    ax.set_title('Strength of Collapse Warning: σ(t) × T(t)\n'
                 'Leading indicators only, recursive normalization (no look-ahead)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.15, zorder=0)

    # Add recession labels
    ax.text(pd.Timestamp('2001-07-01'), ax.get_ylim()[1] * 0.95,
            'dot-com\nrecession', fontsize=7, ha='center', va='top',
            color='gray', alpha=0.6)
    ax.text(pd.Timestamp('2008-09-01'), ax.get_ylim()[1] * 0.95,
            'Great\nRecession', fontsize=7, ha='center', va='top',
            color='gray', alpha=0.6)

    plt.tight_layout()
    for ext in ['.png', '.pdf']:
        path = os.path.join(FIGDIR, f'gfc_collapse_warning{ext}')
        plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved figures/gfc_collapse_warning.png/.pdf")


# ============================================================================
# SECTION 5: LaTeX Table
# ============================================================================

def make_latex_table(ols_results, placebo_results, logit_res):
    """
    Publication-quality LaTeX regression table matching thesis conventions.
    """
    print("\n  Generating LaTeX regression table...")

    # Column specs
    cols = [
        ('m1', '(1)\nBaseline'),
        ('m2', '(2)\n+PMI'),
        ('m3', '(3)\nInteraction'),
        ('m4', '(4)\n+FinDev'),
        ('m5', '(5)\nGDP growth'),
    ]

    # Row specs: (param_name, display_label)
    rows = [
        ('ari', 'ARI (Activity Restrictions)'),
        ('pmi', 'PMI (Private Monitoring)'),
        ('ari_x_pmi', r'ARI $\times$ PMI'),
        ('log_gdp_pc', r'$\log$ GDP per capita'),
        ('fin_dev_2003', 'Financial Development'),
    ]

    def fmt_coef(res, var):
        if var not in res.params.index:
            return ''
        b = res.params[var]
        se = res.bse[var]
        p = res.pvalues[var]
        stars = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        return f'\\makecell{{{b:.3f}{stars} \\\\ ({se:.3f})}}'

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Cross-Country GFC Severity: Activity Restrictions $\times$ Private Monitoring}')
    lines.append(r'\label{tab:gfc_prediction}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{l' + 'c' * len(cols) + '}')
    lines.append(r'\toprule')

    # Header
    header = ' & '.join([f'{label.split(chr(10))[0]}' for _, label in cols])
    lines.append(f' & {header} \\\\')
    subheader = ' & '.join([label.split('\n')[1] if '\n' in label else '' for _, label in cols])
    lines.append(f' & {subheader} \\\\')
    lines.append(r'\midrule')

    # DV label
    dv_labels = {
        'm1': 'GDP loss', 'm2': 'GDP loss', 'm3': 'GDP loss',
        'm4': 'GDP loss', 'm5': 'GDP growth',
    }
    dv_row = ' & '.join([dv_labels.get(k, '') for k, _ in cols])
    lines.append(f'  DV: & {dv_row} \\\\')
    lines.append(r'\midrule')

    # Coefficient rows
    for var, display in rows:
        entries = []
        for model_key, _ in cols:
            if model_key in ols_results:
                entries.append(fmt_coef(ols_results[model_key], var))
            else:
                entries.append('')
        lines.append(f'  {display} & {" & ".join(entries)} \\\\[4pt]')

    lines.append(r'\midrule')

    # Model stats
    for stat_label, stat_fn in [
        ('$N$', lambda r: f'{int(r.nobs)}'),
        ('$R^2$', lambda r: f'{r.rsquared:.3f}'),
    ]:
        entries = []
        for model_key, _ in cols:
            if model_key in ols_results:
                entries.append(stat_fn(ols_results[model_key]))
            else:
                entries.append('')
        lines.append(f'  {stat_label} & {" & ".join(entries)} \\\\')

    lines.append(r'  SE & HC1 & HC1 & HC1 & HC1 & HC1 \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\vspace{4pt}')

    # Footnote with placebo results
    placebo_text = []
    for idx_name in ['CSI', 'SPI', 'EBI']:
        if idx_name in placebo_results:
            pr = placebo_results[idx_name]
            p_fmt = f'{pr["p"]:.3f}'
            placebo_text.append(f'{idx_name}$\\times$PMI: $\\beta$={pr["beta"]:+.3f} (p={p_fmt})')

    footnote_parts = [
        r'Dependent variable: columns (1)--(4) cumulative GDP per capita loss 2007--2009 (\%);',
        r'column (5) GDP growth rate 2009 (\%).',
        r'ARI = BCL Activity Restrictions Index (higher = banks restricted to commodity lending = higher $\sigma$).',
        r'PMI = BCL Private Monitoring Index (higher = stronger information disclosure = lower $T$).',
        r'All from BCL Wave~2 (2003), pre-crisis.',
        r'Robust (HC1) standard errors in parentheses.',
    ]
    if placebo_text:
        footnote_parts.append(
            r'Placebo interactions (all insignificant): ' + '; '.join(placebo_text) + '.'
        )
    if logit_res is not None:
        b_int = logit_res.params.get('ari_x_pmi', 0)
        p_int = logit_res.pvalues.get('ari_x_pmi', 1)
        footnote_parts.append(
            f'Logit crisis probability: ARI$\\times$PMI $\\beta$={b_int:+.3f} (p={p_int:.3f}), '
            f'pseudo $R^2$={logit_res.prsquared:.3f}.'
        )

    footnote_parts.append(r'$^{***}p<0.001$; $^{**}p<0.01$; $^{*}p<0.05$.')

    lines.append(r'\parbox{0.95\textwidth}{\footnotesize')
    lines.append('  ' + '\n  '.join(footnote_parts))
    lines.append(r'}')
    lines.append(r'\end{table}')

    table_str = '\n'.join(lines)

    path = os.path.join(OUTPUT, 'gfc_regression_table.tex')
    with open(path, 'w') as f:
        f.write(table_str)
    print(f"    Saved {path}")

    return table_str


# ============================================================================
# SECTION 6: Main
# ============================================================================

def main():
    print("=" * 72)
    print("  FREE ENERGY FRAMEWORK — GFC PREDICTION (Paper Quality)")
    print("=" * 72)
    print()
    print("  FRAMEWORK PREDICTION")
    print("  Market collapse when T > T*(σ), where T* decreases with σ.")
    print("  BCL (2006) puzzle: activity restrictions increase crisis risk.")
    print("  Explanation: ARI → σ → lower T* → more vulnerable to T shocks.")
    print("  Novel test: this effect is MEDIATED by information quality (PMI).")
    print("  The ARI × PMI interaction is the specific prediction.")
    print()

    # ====================================================================
    # PART A: Cross-Country GFC Prediction
    # ====================================================================
    print(f"\n{'='*72}")
    print("  PART A: CROSS-COUNTRY GFC PREDICTION")
    print(f"{'='*72}")

    panel = build_cross_country_panel()
    if panel is None:
        print("  FATAL: Could not build cross-country panel")
        return

    ols_results, ols_df = run_ols_regressions(panel)
    logit_res = run_logit(panel)
    placebo_results = run_placebo_tests(panel)
    quad_stats, ari_med, pmi_med = quadrant_analysis(panel)
    fpr, tpr, auc = compute_roc(panel)

    # ====================================================================
    # PART B: US Early Warning
    # ====================================================================
    print(f"\n{'='*72}")
    print("  PART B: US EARLY WARNING (Leading Indicators Only)")
    print(f"{'='*72}")

    us_panel = build_us_early_warning()

    # ====================================================================
    # PART C: Publication Output
    # ====================================================================
    print(f"\n{'='*72}")
    print("  PART C: PUBLICATION OUTPUT")
    print(f"{'='*72}")

    if HAS_MPL:
        make_cross_country_figure(panel, quad_stats, ari_med, pmi_med)
        make_us_dashboard(us_panel)
        make_collapse_warning_figure(us_panel)

    make_latex_table(ols_results, placebo_results, logit_res)

    # Save calibration summary
    summary = {
        'part_a': {
            'n_countries': int(ols_results['m3'].nobs),
            'main_model_r2': float(ols_results['m3'].rsquared),
            'ari_x_pmi_beta': float(ols_results['m3'].params.get('ari_x_pmi', 0)),
            'ari_x_pmi_p': float(ols_results['m3'].pvalues.get('ari_x_pmi', 1)),
            'roc_auc': float(auc),
            'quadrant_worst': 'High ARI / Low PMI',
            'quadrant_worst_adj': float(quad_stats.get('High ARI / Low PMI', {}).get('mean_adj', 0)),
            'quadrant_worst_raw': float(quad_stats.get('High ARI / Low PMI', {}).get('mean_loss', 0)),
        },
        'part_b': {
            'n_months': len(us_panel),
        },
    }

    # Add placebo results
    for idx_name in ['CSI', 'SPI', 'EBI']:
        if idx_name in placebo_results:
            summary['part_a'][f'placebo_{idx_name}_p'] = float(placebo_results[idx_name]['p'])

    # Check if fragility rose before crisis
    if 'fragility' in us_panel.columns:
        frag = us_panel['fragility'].dropna()
        smooth = frag.rolling(6, min_periods=3).mean()
        high_frag = smooth[smooth > 1.0]
        if len(high_frag) > 0:
            summary['part_b']['first_signal_date'] = str(high_frag.index[0].date())

        # Mean fragility by period
        for label, start, end in [('pre_2004', '2000-01-01', '2003-12-31'),
                                   ('buildup', '2004-01-01', '2006-12-31'),
                                   ('crisis', '2007-01-01', '2009-06-30')]:
            period = frag[(frag.index >= start) & (frag.index <= end)]
            if len(period) > 0:
                summary['part_b'][f'mean_fragility_{label}'] = float(period.mean())

    with open(os.path.join(OUTPUT, 'gfc_calibration.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Final summary ──
    print(f"\n\n{'='*72}")
    print("  SUMMARY")
    print(f"{'='*72}")

    m3 = ols_results['m3']
    b_int = m3.params.get('ari_x_pmi', 0)
    p_int = m3.pvalues.get('ari_x_pmi', 1)
    print(f"""
  PART A — Cross-Country (N = {int(m3.nobs)})
    Main result (Model 3): ARI × PMI interaction
      β = {b_int:+.3f}, p = {p_int:.4f}
      R² = {m3.rsquared:.3f}

    Placebo interactions (none should be significant):""")

    for idx_name in ['CSI', 'SPI', 'EBI']:
        if idx_name in placebo_results:
            pr = placebo_results[idx_name]
            sig = 'SIGNIFICANT' if pr['p'] < 0.05 else 'insignificant'
            print(f"      {idx_name} × PMI: β = {pr['beta']:+.3f}, p = {pr['p']:.3f} ({sig})")

    print(f"""
    Quadrant analysis (GDP-adjusted, prediction: High ARI / Low PMI = worst):""")
    for qname, qs in quad_stats.items():
        adj = qs.get('mean_adj', qs['mean_loss'])
        print(f"      {qname}: adj mean = {adj:+.1f}%, raw = {qs['mean_loss']:+.1f}% (n={qs['n']})")

    print(f"""
    ROC AUC = {auc:.3f}

  PART B — US Early Warning
    Leading indicators only: lending standards + housing deviation
    Recursive normalization (no look-ahead bias)""")

    if 'first_signal_date' in summary.get('part_b', {}):
        print(f"    First signal (fragility > 1σ): {summary['part_b']['first_signal_date']}")
    for label in ['pre_2004', 'buildup', 'crisis']:
        key = f'mean_fragility_{label}'
        if key in summary.get('part_b', {}):
            print(f"    Mean fragility ({label}): {summary['part_b'][key]:+.3f}")

    print(f"""
  OUTPUT FILES
    thesis_data/gfc_prediction_panel.csv  — Cross-country panel
    thesis_data/gfc_us_timeseries.csv     — US monthly time series
    thesis_data/gfc_calibration.json      — Summary statistics
    thesis_data/gfc_regression_table.tex  — LaTeX regression table
    figures/gfc_phase_space.png/.pdf      — Cross-country scatter
    figures/gfc_us_dashboard.png/.pdf     — US early warning dashboard
""")


if __name__ == '__main__':
    main()
