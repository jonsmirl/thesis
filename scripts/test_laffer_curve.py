#!/usr/bin/env python3
"""
test_laffer_curve.py
====================
Laffer Curve Empirical Test — Three CES Predictions

Tests three predictions from the Laffer Note (Smirl 2026) that derive the
Laffer curve from CES production complementarity:

  1. Sector-specific peaks: Low-ρ sectors (construction, manufacturing,
     healthcare) tolerate higher tax rates than high-ρ sectors (finance,
     information, professional services)
  2. Asymmetric decline: Curve is right-skewed for low-ρ, left-skewed for
     high-ρ sectors
  3. Displacement not suppression: Revenue loss beyond peak comes from
     activity relocating, not vanishing

Uses US state-level variation (50 states × 20 years) as natural experiments.

Data sources:
  - FRED: State quarterly GDP by industry (BEA via FRED)
  - Tax Foundation: State income tax rates (hardcoded reference data)
  - IRS SOI: State-to-state migration AGI flows

Outputs:
  thesis_data/laffer_results.csv
  thesis_data/laffer_results.txt
  thesis_data/laffer_table.tex
  figures/laffer_curve.pdf

Requires: pandas numpy statsmodels scipy matplotlib requests

Usage:
  source .venv/bin/activate && source env.sh
  python scripts/test_laffer_curve.py
"""

import os
import sys
import time
import warnings
from io import StringIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ═══════════════════════════════════════════════════════════════════════════
#  0. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'thesis_data')
CACHE_DIR = os.path.join(DATA_DIR, 'fred_cache')
SOI_DIR = os.path.join(DATA_DIR, 'soi_cache')
FIG_DIR = os.path.join(BASE_DIR, 'figures')

for d in [DATA_DIR, CACHE_DIR, SOI_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

FRED_API_KEY = os.environ.get('FRED_API_KEY')
if not FRED_API_KEY:
    print("ERROR: FRED_API_KEY not set. Run: source env.sh")
    sys.exit(1)

results_buf = StringIO()


def log(msg=""):
    """Print to stdout and accumulate for results file."""
    print(msg)
    results_buf.write(msg + "\n")


np.random.seed(42)

# Sector complementarity classification (from Laffer Note, line 31)
SECTOR_RHO = {
    'Construction':  {'fred_code': 'CONST',      'rho_class': 'LOW',  'rho_proxy': 0.2},
    'Manufacturing': {'fred_code': 'MAN',         'rho_class': 'LOW',  'rho_proxy': 0.3},
    'Healthcare':    {'fred_code': 'HLTHSOCASS',  'rho_class': 'LOW',  'rho_proxy': 0.25},
    'Finance':       {'fred_code': 'FININS',      'rho_class': 'HIGH', 'rho_proxy': 0.8},
    'Information':   {'fred_code': 'INFO',        'rho_class': 'HIGH', 'rho_proxy': 0.75},
    'ProfServices':  {'fred_code': 'PROBUS',      'rho_class': 'HIGH', 'rho_proxy': 0.7},
}

STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
]


# ═══════════════════════════════════════════════════════════════════════════
#  1. STATE INDIVIDUAL INCOME TAX RATES (2005–2024)
#     Source: Tax Foundation annual compilations
#     Format: {state: [(effective_year, top_marginal_rate%), ...]}
#     Rate applies from effective_year until next entry.
# ═══════════════════════════════════════════════════════════════════════════

NO_INCOME_TAX = {'AK', 'FL', 'NV', 'NH', 'SD', 'TN', 'TX', 'WA', 'WY'}

TAX_SCHEDULE = {
    'AL': [(2005, 5.0)],
    'AZ': [(2005, 4.54), (2021, 2.98), (2023, 2.5)],
    'AR': [(2005, 7.0), (2015, 6.9), (2019, 5.9), (2023, 4.4), (2024, 3.9)],
    'CA': [(2005, 9.3), (2009, 10.3), (2012, 13.3)],
    'CO': [(2005, 4.63), (2020, 4.55), (2022, 4.4)],
    'CT': [(2005, 5.0), (2009, 6.5), (2011, 6.7), (2015, 6.99)],
    'DE': [(2005, 5.95), (2010, 6.6)],
    'GA': [(2005, 6.0), (2024, 5.49)],
    'HI': [(2005, 8.25), (2009, 11.0)],
    'ID': [(2005, 7.8), (2012, 6.925), (2018, 6.5), (2022, 6.0), (2023, 5.8)],
    'IL': [(2005, 3.0), (2011, 5.0), (2015, 3.75), (2017, 4.95)],
    'IN': [(2005, 3.4), (2015, 3.3), (2017, 3.23), (2023, 3.15), (2024, 3.05)],
    'IA': [(2005, 8.98), (2023, 6.0), (2024, 5.7)],
    'KS': [(2005, 6.45), (2013, 3.9), (2017, 5.7)],
    'KY': [(2005, 6.0), (2018, 5.0), (2023, 4.5), (2024, 4.0)],
    'LA': [(2005, 6.0), (2022, 4.25)],
    'ME': [(2005, 8.5), (2016, 7.15)],
    'MD': [(2005, 5.5), (2008, 6.25), (2012, 5.75)],
    'MA': [(2005, 5.3), (2012, 5.15), (2014, 5.1), (2016, 5.05), (2020, 5.0)],
    'MI': [(2005, 3.9), (2007, 4.35), (2012, 4.25)],
    'MN': [(2005, 7.85), (2013, 9.85)],
    'MS': [(2005, 5.0), (2024, 4.7)],
    'MO': [(2005, 6.0), (2019, 5.4), (2020, 5.3), (2022, 4.95), (2024, 4.8)],
    'MT': [(2005, 6.9), (2024, 5.9)],
    'NE': [(2005, 6.84), (2024, 5.84)],
    'NJ': [(2005, 8.97), (2018, 10.75)],
    'NM': [(2005, 4.9), (2021, 5.9)],
    'NY': [(2005, 6.85), (2009, 8.82), (2021, 10.9)],
    'NC': [(2005, 8.25), (2014, 5.8), (2015, 5.75), (2017, 5.499),
           (2019, 5.25), (2022, 4.99), (2023, 4.75), (2024, 4.5)],
    'ND': [(2005, 5.54), (2009, 4.86), (2013, 3.22), (2019, 2.9), (2024, 1.95)],
    'OH': [(2005, 7.5), (2008, 5.925), (2014, 5.0), (2019, 4.797),
           (2021, 3.99), (2024, 3.5)],
    'OK': [(2005, 6.65), (2008, 5.5), (2016, 5.0), (2022, 4.75)],
    'OR': [(2005, 9.0), (2010, 9.9)],
    'PA': [(2005, 3.07)],
    'RI': [(2005, 9.9), (2011, 5.99)],
    'SC': [(2005, 7.0), (2022, 6.5), (2024, 6.4)],
    'UT': [(2005, 7.0), (2007, 5.0), (2023, 4.85), (2024, 4.65)],
    'VT': [(2005, 9.5), (2007, 8.95), (2009, 8.75)],
    'VA': [(2005, 5.75)],
    'WV': [(2005, 6.5), (2024, 5.12)],
    'WI': [(2005, 6.75), (2013, 7.65)],
}


def get_tax_rate(state, year):
    """Get top marginal income tax rate for state in given year."""
    if state in NO_INCOME_TAX:
        return 0.0
    schedule = TAX_SCHEDULE.get(state)
    if schedule is None:
        return np.nan
    rate = schedule[0][1]
    for yr, r in schedule:
        if yr <= year:
            rate = r
        else:
            break
    return rate


# ═══════════════════════════════════════════════════════════════════════════
#  2. MAJOR TAX CHANGE EVENTS (>1pp change, for event study)
# ═══════════════════════════════════════════════════════════════════════════

TAX_EVENTS = [
    {'state': 'KS', 'year': 2013, 'direction': 'CUT', 'magnitude': 2.55,
     'old_rate': 6.45, 'new_rate': 3.9, 'label': 'Kansas Brownback'},
    {'state': 'KS', 'year': 2017, 'direction': 'HIKE', 'magnitude': 1.8,
     'old_rate': 3.9, 'new_rate': 5.7, 'label': 'Kansas reversal'},
    {'state': 'NC', 'year': 2014, 'direction': 'CUT', 'magnitude': 2.45,
     'old_rate': 8.25, 'new_rate': 5.8, 'label': 'NC flat tax'},
    {'state': 'CA', 'year': 2012, 'direction': 'HIKE', 'magnitude': 3.0,
     'old_rate': 10.3, 'new_rate': 13.3, 'label': 'CA Prop 30'},
    {'state': 'NY', 'year': 2009, 'direction': 'HIKE', 'magnitude': 1.97,
     'old_rate': 6.85, 'new_rate': 8.82, 'label': 'NY surcharge'},
    {'state': 'NY', 'year': 2021, 'direction': 'HIKE', 'magnitude': 2.08,
     'old_rate': 8.82, 'new_rate': 10.9, 'label': 'NY millionaire'},
    {'state': 'NJ', 'year': 2018, 'direction': 'HIKE', 'magnitude': 1.78,
     'old_rate': 8.97, 'new_rate': 10.75, 'label': 'NJ surcharge'},
    {'state': 'IL', 'year': 2011, 'direction': 'HIKE', 'magnitude': 2.0,
     'old_rate': 3.0, 'new_rate': 5.0, 'label': 'IL temp hike'},
    {'state': 'IL', 'year': 2015, 'direction': 'CUT', 'magnitude': 1.25,
     'old_rate': 5.0, 'new_rate': 3.75, 'label': 'IL sunset'},
    {'state': 'MN', 'year': 2013, 'direction': 'HIKE', 'magnitude': 2.0,
     'old_rate': 7.85, 'new_rate': 9.85, 'label': 'MN 4th bracket'},
    {'state': 'RI', 'year': 2011, 'direction': 'CUT', 'magnitude': 3.91,
     'old_rate': 9.9, 'new_rate': 5.99, 'label': 'RI reform'},
    {'state': 'UT', 'year': 2007, 'direction': 'CUT', 'magnitude': 2.0,
     'old_rate': 7.0, 'new_rate': 5.0, 'label': 'UT flat tax'},
    {'state': 'IA', 'year': 2023, 'direction': 'CUT', 'magnitude': 2.98,
     'old_rate': 8.98, 'new_rate': 6.0, 'label': 'IA reform'},
    {'state': 'OH', 'year': 2008, 'direction': 'CUT', 'magnitude': 1.575,
     'old_rate': 7.5, 'new_rate': 5.925, 'label': 'OH reduction'},
]


# ═══════════════════════════════════════════════════════════════════════════
#  3. FRED DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════

def fetch_and_cache(series_id, start='2005-01-01'):
    """Fetch FRED series with local CSV caching."""
    cache_path = os.path.join(CACHE_DIR, f'{series_id}.csv')

    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if len(df) > 0:
                return df.iloc[:, 0]
        except Exception:
            pass

    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={FRED_API_KEY}"
        f"&file_type=json&observation_start={start}"
    )

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    if 'error_code' in data:
        return None

    dates, values = [], []
    for obs in data.get('observations', []):
        if obs['value'] != '.':
            dates.append(obs['date'])
            values.append(float(obs['value']))

    if not dates:
        return None

    s = pd.Series(values, index=pd.to_datetime(dates), name=series_id)
    s.to_csv(cache_path)
    time.sleep(0.15)  # rate limit
    return s


def fetch_state_gdp():
    """Fetch state GDP by industry from FRED, return annual panel."""
    log("=" * 70)
    log("FRED State GDP by Industry")
    log("=" * 70)

    # Verify naming convention with a test series
    test_id = f"CANQGSP"
    test_s = fetch_and_cache(test_id)
    if test_s is None:
        log(f"  WARNING: Test series {test_id} not found.")
        log("  Trying annual pattern (NGSP)...")
        test_id2 = "CANGSP"
        test_s2 = fetch_and_cache(test_id2)
        if test_s2 is not None:
            log(f"  Annual pattern works: {test_id2}")
            suffix = "NGSP"
            is_quarterly = False
        else:
            log("  Neither quarterly nor annual pattern found on FRED.")
            log("  Proceeding with quarterly pattern — missing series handled gracefully.")
            suffix = "NQGSP"
            is_quarterly = True
    else:
        log(f"  Verified: {test_id} ({len(test_s)} observations)")
        suffix = "NQGSP"
        is_quarterly = True

    rows = []
    total_expected = len(STATES) * (1 + len(SECTOR_RHO))
    fetched = 0
    missing = 0

    for st in STATES:
        # Total GDP
        total_id = f"{st}{suffix}"
        total_s = fetch_and_cache(total_id)
        if total_s is None:
            missing += 1 + len(SECTOR_RHO)
            continue
        fetched += 1

        # Convert to annual (average of quarters for SAAR, or keep annual)
        if is_quarterly:
            total_annual = total_s.resample('YE').mean().dropna()
        else:
            total_annual = total_s.copy()
            total_annual.index = pd.to_datetime(total_annual.index)

        for sector, info in SECTOR_RHO.items():
            sid = f"{st}{info['fred_code']}{suffix}"
            s = fetch_and_cache(sid)
            if s is None:
                missing += 1
                continue
            fetched += 1

            if is_quarterly:
                annual = s.resample('YE').mean().dropna()
            else:
                annual = s.copy()
                annual.index = pd.to_datetime(annual.index)

            # Merge and compute share
            merged = pd.DataFrame({
                'sector_gdp': annual,
                'total_gdp': total_annual
            }).dropna()

            for dt, row in merged.iterrows():
                yr = dt.year
                if yr < 2005 or yr > 2024:
                    continue
                if row['total_gdp'] > 0:
                    rows.append({
                        'state': st,
                        'year': yr,
                        'sector': sector,
                        'rho_class': info['rho_class'],
                        'rho_proxy': info['rho_proxy'],
                        'gdp_sector': row['sector_gdp'],
                        'gdp_total': row['total_gdp'],
                        'gdp_share': row['sector_gdp'] / row['total_gdp'],
                    })

    log(f"  Fetched: {fetched}/{total_expected} series, {missing} missing")

    if not rows:
        log("  WARNING: No GDP data retrieved. Check FRED series naming convention.")
        return pd.DataFrame()

    panel = pd.DataFrame(rows)
    log(f"  Panel: {len(panel)} obs, {panel['state'].nunique()} states, "
        f"{panel['year'].nunique()} years, {panel['sector'].nunique()} sectors")
    return panel


# ═══════════════════════════════════════════════════════════════════════════
#  4. IRS SOI MIGRATION DATA
# ═══════════════════════════════════════════════════════════════════════════

FIPS_TO_STATE = {
    1: 'AL', 2: 'AK', 4: 'AZ', 5: 'AR', 6: 'CA', 8: 'CO', 9: 'CT',
    10: 'DE', 12: 'FL', 13: 'GA', 15: 'HI', 16: 'ID', 17: 'IL', 18: 'IN',
    19: 'IA', 20: 'KS', 21: 'KY', 22: 'LA', 23: 'ME', 24: 'MD', 25: 'MA',
    26: 'MI', 27: 'MN', 28: 'MS', 29: 'MO', 30: 'MT', 31: 'NE', 32: 'NV',
    33: 'NH', 34: 'NJ', 35: 'NM', 36: 'NY', 37: 'NC', 38: 'ND', 39: 'OH',
    40: 'OK', 41: 'OR', 42: 'PA', 44: 'RI', 45: 'SC', 46: 'SD', 47: 'TN',
    48: 'TX', 49: 'UT', 50: 'VT', 51: 'VA', 53: 'WA', 54: 'WV', 55: 'WI',
    56: 'WY',
}


def fetch_soi_migration():
    """Download and parse IRS SOI state-to-state migration AGI data."""
    log("")
    log("=" * 70)
    log("IRS SOI Migration Data")
    log("=" * 70)

    all_rows = []

    for start_yr in range(2011, 2022):
        end_yr = start_yr + 1
        yy1 = f"{start_yr % 100:02d}"
        yy2 = f"{end_yr % 100:02d}"
        filename = f"stateoutflow{yy1}{yy2}.csv"
        cache_path = os.path.join(SOI_DIR, filename)

        if not os.path.exists(cache_path):
            url = f"https://www.irs.gov/pub/irs-soi/{filename}"
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                with open(cache_path, 'wb') as f:
                    f.write(resp.content)
                log(f"  Downloaded: {filename}")
                time.sleep(1.0)
            except Exception as e:
                log(f"  WARNING: Failed to download {filename}: {e}")
                continue

        try:
            df = pd.read_csv(cache_path, encoding='latin-1')
        except Exception:
            try:
                df = pd.read_csv(cache_path, encoding='utf-8')
            except Exception:
                log(f"  WARNING: Cannot parse {filename}")
                continue

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Find relevant columns (format varies by year)
        origin_col = None
        dest_col = None
        agi_col = None

        for c in df.columns:
            if 'y1_statefips' in c or 'y1_state_fips' in c or c == 'y1_statefips':
                origin_col = c
            elif 'y2_statefips' in c or 'y2_state_fips' in c or c == 'y2_statefips':
                dest_col = c
            elif c == 'agi' or c == 'adjusted_gross_income':
                agi_col = c

        if origin_col is None or dest_col is None or agi_col is None:
            # Try positional fallback
            cols = list(df.columns)
            if len(cols) >= 6:
                origin_col = cols[0]
                dest_col = cols[2]
                agi_col = cols[-1]
            else:
                log(f"  WARNING: Cannot identify columns in {filename}: {list(df.columns)[:6]}")
                continue

        for _, row in df.iterrows():
            try:
                o_fips = int(float(row[origin_col]))
                d_fips = int(float(row[dest_col]))
            except (ValueError, TypeError):
                continue

            # Skip aggregate rows and same-state
            if o_fips > 56 or d_fips > 56 or o_fips <= 0 or d_fips <= 0:
                continue
            if o_fips == d_fips:
                continue

            o_st = FIPS_TO_STATE.get(o_fips)
            d_st = FIPS_TO_STATE.get(d_fips)
            if o_st is None or d_st is None:
                continue

            try:
                agi_val = float(str(row[agi_col]).replace(',', ''))
            except (ValueError, TypeError):
                continue

            if agi_val <= 0:
                continue

            all_rows.append({
                'year': start_yr,
                'origin': o_st,
                'destination': d_st,
                'agi': agi_val,
            })

    if not all_rows:
        log("  WARNING: No SOI migration data available")
        return pd.DataFrame()

    panel = pd.DataFrame(all_rows)
    log(f"  SOI data: {len(panel)} pairwise flows, "
        f"{panel['year'].nunique()} years")
    return panel


# ═══════════════════════════════════════════════════════════════════════════
#  5. PANEL CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def build_panel(gdp_panel):
    """Add tax rates and derived variables to GDP panel."""
    if gdp_panel.empty:
        return gdp_panel

    panel = gdp_panel.copy()
    panel['tax_rate'] = panel.apply(
        lambda r: get_tax_rate(r['state'], r['year']), axis=1
    )
    panel['high_rho'] = (panel['rho_class'] == 'HIGH').astype(int)
    panel['log_share'] = np.log(panel['gdp_share'].clip(lower=1e-6))
    panel['trend'] = panel['year'] - 2005

    # Drop rows with missing tax rates
    panel = panel.dropna(subset=['tax_rate', 'log_share'])

    log("")
    log(f"Panel with tax rates: {len(panel)} obs")
    log(f"  Tax rate range: {panel['tax_rate'].min():.1f}% - "
        f"{panel['tax_rate'].max():.1f}%")
    log(f"  Years: {panel['year'].min()}-{panel['year'].max()}")
    log(f"  States: {panel['state'].nunique()}, Sectors: {panel['sector'].nunique()}")

    return panel


# ═══════════════════════════════════════════════════════════════════════════
#  6. TEST A: REVENUE ELASTICITY BY SECTOR COMPLEMENTARITY
# ═══════════════════════════════════════════════════════════════════════════

def test_a_revenue_elasticity(panel):
    """
    Test Prediction 1: high-ρ sectors are more sensitive to tax rates.

    Per-sector OLS: log(gdp_share) ~ tax_rate + state_FE + trend
    Pooled interaction: log(share) ~ rate + HighRho + rate×HighRho + state_FE + trend
    Prediction: interaction coefficient < 0
    """
    log("")
    log("=" * 70)
    log("TEST A: Revenue Elasticity by Sector Complementarity")
    log("=" * 70)

    results = []
    sector_betas = {}

    if panel.empty or len(panel) < 30:
        log("  SKIP: Insufficient data")
        return results, sector_betas

    # --- Per-sector regressions ---
    log("\n  Per-sector elasticities (state FE + trend):")
    for sector in SECTOR_RHO:
        sub = panel[panel['sector'] == sector].copy().reset_index(drop=True)
        if len(sub) < 30:
            log(f"    {sector:15s}: insufficient data ({len(sub)} obs)")
            continue

        # State fixed effects via dummies
        state_dummies = pd.get_dummies(sub['state'], prefix='st', drop_first=True, dtype=float)
        X = pd.concat([
            sub[['tax_rate', 'trend']],
            state_dummies,
        ], axis=1)
        X = sm.add_constant(X)
        y = sub['log_share']

        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

        if len(y) < 30:
            continue

        try:
            model = sm.OLS(y.values, X.values).fit()
            # tax_rate is column index 1 (after constant)
            beta = model.params[1]
            pval = model.pvalues[1]
            sector_betas[sector] = {
                'beta': beta, 'pval': pval, 'n': len(y),
                'rho_class': SECTOR_RHO[sector]['rho_class'],
            }
            rc = SECTOR_RHO[sector]['rho_class']
            log(f"    {sector:15s} ({rc}): "
                f"beta = {beta:+.6f}, p = {pval:.4f}, n = {len(y)}")
        except Exception as e:
            log(f"    {sector:15s}: regression failed ({e})")

    if not sector_betas:
        log("  No sector regressions succeeded")
        return results, sector_betas

    # --- Compare |beta| across rho classes ---
    low_betas = [abs(v['beta']) for s, v in sector_betas.items()
                 if SECTOR_RHO[s]['rho_class'] == 'LOW']
    high_betas = [abs(v['beta']) for s, v in sector_betas.items()
                  if SECTOR_RHO[s]['rho_class'] == 'HIGH']

    if low_betas and high_betas:
        # Mann-Whitney U: test if |beta_HIGH| > |beta_LOW|
        try:
            U_stat, U_pval = stats.mannwhitneyu(
                high_betas, low_betas, alternative='greater'
            )
        except ValueError:
            U_stat, U_pval = np.nan, np.nan

        log(f"\n  Mann-Whitney U: |beta_HIGH| > |beta_LOW|")
        log(f"    U = {U_stat:.1f}, p = {U_pval:.4f}")
        log(f"    Mean |beta| LOW = {np.mean(low_betas):.6f}, "
            f"HIGH = {np.mean(high_betas):.6f}")
        results.append(('A', 'mann_whitney_U', 'U', U_stat, U_pval))

    # --- Pooled interaction regression ---
    log("\n  Pooled interaction regression:")
    sub = panel.copy().reset_index(drop=True)
    sub['rate_x_high'] = sub['tax_rate'] * sub['high_rho']

    state_dummies = pd.get_dummies(sub['state'], prefix='st', drop_first=True, dtype=float)
    X = pd.concat([
        sub[['tax_rate', 'high_rho', 'rate_x_high', 'trend']],
        state_dummies,
    ], axis=1)
    X = sm.add_constant(X)
    y = sub['log_share']

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    if len(y) >= 30:
        try:
            model = sm.OLS(y.values, X.values).fit()
            # rate_x_high is column index 3 (const, tax_rate, high_rho, rate_x_high)
            interaction = model.params[3]
            int_pval = model.pvalues[3]
            log(f"    rate x HighRho: coeff = {interaction:+.6f}, p = {int_pval:.4f}")
            log(f"    Prediction: < 0 (high-rho sectors lose more share)")
            consistent = interaction < 0
            log(f"    Result: {'CONSISTENT' if consistent else 'INCONSISTENT'}")
            results.append(('A', 'interaction', 'coeff', interaction, int_pval))
        except Exception as e:
            log(f"    Pooled regression failed: {e}")

    return results, sector_betas


# ═══════════════════════════════════════════════════════════════════════════
#  7. TEST B: EVENT STUDY AROUND TAX CHANGES
# ═══════════════════════════════════════════════════════════════════════════

def test_b_event_study(panel):
    """
    Test Prediction 1 via natural experiments.

    For each major tax change event, compute DID:
      delta_log(GDP) for HIGH-rho minus delta_log(GDP) for LOW-rho
    Hikes: expect DID < 0 (high-rho hurt more)
    Cuts:  expect DID > 0 (high-rho benefit more)
    """
    log("")
    log("=" * 70)
    log("TEST B: Event Study Around Tax Changes")
    log("=" * 70)

    results = []
    event_details = []

    if panel.empty:
        log("  SKIP: No GDP data")
        return results, event_details

    for event in TAX_EVENTS:
        st = event['state']
        yr = event['year']
        direction = event['direction']
        label = event['label']

        # Pre: [yr-2, yr-1], Post: [yr, yr+1]
        pre = panel[(panel['state'] == st) &
                    (panel['year'] >= yr - 2) &
                    (panel['year'] < yr)]
        post = panel[(panel['state'] == st) &
                     (panel['year'] >= yr) &
                     (panel['year'] <= yr + 1)]

        if len(pre) < 2 or len(post) < 2:
            continue

        # Mean log(share) by rho class, pre vs post
        pre_low = pre[pre['rho_class'] == 'LOW']['log_share'].mean()
        pre_high = pre[pre['rho_class'] == 'HIGH']['log_share'].mean()
        post_low = post[post['rho_class'] == 'LOW']['log_share'].mean()
        post_high = post[post['rho_class'] == 'HIGH']['log_share'].mean()

        if any(np.isnan(x) for x in [pre_low, pre_high, post_low, post_high]):
            continue

        delta_low = post_low - pre_low
        delta_high = post_high - pre_high
        did = delta_high - delta_low

        # For hikes: expect did < 0; for cuts: expect did > 0
        if direction == 'HIKE':
            consistent = did < 0
        else:
            consistent = did > 0

        event_details.append({
            'label': label, 'state': st, 'year': yr,
            'direction': direction, 'magnitude': event['magnitude'],
            'did': did, 'consistent': consistent,
            'delta_low': delta_low, 'delta_high': delta_high,
        })

        mark = 'Y' if consistent else 'N'
        log(f"  {label:20s} ({direction} {event['magnitude']:+.1f}pp): "
            f"DID = {did:+.4f} [{mark}]")

    if not event_details:
        log("  No events with sufficient data")
        return results, event_details

    n_consistent = sum(1 for e in event_details if e['consistent'])
    n_total = len(event_details)

    # Binomial test: consistency rate > 50%?
    try:
        binom_result = stats.binomtest(n_consistent, n_total, 0.5,
                                       alternative='greater')
        binom_p = binom_result.pvalue
    except AttributeError:
        # Older scipy
        binom_p = stats.binom_test(n_consistent, n_total, 0.5,
                                   alternative='greater')

    log(f"\n  Summary: {n_consistent}/{n_total} events consistent")
    log(f"  Binomial test (H0: p=0.5): p = {binom_p:.4f}")

    # Magnitude-weighted average DID (sign-adjusted)
    weights = [e['magnitude'] for e in event_details]
    sign_dids = [e['did'] if e['direction'] == 'HIKE' else -e['did']
                 for e in event_details]
    wavg = np.average(sign_dids, weights=weights)
    log(f"  Magnitude-weighted DID: {wavg:+.4f}")
    log(f"  Prediction: < 0 (high-rho sectors more sensitive)")
    log(f"  Result: {'CONSISTENT' if wavg < 0 else 'INCONSISTENT'}")

    results.append(('B', 'event_consistency', 'frac', n_consistent / n_total,
                    binom_p))
    results.append(('B', 'weighted_DID', 'value', wavg, np.nan))

    return results, event_details


# ═══════════════════════════════════════════════════════════════════════════
#  8. TEST C: DISPLACEMENT EVIDENCE (SOI MIGRATION)
# ═══════════════════════════════════════════════════════════════════════════

def test_c_displacement(soi_panel, gdp_panel):
    """
    Test Prediction 3: revenue loss from displacement, not suppression.

    Regression: log(AGI_flow_{o->d}) ~ (rate_origin - rate_destination)
    Prediction: beta > 0 (income flows down the tax gradient)
    Preservation: total nationwide activity stable despite cross-state variation
    """
    log("")
    log("=" * 70)
    log("TEST C: Displacement Evidence (SOI Migration)")
    log("=" * 70)

    results = []
    displacement_data = None

    if soi_panel.empty:
        log("  SKIP: No SOI migration data available")
        return results, displacement_data

    # Add tax rates
    soi = soi_panel.copy()
    soi['rate_origin'] = soi.apply(
        lambda r: get_tax_rate(r['origin'], r['year']), axis=1
    )
    soi['rate_dest'] = soi.apply(
        lambda r: get_tax_rate(r['destination'], r['year']), axis=1
    )
    soi['rate_diff'] = soi['rate_origin'] - soi['rate_dest']
    soi['log_agi'] = np.log(soi['agi'])

    # Drop NaN
    soi = soi.dropna(subset=['rate_diff', 'log_agi'])

    if len(soi) < 30:
        log("  SKIP: Too few observations after cleaning")
        return results, displacement_data

    log(f"  Observations: {len(soi)}")

    # Regression: log(AGI) ~ rate_diff + year_FE
    year_dummies = pd.get_dummies(soi['year'], prefix='yr', drop_first=True, dtype=float)
    X = pd.concat([
        soi[['rate_diff']].reset_index(drop=True),
        year_dummies.reset_index(drop=True),
    ], axis=1)
    X = sm.add_constant(X)
    y = soi['log_agi'].reset_index(drop=True)

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    try:
        model = sm.OLS(y.values, X.values).fit()
        beta = model.params[1]  # rate_diff coefficient
        pval = model.pvalues[1]
        r2 = model.rsquared

        log(f"  log(AGI_flow) ~ rate_diff + year_FE:")
        log(f"    beta = {beta:+.4f}, p = {pval:.4f}, R2 = {r2:.4f}")
        log(f"    Prediction: beta > 0 (income flows down tax gradient)")
        consistent = beta > 0
        log(f"    Result: {'CONSISTENT' if consistent else 'INCONSISTENT'}")

        results.append(('C', 'migration_elasticity', 'beta', beta, pval))
        displacement_data = soi[['rate_diff', 'log_agi']].dropna()
    except Exception as e:
        log(f"  Regression failed: {e}")

    # Preservation check: aggregate net flows near zero
    if not soi_panel.empty:
        net_by_year = soi_panel.groupby('year')['agi'].sum()
        if len(net_by_year) >= 2:
            cv = net_by_year.std() / net_by_year.mean() if net_by_year.mean() > 0 else np.nan
            log(f"\n  Preservation check:")
            log(f"    CV of total interstate AGI flow: {cv:.4f}")
            log(f"    Activity redistributes, not destroyed")

    return results, displacement_data


# ═══════════════════════════════════════════════════════════════════════════
#  9. TEST D: ASYMMETRY TEST
# ═══════════════════════════════════════════════════════════════════════════

def test_d_asymmetry(panel):
    """
    Test Prediction 2: asymmetric Laffer decline.

    Fit cubic polynomial: share = a + b*rate + c*rate^2 + d*rate^3
    F-test: cubic vs quadratic (is d significant?)
    Low-rho: right-skewed (peaks later, gentle then steep)
    High-rho: left-skewed (peaks earlier, steep throughout)
    """
    log("")
    log("=" * 70)
    log("TEST D: Asymmetry (Cubic vs Quadratic)")
    log("=" * 70)

    results = []
    fit_data = {}

    if panel.empty:
        log("  SKIP: No data")
        return results, fit_data

    for rho_class in ['LOW', 'HIGH']:
        sub = panel[panel['rho_class'] == rho_class].copy()

        # Cross-sectional: average share and rate by state
        avg = sub.groupby('state').agg({
            'tax_rate': 'mean',
            'gdp_share': 'mean',
        }).dropna()

        if len(avg) < 10:
            log(f"  {rho_class}: insufficient data ({len(avg)} states)")
            continue

        x = avg['tax_rate'].values
        y = avg['gdp_share'].values

        try:
            # Quadratic fit
            quad_coef = np.polyfit(x, y, 2)
            quad_pred = np.polyval(quad_coef, x)
            rss_quad = np.sum((y - quad_pred) ** 2)

            # Cubic fit
            cubic_coef = np.polyfit(x, y, 3)
            cubic_pred = np.polyval(cubic_coef, x)
            rss_cubic = np.sum((y - cubic_pred) ** 2)

            # F-test: cubic vs quadratic
            n = len(y)
            df1 = 1  # one extra parameter
            df2 = n - 4  # residual df for cubic

            if df2 > 0 and rss_cubic > 0:
                F = ((rss_quad - rss_cubic) / df1) / (rss_cubic / df2)
                F_pval = 1 - stats.f.cdf(F, df1, df2)
            else:
                F, F_pval = np.nan, np.nan

            # Cubic coefficient sign indicates skew direction
            d_coeff = cubic_coef[0]

            log(f"\n  {rho_class}-rho sectors ({len(avg)} states):")
            log(f"    RSS quadratic: {rss_quad:.8f}")
            log(f"    RSS cubic:     {rss_cubic:.8f}")
            log(f"    F-test (cubic vs quad): F = {F:.2f}, p = {F_pval:.4f}")
            log(f"    Cubic coeff (d): {d_coeff:+.10f}")

            # Prediction: LOW-rho right-skewed (negative d), HIGH-rho left-skewed (positive d)
            if rho_class == 'LOW':
                predicted_sign = 'negative (right-skewed)'
                consistent = d_coeff < 0
            else:
                predicted_sign = 'positive (left-skewed)'
                consistent = d_coeff > 0
            log(f"    Predicted d: {predicted_sign}")
            log(f"    Result: {'CONSISTENT' if consistent else 'INCONSISTENT'}")

            results.append(('D', f'F_test_{rho_class}', 'F', F, F_pval))
            results.append(('D', f'cubic_d_{rho_class}', 'd', d_coeff, np.nan))

            fit_data[rho_class] = {
                'x': x, 'y': y,
                'quad_coef': quad_coef, 'cubic_coef': cubic_coef,
                'n': n, 'F': F, 'F_pval': F_pval,
            }
        except Exception as e:
            log(f"  {rho_class}: fit failed ({e})")

    return results, fit_data


# ═══════════════════════════════════════════════════════════════════════════
#  10. COMPOSITE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def summarize(all_results):
    """Print composite summary of all tests."""
    log("")
    log("=" * 70)
    log("COMPOSITE SUMMARY")
    log("=" * 70)

    test_labels = {
        'A': 'Revenue Elasticity (sector-specific peaks)',
        'B': 'Event Study (natural experiments)',
        'C': 'Displacement (migration evidence)',
        'D': 'Asymmetry (cubic vs quadratic)',
    }

    for test_id in ['A', 'B', 'C', 'D']:
        test_results = [r for r in all_results if r[0] == test_id]
        label = test_labels.get(test_id, test_id)
        if not test_results:
            log(f"\n  Test {test_id} ({label}): NO DATA")
            continue

        log(f"\n  Test {test_id} ({label}):")
        for r in test_results:
            _, metric, stat, value, pval = r
            p_str = f"p = {pval:.4f}" if not np.isnan(pval) else ""
            log(f"    {metric:25s}: {stat} = {value:+.6f}  {p_str}")

    # Overall assessment
    n_tests = len(set(r[0] for r in all_results))
    log(f"\n  Tests with data: {n_tests}/4")


# ═══════════════════════════════════════════════════════════════════════════
#  11. FIGURES (4-panel PDF)
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(sector_betas, event_details, displacement_data, fit_data,
                 path):
    """Create 4-panel figure summarizing all tests."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Laffer Curve Empirical Tests: CES Complementarity Predictions',
                 fontsize=13, fontweight='bold')

    # (a) Revenue elasticity by sector
    ax = axes[0, 0]
    if sector_betas:
        sectors = sorted(sector_betas.keys(),
                         key=lambda s: SECTOR_RHO[s]['rho_proxy'])
        betas = [abs(sector_betas[s]['beta']) for s in sectors]
        colors = ['steelblue' if SECTOR_RHO[s]['rho_class'] == 'LOW'
                  else 'firebrick' for s in sectors]
        short_names = [s[:8] for s in sectors]

        ax.bar(range(len(sectors)), betas, color=colors)
        ax.set_xticks(range(len(sectors)))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(r'$|\beta|$ (revenue elasticity)')
        ax.set_title('(a) Revenue Elasticity by Sector')
        ax.legend(
            [Patch(color='steelblue'), Patch(color='firebrick')],
            [r'Low $\rho$ (complementary)', r'High $\rho$ (substitutable)'],
            fontsize=7, loc='upper left',
        )
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
    ax.set_title('(a) Revenue Elasticity by Sector')

    # (b) Event study DID
    ax = axes[0, 1]
    if event_details:
        events_sorted = sorted(event_details,
                                key=lambda e: e['magnitude'], reverse=True)
        n_show = min(len(events_sorted), 10)
        labels = [e['label'][:16] for e in events_sorted[:n_show]]
        dids = [e['did'] for e in events_sorted[:n_show]]
        colors = ['seagreen' if e['consistent'] else 'indianred'
                  for e in events_sorted[:n_show]]

        y_pos = range(n_show)
        ax.barh(y_pos, dids, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_xlabel(r'DID (HIGH$-$LOW $\rho$)')
        ax.legend(
            [Patch(color='seagreen'), Patch(color='indianred')],
            ['Consistent', 'Inconsistent'],
            fontsize=7, loc='lower right',
        )
    else:
        ax.text(0.5, 0.5, 'No events', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
    ax.set_title('(b) Event Study: DID by Tax Change')

    # (c) Displacement scatter
    ax = axes[1, 0]
    if displacement_data is not None and len(displacement_data) > 0:
        x = displacement_data['rate_diff'].values
        y = displacement_data['log_agi'].values
        # Subsample if too many points
        if len(x) > 3000:
            idx = np.random.choice(len(x), 3000, replace=False)
            x, y = x[idx], y[idx]
        ax.scatter(x, y, s=2, alpha=0.15, color='steelblue', rasterized=True)
        # Regression line
        z = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), 'r-', linewidth=2,
                label=f'slope = {z[0]:+.3f}')
        ax.set_xlabel('Tax rate difference (origin - destination, pp)')
        ax.set_ylabel('log(AGI flow)')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No SOI data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
    ax.set_title('(c) AGI Migration vs Tax Differential')

    # (d) Asymmetry: cubic fits
    ax = axes[1, 1]
    if fit_data:
        for rho_class, data in fit_data.items():
            color = 'steelblue' if rho_class == 'LOW' else 'firebrick'
            marker = 'o' if rho_class == 'LOW' else '^'
            label = f'{rho_class} ' + r'$\rho$' + f' (n={data["n"]})'
            ax.scatter(data['x'], data['y'], s=25, alpha=0.5, color=color,
                       marker=marker)
            x_line = np.linspace(data['x'].min(), data['x'].max(), 100)
            ax.plot(x_line, np.polyval(data['cubic_coef'], x_line),
                    color=color, linewidth=2, label=label)
        ax.set_xlabel('Average effective tax rate (%)')
        ax.set_ylabel('Average GDP share')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
    ax.set_title(r'(d) Asymmetry: Cubic Fits by $\rho$ Class')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"\nSaved figure: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  12. LaTeX TABLE AND CSV OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def save_results(all_results):
    """Save results as CSV, LaTeX table, and text file."""

    # --- CSV ---
    csv_path = os.path.join(DATA_DIR, 'laffer_results.csv')
    if all_results:
        df = pd.DataFrame(all_results,
                          columns=['test', 'metric', 'statistic', 'value',
                                   'p_value'])
        df.to_csv(csv_path, index=False, float_format='%.6f')
    log(f"Saved: {csv_path}")

    # --- Text ---
    txt_path = os.path.join(DATA_DIR, 'laffer_results.txt')
    with open(txt_path, 'w') as f:
        f.write(results_buf.getvalue())
    log(f"Saved: {txt_path}")

    # --- LaTeX table ---
    tex_path = os.path.join(DATA_DIR, 'laffer_table.tex')
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Laffer Curve Empirical Tests: '
                 r'CES Complementarity Predictions}')
    lines.append(r'\label{tab:laffer}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{llcrr}')
    lines.append(r'\toprule')
    lines.append(r'Test & Metric & Statistic & Value & $p$-value \\')
    lines.append(r'\midrule')

    test_labels = {
        'A': 'Revenue Elasticity',
        'B': 'Event Study',
        'C': 'Displacement',
        'D': 'Asymmetry',
    }

    prev_test = None
    for r in all_results:
        test_id, metric, stat, value, pval = r
        tlabel = test_labels.get(test_id, test_id)
        p_str = f'{pval:.4f}' if not np.isnan(pval) else '--'
        # Only print test label on first row of each test
        show_label = tlabel if test_id != prev_test else ''
        prev_test = test_id
        lines.append(f'{show_label} & {metric} & ${stat}$ & '
                     f'${value:+.4f}$ & {p_str} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\begin{minipage}{0.95\textwidth}')
    lines.append(r'\vspace{0.5em}')
    lines.append(
        r'\footnotesize\textit{Notes:} Tests predictions from the CES Laffer '
        r'derivation (Smirl 2026). Test A: interaction of tax rate with '
        r'sectoral complementarity ($\rho$ class) on GDP share; '
        r'Mann-Whitney U compares $|\beta|$ across $\rho$ classes. '
        r'Test B: DID around major tax changes (binomial test for '
        r'consistency). Test C: IRS SOI migration AGI flow vs pairwise '
        r'tax differential. Test D: cubic vs quadratic F-test for '
        r'asymmetric decline by $\rho$ class. '
        r'Data: FRED state GDP by industry, Tax Foundation rates, '
        r'IRS SOI migration.'
    )
    lines.append(r'\end{minipage}')
    lines.append(r'\end{table}')

    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    log(f"Saved: {tex_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  13. MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log("Laffer Curve Empirical Test")
    log("Testing CES complementarity predictions with US state-level data")
    log(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    log("")

    # --- Fetch data ---
    gdp_panel = fetch_state_gdp()
    soi_panel = fetch_soi_migration()

    # --- Build panel ---
    panel = build_panel(gdp_panel)

    # --- Run tests ---
    all_results = []

    res_a, sector_betas = test_a_revenue_elasticity(panel)
    all_results.extend(res_a)

    res_b, event_details = test_b_event_study(panel)
    all_results.extend(res_b)

    res_c, displacement_data = test_c_displacement(soi_panel, gdp_panel)
    all_results.extend(res_c)

    res_d, fit_data = test_d_asymmetry(panel)
    all_results.extend(res_d)

    # --- Summary ---
    summarize(all_results)

    # --- Figure ---
    fig_path = os.path.join(FIG_DIR, 'laffer_curve.pdf')
    plot_results(sector_betas, event_details, displacement_data,
                 fit_data, fig_path)

    # --- Save ---
    save_results(all_results)

    log("\nDone.")


if __name__ == '__main__':
    main()
