#!/usr/bin/env python3
"""
test_labor_matching.py — Empirical test of CES free energy prediction for labor markets

PREDICTION (from CES free energy framework):
    Industries with lower rho (more complementary worker-firm matching) should have:
    1. Lower matching efficiency (longer search duration per vacancy-unemployment pair)
    2. Steeper Beveridge curves (vacancy-unemployment tradeoff)

    rho is proxied by the INVERSE of an O*NET-based "task complementarity index":
    high teamwork/coordination/interdependence => low rho => workers are hard to substitute.

DATA SOURCES:
    1. JOLTS from FRED API: monthly job openings (vacancies) and hires by industry
    2. CPS from FRED API: unemployment level by industry
    3. O*NET: Work Context + Work Activities for teamwork/coordination scores
    4. OES from BLS: occupation x industry employment crosswalk (SOC -> NAICS -> JOLTS)

METHODOLOGY:
    Stage 1: Estimate industry-level matching functions
        log(H_it) = alpha_i + beta_V * log(V_it) + beta_U * log(U_it) + epsilon_it
    Stage 2: Cross-industry regression
        alpha_i = gamma_0 + gamma_1 * complementarity_i + epsilon_i
        PREDICTION: gamma_1 < 0

Author: Connor Doll / Smirl (2026)
"""

import os
import sys
import json
import time
import io
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

warnings.filterwarnings('ignore', category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

FRED_API_KEY = os.environ.get('FRED_API_KEY', '')
if not FRED_API_KEY:
    print("ERROR: FRED_API_KEY not set. Run: source /home/jonsmirl/thesis/env.sh")
    sys.exit(1)

BASE_DIR = Path('/home/jonsmirl/thesis')
DATA_DIR = BASE_DIR / 'thesis_data'
FIG_DIR = BASE_DIR / 'figures'
CACHE_DIR = DATA_DIR / 'labor_matching_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# JOLTS observation period
OBS_START = '2001-01-01'  # JOLTS starts Dec 2000
OBS_END = '2025-12-01'

# ──────────────────────────────────────────────────────────────────────────────
# Section 1: Industry definitions and FRED series mapping
# ──────────────────────────────────────────────────────────────────────────────

# Each industry maps to: (JOLTS openings, JOLTS hires, Unemployment level, short_name, NAICS prefixes for OES)
# Note: JOLTS uses JTU/JTS prefixes; CPS unemployment uses LNU03 (level) series
# For Trade/Transport/Utilities JOLTS, we combine Wholesale+Retail trade with Transport+Utilities unemployment

INDUSTRIES = {
    'Mining': {
        'jol': 'JTU110099JOL',
        'hil': 'JTU110099HIL',
        'ul':  'LNU03032230',   # Mining unemployment level
        'naics_prefixes': ['21'],  # Mining, Quarrying, Oil & Gas
        'short': 'Mining',
    },
    'Construction': {
        'jol': 'JTS2300JOL',
        'hil': 'JTS2300HIL',
        'ul':  'LNU03032231',
        'naics_prefixes': ['23'],
        'short': 'Construction',
    },
    'Manufacturing': {
        'jol': 'JTS3000JOL',
        'hil': 'JTS3000HIL',
        'ul':  'LNU03032232',
        'naics_prefixes': ['31', '32', '33'],
        'short': 'Manufacturing',
    },
    'Trade_Transport_Util': {
        'jol': 'JTS4000JOL',
        'hil': 'JTS4000HIL',
        'ul':  'LNU03032235',  # Wholesale+Retail trade unemployment
        # We also need LNU03032236 (Transport/Utilities) — will sum both
        'ul2': 'LNU03032236',
        'naics_prefixes': ['42', '44', '45', '48', '49', '22'],  # Wholesale, Retail, Transport, Utilities
        'short': 'TTU',
    },
    'Information': {
        'jol': 'JTU5100JOL',
        'hil': 'JTU5100HIL',
        'ul':  'LNU03032237',
        'naics_prefixes': ['51'],
        'short': 'Information',
    },
    'Financial': {
        'jol': 'JTU510099JOL',
        'hil': 'JTU510099HIL',
        'ul':  'LNU03032238',
        'naics_prefixes': ['52', '53'],  # Finance+Insurance, Real Estate
        'short': 'Financial',
    },
    'Prof_Business': {
        'jol': 'JTS540099JOL',
        'hil': 'JTS540099HIL',
        'ul':  'LNU03032239',
        'naics_prefixes': ['54', '55', '56'],  # Prof/Sci/Tech, Management, Admin/Support
        'short': 'ProfBiz',
    },
    'Edu_Health': {
        'jol': 'JTS6000JOL',
        'hil': 'JTS6000HIL',
        'ul':  'LNU03032240',
        'naics_prefixes': ['61', '62'],  # Education, Healthcare
        'short': 'EduHealth',
    },
    'Leisure_Hospitality': {
        'jol': 'JTS7000JOL',
        'hil': 'JTS7000HIL',
        'ul':  'LNU03032241',
        'naics_prefixes': ['71', '72'],  # Arts/Entertainment, Accommodation/Food
        'short': 'Leisure',
    },
    'Government': {
        'jol': 'JTS9000JOL',
        'hil': 'JTS9000HIL',
        'ul':  None,  # No CPS government unemployment series available
        'naics_prefixes': ['92', '99'],  # Public Administration (will use aggregate if needed)
        'short': 'Govt',
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Section 2: FRED API data fetching
# ──────────────────────────────────────────────────────────────────────────────

def fetch_fred_series(series_id, start=OBS_START, end=OBS_END):
    """Fetch a single FRED series, with caching."""
    cache_file = CACHE_DIR / f'fred_{series_id}.csv'
    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=['date'])
        return df

    url = 'https://api.stlouisfed.org/fred/series/observations'
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'observation_start': start,
        'observation_end': end,
    }

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if 'observations' not in data:
                print(f"  WARNING: No observations for {series_id}: {data.get('error_message', '')}")
                return None
            obs = data['observations']
            df = pd.DataFrame(obs)[['date', 'value']]
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            df.to_csv(cache_file, index=False)
            return df
        except Exception as e:
            print(f"  Retry {attempt+1}/3 for {series_id}: {e}")
            time.sleep(2)

    print(f"  ERROR: Failed to fetch {series_id}")
    return None


def fetch_all_jolts_and_unemployment():
    """Fetch all JOLTS and unemployment series for all industries."""
    print("\n=== Section 2: Fetching JOLTS and CPS unemployment data from FRED ===")

    industry_data = {}

    for name, info in INDUSTRIES.items():
        print(f"\n  Fetching {name}...")

        # Job openings
        jol_df = fetch_fred_series(info['jol'])
        if jol_df is not None:
            jol_df = jol_df.rename(columns={'value': 'vacancies'})
            print(f"    Job openings: {len(jol_df)} obs")
        else:
            print(f"    WARNING: No job openings data for {name}")
            continue

        # Hires
        hil_df = fetch_fred_series(info['hil'])
        if hil_df is not None:
            hil_df = hil_df.rename(columns={'value': 'hires'})
            print(f"    Hires: {len(hil_df)} obs")
        else:
            print(f"    WARNING: No hires data for {name}")
            continue

        # Unemployment
        if info.get('ul') is None:
            print(f"    Skipping {name} — no unemployment series")
            continue

        ul_df = fetch_fred_series(info['ul'])
        if ul_df is not None:
            ul_df = ul_df.rename(columns={'value': 'unemployment'})
            print(f"    Unemployment: {len(ul_df)} obs")
        else:
            print(f"    WARNING: No unemployment data for {name}")
            continue

        # For Trade/Transport/Utilities: sum two unemployment series
        if 'ul2' in info and info['ul2']:
            ul2_df = fetch_fred_series(info['ul2'])
            if ul2_df is not None:
                ul2_df = ul2_df.rename(columns={'value': 'unemployment2'})
                # Merge and sum
                ul_merged = pd.merge(ul_df, ul2_df, on='date', how='outer')
                ul_merged['unemployment'] = ul_merged['unemployment'].fillna(0) + ul_merged['unemployment2'].fillna(0)
                ul_df = ul_merged[['date', 'unemployment']].dropna()
                print(f"    Combined TTU unemployment: {len(ul_df)} obs")

        # Merge all three
        merged = pd.merge(jol_df, hil_df, on='date', how='inner')
        merged = pd.merge(merged, ul_df, on='date', how='inner')

        # Drop zeros (can't take log)
        merged = merged[(merged['vacancies'] > 0) &
                        (merged['hires'] > 0) &
                        (merged['unemployment'] > 0)]

        industry_data[name] = merged
        print(f"    Merged panel: {len(merged)} obs, {merged['date'].min().strftime('%Y-%m')} to {merged['date'].max().strftime('%Y-%m')}")

    return industry_data


# ──────────────────────────────────────────────────────────────────────────────
# Section 3: O*NET complementarity index construction
# ──────────────────────────────────────────────────────────────────────────────

def download_onet_file(filename):
    """Download an O*NET database file with caching."""
    cache_file = CACHE_DIR / f'onet_{filename.replace(" ", "_").replace("/", "_")}'
    if cache_file.exists():
        return pd.read_csv(cache_file, sep='\t')

    url = f'https://www.onetcenter.org/dl_files/database/db_29_1_text/{requests.utils.quote(filename)}'
    print(f"    Downloading O*NET: {filename}...")
    r = requests.get(url, timeout=60, headers={'User-Agent': 'Mozilla/5.0 (thesis research)'})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep='\t')
    df.to_csv(cache_file, sep='\t', index=False)
    return df


def build_onet_complementarity_index():
    """
    Build a task complementarity index per O*NET-SOC occupation.

    Higher index = more teamwork, coordination, interdependence = LOWER rho (more complementary).

    Components (all from O*NET 29.1):

    Work Context (scale CX = context, 1-5 range):
        4.C.1.a.4   Contact With Others
        4.C.1.b.1.e Work With Work Group or Team
        4.C.1.b.1.g Coordinate or Lead Others
        4.C.1.a.2.l Face-to-Face Discussions

    Work Activities (scale IM = importance, 1-5 range):
        4.A.4.a.2   Communicating with Supervisors, Peers, or Subordinates
        4.A.4.b.1   Coordinating the Work and Activities of Others
        4.A.4.b.2   Developing and Building Teams
        4.A.4.b.4   Guiding, Directing, and Motivating Subordinates

    We average these 8 items (standardized to 0-1 scale) to get the index.
    """
    print("\n=== Section 3: Building O*NET complementarity index ===")

    # Work Context items
    wc_df = download_onet_file('Work Context.txt')
    wc_elements = {
        '4.C.1.a.4':   'Contact With Others',
        '4.C.1.b.1.e': 'Work With Work Group or Team',
        '4.C.1.b.1.g': 'Coordinate or Lead Others',
        '4.C.1.a.2.l': 'Face-to-Face Discussions',
    }

    # Filter to CX scale (context level, 1-5) and our elements
    wc_scores = wc_df[
        (wc_df['Scale ID'] == 'CX') &
        (wc_df['Element ID'].isin(wc_elements.keys())) &
        (wc_df['Recommend Suppress'] != 'Y')
    ][['O*NET-SOC Code', 'Element ID', 'Data Value']].copy()
    wc_scores['Data Value'] = pd.to_numeric(wc_scores['Data Value'], errors='coerce')
    # Normalize to 0-1 (CX scale is 1-5)
    wc_scores['norm_value'] = (wc_scores['Data Value'] - 1) / 4
    print(f"  Work Context items: {len(wc_scores)} rows, {wc_scores['O*NET-SOC Code'].nunique()} occupations")

    # Work Activities items
    wa_df = download_onet_file('Work Activities.txt')
    wa_elements = {
        '4.A.4.a.2': 'Communicating with Supervisors, Peers, or Subordinates',
        '4.A.4.b.1': 'Coordinating the Work and Activities of Others',
        '4.A.4.b.2': 'Developing and Building Teams',
        '4.A.4.b.4': 'Guiding, Directing, and Motivating Subordinates',
    }

    wa_scores = wa_df[
        (wa_df['Scale ID'] == 'IM') &
        (wa_df['Element ID'].isin(wa_elements.keys())) &
        (wa_df['Recommend Suppress'] != 'Y')
    ][['O*NET-SOC Code', 'Element ID', 'Data Value']].copy()
    wa_scores['Data Value'] = pd.to_numeric(wa_scores['Data Value'], errors='coerce')
    # Normalize to 0-1 (IM scale is 1-5)
    wa_scores['norm_value'] = (wa_scores['Data Value'] - 1) / 4
    print(f"  Work Activities items: {len(wa_scores)} rows, {wa_scores['O*NET-SOC Code'].nunique()} occupations")

    # Combine
    all_scores = pd.concat([
        wc_scores[['O*NET-SOC Code', 'Element ID', 'norm_value']],
        wa_scores[['O*NET-SOC Code', 'Element ID', 'norm_value']],
    ])

    # Average across all 8 items per occupation
    occ_index = all_scores.groupby('O*NET-SOC Code')['norm_value'].mean().reset_index()
    occ_index.columns = ['onet_soc', 'complementarity_index']

    # Extract 6-digit SOC code from O*NET-SOC (format: XX-XXXX.XX -> XX-XXXX)
    occ_index['soc_6'] = occ_index['onet_soc'].str[:7]

    print(f"  Complementarity index: {len(occ_index)} O*NET occupations")
    print(f"  Range: {occ_index['complementarity_index'].min():.3f} - {occ_index['complementarity_index'].max():.3f}")
    print(f"  Mean: {occ_index['complementarity_index'].mean():.3f}, Std: {occ_index['complementarity_index'].std():.3f}")

    return occ_index


# ──────────────────────────────────────────────────────────────────────────────
# Section 4: OES crosswalk — aggregate O*NET scores to JOLTS industries
# ──────────────────────────────────────────────────────────────────────────────

def download_oes_crosswalk():
    """Download OES sector-level data for occupation x industry employment weights.

    Uses oesm23in4.zip which contains natsector_M2023_dl.xlsx — the sector-level
    (2-digit NAICS) occupation x industry employment matrix.
    """
    cache_file = CACHE_DIR / 'oes_natsector_2023.csv'
    if cache_file.exists():
        return pd.read_csv(cache_file)

    zip_cache = CACHE_DIR / 'oesm23in4.zip'
    if not zip_cache.exists():
        print("    Downloading OES sector-level industry-occupation data (31 MB)...")
        url = 'https://www.bls.gov/oes/special-requests/oesm23in4.zip'
        r = requests.get(url, timeout=180, headers={'User-Agent': 'Mozilla/5.0 (thesis research)'})
        r.raise_for_status()
        with open(zip_cache, 'wb') as f:
            f.write(r.content)
        print(f"    Downloaded: {len(r.content):,} bytes")
    else:
        print("    Using cached OES ZIP")

    z = zipfile.ZipFile(zip_cache)
    target = 'oesm23in4/natsector_M2023_dl.xlsx'
    print(f"    Reading {target}...")
    with z.open(target) as f:
        df = pd.read_excel(f, engine='openpyxl')

    print(f"    OES sector data: {df.shape[0]} rows, {df['NAICS'].nunique()} sectors")
    df.to_csv(cache_file, index=False)
    return df


def aggregate_complementarity_to_industries(occ_index):
    """
    Aggregate O*NET complementarity index from occupation to JOLTS industry level,
    using OES employment weights from the BLS sector-level staffing patterns.

    The OES sector data has NAICS codes like '21', '23', '31-33', '44-45', '48-49', '51', etc.
    We map these to JOLTS industries (which are roughly 2-digit NAICS supersectors).
    """
    print("\n=== Section 4: Aggregating complementarity index to JOLTS industries ===")

    oes_df = download_oes_crosswalk()
    if oes_df is None:
        print("  WARNING: OES data unavailable, falling back to simple mapping")
        return None

    # Keep only detailed occupations (not summary lines like 00-0000, 11-0000)
    oes_df = oes_df[oes_df['O_GROUP'] == 'detailed'].copy()
    print(f"  Detailed occupation rows: {len(oes_df)}")

    # Clean employment column
    oes_df['TOT_EMP'] = pd.to_numeric(
        oes_df['TOT_EMP'].astype(str).str.replace(',', '').str.replace('**', '', regex=False),
        errors='coerce'
    )

    # Clean SOC code to match O*NET 6-digit format (XX-XXXX)
    oes_df['soc_clean'] = oes_df['OCC_CODE'].astype(str).str.strip()

    # O*NET uses XX-XXXX.XX format; average across sub-occupations to 6-digit SOC
    occ_6d = occ_index.groupby('soc_6')['complementarity_index'].mean().reset_index()

    # Merge OES with O*NET
    merged = pd.merge(oes_df, occ_6d, left_on='soc_clean', right_on='soc_6', how='inner')
    print(f"  Matched {len(merged)} OES-ONET occupation-industry pairs ({merged['soc_clean'].nunique()} unique SOC codes)")

    if len(merged) == 0:
        print("  WARNING: No matches found")
        return None

    # Map OES NAICS sectors to JOLTS industries
    # OES uses sector codes: '11', '21', '22', '23', '31-33', '42', '44-45', '48-49',
    #   '51', '52', '53', '54', '55', '56', '61', '62', '71', '72', '81', '99'
    # JOLTS industries aggregate these into supersectors

    naics_to_jolts = {
        '21': 'Mining',
        '23': 'Construction',
        '31-33': 'Manufacturing',
        '42': 'Trade_Transport_Util',
        '44-45': 'Trade_Transport_Util',
        '48-49': 'Trade_Transport_Util',
        '22': 'Trade_Transport_Util',  # Utilities grouped with TTU in JOLTS
        '51': 'Information',
        '52': 'Financial',
        '53': 'Financial',  # Real Estate grouped with Financial in JOLTS
        '54': 'Prof_Business',
        '55': 'Prof_Business',
        '56': 'Prof_Business',
        '61': 'Edu_Health',
        '62': 'Edu_Health',
        '71': 'Leisure_Hospitality',
        '72': 'Leisure_Hospitality',
        '99': 'Government',
    }

    # Convert NAICS to string for matching
    merged['naics_str'] = merged['NAICS'].astype(str).str.strip()
    merged['jolts_industry'] = merged['naics_str'].map(naics_to_jolts)

    matched = merged.dropna(subset=['jolts_industry'])
    print(f"  Mapped to JOLTS industries: {len(matched)} rows")

    # Aggregate to JOLTS industry level using employment weights
    industry_complementarity = {}

    for ind_name in INDUSTRIES:
        if INDUSTRIES[ind_name].get('ul') is None:
            continue

        ind_data = matched[matched['jolts_industry'] == ind_name].copy()

        if len(ind_data) == 0:
            print(f"  WARNING: No OES data for {ind_name}")
            continue

        # Employment-weighted average
        valid = ind_data.dropna(subset=['TOT_EMP', 'complementarity_index'])
        if len(valid) == 0 or valid['TOT_EMP'].sum() == 0:
            ci = ind_data['complementarity_index'].mean()
            n_occs = len(ind_data)
        else:
            ci = np.average(valid['complementarity_index'], weights=valid['TOT_EMP'])
            n_occs = len(valid)

        industry_complementarity[ind_name] = ci
        print(f"  {ind_name:25s}: complementarity = {ci:.4f} (emp-weighted, {n_occs} occs)")

    return industry_complementarity


def build_fallback_complementarity():
    """
    If OES crosswalk fails, construct industry complementarity from O*NET occupation
    averages using a manual SOC-to-JOLTS industry mapping.
    """
    print("\n  Building fallback complementarity index using SOC major group mapping...")

    # SOC major groups -> approximate JOLTS industry mapping
    # See https://www.bls.gov/soc/2018/major_groups.htm
    soc_to_jolts = {
        # Mining: 47 (some construction overlap)
        'Mining': ['47-5'],  # Mining machine operators etc.
        # Construction: 47 (construction trades)
        'Construction': ['47-1', '47-2', '47-3', '47-4'],
        # Manufacturing: 51 (production), some 17 (engineering)
        'Manufacturing': ['51-'],
        # Trade/Transport/Utilities: 41 (sales), 43 (office), 53 (transport)
        'Trade_Transport_Util': ['41-', '53-'],
        # Information: 15 (computer), 27 (media)
        'Information': ['15-', '27-'],
        # Financial: 13 (business/financial)
        'Financial': ['13-'],
        # Professional/Business: 11 (management), 23 (legal), some 15, 17
        'Prof_Business': ['11-', '23-', '17-'],
        # Education/Health: 21 (community/social), 25 (education), 29 (healthcare)
        'Edu_Health': ['21-', '25-', '29-'],
        # Leisure/Hospitality: 35 (food), 39 (personal care), some 27
        'Leisure_Hospitality': ['35-', '39-'],
    }

    return soc_to_jolts


# ──────────────────────────────────────────────────────────────────────────────
# Section 5: Stage 1 — Estimate industry matching functions
# ──────────────────────────────────────────────────────────────────────────────

def estimate_matching_functions(industry_data):
    """
    For each industry, estimate:
        log(H_it) = alpha_i + beta_V * log(V_it) + beta_U * log(U_it) + epsilon_it

    Also estimate Beveridge curve slope:
        log(V_it) = a_i + b_i * log(U_it) + epsilon_it

    Returns dict of {industry: results_dict}
    """
    print("\n=== Section 5: Estimating industry-level matching functions ===")

    results = {}

    for name, df in industry_data.items():
        if len(df) < 24:
            print(f"  Skipping {name}: only {len(df)} obs")
            continue

        # Take logs
        df = df.copy()
        df['log_H'] = np.log(df['hires'])
        df['log_V'] = np.log(df['vacancies'])
        df['log_U'] = np.log(df['unemployment'])

        # Matching function: log(H) = alpha + beta_V * log(V) + beta_U * log(U)
        X = sm.add_constant(df[['log_V', 'log_U']])
        y = df['log_H']

        try:
            model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

            alpha_i = model.params['const']
            beta_V = model.params['log_V']
            beta_U = model.params['log_U']

            # Matching efficiency: alpha measures TFP of the matching function
            # Higher alpha = more hires for given V,U = more efficient matching

            # Also compute vacancy yield = H/V (simpler efficiency measure)
            vacancy_yield = (df['hires'] / df['vacancies']).mean()

            # Also compute average matching rate: H / (V * U)^0.5
            matching_rate = (df['hires'] / np.sqrt(df['vacancies'] * df['unemployment'])).mean()

            # Beveridge curve: log(V) = a + b * log(U)
            X_bev = sm.add_constant(df['log_U'])
            y_bev = df['log_V']
            bev_model = OLS(y_bev, X_bev).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
            bev_slope = bev_model.params['log_U']

            results[name] = {
                'alpha': alpha_i,
                'alpha_se': model.bse['const'],
                'alpha_pval': model.pvalues['const'],
                'beta_V': beta_V,
                'beta_V_se': model.bse['log_V'],
                'beta_V_pval': model.pvalues['log_V'],
                'beta_U': beta_U,
                'beta_U_se': model.bse['log_U'],
                'beta_U_pval': model.pvalues['log_U'],
                'R2': model.rsquared,
                'nobs': len(df),
                'vacancy_yield': vacancy_yield,
                'matching_rate': matching_rate,
                'beveridge_slope': bev_slope,
                'beveridge_slope_se': bev_model.bse['log_U'],
                'beveridge_R2': bev_model.rsquared,
            }

            sig_V = '***' if model.pvalues['log_V'] < 0.01 else '**' if model.pvalues['log_V'] < 0.05 else '*' if model.pvalues['log_V'] < 0.1 else ''
            sig_U = '***' if model.pvalues['log_U'] < 0.01 else '**' if model.pvalues['log_U'] < 0.05 else '*' if model.pvalues['log_U'] < 0.1 else ''

            print(f"  {name:25s}: alpha={alpha_i:6.3f}, beta_V={beta_V:.3f}{sig_V}, beta_U={beta_U:.3f}{sig_U}, R2={model.rsquared:.3f}, n={len(df)}")

        except Exception as e:
            print(f"  ERROR estimating {name}: {e}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Section 6: Stage 2 — Cross-industry regression
# ──────────────────────────────────────────────────────────────────────────────

def cross_industry_regressions(matching_results, industry_complementarity):
    """
    Regress matching efficiency on complementarity index across industries.

    PREDICTION: gamma_1 < 0 (more complementary => lower matching efficiency)

    Also test: more complementary => steeper Beveridge curve (more negative slope)
    """
    print("\n=== Section 6: Cross-industry regressions ===")

    # Build cross-industry dataset
    rows = []
    for name in matching_results:
        if name not in industry_complementarity:
            print(f"  WARNING: No complementarity index for {name}, skipping")
            continue
        r = matching_results[name]
        rows.append({
            'industry': name,
            'short': INDUSTRIES[name]['short'],
            'complementarity': industry_complementarity[name],
            'alpha': r['alpha'],
            'alpha_se': r['alpha_se'],
            'beta_V': r['beta_V'],
            'beta_U': r['beta_U'],
            'R2_matching': r['R2'],
            'vacancy_yield': r['vacancy_yield'],
            'matching_rate': r['matching_rate'],
            'beveridge_slope': r['beveridge_slope'],
            'beveridge_slope_se': r['beveridge_slope_se'],
            'nobs': r['nobs'],
        })

    cross_df = pd.DataFrame(rows)
    print(f"\n  Cross-industry dataset: {len(cross_df)} industries")
    print(cross_df[['industry', 'complementarity', 'alpha', 'vacancy_yield', 'matching_rate', 'beveridge_slope']].to_string(index=False))

    if len(cross_df) < 4:
        print("  ERROR: Too few industries for cross-industry regression")
        return cross_df, {}

    regression_results = {}

    # ── Test 1: Matching efficiency (alpha) on complementarity ──
    print("\n  --- Test 1: Matching efficiency (alpha) vs complementarity ---")
    X = sm.add_constant(cross_df['complementarity'])
    y = cross_df['alpha']
    model1 = OLS(y, X).fit()
    gamma_1 = model1.params['complementarity']
    p_val = model1.pvalues['complementarity']
    print(f"  alpha_i = {model1.params['const']:.4f} + {gamma_1:.4f} * complementarity_i")
    print(f"  gamma_1 = {gamma_1:.4f} (SE={model1.bse['complementarity']:.4f}, p={p_val:.4f})")
    print(f"  R-squared = {model1.rsquared:.4f}")
    print(f"  PREDICTION (gamma_1 < 0): {'SUPPORTED' if gamma_1 < 0 and p_val < 0.10 else 'SUPPORTED (weak)' if gamma_1 < 0 else 'NOT SUPPORTED'}")
    regression_results['alpha_on_comp'] = {
        'gamma_1': gamma_1, 'p_value': p_val, 'R2': model1.rsquared,
        'model': model1,
    }

    # ── Test 2: Vacancy yield on complementarity ──
    print("\n  --- Test 2: Vacancy yield vs complementarity ---")
    y2 = cross_df['vacancy_yield']
    model2 = OLS(y2, X).fit()
    g2 = model2.params['complementarity']
    p2 = model2.pvalues['complementarity']
    print(f"  VY_i = {model2.params['const']:.4f} + {g2:.4f} * complementarity_i")
    print(f"  gamma_1 = {g2:.4f} (SE={model2.bse['complementarity']:.4f}, p={p2:.4f})")
    print(f"  R-squared = {model2.rsquared:.4f}")
    print(f"  PREDICTION (gamma_1 < 0): {'SUPPORTED' if g2 < 0 and p2 < 0.10 else 'SUPPORTED (weak)' if g2 < 0 else 'NOT SUPPORTED'}")
    regression_results['vy_on_comp'] = {
        'gamma_1': g2, 'p_value': p2, 'R2': model2.rsquared,
        'model': model2,
    }

    # ── Test 3: Matching rate on complementarity ──
    print("\n  --- Test 3: Matching rate (H/sqrt(VU)) vs complementarity ---")
    y3 = cross_df['matching_rate']
    model3 = OLS(y3, X).fit()
    g3 = model3.params['complementarity']
    p3 = model3.pvalues['complementarity']
    print(f"  MR_i = {model3.params['const']:.4f} + {g3:.4f} * complementarity_i")
    print(f"  gamma_1 = {g3:.4f} (SE={model3.bse['complementarity']:.4f}, p={p3:.4f})")
    print(f"  R-squared = {model3.rsquared:.4f}")
    print(f"  PREDICTION (gamma_1 < 0): {'SUPPORTED' if g3 < 0 and p3 < 0.10 else 'SUPPORTED (weak)' if g3 < 0 else 'NOT SUPPORTED'}")
    regression_results['mr_on_comp'] = {
        'gamma_1': g3, 'p_value': p3, 'R2': model3.rsquared,
        'model': model3,
    }

    # ── Test 4: Beveridge slope on complementarity ──
    print("\n  --- Test 4: Beveridge curve slope vs complementarity ---")
    print("  (Prediction: more complementary => more negative slope, i.e., steeper tradeoff)")
    y4 = cross_df['beveridge_slope']
    model4 = OLS(y4, X).fit()
    g4 = model4.params['complementarity']
    p4 = model4.pvalues['complementarity']
    print(f"  BevSlope_i = {model4.params['const']:.4f} + {g4:.4f} * complementarity_i")
    print(f"  gamma_1 = {g4:.4f} (SE={model4.bse['complementarity']:.4f}, p={p4:.4f})")
    print(f"  R-squared = {model4.rsquared:.4f}")
    print(f"  PREDICTION (gamma_1 < 0): {'SUPPORTED' if g4 < 0 and p4 < 0.10 else 'SUPPORTED (weak)' if g4 < 0 else 'NOT SUPPORTED'}")
    regression_results['bev_on_comp'] = {
        'gamma_1': g4, 'p_value': p4, 'R2': model4.rsquared,
        'model': model4,
    }

    # ── Additional: Spearman rank correlations (non-parametric) ──
    print("\n  --- Spearman rank correlations ---")
    for dep_var, label in [('alpha', 'Matching efficiency (alpha)'),
                            ('vacancy_yield', 'Vacancy yield (H/V)'),
                            ('matching_rate', 'Matching rate (H/sqrt(VU))'),
                            ('beveridge_slope', 'Beveridge slope')]:
        rho_s, p_s = stats.spearmanr(cross_df['complementarity'], cross_df[dep_var])
        print(f"  {label:40s}: rho_S = {rho_s:+.4f} (p = {p_s:.4f})")

    return cross_df, regression_results


# ──────────────────────────────────────────────────────────────────────────────
# Section 7: Figures
# ──────────────────────────────────────────────────────────────────────────────

def create_figures(cross_df, regression_results, industry_data, matching_results):
    """Generate publication-quality figures."""
    print("\n=== Section 7: Creating figures ===")

    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    # ── Figure 1: 4-panel scatter (alpha, VY, MR, Beveridge) vs complementarity ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    panels = [
        ('alpha', 'Matching efficiency ($\\alpha_i$)',
         'alpha_on_comp', 'lower matching\nefficiency', False),
        ('vacancy_yield', 'Vacancy yield ($H/V$)',
         'vy_on_comp', 'lower vacancy\nyield', False),
        ('matching_rate', 'Matching rate ($H/\\sqrt{VU}$)',
         'mr_on_comp', 'lower matching\nrate', False),
        ('beveridge_slope', 'Beveridge curve slope',
         'bev_on_comp', 'steeper\nBeveridge curve', False),
    ]

    for ax, (yvar, ylabel, regkey, pred_label, invert) in zip(axes.flat, panels):
        x = cross_df['complementarity']
        y = cross_df[yvar]

        # Scatter with labels
        ax.scatter(x, y, s=80, c='steelblue', edgecolors='navy', alpha=0.8, zorder=5)
        for _, row in cross_df.iterrows():
            ax.annotate(row['short'], (row['complementarity'], row[yvar]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=9, color='black')

        # OLS fit line
        if regkey in regression_results:
            model = regression_results[regkey]['model']
            x_line = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
            y_line = model.params['const'] + model.params['complementarity'] * x_line
            pval = regression_results[regkey]['p_value']
            gamma = regression_results[regkey]['gamma_1']
            R2 = regression_results[regkey]['R2']

            color = 'darkred' if pval < 0.10 else 'gray'
            ls = '-' if pval < 0.10 else '--'
            ax.plot(x_line, y_line, color=color, linestyle=ls, linewidth=2, zorder=3)

            # Annotation
            sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
            ax.text(0.05, 0.95,
                    f'$\\gamma_1$ = {gamma:.3f}{sig}\n$p$ = {pval:.3f}\n$R^2$ = {R2:.3f}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

        ax.set_xlabel('Task complementarity index (O*NET)')
        ax.set_ylabel(ylabel)

        # Prediction arrow
        ax.annotate(f'Prediction:\n{pred_label}',
                    xy=(0.95, 0.05), xycoords='axes fraction',
                    fontsize=8, ha='right', va='bottom', color='darkred', style='italic')

    fig.suptitle('CES Free Energy Prediction: Labor Market Matching vs Task Complementarity',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path = FIG_DIR / 'labor_matching_test.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'labor_matching_test.pdf', bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close(fig)

    # ── Figure 2: Industry Beveridge curves (V-U space) ──
    fig2, ax2 = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(industry_data)))
    sorted_industries = sorted(industry_data.keys(),
                               key=lambda x: matching_results.get(x, {}).get('beveridge_slope', 0))

    for idx, name in enumerate(sorted_industries):
        if name not in matching_results:
            continue
        df = industry_data[name]
        short = INDUSTRIES[name]['short']
        bslope = matching_results[name]['beveridge_slope']
        ax2.scatter(np.log(df['unemployment']), np.log(df['vacancies']),
                    s=8, alpha=0.3, color=colors[idx], label=f'{short} (b={bslope:.2f})')

        # Add regression line
        x_range = np.linspace(np.log(df['unemployment']).min(), np.log(df['unemployment']).max(), 50)
        bev_intercept = np.log(df['vacancies']).mean() - bslope * np.log(df['unemployment']).mean()
        ax2.plot(x_range, bev_intercept + bslope * x_range,
                 color=colors[idx], linewidth=1.5, alpha=0.8)

    ax2.set_xlabel('log(Unemployment)')
    ax2.set_ylabel('log(Vacancies)')
    ax2.set_title('Industry Beveridge Curves (2001-2025)')
    ax2.legend(fontsize=8, loc='upper right')
    plt.tight_layout()

    fig2_path = FIG_DIR / 'labor_beveridge_curves.png'
    fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
    fig2.savefig(FIG_DIR / 'labor_beveridge_curves.pdf', bbox_inches='tight')
    print(f"  Saved: {fig2_path}")
    plt.close(fig2)

    # ── Figure 3: Complementarity index distribution by industry ──
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    sorted_df = cross_df.sort_values('complementarity')
    colors3 = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(sorted_df)))

    bars = ax3.barh(range(len(sorted_df)), sorted_df['complementarity'],
                    color=colors3, edgecolor='gray', linewidth=0.5)
    ax3.set_yticks(range(len(sorted_df)))
    ax3.set_yticklabels(sorted_df['short'])
    ax3.set_xlabel('Task Complementarity Index (higher = more complementary = lower $\\rho$)')
    ax3.set_title('O*NET Task Complementarity by JOLTS Industry\n(Employment-weighted average of teamwork/coordination scores)')

    # Add value labels
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        ax3.text(row['complementarity'] + 0.003, i, f"{row['complementarity']:.3f}",
                 va='center', fontsize=9)

    plt.tight_layout()
    fig3_path = FIG_DIR / 'labor_complementarity_index.png'
    fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
    fig3.savefig(FIG_DIR / 'labor_complementarity_index.pdf', bbox_inches='tight')
    print(f"  Saved: {fig3_path}")
    plt.close(fig3)


# ──────────────────────────────────────────────────────────────────────────────
# Section 8: Save results
# ──────────────────────────────────────────────────────────────────────────────

def save_results(cross_df, matching_results, regression_results, industry_complementarity):
    """Save all results to thesis_data/."""
    print("\n=== Section 8: Saving results ===")

    # Cross-industry table
    out_path = DATA_DIR / 'labor_matching_cross_industry.csv'
    cross_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    # Matching function estimates
    match_rows = []
    for name, r in matching_results.items():
        row = {'industry': name, 'short': INDUSTRIES[name]['short']}
        row.update(r)
        match_rows.append(row)
    match_df = pd.DataFrame(match_rows)
    out_path2 = DATA_DIR / 'labor_matching_estimates.csv'
    match_df.to_csv(out_path2, index=False)
    print(f"  Saved: {out_path2}")

    # Regression results summary
    out_path3 = DATA_DIR / 'labor_matching_regression_table.tex'
    with open(out_path3, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Cross-Industry Regression: Matching Outcomes on Task Complementarity}\n")
        f.write("\\label{tab:labor_matching}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\\hline\n")
        f.write(" & (1) & (2) & (3) & (4) \\\\\n")
        f.write("Dep. Var. & $\\alpha_i$ & $H/V$ & $H/\\sqrt{VU}$ & Bev. Slope \\\\\n")
        f.write("\\hline\n")

        keys = ['alpha_on_comp', 'vy_on_comp', 'mr_on_comp', 'bev_on_comp']
        # Coefficient
        vals = []
        for k in keys:
            if k in regression_results:
                g = regression_results[k]['gamma_1']
                p = regression_results[k]['p_value']
                sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
                vals.append(f"${g:.4f}${sig}")
            else:
                vals.append("--")
        f.write(f"Complementarity & {' & '.join(vals)} \\\\\n")

        # SE
        vals_se = []
        for k in keys:
            if k in regression_results:
                se = regression_results[k]['model'].bse['complementarity']
                vals_se.append(f"({se:.4f})")
            else:
                vals_se.append("")
        f.write(f" & {' & '.join(vals_se)} \\\\\n")

        # R2
        vals_r2 = []
        for k in keys:
            if k in regression_results:
                vals_r2.append(f"{regression_results[k]['R2']:.3f}")
            else:
                vals_r2.append("--")
        f.write(f"$R^2$ & {' & '.join(vals_r2)} \\\\\n")

        f.write(f"N (industries) & \\multicolumn{{4}}{{c}}{{{len(cross_df)}}} \\\\\n")
        f.write("\\hline\n")
        f.write("Predicted sign & $-$ & $-$ & $-$ & $-$ \\\\\n")
        f.write("\\hline\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write("\\item \\textit{Notes:} Stage 1 estimates industry matching functions: "
                "$\\log H_{it} = \\alpha_i + \\beta_V \\log V_{it} + \\beta_U \\log U_{it} + \\varepsilon_{it}$ "
                "using JOLTS (vacancies, hires) and CPS (unemployment) monthly data, 2001--2025, "
                "with Newey--West (HAC) standard errors (12 lags). "
                "Stage 2 regresses industry-level matching outcomes on an O*NET-based task complementarity "
                "index (employment-weighted average of teamwork, coordination, and interdependence scores "
                "from Work Context and Work Activities databases, aggregated from occupations to industries "
                "via OES employment weights). Higher complementarity $\\approx$ lower $\\rho$ in CES framework. "
                "Column (4) Beveridge slope is from $\\log V_{it} = a_i + b_i \\log U_{it} + \\varepsilon_{it}$. "
                "*, **, *** denote significance at 10\\%, 5\\%, 1\\%.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{table}\n")
    print(f"  Saved: {out_path3}")


# ──────────────────────────────────────────────────────────────────────────────
# Section 9: Summary assessment
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(cross_df, regression_results):
    """Print final assessment of whether the CES prediction holds."""

    print("\n" + "=" * 80)
    print("SUMMARY: CES Free Energy Prediction — Labor Market Matching Test")
    print("=" * 80)

    print(f"\nIndustries tested: {len(cross_df)}")
    print(f"Complementarity range: [{cross_df['complementarity'].min():.3f}, {cross_df['complementarity'].max():.3f}]")

    print("\nPREDICTION: Higher task complementarity (lower rho) => lower matching efficiency")
    print("-" * 70)

    support_count = 0
    total = 0

    for key, label in [('alpha_on_comp', 'Matching efficiency (alpha)'),
                        ('vy_on_comp', 'Vacancy yield (H/V)'),
                        ('mr_on_comp', 'Matching rate (H/sqrt(VU))'),
                        ('bev_on_comp', 'Beveridge slope')]:
        if key not in regression_results:
            continue
        total += 1
        r = regression_results[key]
        sign_correct = r['gamma_1'] < 0
        sig = r['p_value'] < 0.10

        status = 'SUPPORTED' if (sign_correct and sig) else 'SIGN OK (insig)' if sign_correct else 'WRONG SIGN'
        if sign_correct:
            support_count += 1

        stars = '***' if r['p_value'] < 0.01 else '**' if r['p_value'] < 0.05 else '*' if r['p_value'] < 0.1 else ''
        print(f"  {label:40s}: gamma_1 = {r['gamma_1']:+.4f}{stars:3s} (p={r['p_value']:.3f}, R2={r['R2']:.3f}) [{status}]")

    print(f"\nSign prediction correct: {support_count}/{total}")

    if support_count == total:
        print("\nASSESSMENT: Strong support for CES free energy prediction.")
        print("All four measures show the predicted negative relationship between")
        print("task complementarity and matching efficiency.")
    elif support_count >= 3:
        print("\nASSESSMENT: Substantial support for CES free energy prediction.")
        print("Most measures show the predicted negative relationship.")
    elif support_count >= 2:
        print("\nASSESSMENT: Mixed support for CES free energy prediction.")
        print("Some measures show the predicted relationship, others do not.")
    else:
        print("\nASSESSMENT: Weak or no support for CES free energy prediction.")
        print("The cross-industry relationship does not consistently show the")
        print("predicted negative sign.")

    print("\nCAVEATS:")
    print("  1. Small N (9 industries) limits statistical power")
    print("  2. JOLTS industry classification is coarse (1-digit NAICS)")
    print("  3. O*NET complementarity index is a proxy for CES rho parameter")
    print("  4. Matching function specification assumes Cobb-Douglas form")
    print("  5. CPS unemployment by industry may have measurement error")
    print("  6. COVID-19 period (2020-2022) created structural breaks")
    print("=" * 80)


# ──────────────────────────────────────────────────────────────────────────────
# Section 10: Robustness checks
# ──────────────────────────────────────────────────────────────────────────────

def robustness_pre_covid(industry_data, industry_complementarity):
    """
    Re-estimate matching functions and cross-industry regressions
    using only the pre-COVID sample (2001-2019).
    """
    print("\n=== Section 10a: Robustness — Pre-COVID sample (2001-2019) ===")

    pre_covid = {}
    for name, df in industry_data.items():
        df_pre = df[df['date'] < '2020-03-01'].copy()
        if len(df_pre) >= 24:
            pre_covid[name] = df_pre

    matching_pre = estimate_matching_functions(pre_covid)
    cross_pre, reg_pre = cross_industry_regressions(matching_pre, industry_complementarity)
    return cross_pre, reg_pre


def robustness_task_specificity(occ_index, industry_data, industry_complementarity):
    """
    Alternative complementarity measure: within-industry dispersion of complementarity scores.

    Higher dispersion = occupations in the industry are more heterogeneous in their
    teamwork requirements = harder to substitute across roles = lower rho.

    This captures the "heterogeneous agents" aspect of the CES framework more directly
    than the level of complementarity.
    """
    print("\n=== Section 10b: Robustness — Task specificity (within-industry dispersion) ===")

    # Load OES sector data for dispersion calculation
    cache_file = CACHE_DIR / 'oes_natsector_2023.csv'
    if not cache_file.exists():
        print("  WARNING: OES data not cached, skipping specificity measure")
        return None

    oes_df = pd.read_csv(cache_file)
    oes_df = oes_df[oes_df['O_GROUP'] == 'detailed'].copy()
    oes_df['TOT_EMP'] = pd.to_numeric(
        oes_df['TOT_EMP'].astype(str).str.replace(',', '').str.replace('**', '', regex=False),
        errors='coerce'
    )
    oes_df['soc_clean'] = oes_df['OCC_CODE'].astype(str).str.strip()

    # Merge with O*NET
    occ_6d = occ_index.groupby('soc_6')['complementarity_index'].mean().reset_index()
    merged = pd.merge(oes_df, occ_6d, left_on='soc_clean', right_on='soc_6', how='inner')

    naics_to_jolts = {
        '21': 'Mining', '23': 'Construction', '31-33': 'Manufacturing',
        '42': 'Trade_Transport_Util', '44-45': 'Trade_Transport_Util',
        '48-49': 'Trade_Transport_Util', '22': 'Trade_Transport_Util',
        '51': 'Information', '52': 'Financial', '53': 'Financial',
        '54': 'Prof_Business', '55': 'Prof_Business', '56': 'Prof_Business',
        '61': 'Edu_Health', '62': 'Edu_Health',
        '71': 'Leisure_Hospitality', '72': 'Leisure_Hospitality',
    }
    merged['naics_str'] = merged['NAICS'].astype(str).str.strip()
    merged['jolts_industry'] = merged['naics_str'].map(naics_to_jolts)
    matched = merged.dropna(subset=['jolts_industry'])

    # Compute within-industry standard deviation (employment-weighted)
    industry_specificity = {}
    for ind_name in INDUSTRIES:
        if INDUSTRIES[ind_name].get('ul') is None:
            continue
        ind_data = matched[matched['jolts_industry'] == ind_name].copy()
        valid = ind_data.dropna(subset=['TOT_EMP', 'complementarity_index'])
        if len(valid) < 5:
            continue

        # Weighted standard deviation
        weights = valid['TOT_EMP'].values
        values = valid['complementarity_index'].values
        wmean = np.average(values, weights=weights)
        wvar = np.average((values - wmean) ** 2, weights=weights)
        wsd = np.sqrt(wvar)

        # Also compute interquartile range for robustness
        sorted_idx = np.argsort(values)
        cum_w = np.cumsum(weights[sorted_idx]) / weights.sum()
        q25 = values[sorted_idx][np.searchsorted(cum_w, 0.25)]
        q75 = values[sorted_idx][np.searchsorted(cum_w, 0.75)]
        iqr = q75 - q25

        industry_specificity[ind_name] = {
            'task_dispersion': wsd,
            'task_iqr': iqr,
            'n_occs': len(valid),
        }
        print(f"  {ind_name:25s}: dispersion={wsd:.4f}, IQR={iqr:.4f} ({len(valid)} occs)")

    # Run cross-industry regression with dispersion as alternative proxy
    rows = []
    matching_results = estimate_matching_functions(industry_data)
    for name in matching_results:
        if name not in industry_specificity or name not in industry_complementarity:
            continue
        r = matching_results[name]
        rows.append({
            'industry': name,
            'short': INDUSTRIES[name]['short'],
            'complementarity': industry_complementarity[name],
            'task_dispersion': industry_specificity[name]['task_dispersion'],
            'task_iqr': industry_specificity[name]['task_iqr'],
            'alpha': r['alpha'],
            'vacancy_yield': r['vacancy_yield'],
            'matching_rate': r['matching_rate'],
            'beveridge_slope': r['beveridge_slope'],
        })

    spec_df = pd.DataFrame(rows)
    if len(spec_df) < 4:
        print("  Too few industries for specificity regression")
        return spec_df

    print(f"\n  Cross-industry with task dispersion (N={len(spec_df)}):")

    # Regression: alpha on task_dispersion
    X = sm.add_constant(spec_df['task_dispersion'])
    for dep, label in [('alpha', 'Matching efficiency'), ('vacancy_yield', 'Vacancy yield'),
                        ('beveridge_slope', 'Beveridge slope')]:
        model = OLS(spec_df[dep], X).fit()
        g = model.params['task_dispersion']
        p = model.pvalues['task_dispersion']
        stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        print(f"    {label:30s}: gamma = {g:+.4f}{stars} (p={p:.3f}, R2={model.rsquared:.3f})")

    # Also try a combined regression: alpha on both complementarity + dispersion
    print(f"\n  Combined regression: alpha on complementarity + dispersion:")
    X_comb = sm.add_constant(spec_df[['complementarity', 'task_dispersion']])
    model_comb = OLS(spec_df['alpha'], X_comb).fit()
    for var in ['complementarity', 'task_dispersion']:
        g = model_comb.params[var]
        p = model_comb.pvalues[var]
        stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        print(f"    {var:25s}: gamma = {g:+.4f}{stars} (p={p:.3f})")
    print(f"    R-squared: {model_comb.rsquared:.4f}")

    return spec_df


def robustness_vacancy_duration(industry_complementarity):
    """
    Alternative dependent variable: average vacancy fill time.

    From JOLTS: vacancy duration ~ V/H (openings divided by hires per month).
    A higher V/H ratio means vacancies take longer to fill.

    PREDICTION: More complementary industries have higher V/H (longer fill times).
    """
    print("\n=== Section 10c: Robustness — Average vacancy duration proxy ===")

    # V/H ratio is already captured by 1/vacancy_yield
    # This test is just a sign-flip check on Test 2
    print("  (This is the reciprocal of vacancy yield; see Test 2 above)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("CES Free Energy Framework — Labor Market Matching Test")
    print("=" * 80)

    # Stage 0: Fetch data
    industry_data = fetch_all_jolts_and_unemployment()

    # Build complementarity index from O*NET
    occ_index = build_onet_complementarity_index()

    # Aggregate to JOLTS industry level using OES crosswalk
    industry_complementarity = aggregate_complementarity_to_industries(occ_index)

    if industry_complementarity is None or len(industry_complementarity) < 5:
        print("\n  WARNING: OES crosswalk failed or too few matches. Using fallback mapping.")
        # Fallback: use SOC major group averages
        soc_to_jolts = build_fallback_complementarity()
        industry_complementarity = {}
        for ind_name, soc_prefixes in soc_to_jolts.items():
            mask = pd.Series(False, index=occ_index.index)
            for prefix in soc_prefixes:
                mask = mask | occ_index['soc_6'].str.startswith(prefix)
            matched = occ_index[mask]
            if len(matched) > 0:
                industry_complementarity[ind_name] = matched['complementarity_index'].mean()
                print(f"    {ind_name}: {industry_complementarity[ind_name]:.4f} ({len(matched)} occs)")

    # Stage 1: Estimate matching functions
    matching_results = estimate_matching_functions(industry_data)

    # Stage 2: Cross-industry regressions
    cross_df, regression_results = cross_industry_regressions(matching_results, industry_complementarity)

    if len(cross_df) > 0 and len(regression_results) > 0:
        # Create figures
        create_figures(cross_df, regression_results, industry_data, matching_results)

        # Save results
        save_results(cross_df, matching_results, regression_results, industry_complementarity)

        # Robustness checks
        cross_pre, reg_pre = robustness_pre_covid(industry_data, industry_complementarity)
        spec_df = robustness_task_specificity(occ_index, industry_data, industry_complementarity)

        # Summary
        print_summary(cross_df, regression_results)

        # Pre-COVID summary
        if reg_pre:
            print("\n  --- Pre-COVID robustness (2001-2019) ---")
            for key, label in [('alpha_on_comp', 'alpha'), ('bev_on_comp', 'Beveridge')]:
                if key in reg_pre:
                    r = reg_pre[key]
                    stars = '***' if r['p_value'] < 0.01 else '**' if r['p_value'] < 0.05 else '*' if r['p_value'] < 0.1 else ''
                    print(f"    {label:15s}: gamma_1 = {r['gamma_1']:+.4f}{stars} (p={r['p_value']:.3f})")
    else:
        print("\n  ERROR: Insufficient data for analysis")

    return cross_df, regression_results


if __name__ == '__main__':
    cross_df, regression_results = main()
