#!/usr/bin/env python3
"""
Test: Price of Incentive Compatibility (PoIC) in Procurement Auctions
=====================================================================

Prediction from CES free energy framework (Myerson mechanism design):
  The information rent (markup over estimated cost) is higher when goods
  are more differentiated/complementary (low rho) and lower when goods
  are more substitutable/commodity-like (high rho).

Empirical test: In US federal procurement, the markup (award/ceiling)
  should be systematically higher for differentiated product categories
  than for commodity categories.

Data: USAspending.gov bulk download API (FPDS contract data)
  - current_total_value_of_award: actual obligation
  - potential_total_value_of_award: contract ceiling (ex-ante estimate)
  - number_of_offers_received: competitive intensity
  - naics_code: industry classification
  - extent_competed: competition type

Markup = current_total_value / potential_total_value
  Values < 1 mean the award is below ceiling (typical)
  We expect markup closer to 1 (i.e. higher extraction) for differentiated goods

Author: Connor Doll / Smirl (2026)
"""

import os
import sys
import time
import json
import zipfile
import io
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# Suppress warnings for clean output
warnings.filterwarnings('ignore', category=FutureWarning)
try:
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
except AttributeError:
    pass  # Older pandas version

# ============================================================
# Configuration
# ============================================================

DATA_DIR = '/home/jonsmirl/thesis/thesis_data'
FIG_DIR = '/home/jonsmirl/thesis/figures'
CACHE_DIR = os.path.join(DATA_DIR, 'procurement_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# NAICS 2-digit substitutability classification
# Based on product differentiation / specificity
NAICS_SUBSTITUTABILITY = {
    # HIGH rho (commodity-like, standardized, fungible)
    '11': ('HIGH', 'Agriculture/Forestry/Fishing'),
    '21': ('HIGH', 'Mining/Oil/Gas'),
    '22': ('HIGH', 'Utilities'),
    '42': ('HIGH', 'Wholesale Trade'),
    '44': ('HIGH', 'Retail Trade'),
    '45': ('HIGH', 'Retail Trade'),
    # MEDIUM rho (some differentiation)
    '23': ('MEDIUM', 'Construction'),
    '31': ('MEDIUM', 'Manufacturing - Food/Textile'),
    '32': ('MEDIUM', 'Manufacturing - Wood/Chemical'),
    '33': ('MEDIUM', 'Manufacturing - Metal/Machinery/Electronics'),
    '48': ('MEDIUM', 'Transportation'),
    '49': ('MEDIUM', 'Warehousing'),
    '51': ('MEDIUM', 'Information/Media'),
    '52': ('MEDIUM', 'Finance/Insurance'),
    '53': ('MEDIUM', 'Real Estate'),
    '56': ('MEDIUM', 'Administrative/Waste Services'),
    '61': ('MEDIUM', 'Educational Services'),
    '71': ('MEDIUM', 'Arts/Entertainment'),
    '72': ('MEDIUM', 'Accommodation/Food Services'),
    # LOW rho (highly differentiated, customized, relationship-specific)
    '54': ('LOW', 'Professional/Scientific/Technical'),
    '55': ('LOW', 'Management of Companies'),
    '62': ('LOW', 'Healthcare/Social Assistance'),
    '81': ('LOW', 'Other Professional Services'),
    '92': ('LOW', 'Public Administration Services'),
}

# More granular 4-digit NAICS classification for key sectors
NAICS4_SUBSTITUTABILITY = {
    # Definitely HIGH rho (commodity)
    '3241': ('HIGH', 'Petroleum/Coal Products Mfg'),
    '3251': ('HIGH', 'Basic Chemical Mfg'),
    '3253': ('HIGH', 'Pesticide/Fertilizer Mfg'),
    '3261': ('HIGH', 'Plastics Product Mfg'),
    '3221': ('HIGH', 'Pulp/Paper Mills'),
    '4234': ('HIGH', 'Prof/Commercial Equipment Wholesale'),
    '4247': ('HIGH', 'Petroleum Wholesale'),
    '3111': ('HIGH', 'Animal Food Mfg'),
    '3112': ('HIGH', 'Grain/Oilseed Milling'),
    '2111': ('HIGH', 'Oil/Gas Extraction'),
    '2211': ('HIGH', 'Electric Power Generation'),
    '3339': ('HIGH', 'General Purpose Machinery Mfg'),
    # Definitely LOW rho (differentiated)
    '5411': ('LOW', 'Legal Services'),
    '5412': ('LOW', 'Accounting/Bookkeeping'),
    '5413': ('LOW', 'Architecture/Engineering'),
    '5414': ('LOW', 'Specialized Design'),
    '5415': ('LOW', 'Computer Systems Design'),
    '5416': ('LOW', 'Management/Scientific Consulting'),
    '5417': ('LOW', 'Scientific R&D'),
    '5418': ('LOW', 'Advertising/Public Relations'),
    '5419': ('LOW', 'Other Prof/Scientific/Technical'),
    '6211': ('LOW', 'Offices of Physicians'),
    '6214': ('LOW', 'Outpatient Care Centers'),
    '6215': ('LOW', 'Medical/Diagnostic Labs'),
    '6221': ('LOW', 'General Medical/Surgical Hospitals'),
    '8111': ('LOW', 'Automotive Repair/Maintenance'),
    # Medium
    '2361': ('MEDIUM', 'Residential Building Construction'),
    '2362': ('MEDIUM', 'Nonresidential Building Construction'),
    '2371': ('MEDIUM', 'Utility System Construction'),
    '2373': ('MEDIUM', 'Highway/Street Construction'),
    '3341': ('MEDIUM', 'Computer/Peripheral Equipment Mfg'),
    '3342': ('MEDIUM', 'Communications Equipment Mfg'),
    '3344': ('MEDIUM', 'Semiconductor Mfg'),
    '3345': ('MEDIUM', 'Navigation/Measurement Instruments'),
    '3364': ('MEDIUM', 'Aerospace Product Mfg'),
    '5112': ('MEDIUM', 'Software Publishers'),
    '5182': ('MEDIUM', 'Data Processing/Hosting'),
    '5191': ('MEDIUM', 'Other Information Services'),
    '5613': ('MEDIUM', 'Employment Services'),
    '5614': ('MEDIUM', 'Business Support Services'),
    '5616': ('MEDIUM', 'Investigation/Security Services'),
    '5617': ('MEDIUM', 'Services to Buildings/Dwellings'),
}

# Numeric mapping for regression
SUBST_NUMERIC = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}


def classify_naics(naics_code):
    """Classify a NAICS code by substitutability, trying 4-digit first, then 2-digit."""
    if pd.isna(naics_code):
        return None, None
    code = str(int(float(naics_code))) if not isinstance(naics_code, str) else naics_code
    code = code.strip()

    # Try 4-digit first
    code4 = code[:4]
    if code4 in NAICS4_SUBSTITUTABILITY:
        cat, desc = NAICS4_SUBSTITUTABILITY[code4]
        return cat, desc

    # Fall back to 2-digit
    code2 = code[:2]
    if code2 in NAICS_SUBSTITUTABILITY:
        cat, desc = NAICS_SUBSTITUTABILITY[code2]
        return cat, desc

    return None, None


# ============================================================
# Section 1: Data Download from USAspending (direct API)
# ============================================================

def fetch_top_naics(year=2023, total_wanted=200):
    """Get top NAICS codes by spending volume using the category API."""
    results = []
    for page in range(1, (total_wanted // 100) + 2):
        url = 'https://api.usaspending.gov/api/v2/search/spending_by_category/naics/'
        payload = {
            'filters': {
                'time_period': [{'start_date': f'{year}-01-01', 'end_date': f'{year}-12-31'}],
                'award_type_codes': ['A', 'B', 'C', 'D'],
            },
            'limit': 100,
            'page': page,
        }
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code == 200:
            page_results = r.json().get('results', [])
            results.extend(page_results)
            if len(page_results) < 100:
                break
        else:
            print(f"  Category API failed on page {page}: {r.status_code}")
            break
        time.sleep(0.3)
    return results[:total_wanted]


def fetch_awards_for_naics(naics_code, year=2023, pages=3, per_page=100):
    """Fetch award IDs for a specific NAICS code."""
    url = 'https://api.usaspending.gov/api/v2/search/spending_by_award/'
    ids = []
    for page in range(1, pages + 1):
        payload = {
            'filters': {
                'time_period': [{'start_date': f'{year}-01-01', 'end_date': f'{year}-12-31'}],
                'award_type_codes': ['A', 'B', 'C', 'D'],
                'naics_codes': {'require': [naics_code]},
            },
            'fields': ['Award ID', 'Award Amount', 'generated_internal_id'],
            'page': page, 'limit': per_page,
            'sort': 'Award Amount', 'order': 'desc', 'subawards': False,
        }
        try:
            r = requests.post(url, json=payload, timeout=60)
            if r.status_code == 200:
                results = r.json().get('results', [])
                ids.extend([x.get('generated_internal_id') for x in results if x.get('generated_internal_id')])
                if len(results) < per_page:
                    break
            else:
                break
        except Exception:
            break
        time.sleep(0.2)
    return ids


def fetch_award_detail(award_id):
    """Fetch detailed award info: ceiling, offers, competition type, NAICS."""
    url = f'https://api.usaspending.gov/api/v2/awards/{award_id}/'
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return None


def fetch_procurement_data():
    """
    Fetch procurement data using a two-step approach:
    1. Get top NAICS codes by spending volume
    2. For a sample of NAICS codes in each category, fetch award details
    """
    cache_path = os.path.join(CACHE_DIR, 'procurement_2023_all.csv')
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, low_memory=False)
        print(f"  Loaded {len(df)} rows")
        return df

    # Step 1: Get top NAICS codes
    print("  Step 1: Getting top NAICS codes by spending...")
    naics_list = fetch_top_naics(total_wanted=200)
    print(f"  Got {len(naics_list)} NAICS codes")

    # Classify each NAICS code
    naics_by_cat = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
    for item in naics_list:
        code = str(item.get('code', ''))
        cat, _ = classify_naics(code)
        if cat and cat in naics_by_cat:
            naics_by_cat[cat].append(code)

    for cat in ['HIGH', 'MEDIUM', 'LOW']:
        print(f"  {cat}: {len(naics_by_cat[cat])} NAICS codes")

    # Step 2: Sample award IDs from each category
    # Take up to 8 NAICS codes per category, 50 awards each = ~400 per category
    print("\n  Step 2: Fetching award IDs by NAICS...")
    award_ids_by_cat = {}
    for cat in ['HIGH', 'MEDIUM', 'LOW']:
        codes = naics_by_cat[cat][:8]
        all_ids = []
        for code in codes:
            ids = fetch_awards_for_naics(code, pages=1, per_page=50)
            all_ids.extend(ids)
            print(f"    NAICS {code} ({cat}): {len(ids)} awards")
        award_ids_by_cat[cat] = all_ids
        print(f"  {cat} total: {len(all_ids)} award IDs")

    # Step 3: Fetch details for all award IDs
    all_ids = []
    for cat in ['HIGH', 'MEDIUM', 'LOW']:
        all_ids.extend(award_ids_by_cat[cat])
    print(f"\n  Step 3: Fetching details for {len(all_ids)} awards...")

    detail_rows = []
    for idx, aid in enumerate(all_ids):
        if idx % 100 == 0:
            print(f"    Progress: {idx}/{len(all_ids)}")
        d = fetch_award_detail(aid)
        if d is None:
            continue
        tc = d.get('latest_transaction_contract_data') or {}
        naics = tc.get('naics', '')
        detail_rows.append({
            'award_id': aid,
            'current_total_value_of_award': d.get('total_obligation'),
            'potential_total_value_of_award': d.get('base_and_all_options'),
            'naics_code': naics,
            'number_of_offers_received': tc.get('number_of_offers_received'),
            'extent_competed': tc.get('extent_competed', ''),
            'type_of_contract_pricing': tc.get('type_of_contract_pricing', ''),
            'awarding_agency_name': d.get('awarding_agency', {}).get('toptier_agency', {}).get('name', '') if isinstance(d.get('awarding_agency'), dict) else '',
            'product_or_service_code': tc.get('product_or_service_code', ''),
        })
        time.sleep(0.1)

    if not detail_rows:
        print("ERROR: No award details fetched")
        return None

    df = pd.DataFrame(detail_rows)
    df.to_csv(cache_path, index=False)
    print(f"\n  Saved {len(df)} detailed awards to cache")
    return df


# ============================================================
# Section 2: Data Cleaning and Markup Computation
# ============================================================

def clean_and_compute_markup(df):
    """
    Clean data and compute markup ratio.

    Markup = current_total_value_of_award / potential_total_value_of_award

    Where:
      current_total_value = actual amount obligated/spent
      potential_total_value = contract ceiling (ex-ante estimated max)

    A markup close to 1.0 means the contractor extracted nearly the full
    contract ceiling. Values < 1 mean underrun. Values > 1 are rare
    (contract modifications exceeding ceiling).
    """
    print("\n=== Data Cleaning ===")
    print(f"Starting rows: {len(df)}")

    # Key columns
    cols_needed = [
        'current_total_value_of_award',
        'potential_total_value_of_award',
        'naics_code',
        'number_of_offers_received',
        'extent_competed',
        'type_of_contract_pricing',
        'awarding_agency_name',
        'product_or_service_code',
    ]

    # Check columns exist
    for c in cols_needed:
        if c not in df.columns:
            print(f"  WARNING: Column {c} not found")

    # Convert to numeric
    df['award_value'] = pd.to_numeric(df['current_total_value_of_award'], errors='coerce')
    df['ceiling_value'] = pd.to_numeric(df['potential_total_value_of_award'], errors='coerce')
    df['num_offers'] = pd.to_numeric(df['number_of_offers_received'], errors='coerce')

    # Filter: need both values, positive
    mask = (
        df['award_value'].notna() &
        df['ceiling_value'].notna() &
        (df['award_value'] > 0) &
        (df['ceiling_value'] > 0) &
        df['naics_code'].notna()
    )
    df = df[mask].copy()
    print(f"After requiring award, ceiling, NAICS: {len(df)}")

    # Compute markup
    df['markup'] = df['award_value'] / df['ceiling_value']

    # Filter outliers: keep markup in [0.01, 5.0]
    # Extreme values indicate data errors or unusual contract modifications
    mask_markup = (df['markup'] >= 0.01) & (df['markup'] <= 5.0)
    df = df[mask_markup].copy()
    print(f"After markup filter [0.01, 5.0]: {len(df)}")

    # Filter: minimum contract size ($10K) to exclude micro-purchases
    mask_size = df['ceiling_value'] >= 10000
    df = df[mask_size].copy()
    print(f"After minimum size ($10K): {len(df)}")

    # Classify NAICS
    df['naics_str'] = df['naics_code'].astype(str).str.split('.').str[0]
    df['naics_2digit'] = df['naics_str'].str[:2]
    df['naics_4digit'] = df['naics_str'].str[:4]

    classifications = df['naics_str'].apply(classify_naics)
    df['subst_category'] = classifications.apply(lambda x: x[0])
    df['naics_sector_name'] = classifications.apply(lambda x: x[1])

    # Drop unclassified
    df = df[df['subst_category'].notna()].copy()
    print(f"After NAICS classification: {len(df)}")

    # Numeric substitutability
    df['subst_numeric'] = df['subst_category'].map(SUBST_NUMERIC)

    # Log number of offers (add 1 to handle zeros)
    df['log_offers'] = np.log1p(df['num_offers'].fillna(0))

    # Competition type dummies
    df['fully_competed'] = df['extent_competed'].str.contains('FULL', na=False).astype(int)

    # Contract pricing type
    df['fixed_price'] = df['type_of_contract_pricing'].str.contains('FIXED', na=False).astype(int)

    # Log contract size control
    df['log_ceiling'] = np.log(df['ceiling_value'])

    print(f"\nFinal dataset: {len(df)} contracts")
    print(f"\nSubstitutability distribution:")
    for cat in ['HIGH', 'MEDIUM', 'LOW']:
        n = (df['subst_category'] == cat).sum()
        med = df.loc[df['subst_category'] == cat, 'markup'].median()
        print(f"  {cat:8s}: {n:>8,} contracts, median markup = {med:.4f}")

    return df


# ============================================================
# Section 3: Statistical Analysis
# ============================================================

def run_analysis(df):
    """Run the PoIC regression analysis."""
    print("\n" + "=" * 70)
    print("EMPIRICAL TEST: Price of Incentive Compatibility (PoIC)")
    print("=" * 70)

    # ----------------------------------------------------------
    # 3a. Summary statistics by category
    # ----------------------------------------------------------
    print("\n--- Summary Statistics by Substitutability Category ---")
    summary = df.groupby('subst_category').agg(
        n_contracts=('markup', 'count'),
        mean_markup=('markup', 'mean'),
        median_markup=('markup', 'median'),
        std_markup=('markup', 'std'),
        p25_markup=('markup', lambda x: x.quantile(0.25)),
        p75_markup=('markup', lambda x: x.quantile(0.75)),
        mean_offers=('num_offers', 'mean'),
        median_ceiling=('ceiling_value', 'median'),
        pct_fully_competed=('fully_competed', 'mean'),
        pct_fixed_price=('fixed_price', 'mean'),
    ).reindex(['LOW', 'MEDIUM', 'HIGH'])

    print(summary.to_string())

    # ----------------------------------------------------------
    # 3b. Non-parametric test: Kruskal-Wallis
    # ----------------------------------------------------------
    print("\n--- Non-parametric Tests ---")
    groups = [df.loc[df['subst_category'] == c, 'markup'].values
              for c in ['LOW', 'MEDIUM', 'HIGH']]
    H, p_kw = stats.kruskal(*groups)
    print(f"Kruskal-Wallis H = {H:.4f}, p = {p_kw:.2e}")

    # Pairwise Mann-Whitney
    for (c1, g1), (c2, g2) in [
        (('LOW', groups[0]), ('MEDIUM', groups[1])),
        (('LOW', groups[0]), ('HIGH', groups[2])),
        (('MEDIUM', groups[1]), ('HIGH', groups[2])),
    ]:
        U, p_mw = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        print(f"  Mann-Whitney {c1:6s} vs {c2:6s}: U = {U:.0f}, p = {p_mw:.2e}")

    # Monotonicity test: LOW > MEDIUM > HIGH?
    med_low = np.median(groups[0])
    med_med = np.median(groups[1])
    med_high = np.median(groups[2])
    monotonic = med_low > med_med > med_high
    print(f"\nMonotonicity check (median): LOW ({med_low:.4f}) > MEDIUM ({med_med:.4f}) > HIGH ({med_high:.4f})")
    print(f"Monotonicity holds: {monotonic}")

    # Jonckheere-Terpstra trend test (manual implementation)
    # Tests for ordered alternative: LOW > MEDIUM > HIGH
    # which is equivalent to markup decreasing with substitutability
    print("\n--- Jonckheere-Terpstra Trend Test ---")
    # Count pairs where observation from less-substitutable group > more-substitutable group
    jt_stat = 0
    n_pairs = 0
    # LOW vs MEDIUM
    for xi in groups[0]:
        jt_stat += np.sum(xi > groups[1]) + 0.5 * np.sum(xi == groups[1])
        n_pairs += len(groups[1])
    # LOW vs HIGH
    for xi in groups[0]:
        jt_stat += np.sum(xi > groups[2]) + 0.5 * np.sum(xi == groups[2])
        n_pairs += len(groups[2])
    # MEDIUM vs HIGH
    for xi in groups[1]:
        jt_stat += np.sum(xi > groups[2]) + 0.5 * np.sum(xi == groups[2])
        n_pairs += len(groups[2])

    # Normal approximation for JT
    n = [len(g) for g in groups]
    N = sum(n)
    E_jt = (N**2 - sum(ni**2 for ni in n)) / 4
    V_jt = (N**2 * (2*N + 3) - sum(ni**2 * (2*ni + 3) for ni in n)) / 72
    z_jt = (jt_stat - E_jt) / np.sqrt(V_jt)
    p_jt = 1 - stats.norm.cdf(z_jt)  # one-sided: greater
    print(f"JT statistic = {jt_stat:.0f}, E[JT] = {E_jt:.0f}, z = {z_jt:.4f}, p (one-sided) = {p_jt:.2e}")

    # ----------------------------------------------------------
    # 3c. OLS Regression
    # ----------------------------------------------------------
    print("\n--- OLS Regression ---")
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        # Model 1: Bivariate
        model1 = smf.ols('markup ~ subst_numeric', data=df).fit(cov_type='HC1')
        print("\nModel 1: markup ~ substitutability")
        print(f"  beta_subst = {model1.params['subst_numeric']:.6f} "
              f"(SE = {model1.bse['subst_numeric']:.6f}, "
              f"t = {model1.tvalues['subst_numeric']:.4f}, "
              f"p = {model1.pvalues['subst_numeric']:.2e})")
        print(f"  R^2 = {model1.rsquared:.6f}, N = {model1.nobs:.0f}")

        # Model 2: With competition controls
        model2 = smf.ols('markup ~ subst_numeric + log_offers + fully_competed',
                         data=df.dropna(subset=['log_offers'])).fit(cov_type='HC1')
        print("\nModel 2: markup ~ substitutability + log(offers) + fully_competed")
        for var in ['subst_numeric', 'log_offers', 'fully_competed']:
            print(f"  beta_{var} = {model2.params[var]:.6f} "
                  f"(SE = {model2.bse[var]:.6f}, "
                  f"t = {model2.tvalues[var]:.4f}, "
                  f"p = {model2.pvalues[var]:.2e})")
        print(f"  R^2 = {model2.rsquared:.6f}, N = {model2.nobs:.0f}")

        # Model 3: Full controls (competition + size + pricing type)
        model3 = smf.ols('markup ~ subst_numeric + log_offers + fully_competed + log_ceiling + fixed_price',
                         data=df.dropna(subset=['log_offers'])).fit(cov_type='HC1')
        print("\nModel 3: markup ~ substitutability + log(offers) + fully_competed + log(ceiling) + fixed_price")
        for var in ['subst_numeric', 'log_offers', 'fully_competed', 'log_ceiling', 'fixed_price']:
            print(f"  beta_{var} = {model3.params[var]:.6f} "
                  f"(SE = {model3.bse[var]:.6f}, "
                  f"t = {model3.tvalues[var]:.4f}, "
                  f"p = {model3.pvalues[var]:.2e})")
        print(f"  R^2 = {model3.rsquared:.6f}, N = {model3.nobs:.0f}")

        # Model 4: Category dummies (more flexible, no linearity assumption)
        df_reg = df.dropna(subset=['log_offers']).copy()
        df_reg['cat_LOW'] = (df_reg['subst_category'] == 'LOW').astype(int)
        df_reg['cat_MEDIUM'] = (df_reg['subst_category'] == 'MEDIUM').astype(int)
        # HIGH is reference category
        model4 = smf.ols('markup ~ cat_LOW + cat_MEDIUM + log_offers + fully_competed + log_ceiling + fixed_price',
                         data=df_reg).fit(cov_type='HC1')
        print("\nModel 4: Category dummies (ref=HIGH) + full controls")
        for var in ['cat_LOW', 'cat_MEDIUM', 'log_offers', 'fully_competed', 'log_ceiling', 'fixed_price']:
            print(f"  beta_{var} = {model4.params[var]:.6f} "
                  f"(SE = {model4.bse[var]:.6f}, "
                  f"t = {model4.tvalues[var]:.4f}, "
                  f"p = {model4.pvalues[var]:.2e})")
        print(f"  R^2 = {model4.rsquared:.6f}, N = {model4.nobs:.0f}")

        # Store for later use
        models = {'m1': model1, 'm2': model2, 'm3': model3, 'm4': model4}

    except ImportError:
        print("  statsmodels not available, skipping OLS")
        models = None

    # ----------------------------------------------------------
    # 3d. NAICS 4-digit granular analysis
    # ----------------------------------------------------------
    print("\n--- Top/Bottom NAICS Sectors by Median Markup ---")
    sector_stats = df.groupby(['naics_4digit', 'naics_sector_name', 'subst_category']).agg(
        n=('markup', 'count'),
        median_markup=('markup', 'median'),
        mean_markup=('markup', 'mean'),
        median_offers=('num_offers', 'median'),
    ).reset_index()
    sector_stats = sector_stats[sector_stats['n'] >= 50].sort_values('median_markup', ascending=False)

    print("\nTop 15 sectors by median markup (>= 50 contracts):")
    for _, row in sector_stats.head(15).iterrows():
        print(f"  {row['naics_4digit']} ({row['subst_category']:6s}) "
              f"{row['naics_sector_name']:40s} "
              f"n={row['n']:>6,}  markup={row['median_markup']:.4f}  "
              f"offers={row['median_offers']:.1f}")

    print("\nBottom 15 sectors by median markup (>= 50 contracts):")
    for _, row in sector_stats.tail(15).iterrows():
        print(f"  {row['naics_4digit']} ({row['subst_category']:6s}) "
              f"{row['naics_sector_name']:40s} "
              f"n={row['n']:>6,}  markup={row['median_markup']:.4f}  "
              f"offers={row['median_offers']:.1f}")

    # ----------------------------------------------------------
    # 3e. Robustness: Fixed-price contracts only
    # ----------------------------------------------------------
    print("\n--- Robustness: Fixed-Price Contracts Only ---")
    df_fp = df[df['fixed_price'] == 1]
    if len(df_fp) > 100:
        for cat in ['LOW', 'MEDIUM', 'HIGH']:
            subset = df_fp[df_fp['subst_category'] == cat]
            if len(subset) > 0:
                print(f"  {cat:8s}: n={len(subset):>8,}, median markup={subset['markup'].median():.4f}")
        if models is not None:
            model_fp = smf.ols('markup ~ subst_numeric + log_offers + log_ceiling',
                               data=df_fp.dropna(subset=['log_offers'])).fit(cov_type='HC1')
            print(f"  Fixed-price only: beta_subst = {model_fp.params['subst_numeric']:.6f} "
                  f"(p = {model_fp.pvalues['subst_numeric']:.2e}), "
                  f"R^2 = {model_fp.rsquared:.6f}, N = {model_fp.nobs:.0f}")

    # ----------------------------------------------------------
    # 3f. Robustness: Fully competed contracts only
    # ----------------------------------------------------------
    print("\n--- Robustness: Fully Competed Contracts Only ---")
    df_fc = df[df['fully_competed'] == 1]
    if len(df_fc) > 100:
        for cat in ['LOW', 'MEDIUM', 'HIGH']:
            subset = df_fc[df_fc['subst_category'] == cat]
            if len(subset) > 0:
                print(f"  {cat:8s}: n={len(subset):>8,}, median markup={subset['markup'].median():.4f}")
        if models is not None:
            model_fc = smf.ols('markup ~ subst_numeric + log_offers + log_ceiling + fixed_price',
                               data=df_fc.dropna(subset=['log_offers'])).fit(cov_type='HC1')
            print(f"  Fully competed only: beta_subst = {model_fc.params['subst_numeric']:.6f} "
                  f"(p = {model_fc.pvalues['subst_numeric']:.2e}), "
                  f"R^2 = {model_fc.rsquared:.6f}, N = {model_fc.nobs:.0f}")

    return summary, models, sector_stats


# ============================================================
# Section 4: Visualization
# ============================================================

def make_figures(df, summary, models, sector_stats):
    """Generate publication-quality figures."""

    fig_path_base = os.path.join(FIG_DIR, 'procurement_poic')

    # ----------------------------------------------------------
    # Figure 1: Box plots by substitutability category
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Full box plot
    ax = axes[0]
    categories = ['LOW', 'MEDIUM', 'HIGH']
    cat_labels = ['LOW $\\rho$\n(Differentiated)', 'MEDIUM $\\rho$', 'HIGH $\\rho$\n(Commodity)']
    colors = ['#d73027', '#fee08b', '#1a9850']

    data_groups = [df.loc[df['subst_category'] == c, 'markup'].values for c in categories]
    bp = ax.boxplot(data_groups, labels=cat_labels, patch_artist=True,
                    showfliers=False, widths=0.6,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Markup (Award / Ceiling)', fontsize=12)
    ax.set_title('A. Distribution of Procurement Markup\nby Product Substitutability', fontsize=13)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ceiling = Award')

    # Add median annotations
    for i, (cat, g) in enumerate(zip(categories, data_groups)):
        med = np.median(g)
        ax.annotate(f'Median: {med:.3f}', xy=(i + 1, med),
                    xytext=(i + 1 + 0.35, med + 0.02),
                    fontsize=9, ha='left',
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    ax.legend(loc='upper right', fontsize=9)

    # Panel B: Median markup by NAICS 2-digit sector
    ax = axes[1]
    sector_2d = df.groupby(['naics_2digit', 'subst_category']).agg(
        median_markup=('markup', 'median'),
        n=('markup', 'count'),
    ).reset_index()
    sector_2d = sector_2d[sector_2d['n'] >= 100].sort_values('median_markup', ascending=True)

    color_map = {'LOW': '#d73027', 'MEDIUM': '#fee08b', 'HIGH': '#1a9850'}
    bar_colors = [color_map.get(c, 'gray') for c in sector_2d['subst_category']]
    bars = ax.barh(range(len(sector_2d)),
                   sector_2d['median_markup'].values,
                   color=bar_colors, edgecolor='black', linewidth=0.3, alpha=0.8)

    # Add NAICS labels
    naics_labels = []
    for _, row in sector_2d.iterrows():
        naics2 = row['naics_2digit']
        desc = NAICS_SUBSTITUTABILITY.get(naics2, ('?', naics2))[1]
        naics_labels.append(f"{naics2}: {desc[:25]}")
    ax.set_yticks(range(len(sector_2d)))
    ax.set_yticklabels(naics_labels, fontsize=8)
    ax.set_xlabel('Median Markup (Award / Ceiling)', fontsize=11)
    ax.set_title('B. Median Markup by NAICS 2-Digit Sector', fontsize=13)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d73027', alpha=0.7, edgecolor='black', label='LOW $\\rho$ (differentiated)'),
        Patch(facecolor='#fee08b', alpha=0.7, edgecolor='black', label='MEDIUM $\\rho$'),
        Patch(facecolor='#1a9850', alpha=0.7, edgecolor='black', label='HIGH $\\rho$ (commodity)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{fig_path_base}_boxplot.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{fig_path_base}_boxplot.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path_base}_boxplot.png/pdf")

    # ----------------------------------------------------------
    # Figure 2: Scatter plot with regression line
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Jittered scatter of markup vs substitutability
    ax = axes[0]
    np.random.seed(42)
    for cat, color, label in zip(
        ['LOW', 'MEDIUM', 'HIGH'],
        ['#d73027', '#fee08b', '#1a9850'],
        ['LOW (differentiated)', 'MEDIUM', 'HIGH (commodity)']
    ):
        subset = df[df['subst_category'] == cat].sample(min(2000, len(df[df['subst_category'] == cat])))
        jitter = np.random.uniform(-0.15, 0.15, len(subset))
        x_val = SUBST_NUMERIC[cat]
        ax.scatter(x_val + jitter, subset['markup'],
                   alpha=0.08, s=5, color=color, label=label)

    # Add regression line
    if models is not None and 'm1' in models:
        m = models['m1']
        x_line = np.array([0.5, 3.5])
        y_line = m.params['Intercept'] + m.params['subst_numeric'] * x_line
        ax.plot(x_line, y_line, 'k-', linewidth=2,
                label=f'OLS: $\\beta$ = {m.params["subst_numeric"]:.4f} (p={m.pvalues["subst_numeric"]:.1e})')

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['LOW $\\rho$\n(Differentiated)', 'MEDIUM $\\rho$', 'HIGH $\\rho$\n(Commodity)'])
    ax.set_ylabel('Markup (Award / Ceiling)', fontsize=12)
    ax.set_title('A. Markup vs Product Substitutability\n(subsample, N=6000)', fontsize=13)
    ax.set_ylim(0, 2.5)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, loc='upper left')

    # Panel B: NAICS 4-digit sector medians with trend
    ax = axes[1]
    if sector_stats is not None:
        ss = sector_stats[sector_stats['n'] >= 100].copy()
        color_map_cat = {'LOW': '#d73027', 'MEDIUM': '#fee08b', 'HIGH': '#1a9850'}
        for cat in ['LOW', 'MEDIUM', 'HIGH']:
            subset = ss[ss['subst_category'] == cat]
            ax.scatter(subset['n'], subset['median_markup'],
                       c=color_map_cat[cat], edgecolors='black', linewidth=0.5,
                       s=60, alpha=0.7, label=f'{cat} $\\rho$',
                       zorder=3)

        ax.set_xscale('log')
        ax.set_xlabel('Number of Contracts (log scale)', fontsize=11)
        ax.set_ylabel('Median Markup', fontsize=11)
        ax.set_title('B. NAICS 4-digit Sector Median Markup\nvs Category Size', fontsize=13)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{fig_path_base}_scatter.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{fig_path_base}_scatter.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path_base}_scatter.png/pdf")

    # ----------------------------------------------------------
    # Figure 3: Regression coefficient comparison
    # ----------------------------------------------------------
    if models is not None:
        fig, ax = plt.subplots(figsize=(8, 5))

        model_names = ['Bivariate', '+ Competition\ncontrols', '+ Full\ncontrols', 'Category\ndummies']
        betas = []
        ci_low = []
        ci_high = []
        pvals = []

        for key in ['m1', 'm2', 'm3']:
            m = models[key]
            b = m.params['subst_numeric']
            se = m.bse['subst_numeric']
            betas.append(b)
            ci_low.append(b - 1.96 * se)
            ci_high.append(b + 1.96 * se)
            pvals.append(m.pvalues['subst_numeric'])

        # For model 4, show LOW dummy coefficient (HIGH is reference)
        m4 = models['m4']
        b4 = m4.params['cat_LOW']
        se4 = m4.bse['cat_LOW']
        betas.append(b4)
        ci_low.append(b4 - 1.96 * se4)
        ci_high.append(b4 + 1.96 * se4)
        pvals.append(m4.pvalues['cat_LOW'])

        y_pos = np.arange(len(model_names))
        errors = [[b - cl for b, cl in zip(betas, ci_low)],
                  [ch - b for b, ch in zip(betas, ci_high)]]

        colors_bar = ['#4393c3' if p < 0.05 else '#d6604d' for p in pvals]
        ax.barh(y_pos, betas, xerr=errors, color=colors_bar, edgecolor='black',
                alpha=0.7, height=0.5, capsize=4)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names, fontsize=11)
        ax.set_xlabel('Coefficient on Substitutability\n(Models 1-3: linear; Model 4: LOW vs HIGH dummy)', fontsize=11)
        ax.set_title('Regression Coefficients Across Specifications\n'
                      'Prediction: $\\beta_{subst}$ < 0 (more substitutable $\\Rightarrow$ lower markup)',
                      fontsize=12)

        # Add p-value annotations
        for i, (b, p) in enumerate(zip(betas, pvals)):
            star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            ax.annotate(f'p={p:.1e} {star}', xy=(b, i),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=9, va='center')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4393c3', alpha=0.7, edgecolor='black', label='p < 0.05'),
            Patch(facecolor='#d6604d', alpha=0.7, edgecolor='black', label='p >= 0.05'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{fig_path_base}_coefficients.png', dpi=200, bbox_inches='tight')
        plt.savefig(f'{fig_path_base}_coefficients.pdf', bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path_base}_coefficients.png/pdf")

    # ----------------------------------------------------------
    # Figure 4: Competition interaction effect
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # Bin offers and show markup by category
    df_plot = df[df['num_offers'].notna() & (df['num_offers'] > 0)].copy()
    df_plot['offer_bin'] = pd.cut(df_plot['num_offers'],
                                   bins=[0, 1, 2, 3, 5, 10, 100],
                                   labels=['1', '2', '3', '4-5', '6-10', '11+'])

    for cat, color, ls in zip(
        ['LOW', 'MEDIUM', 'HIGH'],
        ['#d73027', '#fee08b', '#1a9850'],
        ['-', '--', ':']
    ):
        subset = df_plot[df_plot['subst_category'] == cat]
        medians = subset.groupby('offer_bin')['markup'].median()
        counts = subset.groupby('offer_bin')['markup'].count()
        # Only plot bins with >= 30 observations
        valid = counts >= 30
        if valid.sum() > 0:
            ax.plot(medians.index[valid], medians.values[valid],
                    f'{ls}o', color=color, linewidth=2, markersize=8,
                    label=f'{cat} $\\rho$', markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel('Number of Offers Received', fontsize=12)
    ax.set_ylabel('Median Markup (Award / Ceiling)', fontsize=12)
    ax.set_title('Markup by Number of Offers and Product Type\n'
                 '(PoIC prediction: gap between LOW and HIGH persists across offer bins)',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{fig_path_base}_competition.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{fig_path_base}_competition.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path_base}_competition.png/pdf")


# ============================================================
# Section 5: Save Results
# ============================================================

def save_results(df, summary, models, sector_stats):
    """Save regression tables and processed data."""

    # Save processed contract data
    output_cols = [
        'contract_award_unique_key', 'naics_str', 'naics_2digit', 'naics_4digit',
        'naics_sector_name', 'subst_category', 'subst_numeric',
        'award_value', 'ceiling_value', 'markup',
        'num_offers', 'log_offers', 'fully_competed', 'fixed_price',
        'log_ceiling', 'extent_competed', 'type_of_contract_pricing',
        'awarding_agency_name',
    ]
    existing_cols = [c for c in output_cols if c in df.columns]
    df[existing_cols].to_csv(
        os.path.join(DATA_DIR, 'procurement_poic_panel.csv'),
        index=False
    )
    print(f"  Saved: {DATA_DIR}/procurement_poic_panel.csv ({len(df)} rows)")

    # Save sector statistics
    if sector_stats is not None:
        sector_stats.to_csv(
            os.path.join(DATA_DIR, 'procurement_poic_sectors.csv'),
            index=False
        )
        print(f"  Saved: {DATA_DIR}/procurement_poic_sectors.csv")

    # Save regression table as LaTeX
    if models is not None:
        latex_lines = [
            r'\begin{table}[ht]',
            r'\centering',
            r'\caption{Price of Incentive Compatibility: Procurement Markup Regressions}',
            r'\label{tab:poic_regression}',
            r'\begin{tabular}{lcccc}',
            r'\hline\hline',
            r' & (1) & (2) & (3) & (4) \\',
            r' & Bivariate & + Competition & + Full Controls & Category Dummies \\',
            r'\hline',
        ]

        # Substitutability coefficient
        for key, label in [('subst_numeric', 'Substitutability ($\\rho$ proxy)')]:
            vals = []
            for mk in ['m1', 'm2', 'm3']:
                m = models[mk]
                if key in m.params:
                    b = m.params[key]
                    se = m.bse[key]
                    p = m.pvalues[key]
                    star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    vals.append(f'{b:.4f}{star}')
                else:
                    vals.append('')
            # Model 4: show LOW dummy
            m4 = models['m4']
            b4 = m4.params.get('cat_LOW', np.nan)
            se4 = m4.bse.get('cat_LOW', np.nan)
            p4 = m4.pvalues.get('cat_LOW', 1)
            star4 = '***' if p4 < 0.001 else '**' if p4 < 0.01 else '*' if p4 < 0.05 else ''
            vals.append(f'{b4:.4f}{star4}')
            latex_lines.append(f'{label} & {" & ".join(vals)} \\\\')

            # SE row
            se_vals = []
            for mk in ['m1', 'm2', 'm3']:
                m = models[mk]
                if key in m.bse:
                    se_vals.append(f'({m.bse[key]:.4f})')
                else:
                    se_vals.append('')
            se_vals.append(f'({se4:.4f})')
            latex_lines.append(f' & {" & ".join(se_vals)} \\\\')

        # Other controls
        for key, label in [
            ('log_offers', 'Log(offers)'),
            ('fully_competed', 'Fully competed'),
            ('log_ceiling', 'Log(ceiling)'),
            ('fixed_price', 'Fixed price'),
        ]:
            vals = []
            for mk in ['m1', 'm2', 'm3', 'm4']:
                m = models[mk]
                if key in m.params:
                    b = m.params[key]
                    p = m.pvalues[key]
                    star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    vals.append(f'{b:.4f}{star}')
                else:
                    vals.append('')
            latex_lines.append(f'{label} & {" & ".join(vals)} \\\\')

        latex_lines.extend([
            r'\hline',
            f'$R^2$ & {models["m1"].rsquared:.4f} & {models["m2"].rsquared:.4f} & '
            f'{models["m3"].rsquared:.4f} & {models["m4"].rsquared:.4f} \\\\',
            f'N & {models["m1"].nobs:.0f} & {models["m2"].nobs:.0f} & '
            f'{models["m3"].nobs:.0f} & {models["m4"].nobs:.0f} \\\\',
            r'\hline\hline',
            r'\multicolumn{5}{l}{\small HC1 robust standard errors in parentheses.} \\',
            r'\multicolumn{5}{l}{\small $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$} \\',
            r'\multicolumn{5}{l}{\small Model 4: LOW and MEDIUM dummies; HIGH is reference category.} \\',
            r'\end{tabular}',
            r'\end{table}',
        ])

        latex_table = '\n'.join(latex_lines)
        with open(os.path.join(DATA_DIR, 'procurement_poic_table.tex'), 'w') as f:
            f.write(latex_table)
        print(f"  Saved: {DATA_DIR}/procurement_poic_table.tex")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("PROCUREMENT POIC TEST")
    print("Testing CES Free Energy Framework Prediction:")
    print("  Differentiated goods -> higher information rent (markup)")
    print("  Commodity goods -> lower information rent (markup)")
    print("=" * 70)

    # Step 1: Fetch data
    print("\n=== Step 1: Fetching USAspending Procurement Data (FY2023) ===")
    df = fetch_procurement_data()
    if df is None or len(df) == 0:
        print("FATAL: No data fetched")
        sys.exit(1)

    # Step 2: Clean and compute markup
    print("\n=== Step 2: Cleaning Data and Computing Markup ===")
    df = clean_and_compute_markup(df)
    if len(df) < 100:
        print("FATAL: Too few observations after cleaning")
        sys.exit(1)

    # Step 3: Statistical analysis
    print("\n=== Step 3: Statistical Analysis ===")
    summary, models, sector_stats = run_analysis(df)

    # Step 4: Figures
    print("\n=== Step 4: Generating Figures ===")
    make_figures(df, summary, models, sector_stats)

    # Step 5: Save
    print("\n=== Step 5: Saving Results ===")
    save_results(df, summary, models, sector_stats)

    # ----------------------------------------------------------
    # Final Assessment
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("ASSESSMENT: PoIC Prediction Test")
    print("=" * 70)

    if models is not None:
        m3 = models['m3']
        beta = m3.params['subst_numeric']
        p = m3.pvalues['subst_numeric']
        m4 = models['m4']
        beta_low = m4.params['cat_LOW']
        p_low = m4.pvalues['cat_LOW']

        print(f"\nPrediction: beta_substitutability < 0")
        print(f"  (More substitutable products -> lower markup / less information rent)")
        print(f"\nResults (full controls, Model 3):")
        print(f"  beta_subst = {beta:.6f}, p = {p:.2e}")
        if beta < 0:
            if p < 0.05:
                verdict = "SUPPORTED (significant, correct sign)"
            else:
                verdict = "WEAKLY SUPPORTED (correct sign, not significant)"
        else:
            if p < 0.05:
                verdict = "REJECTED (significant, wrong sign)"
            else:
                verdict = "INCONCLUSIVE (wrong sign, not significant)"
        print(f"  Verdict: {verdict}")

        print(f"\nCategory dummies (Model 4, ref=HIGH):")
        print(f"  LOW vs HIGH: {beta_low:.6f} (p = {p_low:.2e})")
        if beta_low > 0:
            print(f"  -> Differentiated products have HIGHER markup than commodities")
            print(f"  -> CONSISTENT with PoIC prediction")
        else:
            print(f"  -> Differentiated products have LOWER markup than commodities")
            print(f"  -> INCONSISTENT with PoIC prediction")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == '__main__':
    main()
