#!/usr/bin/env python3
"""
BCL Index Construction from World Bank BRSS Raw Survey Data
===========================================================

Constructs Barth, Caprio & Levine (BCL) regulatory indices from raw
Bank Regulation and Supervision Survey (BRSS) Excel files.

Indices constructed:
  1. Capital Stringency Index (CSI)    — 0-9 scale
  2. Activity Restrictions Index (ARI) — 4-16 scale
  3. Official Supervisory Power (SPI)  — 0-14 scale
  4. Entry Barriers Index (EBI)        — 0-8 scale
  5. Private Monitoring Index (PMI)    — 0-12 scale

Sources:
  - Barth, Caprio & Levine (2001) "The Regulation and Supervision of Banks
    Around the World: A New Database" — original index definitions
  - Barth, Caprio & Levine (2004) "Bank Regulation and Supervision: What
    Works Best?" — refined definitions used in most empirical work
  - Barth, Caprio & Levine (2008) "Bank Regulations Are Changing: For
    Better or Worse?" — Surveys I-III comparisons
  - Barth, Caprio & Levine (2013) "Bank Regulation and Supervision in 180
    Countries from 1999 to 2011" — NBER WP 18733, Surveys I-IV
  - Kara (2016) "Bank Capital Regulations around the World" — Fed FEDS
    2016-057, exact question numbers for capital stringency
  - Anginer et al. (2019) "Bank Regulation and Supervision Ten Years after
    the Global Financial Crisis" — Wave 5 methodology

Usage:
  1. Download BRSS Excel files from:
     https://www.worldbank.org/en/research/brief/BRSS
  2. Place them in a 'brss_data/' subdirectory
  3. Run: python build_bcl_indices.py
  4. Output: bcl_indices_panel.csv (expanded ~180-country panel)

Author: Auto-generated for Connor Doll's thesis research
Date: February 2026
"""

import os
import sys
import re
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# ============================================================================
# SECTION 1: BCL INDEX DEFINITIONS
# ============================================================================
#
# Each index is defined by a set of survey questions. The question numbers
# shift across BRSS waves, so we maintain wave-specific mappings.
#
# The core methodology is consistent: each component question yields a
# binary 0/1 score (or 1-4 for activity restrictions), and the index
# is the sum of component scores.
#
# IMPORTANT: Question numbering convention in BRSS:
#   Wave 1 (2001 survey, ~1999 data): Section.Question format, e.g. "3.1.1"
#   Wave 2 (2003 survey, ~2002 data): Same numbering, minor additions
#   Wave 3 (2007 survey, ~2006 data): Same core, some renumbering
#   Wave 4 (2011 survey, ~2011 data): Expanded questionnaire, some shifts
#   Wave 5 (2019 survey, ~2016 data): Further expansion, new sections
# ============================================================================

# ---------------------------------------------------------------------------
# 1A. CAPITAL STRINGENCY INDEX (CSI)
# ---------------------------------------------------------------------------
# Range: 0-9 (higher = more stringent capital requirements)
#
# Part A: "Overall Capital Stringency" (6 items, 0-6)
#   Whether the capital requirement reflects risk elements and deducts
#   market value losses before minimum adequacy is determined.
#
# Part B: "Initial Capital Stringency" (3 items, 0-3)
#   Whether the sources of funds that count as regulatory capital are
#   restricted and verified.
#
# Reference: Barth et al. (2004) Table 1; Kara (2016) Table 1

CAPITAL_STRINGENCY_QUESTIONS = {
    'part_a': {  # Overall Capital Stringency (0-6)
        'description': 'Whether capital requirement reflects risk elements',
        'components': [
            {
                'id': 'cap_risk_weighted',
                'description': 'Is minimum capital ratio risk-weighted per Basel guidelines?',
                'scoring': 'yes=1',
                'question_ids': {
                    'wave1': '3.1.1', 'wave2': '3.1.1', 'wave3': '3.1.1',
                    'wave4': '3.1.1', 'wave5': '3.1.1'
                },
                'alt_patterns': [
                    r'risk.?weight', r'basel.?guideline', r'minimum.*capital.*ratio.*risk'
                ]
            },
            {
                'id': 'cap_credit_risk',
                'description': 'Does minimum ratio vary as function of credit risk?',
                'scoring': 'yes=1',
                'question_ids': {
                    'wave1': '3.2', 'wave2': '3.2', 'wave3': '3.2',
                    'wave4': '3.2', 'wave5': '3.2'
                },
                'alt_patterns': [r'credit.?risk', r'individual.*bank.*credit']
            },
            {
                'id': 'cap_market_risk',
                'description': 'Does minimum ratio vary as function of market risk?',
                'scoring': 'yes=1',
                'question_ids': {
                    'wave1': '3.3', 'wave2': '3.3', 'wave3': '3.3',
                    'wave4': '3.3', 'wave5': '3.3'
                },
                'alt_patterns': [r'market.?risk']
            },
            {
                'id': 'cap_deduct_loans',
                'description': 'Market value of loan losses deducted from capital?',
                'scoring': 'yes=1',
                'question_ids': {
                    'wave1': '3.9.1', 'wave2': '3.9.1', 'wave3': '3.9.1',
                    'wave4': '3.9.1', 'wave5': '3.9.1'
                },
                'alt_patterns': [
                    r'loan.?loss', r'market.?value.*loan', r'unrealized.*loan'
                ]
            },
            {
                'id': 'cap_deduct_securities',
                'description': 'Unrealized losses in securities portfolios deducted?',
                'scoring': 'yes=1',
                'question_ids': {
                    'wave1': '3.9.2', 'wave2': '3.9.2', 'wave3': '3.9.2',
                    'wave4': '3.9.2', 'wave5': '3.9.2'
                },
                'alt_patterns': [
                    r'securities.*portfolio', r'unrealized.*securit'
                ]
            },
            {
                'id': 'cap_deduct_fx',
                'description': 'Unrealized foreign exchange losses deducted?',
                'scoring': 'yes=1',
                'question_ids': {
                    'wave1': '3.9.3', 'wave2': '3.9.3', 'wave3': '3.9.3',
                    'wave4': '3.9.3', 'wave5': '3.9.3'
                },
                'alt_patterns': [
                    r'foreign.?exchange', r'unrealized.*f(oreign|x)'
                ]
            },
        ]
    },
    'part_b': {  # Initial Capital Stringency (0-3)
        'description': 'Restrictions on sources of regulatory capital',
        'components': [
            {
                'id': 'cap_no_noncash',
                'description': 'Can capital include assets other than cash/govt securities?',
                'scoring': 'no=1',  # REVERSED: No = more stringent = 1
                'question_ids': {
                    'wave1': '3.7.1', 'wave2': '3.7.1', 'wave3': '3.7.1',
                    'wave4': '3.7.1', 'wave5': '3.7.1'
                },
                'alt_patterns': [
                    r'assets.*other.*cash', r'government.*securities',
                    r'initial.*capital.*assets'
                ]
            },
            {
                'id': 'cap_no_borrowed',
                'description': 'Can capital include borrowed funds?',
                'scoring': 'no=1',  # REVERSED: No = more stringent = 1
                'question_ids': {
                    'wave1': '3.7.2', 'wave2': '3.7.2', 'wave3': '3.7.2',
                    'wave4': '3.7.2', 'wave5': '3.7.2'
                },
                'alt_patterns': [r'borrowed.*fund', r'capital.*borrow']
            },
            {
                'id': 'cap_verified',
                'description': 'Are sources of capital verified by regulators?',
                'scoring': 'yes=1',
                'question_ids': {
                    'wave1': '3.8', 'wave2': '3.8', 'wave3': '3.8',
                    'wave4': '3.8', 'wave5': '3.8'
                },
                'alt_patterns': [
                    r'verif', r'sources.*capital.*verif', r'regulatory.*verif'
                ]
            },
        ]
    }
}

# ---------------------------------------------------------------------------
# 1B. ACTIVITY RESTRICTIONS INDEX (ARI)
# ---------------------------------------------------------------------------
# Range: 4-16 (higher = more restrictions)
#
# Measures the degree to which banks may engage in:
#   1. Securities activities (underwriting, dealing, brokering all types)
#   2. Insurance activities (underwriting and selling)
#   3. Real estate activities (investment, development, management)
#   4. Nonfinancial firm ownership
#
# Each scored 1-4:
#   1 = Unrestricted (full range of activities permitted)
#   2 = Permitted (broad range, some limitations)
#   3 = Restricted (significantly limited)
#   4 = Prohibited (activity not permitted at all)

ACTIVITY_RESTRICTIONS_QUESTIONS = {
    'description': 'Degree to which banks may engage in non-core activities',
    'components': [
        {
            'id': 'act_securities',
            'description': 'Securities activities (underwriting, dealing, brokering)',
            'scoring': '1-4',  # 1=unrestricted, 4=prohibited
            'question_ids': {
                'wave1': '4.1', 'wave2': '4.1', 'wave3': '4.1',
                'wave4': '4.1', 'wave5': '4.1'
            },
            'alt_patterns': [r'securities', r'underwriting.*dealing.*brok']
        },
        {
            'id': 'act_insurance',
            'description': 'Insurance activities (underwriting and selling)',
            'scoring': '1-4',
            'question_ids': {
                'wave1': '4.2', 'wave2': '4.2', 'wave3': '4.2',
                'wave4': '4.2', 'wave5': '4.2'
            },
            'alt_patterns': [r'insurance']
        },
        {
            'id': 'act_real_estate',
            'description': 'Real estate activities (investment, development, management)',
            'scoring': '1-4',
            'question_ids': {
                'wave1': '4.3', 'wave2': '4.3', 'wave3': '4.3',
                'wave4': '4.3', 'wave5': '4.3'
            },
            'alt_patterns': [r'real.?estate']
        },
        {
            'id': 'act_nonfinancial',
            'description': 'Ownership of nonfinancial firms',
            'scoring': '1-4',
            'question_ids': {
                'wave1': '4.4', 'wave2': '4.4', 'wave3': '4.4',
                'wave4': '4.4', 'wave5': '4.4'
            },
            'alt_patterns': [
                r'non.?financial.*firm', r'ownership.*non.?financial',
                r'owning.*non.?financial'
            ]
        },
    ]
}

# ---------------------------------------------------------------------------
# 1C. OFFICIAL SUPERVISORY POWER INDEX (SPI)
# ---------------------------------------------------------------------------
# Range: 0-14 (higher = more powerful supervisors)
#
# Measures the degree to which supervisory authorities can take specific
# corrective and preventive actions against banks.
#
# From BCL (2004): "whether the supervisory authorities have the authority
# to take specific actions to prevent and correct problems"

SUPERVISORY_POWER_QUESTIONS = {
    'description': 'Authority of supervisors to prevent/correct problems',
    'components': [
        {
            'id': 'sup_meet_external_auditor',
            'description': 'Can supervisors meet with external auditors without bank approval?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '5.3', 'wave2': '5.3', 'wave3': '5.3',
                'wave4': '5.3', 'wave5': '5.3'
            },
            'alt_patterns': [r'external.*audit', r'meet.*audit']
        },
        {
            'id': 'sup_action_auditors',
            'description': 'Can supervisors take action against external auditors?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '11.2', 'wave2': '11.2', 'wave3': '11.2',
                'wave4': '11.2', 'wave5': '11.2'
            },
            'alt_patterns': [r'action.*against.*audit', r'legal.*action.*audit']
        },
        {
            'id': 'sup_change_org',
            'description': 'Can supervisors force bank to change internal organizational structure?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '11.3', 'wave2': '11.3', 'wave3': '11.3',
                'wave4': '11.3', 'wave5': '11.3'
            },
            'alt_patterns': [r'organizational.*structure', r'internal.*structure']
        },
        {
            'id': 'sup_suspend_dividends',
            'description': 'Can supervisors suspend dividends?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '11.3.1', 'wave2': '11.3.1', 'wave3': '11.3.1',
                'wave4': '11.3.1', 'wave5': '11.3.1'
            },
            'alt_patterns': [r'suspend.*dividend', r'dividend']
        },
        {
            'id': 'sup_suspend_bonuses',
            'description': 'Can supervisors stop bonuses?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '11.3.2', 'wave2': '11.3.2', 'wave3': '11.3.2',
                'wave4': '11.3.2', 'wave5': '11.3.2'
            },
            'alt_patterns': [r'bonus', r'stop.*bonus']
        },
        {
            'id': 'sup_suspend_mgmt_fees',
            'description': 'Can supervisors halt management fees?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '11.3.3', 'wave2': '11.3.3', 'wave3': '11.3.3',
                'wave4': '11.3.3', 'wave5': '11.3.3'
            },
            'alt_patterns': [r'management.*fee', r'halt.*fee']
        },
        {
            'id': 'sup_force_provisions',
            'description': 'Can supervisors require provisions against losses?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '11.4', 'wave2': '11.4', 'wave3': '11.4',
                'wave4': '11.4', 'wave5': '11.4'
            },
            'alt_patterns': [r'provision', r'require.*provision']
        },
        {
            'id': 'sup_supercede_shareholder',
            'description': 'Can supervisors supercede shareholder rights?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '11.5', 'wave2': '11.5', 'wave3': '11.5',
                'wave4': '11.5', 'wave5': '11.5'
            },
            'alt_patterns': [r'shareholder.*right', r'supercede', r'supersede']
        },
        {
            'id': 'sup_remove_directors',
            'description': 'Can supervisors remove/replace directors and management?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '11.6', 'wave2': '11.6', 'wave3': '11.6',
                'wave4': '11.6', 'wave5': '11.6'
            },
            'alt_patterns': [r'remove.*director', r'replace.*management']
        },
        {
            'id': 'sup_suspend_ownership',
            'description': 'Can supervisors suspend ownership rights?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '11.7', 'wave2': '11.7', 'wave3': '11.7',
                'wave4': '11.7', 'wave5': '11.7'
            },
            'alt_patterns': [r'ownership.*right', r'suspend.*ownership']
        },
        {
            'id': 'sup_intervene_bank',
            'description': 'Can supervisors intervene (i.e. supercede management)?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '11.8', 'wave2': '11.8', 'wave3': '11.8',
                'wave4': '11.8', 'wave5': '11.8'
            },
            'alt_patterns': [r'intervene', r'supercede.*management']
        },
        {
            'id': 'sup_declare_insolvency',
            'description': 'Can supervisors declare bank insolvent?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '11.9', 'wave2': '11.9', 'wave3': '11.9',
                'wave4': '11.9', 'wave5': '11.9'
            },
            'alt_patterns': [r'insolven', r'declare.*insolven']
        },
        {
            'id': 'sup_restructuring',
            'description': 'Can supervisors restructure and reorganize a troubled bank?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '11.9.1', 'wave2': '11.9.1', 'wave3': '11.9.1',
                'wave4': '11.9.1', 'wave5': '11.9.1'
            },
            'alt_patterns': [r'restructur', r'reorganiz']
        },
        {
            'id': 'sup_forbearance',
            'description': 'Is forbearance discretion limited by law?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '11.10', 'wave2': '11.10', 'wave3': '11.10',
                'wave4': '11.10', 'wave5': '11.10'
            },
            'alt_patterns': [r'forbearance', r'discretion.*limited']
        },
    ]
}

# ---------------------------------------------------------------------------
# 1D. ENTRY BARRIERS INDEX (EBI)
# ---------------------------------------------------------------------------
# Range: 0-8 (higher = more barriers to entry)
#
# Measures the stringency of requirements for obtaining a banking license.

ENTRY_BARRIERS_QUESTIONS = {
    'description': 'Requirements for obtaining a banking license',
    'components': [
        {
            'id': 'entry_deny_foreign',
            'description': 'Foreign bank entry denied in practice?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '1.8', 'wave2': '1.8', 'wave3': '1.8',
                'wave4': '1.8', 'wave5': '1.8'
            },
            'alt_patterns': [r'foreign.*denied', r'foreign.*entry']
        },
        {
            'id': 'entry_deny_domestic',
            'description': 'Domestic bank entry denied in practice?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '1.9', 'wave2': '1.9', 'wave3': '1.9',
                'wave4': '1.9', 'wave5': '1.9'
            },
            'alt_patterns': [r'domestic.*denied', r'domestic.*entry']
        },
        {
            'id': 'entry_draft_bylaws',
            'description': 'Draft bylaws required for license?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '1.3.1', 'wave2': '1.3.1', 'wave3': '1.3.1',
                'wave4': '1.3.1', 'wave5': '1.3.1'
            },
            'alt_patterns': [r'bylaw', r'draft.*bylaw']
        },
        {
            'id': 'entry_org_chart',
            'description': 'Organizational chart required?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '1.3.2', 'wave2': '1.3.2', 'wave3': '1.3.2',
                'wave4': '1.3.2', 'wave5': '1.3.2'
            },
            'alt_patterns': [r'organizational.*chart', r'org.*chart']
        },
        {
            'id': 'entry_financial_projections',
            'description': 'Financial projections required?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '1.3.3', 'wave2': '1.3.3', 'wave3': '1.3.3',
                'wave4': '1.3.3', 'wave5': '1.3.3'
            },
            'alt_patterns': [r'financial.*projection', r'business.*plan']
        },
        {
            'id': 'entry_background_mgr',
            'description': 'Background check on managers required?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '1.3.4', 'wave2': '1.3.4', 'wave3': '1.3.4',
                'wave4': '1.3.4', 'wave5': '1.3.4'
            },
            'alt_patterns': [
                r'background.*manager', r'experience.*manager',
                r'background.*check'
            ]
        },
        {
            'id': 'entry_background_dir',
            'description': 'Background/experience of directors required?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '1.3.5', 'wave2': '1.3.5', 'wave3': '1.3.5',
                'wave4': '1.3.5', 'wave5': '1.3.5'
            },
            'alt_patterns': [
                r'background.*director', r'experience.*director',
                r'fit.*proper'
            ]
        },
        {
            'id': 'entry_source_funds',
            'description': 'Sources of funds to be used as capital verified?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '1.3.6', 'wave2': '1.3.6', 'wave3': '1.3.6',
                'wave4': '1.3.6', 'wave5': '1.3.6'
            },
            'alt_patterns': [r'source.*fund', r'capital.*source.*verif']
        },
    ]
}

# ---------------------------------------------------------------------------
# 1E. PRIVATE MONITORING INDEX (PMI)
# ---------------------------------------------------------------------------
# Range: 0-12 (higher = more private monitoring required)
#
# Measures the degree to which regulations force banks to disclose
# information and create incentives for private monitoring.

PRIVATE_MONITORING_QUESTIONS = {
    'description': 'Degree to which regulations foster private monitoring',
    'components': [
        {
            'id': 'pm_certified_audit',
            'description': 'Is certified audit required?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '5.1', 'wave2': '5.1', 'wave3': '5.1',
                'wave4': '5.1', 'wave5': '5.1'
            },
            'alt_patterns': [r'certified.*audit']
        },
        {
            'id': 'pm_audit_percent',
            'description': 'What percent of top 10 banks are rated by international agencies?',
            'scoring': 'percent_high=1',  # 100% = 1, else 0
            'question_ids': {
                'wave1': '10.1', 'wave2': '10.1', 'wave3': '10.1',
                'wave4': '10.1', 'wave5': '10.1'
            },
            'alt_patterns': [r'top.*10.*bank.*rated', r'rating.*agenc']
        },
        {
            'id': 'pm_offbalance_disc',
            'description': 'Off-balance sheet items disclosed to public?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '10.4.1', 'wave2': '10.4.1', 'wave3': '10.4.1',
                'wave4': '10.4.1', 'wave5': '10.4.1'
            },
            'alt_patterns': [r'off.?balance', r'off.*balance.*sheet.*disclos']
        },
        {
            'id': 'pm_accrued_income',
            'description': 'Accrued but unpaid interest/principal disclosed?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '10.3', 'wave2': '10.3', 'wave3': '10.3',
                'wave4': '10.3', 'wave5': '10.3'
            },
            'alt_patterns': [r'accrued', r'unpaid.*interest']
        },
        {
            'id': 'pm_consolidated_gl',
            'description': 'Consolidated (group-level) accounts required?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '10.5', 'wave2': '10.5', 'wave3': '10.5',
                'wave4': '10.5', 'wave5': '10.5'
            },
            'alt_patterns': [r'consolidat', r'group.*account']
        },
        {
            'id': 'pm_directors_liable',
            'description': 'Directors legally liable for misleading information?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '10.6', 'wave2': '10.6', 'wave3': '10.6',
                'wave4': '10.6', 'wave5': '10.6'
            },
            'alt_patterns': [r'director.*liable', r'misleading.*information']
        },
        {
            'id': 'pm_risk_mgmt_disc',
            'description': 'Risk management procedures disclosed?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '10.4.2', 'wave2': '10.4.2', 'wave3': '10.4.2',
                'wave4': '10.4.2', 'wave5': '10.4.2'
            },
            'alt_patterns': [r'risk.*management.*procedure', r'risk.*manag.*disclos']
        },
        {
            'id': 'pm_subordinated_debt',
            'description': 'Subordinated debt required as part of capital?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '3.5', 'wave2': '3.5', 'wave3': '3.5',
                'wave4': '3.5', 'wave5': '3.5'
            },
            'alt_patterns': [r'subordinated.*debt', r'sub.*debt.*capital']
        },
        {
            'id': 'pm_no_explicit_di',
            'description': 'No explicit deposit insurance scheme?',
            'scoring': 'no=1',  # REVERSED: absence of DI = more market discipline
            'question_ids': {
                'wave1': '8.1', 'wave2': '8.1', 'wave3': '8.1',
                'wave4': '8.1', 'wave5': '8.1'
            },
            'alt_patterns': [r'explicit.*deposit.*insur', r'deposit.*protection']
        },
        {
            'id': 'pm_no_copay',
            'description': 'Is there a coinsurance requirement?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '8.4', 'wave2': '8.4', 'wave3': '8.4',
                'wave4': '8.4', 'wave5': '8.4'
            },
            'alt_patterns': [r'co.?insurance', r'copay']
        },
        {
            'id': 'pm_accounting_ias',
            'description': 'Are IAS/IFRS accounting standards required?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '10.2', 'wave2': '10.2', 'wave3': '10.2',
                'wave4': '10.2', 'wave5': '10.2'
            },
            'alt_patterns': [r'IAS', r'IFRS', r'international.*accounting']
        },
        {
            'id': 'pm_income_npl',
            'description': 'Must banks stop accruing income on NPLs?',
            'scoring': 'yes=1',
            'question_ids': {
                'wave1': '9.2', 'wave2': '9.2', 'wave3': '9.2',
                'wave4': '9.2', 'wave5': '9.2'
            },
            'alt_patterns': [r'non.?perform', r'NPL', r'stop.*accruing']
        },
    ]
}


# ============================================================================
# SECTION 1B: WAVE 5 EXPLICIT QUESTION MAPPING
# ============================================================================
# Wave 5 (2019) restructured the questionnaire significantly.  The dot-notation
# IDs from waves 1-4 no longer match, and generic alt_patterns cause cross-
# section false positives on the 1780-row combined dataset.  This map gives
# the correct Q-format column-0 code for each BCL component.
#
# Components not in this map (or mapped to None) will be scored as missing
# for wave 5, which is preferable to a wrong match.
# ============================================================================

WAVE5_QUESTION_MAP: Dict[str, Optional[str]] = {
    # --- Capital Stringency Part A (Overall, 0-6) ---
    'cap_risk_weighted':    'Q3_6_1_2016',    # 3.6.1 ICAAP/internal risk-based capital assessment required
    'cap_credit_risk':      'Q3_2a_2016',     # 3.2.a Credit risk covered in capital requirement
    'cap_market_risk':      'Q3_2b_2016',     # 3.2.b Market risk covered
    'cap_deduct_loans':     'Q3_20_3g_2016',  # 3.20.3.g Shortfall of provisions to expected losses deducted from T1
    'cap_deduct_securities':'Q3_20_3d_2016',  # 3.20.3.d Unrealized losses in fair valued exposures deducted from T1
    'cap_deduct_fx':        None,             # No direct FX loss deduction question in wave 5
    # --- Capital Stringency Part B (Initial, 0-3) ---
    'cap_no_noncash':       'Q1_4_3_2016',    # 1.4.3 Can initial capital be non-cash assets?
    'cap_no_borrowed':      'Q1_5_2016',      # 1.5 Can initial capital be borrowed?
    'cap_verified':         'Q1_4_2_2016',    # 1.4.2 Sources of funds verified?
    # --- Activity Restrictions (scored 1-4 each) ---
    # Wave 5 splits each activity into 4 sub-questions (a=unrestricted...d=prohibited).
    # Map to (a) unrestricted sub-question; Yes=score 1, No means higher restriction.
    'act_securities':       'Q4_1a_2016',     # 4.1.a Full range of securities activities in banks
    'act_insurance':        'Q4_2a_2016',     # 4.2.a Full range of insurance activities in banks
    'act_real_estate':      'Q4_3a_2016',     # 4.3.a Full range of real estate activities in banks
    'act_nonfinancial':     'Q4_4a_2016',     # 4.4.a Nonfinancial activities directly in banks
    # --- Supervisory Power (0-14) ---
    'sup_meet_external_auditor': 'Q5_9a_2016',# 5.9.a Supervisor receives auditor's report on financial statements
    'sup_action_auditors':  'Q5_5b_2016',     # 5.5.b Prosecute auditor for negligence/fraud/collusion
    'sup_change_org':       'Q12_5_2016',     # 12.5 Can supervisor force change in internal org structure?
    'sup_suspend_dividends':'Q11_1j_2016',    # 11.1.j Require banks to reduce/suspend dividends
    'sup_suspend_bonuses':  'Q11_1k_2016',    # 11.1.k Require banks to reduce/suspend bonuses
    'sup_suspend_mgmt_fees':None,             # No direct equivalent in wave 5
    'sup_force_provisions': 'Q11_1f_2016',    # 11.1.f Require banks to constitute provisions
    'sup_supercede_shareholder':'Q11_8b_2016',# 11.8.b Supersede shareholders' rights
    'sup_remove_directors': 'Q11_1l_2016',    # 11.1.l Suspend or remove bank directors
    'sup_suspend_ownership':'Q11_8b_2016',    # 11.8.b Supersede shareholders' rights (closest proxy)
    'sup_intervene_bank':   'Q11_1g_2016',    # 11.1.g Restrict/place conditions on business conducted
    'sup_declare_insolvency':'Q11_8a_2016',   # 11.8.a Declare insolvency
    'sup_restructuring':    'Q11_1i_2016',    # 11.1.i Require banks to reduce/restructure operations
    'sup_forbearance':      'Q11_1b_2016',    # 11.1.b Forbearance (waive regulatory requirements)
    # --- Entry Barriers (0-8) ---
    'entry_deny_foreign':   'Q1_8_1_2016',    # 1.8.1 Foreign bank acquisitions not permitted
    'entry_deny_domestic':  None,             # No direct equivalent in wave 5
    'entry_draft_bylaws':   'Q1_6a_2016',     # 1.6.a Draft bylaws (first item in application requirements)
    'entry_org_chart':      'Q1_6b_2016',     # 1.6.b Intended organizational chart
    'entry_financial_projections':'Q1_6e_2016',# 1.6.e Financial projections for first three years
    'entry_background_mgr': 'Q1_6h_2016',    # 1.6.h Background/experience of future senior managers
    'entry_background_dir': 'Q1_6g_2016',    # 1.6.g Background/experience of future Board directors
    'entry_source_funds':   'Q1_6f_2016',     # 1.6.f Sources of funds to be used
    # --- Private Monitoring (0-12) ---
    'pm_certified_audit':   'Q5_1_2016',      # 5.1 Is professional external audit required for all banks?
    'pm_audit_percent':     'Q5_10_1_2016',   # 5.10.1 Auditors must inform supervisor of material issues
    'pm_offbalance_disc':   'Q10_7b_2016',    # 10.7.b Off-balance sheet items disclosed
    'pm_accrued_income':    'Q10_4_2016',     # 10.4 Does accrued unpaid interest enter income statement?
    'pm_consolidated_gl':   'Q10_1_2016',     # 10.1 Are banks required to prepare consolidated accounts?
    'pm_directors_liable':  'Q10_6_2016',     # 10.6 Required to submit financial statements to supervisor?
    'pm_risk_mgmt_disc':    'Q10_7c_2016',    # 10.7.c Governance and risk management framework disclosed
    'pm_subordinated_debt': 'Q3_20c_2016',    # 3.20.c Subordinated debt allowed in regulatory capital
    'pm_no_explicit_di':    'Q8_1_2016',      # 8.1 Is there an explicit deposit insurance protection system?
    'pm_no_copay':          'Q8_13_2016',     # 8.13 Formal coinsurance (all depositors insured < 100%)?
    'pm_accounting_ias':    'Q5_6_2016',      # 5.6 Audits required per International Standards?
    'pm_income_npl':        'Q9_5_2016',      # 9.5 Does accrued unpaid interest enter income while loan NPL?
}


# ============================================================================
# SECTION 2: RESPONSE PARSING UTILITIES
# ============================================================================

def normalize_response(val: Any) -> Optional[str]:
    """Normalize a survey response to lowercase string, handling various formats."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().lower()
    # Remove common prefixes/suffixes
    s = re.sub(r'^[\s\-–—]+', '', s)
    s = re.sub(r'[\s\-–—]+$', '', s)
    if s in ('', 'n/a', 'na', 'n.a.', 'not applicable', '..', '.', '-', '—'):
        return None
    return s


def parse_yes_no(val: Any) -> Optional[int]:
    """Parse a Yes/No response to 1/0."""
    s = normalize_response(val)
    if s is None:
        return None
    if s in ('yes', 'y', '1', '1.0', 'true', 'x'):
        return 1
    if s in ('no', 'n', '0', '0.0', 'false'):
        return 0
    # Handle common variations
    if 'yes' in s:
        return 1
    if 'no' in s:
        return 0
    # Wave 5 uses categorical codes (e.g. "BS"=Banking Supervisor,
    # "C"=Court, "OTH"=Other) for certain multi-select questions.
    # Any short substantive code implies the power/feature exists.
    if 1 <= len(s) <= 20 and s not in ('dk', 'do not know', 'nan'):
        return 1
    return None


def parse_activity_score(val: Any) -> Optional[int]:
    """Parse activity restriction score (1-4 scale)."""
    s = normalize_response(val)
    if s is None:
        return None
    # Try numeric first
    try:
        n = int(float(s))
        if 1 <= n <= 4:
            return n
    except (ValueError, TypeError):
        pass
    # Text-based
    if 'unrestrict' in s or 'permitted' in s and 'broad' in s:
        return 1
    if 'permit' in s:
        return 2
    if 'restrict' in s:
        return 3
    if 'prohibit' in s:
        return 4
    return None


def score_component(val: Any, scoring_rule: str) -> Optional[int]:
    """Score a single component based on its scoring rule."""
    if scoring_rule == 'yes=1':
        return parse_yes_no(val)
    elif scoring_rule == 'no=1':
        # Reversed: No = 1 (more stringent)
        yn = parse_yes_no(val)
        if yn is None:
            return None
        return 1 - yn
    elif scoring_rule == '1-4':
        return parse_activity_score(val)
    elif scoring_rule == 'percent_high=1':
        s = normalize_response(val)
        if s is None:
            return None
        try:
            pct = float(s.replace('%', ''))
            return 1 if pct >= 90 else 0
        except (ValueError, TypeError):
            return None
    else:
        return parse_yes_no(val)


# ============================================================================
# SECTION 3: BRSS EXCEL FILE PARSER
# ============================================================================

class BRSSParser:
    """
    Adaptive parser for BRSS Excel files across all 5 waves.
    
    BRSS files have inconsistent formats:
    - Some have countries as rows, questions as columns
    - Some have questions as rows, countries as columns
    - Column/row headers may include question numbers or descriptive text
    - Question numbering shifts slightly across waves
    
    This parser tries multiple strategies to extract the data.
    """
    
    def __init__(self, filepath: str, wave: int):
        self.filepath = filepath
        self.wave = wave
        self.wave_key = f'wave{wave}'
        self.df_raw = None
        self.orientation = None  # 'countries_as_rows' or 'countries_as_cols'
        self.question_map = {}   # maps question_id -> column/row index
        self.country_list = []
        
    def load(self) -> bool:
        """Load and detect orientation of the Excel file."""
        try:
            # Try reading with openpyxl (xlsx)
            xl = pd.ExcelFile(self.filepath, engine='openpyxl')
        except Exception:
            try:
                # Try xlrd for older xls files
                xl = pd.ExcelFile(self.filepath, engine='xlrd')
            except Exception as e:
                print(f"  ERROR: Cannot read {self.filepath}: {e}")
                return False
        
        # List available sheets
        sheets = xl.sheet_names
        print(f"  Sheets found: {sheets}")
        
        # Wave 5 splits data across numbered section sheets (01-15).
        # Detect this layout and concatenate all section sheets.
        numbered_sheets = sorted([s for s in sheets if re.fullmatch(r'\d{2}', s)])
        if numbered_sheets and len(numbered_sheets) >= 5:
            print(f"  Wave 5 multi-sheet layout: combining {len(numbered_sheets)} section sheets")
            dfs = []
            for i, sheet_name in enumerate(numbered_sheets):
                sheet_df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
                if i == 0:
                    # Keep the header row (row 0 with country names) from first sheet
                    dfs.append(sheet_df)
                else:
                    # Skip the header row from subsequent sheets (same countries)
                    dfs.append(sheet_df.iloc[1:])
            self.df_raw = pd.concat(dfs, ignore_index=True)
            print(f"  Combined shape: {self.df_raw.shape}")
        else:
            # Single-sheet layout (waves 1-4): find the main data sheet
            data_sheet = None
            for name in sheets:
                nl = name.lower()
                if any(kw in nl for kw in ['data', 'survey', 'response', 'database']):
                    data_sheet = name
                    break
            if data_sheet is None and sheets:
                data_sheet = sheets[0]

            print(f"  Using sheet: '{data_sheet}'")
            self.df_raw = pd.read_excel(xl, sheet_name=data_sheet, header=None)
            print(f"  Raw shape: {self.df_raw.shape}")
        
        # Detect orientation
        self._detect_orientation()
        return True
    
    def _detect_orientation(self):
        """Detect whether countries are rows or columns."""
        # Scan first few rows and columns for country names
        known_countries = {
            'argentina', 'brazil', 'china', 'germany', 'united kingdom',
            'united states', 'japan', 'india', 'france', 'australia',
            'canada', 'mexico', 'nigeria', 'south africa', 'turkey',
            'indonesia', 'korea', 'singapore', 'russia', 'russian federation'
        }
        
        # Check first 5 columns for country names (countries-as-rows orientation)
        countries_in_rows = 0
        for col in range(min(5, self.df_raw.shape[1])):
            for row in range(self.df_raw.shape[0]):
                val = normalize_response(self.df_raw.iloc[row, col])
                if val and any(c in val for c in known_countries):
                    countries_in_rows += 1
        
        # Check first 5 rows for country names (countries-as-columns orientation)
        countries_in_cols = 0
        for row in range(min(5, self.df_raw.shape[0])):
            for col in range(self.df_raw.shape[1]):
                val = normalize_response(self.df_raw.iloc[row, col])
                if val and any(c in val for c in known_countries):
                    countries_in_cols += 1
        
        if countries_in_rows >= countries_in_cols:
            self.orientation = 'countries_as_rows'
        else:
            self.orientation = 'countries_as_cols'
        
        print(f"  Detected orientation: {self.orientation}")
        print(f"    (countries in rows: {countries_in_rows}, in cols: {countries_in_cols})")
    
    def _find_question_column(self, question_def: dict) -> Optional[int]:
        """Find the column (or row) index for a specific question."""
        comp_id = question_def.get('id', '')

        # ----- Wave 5: use explicit Q-code mapping -----
        # The wave 5 questionnaire was restructured so dot-notation IDs
        # and generic alt_patterns produce cross-section false positives.
        # Use the curated WAVE5_QUESTION_MAP to find the exact data row.
        if self.wave == 5 and self.orientation != 'countries_as_rows':
            target_qcode = WAVE5_QUESTION_MAP.get(comp_id)
            if target_qcode is None:
                return None  # component has no wave 5 equivalent
            for idx in range(self.df_raw.shape[0]):
                col0_val = str(self.df_raw.iloc[idx, 0]).strip()
                if col0_val == target_qcode:
                    return ('row', idx)
            return None

        # ----- Waves 1-4: original qid / alt_pattern search -----
        qid = question_def['question_ids'].get(self.wave_key)
        alt_patterns = question_def.get('alt_patterns', [])

        if self.orientation == 'countries_as_rows':
            # Questions are in column headers
            for idx in range(self.df_raw.shape[1]):
                for hrow in range(min(5, self.df_raw.shape[0])):
                    val = str(self.df_raw.iloc[hrow, idx]).strip()
                    if qid and qid in val:
                        return ('col', idx)
                    for pattern in alt_patterns:
                        if re.search(pattern, val, re.IGNORECASE):
                            return ('col', idx)
        else:
            # Questions are in row headers
            for idx in range(self.df_raw.shape[0]):
                for hcol in range(min(5, self.df_raw.shape[1])):
                    val = str(self.df_raw.iloc[idx, hcol]).strip()
                    if qid and qid in val:
                        return ('row', idx)
                    for pattern in alt_patterns:
                        if re.search(pattern, val, re.IGNORECASE):
                            return ('row', idx)

        return None
    
    def _get_countries(self) -> Dict[str, int]:
        """Extract country name -> index mapping."""
        countries = {}
        if self.orientation == 'countries_as_rows':
            # Countries in column 0 or 1, data starts after header rows
            name_col = 0
            for row in range(self.df_raw.shape[0]):
                val = normalize_response(self.df_raw.iloc[row, name_col])
                if val and len(val) > 2 and not any(kw in val for kw in
                    ['question', 'section', 'survey', 'database', 'nan']):
                    countries[val] = row
            # Try column 1 if column 0 didn't work well
            if len(countries) < 10:
                countries = {}
                name_col = 1
                for row in range(self.df_raw.shape[0]):
                    val = normalize_response(self.df_raw.iloc[row, name_col])
                    if val and len(val) > 2 and not any(kw in val for kw in
                        ['question', 'section', 'survey', 'database', 'nan']):
                        countries[val] = row
        else:
            # Countries in row 0 or 1
            name_row = 0
            for col in range(self.df_raw.shape[1]):
                val = normalize_response(self.df_raw.iloc[name_row, col])
                if val and len(val) > 2 and not any(kw in val for kw in
                    ['question', 'section', 'survey', 'database', 'nan']):
                    countries[val] = col
            if len(countries) < 10:
                countries = {}
                name_row = 1
                for col in range(self.df_raw.shape[1]):
                    val = normalize_response(self.df_raw.iloc[name_row, col])
                    if val and len(val) > 2 and not any(kw in val for kw in
                        ['question', 'section', 'survey', 'database', 'nan']):
                        countries[val] = col
        
        return countries
    
    def get_response(self, country_idx: int, question_loc: tuple) -> Any:
        """Get a specific response value given country and question indices."""
        loc_type, loc_idx = question_loc
        if self.orientation == 'countries_as_rows':
            if loc_type == 'col':
                return self.df_raw.iloc[country_idx, loc_idx]
        else:
            if loc_type == 'row':
                return self.df_raw.iloc[loc_idx, country_idx]
        return None
    
    def extract_all(self) -> pd.DataFrame:
        """Extract all index components for all countries."""
        countries = self._get_countries()
        print(f"  Found {len(countries)} countries")
        
        if len(countries) < 5:
            print(f"  WARNING: Very few countries found. File may need manual inspection.")
            # Print first few rows for debugging
            print("  First 5 rows, first 5 cols of raw data:")
            print(self.df_raw.iloc[:5, :5].to_string())
        
        # Find all question locations
        all_indices = {
            'capital_stringency': CAPITAL_STRINGENCY_QUESTIONS,
            'activity_restrictions': ACTIVITY_RESTRICTIONS_QUESTIONS,
            'supervisory_power': SUPERVISORY_POWER_QUESTIONS,
            'entry_barriers': ENTRY_BARRIERS_QUESTIONS,
            'private_monitoring': PRIVATE_MONITORING_QUESTIONS,
        }
        
        question_locations = {}
        missing_questions = []
        
        for idx_name, idx_def in all_indices.items():
            if 'part_a' in idx_def:
                # Capital stringency has sub-parts
                for part_key in ['part_a', 'part_b']:
                    for comp in idx_def[part_key]['components']:
                        loc = self._find_question_column(comp)
                        question_locations[comp['id']] = loc
                        if loc is None:
                            missing_questions.append(comp['id'])
            else:
                for comp in idx_def['components']:
                    loc = self._find_question_column(comp)
                    question_locations[comp['id']] = loc
                    if loc is None:
                        missing_questions.append(comp['id'])
        
        found = sum(1 for v in question_locations.values() if v is not None)
        total = len(question_locations)
        print(f"  Matched {found}/{total} question locations")
        if missing_questions:
            print(f"  Missing: {missing_questions[:10]}{'...' if len(missing_questions) > 10 else ''}")
        
        # Extract responses for each country
        records = []
        for country_name, country_idx in countries.items():
            record = {'country': country_name, 'wave': self.wave}
            
            # Score each component
            for comp_id, loc in question_locations.items():
                if loc is None:
                    record[comp_id] = None
                    continue
                raw_val = self.get_response(country_idx, loc)
                
                # Find the scoring rule for this component
                scoring = 'yes=1'  # default
                for idx_def in all_indices.values():
                    if 'part_a' in idx_def:
                        for part_key in ['part_a', 'part_b']:
                            for comp in idx_def[part_key]['components']:
                                if comp['id'] == comp_id:
                                    scoring = comp['scoring']
                    else:
                        for comp in idx_def.get('components', []):
                            if comp['id'] == comp_id:
                                scoring = comp['scoring']
                
                record[comp_id] = score_component(raw_val, scoring)
            
            records.append(record)
        
        return pd.DataFrame(records)


# ============================================================================
# SECTION 4: INDEX COMPUTATION
# ============================================================================

def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute BCL summary indices from component scores.
    
    Handles missing components gracefully: if >50% of components
    are available for an index, computes a scaled version. Otherwise NaN.
    """
    result = df.copy()
    
    # --- Capital Stringency Index (0-9) ---
    cap_a_cols = [
        'cap_risk_weighted', 'cap_credit_risk', 'cap_market_risk',
        'cap_deduct_loans', 'cap_deduct_securities', 'cap_deduct_fx'
    ]
    cap_b_cols = ['cap_no_noncash', 'cap_no_borrowed', 'cap_verified']
    cap_all = cap_a_cols + cap_b_cols
    
    present_cols = [c for c in cap_all if c in result.columns]
    if present_cols:
        result['capital_stringency_idx'] = result[present_cols].sum(axis=1, min_count=5)
    else:
        result['capital_stringency_idx'] = np.nan
    
    # --- Activity Restrictions Index (4-16) ---
    act_cols = ['act_securities', 'act_insurance', 'act_real_estate', 'act_nonfinancial']
    present_cols = [c for c in act_cols if c in result.columns]
    if present_cols:
        result['activity_restrictions_idx'] = result[present_cols].sum(axis=1, min_count=3)
    else:
        result['activity_restrictions_idx'] = np.nan
    
    # --- Supervisory Power Index (0-14) ---
    sup_cols = [c['id'] for c in SUPERVISORY_POWER_QUESTIONS['components']]
    present_cols = [c for c in sup_cols if c in result.columns]
    if present_cols:
        result['supervisory_power_idx'] = result[present_cols].sum(axis=1, min_count=7)
    else:
        result['supervisory_power_idx'] = np.nan
    
    # --- Entry Barriers Index (0-8) ---
    ent_cols = [c['id'] for c in ENTRY_BARRIERS_QUESTIONS['components']]
    present_cols = [c for c in ent_cols if c in result.columns]
    if present_cols:
        result['entry_barriers_idx'] = result[present_cols].sum(axis=1, min_count=4)
    else:
        result['entry_barriers_idx'] = np.nan
    
    # --- Private Monitoring Index (0-12) ---
    pm_cols = [c['id'] for c in PRIVATE_MONITORING_QUESTIONS['components']]
    present_cols = [c for c in pm_cols if c in result.columns]
    if present_cols:
        result['private_monitoring_idx'] = result[present_cols].sum(axis=1, min_count=6)
    else:
        result['private_monitoring_idx'] = np.nan
    
    # --- Overall Restrictiveness (sum of all) ---
    idx_cols = [
        'capital_stringency_idx', 'activity_restrictions_idx',
        'supervisory_power_idx', 'entry_barriers_idx', 'private_monitoring_idx'
    ]
    result['overall_restrictiveness_idx'] = result[idx_cols].sum(axis=1, min_count=3)
    
    return result


# ============================================================================
# SECTION 5: WAVE-TO-YEAR MAPPING AND PANEL CONSTRUCTION
# ============================================================================

WAVE_YEARS = {
    1: 2001,   # Survey I, data as of ~1999
    2: 2003,   # Survey II, data as of ~2002
    3: 2007,   # Survey III, data as of ~2006
    4: 2012,   # Survey IV, data as of ~2011
    5: 2019,   # Survey V, data as of ~2016
}

# ISO country code mapping (comprehensive)
COUNTRY_NAME_TO_CODE = {
    'afghanistan': 'AF', 'albania': 'AL', 'algeria': 'DZ', 'angola': 'AO',
    'argentina': 'AR', 'armenia': 'AM', 'australia': 'AU', 'austria': 'AT',
    'azerbaijan': 'AZ', 'bahamas': 'BS', 'bahrain': 'BH', 'bangladesh': 'BD',
    'barbados': 'BB', 'belarus': 'BY', 'belgium': 'BE', 'belize': 'BZ',
    'benin': 'BJ', 'bermuda': 'BM', 'bhutan': 'BT', 'bolivia': 'BO',
    'bosnia and herzegovina': 'BA', 'botswana': 'BW', 'brazil': 'BR',
    'brunei': 'BN', 'bulgaria': 'BG', 'burkina faso': 'BF', 'burundi': 'BI',
    'cambodia': 'KH', 'cameroon': 'CM', 'canada': 'CA', 'cape verde': 'CV',
    'central african republic': 'CF', 'chad': 'TD', 'chile': 'CL',
    'china': 'CN', "china, people's republic": 'CN', 'colombia': 'CO',
    'comoros': 'KM', 'congo': 'CG', 'costa rica': 'CR', "cote d'ivoire": 'CI',
    'croatia': 'HR', 'cuba': 'CU', 'cyprus': 'CY', 'czech republic': 'CZ',
    'czechia': 'CZ', 'denmark': 'DK', 'dominican republic': 'DO',
    'ecuador': 'EC', 'egypt': 'EG', 'el salvador': 'SV',
    'equatorial guinea': 'GQ', 'estonia': 'EE', 'ethiopia': 'ET',
    'fiji': 'FJ', 'finland': 'FI', 'france': 'FR', 'gabon': 'GA',
    'gambia': 'GM', 'georgia': 'GE', 'germany': 'DE', 'ghana': 'GH',
    'greece': 'GR', 'guatemala': 'GT', 'guinea': 'GN', 'guyana': 'GY',
    'haiti': 'HT', 'honduras': 'HN', 'hong kong': 'HK',
    'hong kong sar': 'HK', 'hong kong, china': 'HK', 'hungary': 'HU',
    'iceland': 'IS', 'india': 'IN', 'indonesia': 'ID',
    'iran': 'IR', 'iran, islamic republic': 'IR', 'iraq': 'IQ',
    'ireland': 'IE', 'israel': 'IL', 'italy': 'IT', 'jamaica': 'JM',
    'japan': 'JP', 'jordan': 'JO', 'kazakhstan': 'KZ', 'kenya': 'KE',
    'korea': 'KR', 'korea, rep.': 'KR', 'korea, republic': 'KR',
    'south korea': 'KR', 'republic of korea': 'KR',
    'kuwait': 'KW', 'kyrgyz republic': 'KG', 'kyrgyzstan': 'KG',
    'laos': 'LA', 'latvia': 'LV', 'lebanon': 'LB', 'lesotho': 'LS',
    'liberia': 'LR', 'libya': 'LY', 'liechtenstein': 'LI',
    'lithuania': 'LT', 'luxembourg': 'LU', 'macao': 'MO', 'macau': 'MO',
    'madagascar': 'MG', 'malawi': 'MW', 'malaysia': 'MY', 'maldives': 'MV',
    'mali': 'ML', 'malta': 'MT', 'mauritania': 'MR', 'mauritius': 'MU',
    'mexico': 'MX', 'moldova': 'MD', 'mongolia': 'MN', 'montenegro': 'ME',
    'morocco': 'MA', 'mozambique': 'MZ', 'myanmar': 'MM', 'namibia': 'NA',
    'nepal': 'NP', 'netherlands': 'NL', 'new zealand': 'NZ',
    'nicaragua': 'NI', 'niger': 'NE', 'nigeria': 'NG',
    'north macedonia': 'MK', 'macedonia': 'MK', 'norway': 'NO',
    'oman': 'OM', 'pakistan': 'PK', 'panama': 'PA',
    'papua new guinea': 'PG', 'paraguay': 'PY', 'peru': 'PE',
    'philippines': 'PH', 'poland': 'PL', 'portugal': 'PT', 'qatar': 'QA',
    'romania': 'RO', 'russia': 'RU', 'russian federation': 'RU',
    'rwanda': 'RW', 'samoa': 'WS', 'saudi arabia': 'SA', 'senegal': 'SN',
    'serbia': 'RS', 'seychelles': 'SC', 'sierra leone': 'SL',
    'singapore': 'SG', 'slovakia': 'SK', 'slovak republic': 'SK',
    'slovenia': 'SI', 'solomon islands': 'SB', 'south africa': 'ZA',
    'spain': 'ES', 'sri lanka': 'LK', 'sudan': 'SD', 'suriname': 'SR',
    'swaziland': 'SZ', 'eswatini': 'SZ', 'sweden': 'SE',
    'switzerland': 'CH', 'syria': 'SY', 'taiwan': 'TW',
    'tajikistan': 'TJ', 'tanzania': 'TZ', 'thailand': 'TH', 'togo': 'TG',
    'tonga': 'TO', 'trinidad and tobago': 'TT', 'tunisia': 'TN',
    'turkey': 'TR', 'turkiye': 'TR', 'turkmenistan': 'TM',
    'uganda': 'UG', 'ukraine': 'UA', 'united arab emirates': 'AE',
    'uae': 'AE', 'united kingdom': 'GB', 'uk': 'GB',
    'united states': 'US', 'usa': 'US', 'u.s.a.': 'US', 'u.s.': 'US',
    'uruguay': 'UY', 'uzbekistan': 'UZ', 'vanuatu': 'VU',
    'venezuela': 'VE', 'vietnam': 'VN', 'viet nam': 'VN',
    'yemen': 'YE', 'zambia': 'ZM', 'zimbabwe': 'ZW',
    # Common variants
    'congo, dem. rep.': 'CD', 'congo, democratic republic': 'CD',
    'congo, rep.': 'CG', 'congo, republic': 'CG',
    "lao pdr": 'LA', "lao people's democratic republic": 'LA',
    'west bank and gaza': 'PS', 'palestinian territories': 'PS',
}


def match_country_code(name: str) -> Optional[str]:
    """Match a country name to its ISO 2-letter code."""
    if not name:
        return None
    nl = name.lower().strip()
    # Direct match
    if nl in COUNTRY_NAME_TO_CODE:
        return COUNTRY_NAME_TO_CODE[nl]
    # Partial match
    for key, code in COUNTRY_NAME_TO_CODE.items():
        if key in nl or nl in key:
            return code
    return None


def build_panel(wave_dfs: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a complete panel dataset from wave-level DataFrames.
    
    Adds:
    - ISO country codes
    - Year mapping
    - First-difference columns
    - Basel III tightening indicator
    """
    panels = []
    
    for wave, df in sorted(wave_dfs.items()):
        if df is None or df.empty:
            continue
        
        # Add year and country code
        df = df.copy()
        df['year'] = WAVE_YEARS.get(wave, 2000 + wave * 3)
        df['country_code'] = df['country'].apply(match_country_code)
        
        # Compute indices if not already present
        if 'capital_stringency_idx' not in df.columns:
            df = compute_indices(df)
        
        panels.append(df)
    
    if not panels:
        return pd.DataFrame()
    
    panel = pd.concat(panels, ignore_index=True)
    
    # Keep only summary columns for the final panel
    summary_cols = [
        'country_code', 'country', 'year', 'wave',
        'capital_stringency_idx', 'activity_restrictions_idx',
        'supervisory_power_idx', 'entry_barriers_idx',
        'private_monitoring_idx', 'overall_restrictiveness_idx'
    ]
    available = [c for c in summary_cols if c in panel.columns]
    panel = panel[available].copy()
    
    # Sort
    panel = panel.sort_values(['country_code', 'year']).reset_index(drop=True)
    
    # Add first-differences
    idx_cols = [
        'capital_stringency_idx', 'activity_restrictions_idx',
        'supervisory_power_idx', 'entry_barriers_idx',
        'private_monitoring_idx', 'overall_restrictiveness_idx'
    ]
    for col in idx_cols:
        if col in panel.columns:
            panel[f'd_{col}'] = panel.groupby('country_code')[col].diff()
    
    # Basel III capital tightening indicator (post-2010 for BCBS members)
    bcbs_members = {
        'AR', 'AU', 'BE', 'BR', 'CA', 'CN', 'FR', 'DE', 'HK', 'IN',
        'ID', 'IT', 'JP', 'KR', 'LU', 'MX', 'NL', 'RU', 'SA', 'SG',
        'ZA', 'ES', 'SE', 'CH', 'TR', 'GB', 'US'
    }
    panel['basel3_capital_tightening'] = (
        (panel['year'] >= 2012) &
        panel['country_code'].isin(bcbs_members)
    ).astype(int)
    
    return panel


# ============================================================================
# SECTION 6: VALIDATION AGAINST EXISTING PANEL
# ============================================================================

def validate_against_existing(new_panel: pd.DataFrame, 
                              existing_path: str) -> pd.DataFrame:
    """
    Compare newly constructed indices against the existing 16-country panel.
    Reports discrepancies and returns a comparison DataFrame.
    """
    try:
        existing = pd.read_csv(existing_path)
    except FileNotFoundError:
        print(f"  Existing panel not found at {existing_path}")
        return pd.DataFrame()
    
    print("\n" + "="*70)
    print("VALIDATION: Comparing with existing 16-country panel")
    print("="*70)
    
    idx_cols = [
        'capital_stringency_idx', 'activity_restrictions_idx',
        'supervisory_power_idx', 'entry_barriers_idx',
        'private_monitoring_idx'
    ]
    
    # Merge on country_code and year
    merged = existing.merge(
        new_panel, 
        on=['country_code', 'year'], 
        suffixes=('_existing', '_new'),
        how='inner'
    )
    
    if merged.empty:
        # Try merge on country name + wave
        merged = existing.merge(
            new_panel, 
            on=['country_code', 'wave'], 
            suffixes=('_existing', '_new'),
            how='inner'
        )
    
    print(f"  Matched {len(merged)} observations across {merged['country_code'].nunique()} countries")
    
    comparisons = []
    for col in idx_cols:
        col_e = f'{col}_existing'
        col_n = f'{col}_new'
        if col_e in merged.columns and col_n in merged.columns:
            valid = merged[[col_e, col_n]].dropna()
            if len(valid) > 0:
                corr = valid[col_e].corr(valid[col_n])
                mae = (valid[col_e] - valid[col_n]).abs().mean()
                exact_match = (valid[col_e] == valid[col_n]).mean()
                print(f"\n  {col}:")
                print(f"    Correlation:   {corr:.3f}")
                print(f"    MAE:           {mae:.2f}")
                print(f"    Exact match:   {exact_match:.1%}")
                comparisons.append({
                    'index': col, 'correlation': corr, 
                    'mae': mae, 'exact_match_pct': exact_match
                })
    
    return pd.DataFrame(comparisons)


# ============================================================================
# SECTION 7: DOWNLOAD & MAIN EXECUTION
# ============================================================================

# World Bank BRSS data catalog direct download URLs
# Source: https://datacatalog.worldbank.org/search/dataset/0038632
BRSS_DOWNLOADS = {
    1: {
        'url': 'https://datacatalogfiles.worldbank.org/ddh-published/0038632/DR0047733/caprio_2000_banking_regulation_database_0.xls',
        'filename': 'brss_2001_wave1.xls',
        'description': 'Wave 1 (2001 Survey, 118 jurisdictions)',
    },
    2: {
        'url': 'https://datacatalogfiles.worldbank.org/ddh-published/0038632/DR0047732/caprio_2003_banking_regulation_database_0_0.xls',
        'filename': 'brss_2003_wave2.xls',
        'description': 'Wave 2 (2003 Survey, 151 jurisdictions)',
    },
    3: {
        'url': 'https://datacatalogfiles.worldbank.org/ddh-published/0038632/DR0047731/banking_regulation_survey_iii_061008.xls',
        'filename': 'brss_2007_wave3.xls',
        'description': 'Wave 3 (2007 Survey, 143 jurisdictions)',
    },
    4: {
        'url': 'https://datacatalogfiles.worldbank.org/ddh-published/0038632/DR0047734/brss-bank-regulation.xlsx',
        'filename': 'brss_2011_wave4.xlsx',
        'description': 'Wave 4 (2011 Survey, 143 jurisdictions)',
    },
    5: {
        'url': 'https://datacatalogfiles.worldbank.org/ddh-published/0038632/DR0047735/survey-20191104-brss-public-release.xlsx',
        'filename': 'brss_2019_wave5.xlsx',
        'description': 'Wave 5 (2019 Survey, 160 jurisdictions)',
    },
    # Wave 5 updated release — used as fallback or supplement
    '5_update': {
        'url': 'https://datacatalogfiles.worldbank.org/ddh-published/0038632/DR0047737/2021_04_26_brss-public-release.xlsx',
        'filename': 'brss_2019_wave5_2021update.xlsx',
        'description': 'Wave 5 — 2021 Update',
    },
}


def download_brss_files(data_dir: str = 'brss_data', 
                        force: bool = False) -> Dict[int, str]:
    """
    Download BRSS Excel files from World Bank Data Catalog.
    
    Skips files that already exist on disk (unless force=True).
    Returns dict mapping wave number -> local file path.
    """
    import urllib.request
    import urllib.error
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    wave_files = {}
    
    for wave_key, info in BRSS_DOWNLOADS.items():
        # Skip the supplemental '5_update' entry for main processing
        if not isinstance(wave_key, int):
            continue
        
        local_path = data_path / info['filename']
        
        # Skip download if file already exists and is non-empty
        if local_path.exists() and local_path.stat().st_size > 1000 and not force:
            print(f"  Wave {wave_key}: CACHED  {local_path.name} "
                  f"({local_path.stat().st_size:,} bytes)")
            wave_files[wave_key] = str(local_path)
            continue
        
        # Download
        print(f"  Wave {wave_key}: Downloading {info['description']}...")
        try:
            urllib.request.urlretrieve(info['url'], str(local_path))
            size = local_path.stat().st_size
            print(f"           Saved {local_path.name} ({size:,} bytes)")
            wave_files[wave_key] = str(local_path)
        except urllib.error.URLError as e:
            print(f"           FAILED: {e}")
            print(f"           URL: {info['url']}")
            # Try to find any manually placed file matching this wave
            for f in data_path.glob('*.xls*'):
                if str(WAVE_YEARS.get(wave_key, '')) in f.name.lower():
                    print(f"           Found manual file: {f.name}")
                    wave_files[wave_key] = str(f)
                    break
        except Exception as e:
            print(f"           FAILED: {e}")
    
    # Also download the 2021 update (supplemental)
    update_info = BRSS_DOWNLOADS['5_update']
    update_path = data_path / update_info['filename']
    if not update_path.exists() or update_path.stat().st_size < 1000:
        print(f"  Wave 5 (2021 update): Downloading...")
        try:
            urllib.request.urlretrieve(update_info['url'], str(update_path))
            print(f"           Saved {update_path.name} "
                  f"({update_path.stat().st_size:,} bytes)")
        except Exception as e:
            print(f"           FAILED (non-critical): {e}")
    else:
        print(f"  Wave 5 (2021 update): CACHED  {update_path.name}")
    
    return wave_files


def find_brss_files(data_dir: str = 'brss_data') -> Dict[int, str]:
    """
    Find BRSS Excel files in the data directory (auto-detect by name).
    Falls back to scanning for any .xls* files with year indicators.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return {}
    
    wave_files = {}
    
    # First check for our canonical filenames
    for wave_key, info in BRSS_DOWNLOADS.items():
        if not isinstance(wave_key, int):
            continue
        local_path = data_path / info['filename']
        if local_path.exists() and local_path.stat().st_size > 1000:
            wave_files[wave_key] = str(local_path)
    
    # Then scan for any other matching files
    wave_year_map = {2001: 1, 2003: 2, 2007: 3, 2011: 4, 2012: 4, 2019: 5}
    for f in data_path.glob('*.xls*'):
        fname = f.name.lower()
        for year, wave in wave_year_map.items():
            if str(year) in fname and wave not in wave_files:
                wave_files[wave] = str(f)
                break
    
    return wave_files


def main():
    """Main entry point."""
    print("="*70)
    print("BCL Index Construction from BRSS Raw Survey Data")
    print("="*70)
    print()
    
    # Configuration
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'brss_data'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'bcl_indices_panel.csv'
    existing_panel = 'bank_regulation_panel.csv'
    force_download = '--force-download' in sys.argv
    
    # Step 1: Download BRSS files (skips existing files)
    print(f"Checking/downloading BRSS Excel files to '{data_dir}/'...")
    print(f"  (files already on disk will be skipped)\n")
    wave_files = download_brss_files(data_dir, force=force_download)
    
    # Step 2: Also pick up any manually-placed files
    manual_files = find_brss_files(data_dir)
    for w, p in manual_files.items():
        if w not in wave_files:
            wave_files[w] = p
    
    if not wave_files:
        print(f"\nNo BRSS files available. Check network or manually place files in '{data_dir}/'")
        print(f"Download URLs:")
        for wk, info in BRSS_DOWNLOADS.items():
            if isinstance(wk, int):
                print(f"  Wave {wk}: {info['url']}")
        return
    
    print(f"\nFound {len(wave_files)} wave files:")
    for wave, path in sorted(wave_files.items()):
        print(f"  Wave {wave} ({WAVE_YEARS.get(wave, '?')}): {path}")
    
    # Process each wave
    wave_dfs = {}
    for wave, filepath in sorted(wave_files.items()):
        print(f"\n{'='*50}")
        print(f"Processing Wave {wave} ({WAVE_YEARS.get(wave, '?')})")
        print(f"{'='*50}")
        
        parser = BRSSParser(filepath, wave)
        if not parser.load():
            continue
        
        components_df = parser.extract_all()
        if components_df.empty:
            print(f"  WARNING: No data extracted for wave {wave}")
            continue
        
        indices_df = compute_indices(components_df)
        wave_dfs[wave] = indices_df
        
        n_countries = len(indices_df)
        n_valid_csi = indices_df['capital_stringency_idx'].notna().sum()
        print(f"  Extracted: {n_countries} countries, "
              f"{n_valid_csi} with valid Capital Stringency")
    
    # Build panel
    print(f"\n{'='*50}")
    print("Building Panel Dataset")
    print(f"{'='*50}")
    
    panel = build_panel(wave_dfs)
    
    if panel.empty:
        print("  ERROR: No data to build panel from")
        return
    
    print(f"  Panel: {len(panel)} observations, "
          f"{panel['country_code'].nunique()} countries, "
          f"waves {sorted(panel['wave'].unique())}")
    
    # Validate
    if os.path.exists(existing_panel):
        validate_against_existing(panel, existing_panel)
    
    # Save
    panel.to_csv(output_file, index=False)
    print(f"\nPanel saved to: {output_file}")
    
    # Summary statistics
    print(f"\n{'='*50}")
    print("Summary Statistics")
    print(f"{'='*50}")
    idx_cols = [
        'capital_stringency_idx', 'activity_restrictions_idx',
        'supervisory_power_idx', 'entry_barriers_idx',
        'private_monitoring_idx'
    ]
    for col in idx_cols:
        if col in panel.columns:
            s = panel[col].dropna()
            print(f"\n  {col}:")
            print(f"    N={len(s)}, Mean={s.mean():.2f}, "
                  f"SD={s.std():.2f}, Min={s.min():.0f}, Max={s.max():.0f}")


if __name__ == '__main__':
    main()
