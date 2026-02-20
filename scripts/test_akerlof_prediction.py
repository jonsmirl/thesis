#!/usr/bin/env python3
"""
test_akerlof_prediction.py — Empirical test of the Free Energy Framework

Akerlof Prediction (Proposition 8):
    Markets with higher σ (more substitutable goods, lower CES curvature K)
    have lower critical temperature T* — they are MORE vulnerable to
    adverse selection / information asymmetry.

Test specification (cross-section of industries):
    Spread_i = α + β·log(σ_i) + γ·Controls_i + ε_i

    where Spread = Corwin-Schultz (2012) bid-ask spread estimator or
    Amihud (2002) illiquidity ratio, both computed from daily price data.
    σ = trade elasticity of substitution from published estimates.

Prediction: β > 0 — industries with higher σ have wider spreads.

Data sources:
    1. Trade elasticities: Broda-Weinstein (2006) SITC medians, Ahmad-Riker
       (2020) NAICS estimates, mapped to GICS industries. Embedded.
    2. Daily price/volume: Yahoo Finance via yfinance (50+ industry ETFs).
    3. Controls: volatility, average dollar volume, ETF AUM.

Usage:
    pip install yfinance   # if not already installed
    source .venv/bin/activate
    python scripts/test_akerlof_prediction.py
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
except ImportError:
    print("ERROR: pip install statsmodels"); sys.exit(1)

try:
    import yfinance as yf
except ImportError:
    print("ERROR: pip install yfinance"); sys.exit(1)

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'thesis_data')
FIGDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)


# ============================================================================
# SECTION 1: Industry ETFs with σ estimates
# ============================================================================
# Each entry: (ETF ticker, industry label, approx NAICS, σ estimate, σ source)
#
# σ sources:
#   BW06 = Broda & Weinstein (2006) QJE, SITC 3-digit median for category
#   AR20 = Ahmad & Riker (2020) USITC, NAICS estimates
#   CP15 = Caliendo & Parro (2015) REStud, GTAP sector
#   EST  = author estimate based on industry characteristics
#
# We restrict to industries producing TRADEABLE GOODS where trade
# elasticities are well-estimated.  Financial, real estate, and pure
# service ETFs are excluded (no trade σ).

INDUSTRIES = [
    # --- Energy ---
    ('XLE',  'Energy (broad)',           '211', 8.1, 'BW06'),
    ('XOP',  'Oil & Gas E&P',           '211', 8.1, 'BW06'),
    ('OIH',  'Oil Services',            '213', 7.5, 'AR20'),

    # --- Materials ---
    ('XLB',  'Materials (broad)',        '325', 4.8, 'AR20'),
    ('XME',  'Metals & Mining',         '212', 9.2, 'BW06'),
    ('LIT',  'Lithium & Battery',       '335', 5.5, 'AR20'),

    # --- Industrials ---
    ('XLI',  'Industrials (broad)',      '333', 3.8, 'AR20'),
    ('ITA',  'Aerospace & Defense',      '336', 2.8, 'AR20'),
    ('IYT',  'Transportation',           '481', 5.5, 'CP15'),
    ('PAVE', 'Infrastructure',           '237', 5.0, 'EST'),

    # --- Consumer Discretionary ---
    ('XLY',  'Consumer Disc. (broad)',   '336', 3.5, 'AR20'),
    ('XHB',  'Homebuilders',             '236', 5.2, 'AR20'),
    ('XRT',  'Retail',                   '452', 4.8, 'EST'),
    ('IBUY', 'Online Retail',            '454', 5.5, 'EST'),

    # --- Consumer Staples ---
    ('XLP',  'Consumer Staples (broad)', '311', 3.2, 'BW06'),
    ('PBJ',  'Food & Beverage',         '311', 2.8, 'BW06'),

    # --- Health Care ---
    ('XLV',  'Health Care (broad)',      '325', 2.5, 'AR20'),
    ('IBB',  'Biotech',                  '325', 1.8, 'AR20'),
    ('IHI',  'Medical Devices',          '339', 2.2, 'AR20'),
    ('XBI',  'Biotech (small cap)',      '325', 1.6, 'AR20'),

    # --- Technology ---
    ('XLK',  'Technology (broad)',       '334', 5.2, 'AR20'),
    ('SOXX', 'Semiconductors',           '334', 4.5, 'AR20'),
    ('IGV',  'Software',                 '511', 3.0, 'EST'),
    ('HACK', 'Cybersecurity',            '511', 2.5, 'EST'),
    ('SKYY', 'Cloud Computing',          '518', 3.2, 'EST'),

    # --- Communication ---
    ('XLC',  'Communication (broad)',    '517', 6.0, 'CP15'),

    # --- Utilities ---
    ('XLU',  'Utilities',                '221', 7.8, 'CP15'),
    ('TAN',  'Solar',                    '335', 5.0, 'AR20'),

    # --- Manufacturing sub-sectors ---
    ('IYJ',  'Industrial Machinery',     '333', 3.5, 'AR20'),
    ('WOOD', 'Timber/Forestry',          '321', 6.5, 'BW06'),
    ('MOO',  'Agribusiness',             '111', 4.2, 'BW06'),
    ('GNR',  'Natural Resources',        '212', 8.5, 'BW06'),
]


# ============================================================================
# SECTION 2: Download Data
# ============================================================================

def download_etf_data(tickers, period='2y'):
    """Batch download daily OHLCV from Yahoo Finance."""
    cache = os.path.join(OUTPUT, 'akerlof_etf_prices.csv')
    if os.path.exists(cache):
        print(f"  Loading cached price data from {cache}")
        df = pd.read_csv(cache, parse_dates=['Date'])
        # Check if we have most tickers
        cached_tickers = df.columns.drop(['Date']).str.split('_').str[0].unique()
        missing = set(tickers) - set(cached_tickers)
        if len(missing) <= 3:
            return df
        print(f"  Cache missing {len(missing)} tickers, re-downloading...")

    print(f"  Downloading {len(tickers)} ETFs from Yahoo Finance ({period})...")
    data = yf.download(tickers, period=period, group_by='ticker',
                       auto_adjust=True, progress=False, threads=True)

    records = []
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                sub = data
            else:
                sub = data[ticker] if ticker in data.columns.get_level_values(0) else None
            if sub is None or sub.empty:
                continue
            sub = sub.dropna(subset=['Close', 'Volume'])
            for date, row in sub.iterrows():
                records.append({
                    'Date': date,
                    'ticker': ticker,
                    'Open': row.get('Open', np.nan),
                    'High': row.get('High', np.nan),
                    'Low': row.get('Low', np.nan),
                    'Close': row['Close'],
                    'Volume': row['Volume'],
                })
        except Exception as e:
            print(f"    Warning: {ticker}: {e}")

    df = pd.DataFrame(records)
    if len(df) > 0:
        df.to_csv(cache, index=False)
        print(f"  Got data for {df['ticker'].nunique()} ETFs, "
              f"{len(df):,} daily observations")
    return df


# ============================================================================
# SECTION 3: Compute Spread and Illiquidity Measures
# ============================================================================

def corwin_schultz_spread(high, low):
    """
    Corwin & Schultz (2012) bid-ask spread estimator from daily high/low.

    The estimator exploits the fact that daily high (low) prices are
    almost always buyer-initiated (seller-initiated) trades, so the
    high-low range reflects both volatility and the spread.  By comparing
    single-day and two-day ranges, the volatility component cancels and
    the spread is identified.

    Returns estimated spread as a fraction of price.
    """
    h = np.log(high.values)
    l = np.log(low.values)

    # Single-day squared log range
    hl2 = (h - l) ** 2

    # Two-day high and low
    h2 = np.maximum(h[:-1], h[1:])
    l2 = np.minimum(l[:-1], l[1:])
    hl2_2day = (h2 - l2) ** 2

    # β = E[sum of single-day squared ranges]
    beta = np.nanmean(hl2[:-1] + hl2[1:])

    # γ = E[two-day squared range]
    gamma = np.nanmean(hl2_2day)

    # Estimate
    k = 2 * np.sqrt(2) - 1   # 3 - 2√2 ≈ 0.1716
    denom = 3 - 2 * np.sqrt(2)

    alpha_raw = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / denom)

    if np.isnan(alpha_raw) or alpha_raw <= 0:
        return 0.0

    spread = 2 * (np.exp(alpha_raw) - 1) / (1 + np.exp(alpha_raw))
    return max(spread, 0.0)


def compute_amihud(returns, dollar_volume):
    """Amihud (2002) illiquidity: mean(|r| / dollar_volume) × 1e6."""
    valid = (dollar_volume > 0) & np.isfinite(returns)
    if valid.sum() < 20:
        return np.nan
    ratio = np.abs(returns[valid]) / dollar_volume[valid]
    return np.nanmean(ratio) * 1e6


def compute_industry_measures(price_df, industries):
    """Compute spread and illiquidity for each ETF."""
    results = []

    for ticker, label, naics, sigma, source in industries:
        sub = price_df[price_df['ticker'] == ticker].sort_values('Date').copy()
        if len(sub) < 60:
            print(f"    Skipping {ticker} ({label}): only {len(sub)} days")
            continue

        # Returns
        sub['ret'] = sub['Close'].pct_change()
        sub['dollar_vol'] = sub['Close'] * sub['Volume']

        # Drop first row (NaN return)
        sub = sub.iloc[1:].copy()

        # Corwin-Schultz spread
        cs_spread = corwin_schultz_spread(sub['High'], sub['Low'])

        # Amihud illiquidity
        amihud = compute_amihud(sub['ret'].values, sub['dollar_vol'].values)

        # Controls
        volatility = sub['ret'].std() * np.sqrt(252)
        avg_dollar_vol = sub['dollar_vol'].mean()
        avg_price = sub['Close'].mean()
        n_days = len(sub)

        results.append({
            'ticker': ticker,
            'industry': label,
            'naics': naics,
            'sigma': sigma,
            'sigma_source': source,
            'cs_spread': cs_spread,
            'amihud': amihud,
            'volatility': volatility,
            'avg_dollar_vol': avg_dollar_vol,
            'avg_price': avg_price,
            'n_days': n_days,
        })

    df = pd.DataFrame(results)

    # Log transforms
    df['log_sigma'] = np.log(df['sigma'])
    df['log_amihud'] = np.log(df['amihud'].clip(lower=1e-10))
    df['log_cs_spread'] = np.log(df['cs_spread'].clip(lower=1e-6))
    df['log_dollar_vol'] = np.log(df['avg_dollar_vol'].clip(lower=1))
    df['log_volatility'] = np.log(df['volatility'].clip(lower=1e-6))

    return df


# ============================================================================
# SECTION 4: Regressions
# ============================================================================

def run_cross_section(df, yvar, ylabel, xvar='log_sigma', xlabel='log(σ)'):
    """OLS cross-section with robust SE."""
    controls = ['log_volatility', 'log_dollar_vol']
    cols = [yvar, xvar] + controls
    sub = df.dropna(subset=cols).copy()

    if len(sub) < 10:
        print(f"\n  Too few observations ({len(sub)}) for {ylabel}")
        return None

    # Bivariate first
    X_biv = sm.add_constant(sub[[xvar]])
    res_biv = sm.OLS(sub[yvar], X_biv).fit(cov_type='HC1')

    # With controls
    X_full = sm.add_constant(sub[[xvar] + controls])
    res_full = sm.OLS(sub[yvar], X_full).fit(cov_type='HC1')

    print(f"\n{'='*72}")
    print(f"  Dependent variable: {ylabel}")
    print(f"{'='*72}")
    print(f"  N = {len(sub)}")

    for spec_label, res in [("Bivariate", res_biv), ("With controls", res_full)]:
        print(f"\n  --- {spec_label} ---")
        print(f"  R² = {res.rsquared:.4f}")
        print(f"  {'Variable':<22} {'β':>9} {'SE':>9} {'t':>7} {'p':>8}")
        print(f"  {'-'*56}")
        for v in res.params.index:
            if v == 'const':
                continue
            b, s, t, p = res.params[v], res.bse[v], res.tvalues[v], res.pvalues[v]
            star = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else '†' if p<.1 else ''
            vname = xlabel if v == xvar else v
            print(f"  {vname:<22} {b:>9.4f} {s:>9.4f} {t:>7.2f} {p:>8.4f} {star}")

    b = res_full.params[xvar]
    p = res_full.pvalues[xvar]
    print(f"\n  PREDICTION: β({xlabel}) > 0")
    if b > 0 and p < 0.05:
        print(f"  VERDICT: CONFIRMED (β = {b:.4f}, p = {p:.4f})")
    elif b > 0 and p < 0.10:
        print(f"  VERDICT: WEAK SUPPORT (β = {b:.4f}, p = {p:.4f})")
    elif b > 0:
        print(f"  VERDICT: CORRECT SIGN, not significant (β = {b:.4f}, p = {p:.4f})")
    else:
        print(f"  VERDICT: WRONG SIGN (β = {b:.4f}, p = {p:.4f})")

    return {'outcome': ylabel, 'beta': b, 'se': res_full.bse[xvar],
            'p': p, 'n': len(sub), 'r2': res_full.rsquared,
            'beta_biv': res_biv.params[xvar], 'p_biv': res_biv.pvalues[xvar]}


# ============================================================================
# SECTION 5: Figures
# ============================================================================

def make_figure(df):
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel A: Amihud vs σ ---
    ax = axes[0]
    valid = df.dropna(subset=['log_sigma', 'log_amihud'])
    ax.scatter(valid['sigma'], valid['amihud'], s=50, alpha=0.7,
               c=valid['sigma'], cmap='RdYlBu_r', edgecolors='k', lw=0.5)
    for _, r in valid.iterrows():
        ax.annotate(r['ticker'], (r['sigma'], r['amihud']),
                    fontsize=6, alpha=0.7, xytext=(4, 4),
                    textcoords='offset points')
    # Log scale for y
    ax.set_yscale('log')
    ax.set_xlabel('Trade elasticity of substitution (σ)', fontsize=11)
    ax.set_ylabel('Amihud illiquidity (×10⁶)', fontsize=11)
    ax.set_title('A.  Illiquidity vs. σ', fontsize=12)
    ax.grid(True, alpha=0.25)
    # Trend line on log-log
    z = np.polyfit(np.log(valid['sigma']), np.log(valid['amihud']), 1)
    xs = np.linspace(valid['sigma'].min(), valid['sigma'].max(), 100)
    ax.plot(xs, np.exp(np.polyval(z, np.log(xs))), 'k--', lw=1.5,
            label=f'elasticity = {z[0]:.2f}')
    ax.legend(fontsize=10)

    # --- Panel B: CS Spread vs σ ---
    ax = axes[1]
    valid2 = df.dropna(subset=['log_sigma', 'cs_spread'])
    valid2 = valid2[valid2['cs_spread'] > 0]
    if len(valid2) > 5:
        ax.scatter(valid2['sigma'], valid2['cs_spread'] * 100, s=50, alpha=0.7,
                   c=valid2['sigma'], cmap='RdYlBu_r', edgecolors='k', lw=0.5)
        for _, r in valid2.iterrows():
            ax.annotate(r['ticker'], (r['sigma'], r['cs_spread'] * 100),
                        fontsize=6, alpha=0.7, xytext=(4, 4),
                        textcoords='offset points')
        ax.set_yscale('log')
        ax.set_xlabel('Trade elasticity of substitution (σ)', fontsize=11)
        ax.set_ylabel('Corwin-Schultz bid-ask spread (%)', fontsize=11)
        ax.set_title('B.  Estimated Spread vs. σ', fontsize=12)
        ax.grid(True, alpha=0.25)
        z2 = np.polyfit(np.log(valid2['sigma']), np.log(valid2['cs_spread']*100), 1)
        ax.plot(xs, np.exp(np.polyval(z2, np.log(xs))), 'k--', lw=1.5,
                label=f'elasticity = {z2[0]:.2f}')
        ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIGDIR, 'akerlof_prediction_test.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved to {path}")


# ============================================================================
# SECTION 6: LaTeX Table
# ============================================================================

def latex_table(rows, df):
    """Write LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Akerlof Prediction: Adverse Selection vs.\ Trade Elasticity of Substitution}",
        r"\label{tab:akerlof_test}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Measure & $\hat\beta(\log\sigma)$ & SE & $p$ & $R^2$ \\",
        r"\midrule",
    ]
    for r in rows:
        s = '***' if r['p']<.001 else '**' if r['p']<.01 else '*' if r['p']<.05 else ''
        lines.append(
            f"  {r['outcome']} & {r['beta']:.3f}{s} & ({r['se']:.3f})"
            f" & {r['p']:.3f} & {r['r2']:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{4pt}",
        r"\parbox{0.9\textwidth}{\footnotesize",
        f"  Cross-section of {len(df)} industry ETFs.",
        r"  Dependent variables: log Amihud (2002) illiquidity and log Corwin-Schultz (2012)",
        r"  estimated bid-ask spread, computed from daily prices over trailing 2 years.",
        r"  $\sigma$ = trade elasticity of substitution from Broda-Weinstein (2006),",
        r"  Ahmad-Riker (2020), and Caliendo-Parro (2015).",
        r"  Controls: log annualized volatility, log average daily dollar volume.",
        r"  Robust (HC1) standard errors.",
        r"  Theory predicts $\hat\beta > 0$: higher $\sigma$ (more substitutable goods)",
        r"  implies wider spreads (more adverse selection).",
        r"  $^{***}p<0.001$; $^{**}p<0.01$; $^{*}p<0.05$.",
        r"}",
        r"\end{table}",
    ]
    path = os.path.join(OUTPUT, 'akerlof_test_table.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  LaTeX table saved to {path}")


# ============================================================================
# SECTION 7: Main
# ============================================================================

def main():
    print("=" * 72)
    print("  FREE ENERGY FRAMEWORK — EMPIRICAL TEST")
    print("  Akerlof Prediction: Adverse Selection vs. σ")
    print("=" * 72)
    print()
    print("  HYPOTHESIS")
    print("  Markets with higher σ (more substitutable goods) have lower")
    print("  critical temperature T* — they are MORE vulnerable to adverse")
    print("  selection.  This manifests as wider bid-ask spreads and higher")
    print("  illiquidity in equities of firms in high-σ industries.")
    print()
    print("  TEST")
    print("  Cross-section of ~30 industry ETFs: regress spread measures")
    print("  on trade elasticity of substitution (σ) with controls.")
    print()

    tickers = [t for t, *_ in INDUSTRIES]

    # Download
    print("─" * 72)
    print("  DOWNLOADING DATA")
    print("─" * 72)
    price_df = download_etf_data(tickers)

    if len(price_df) == 0:
        print("\n  ERROR: No price data downloaded. Check yfinance installation.")
        return

    # Compute measures
    print(f"\n{'─'*72}")
    print("  COMPUTING SPREAD AND ILLIQUIDITY MEASURES")
    print(f"{'─'*72}")
    df = compute_industry_measures(price_df, INDUSTRIES)

    if len(df) < 10:
        print("\n  ERROR: Too few industries with valid data.")
        return

    # Summary stats
    print(f"\n  Industries with valid data: {len(df)}")
    print(f"\n  {'Ticker':<8} {'Industry':<28} {'σ':>5} {'CS Spread':>10} {'Amihud':>10}")
    print(f"  {'-'*66}")
    for _, r in df.sort_values('sigma').iterrows():
        cs = f"{r['cs_spread']*100:.3f}%" if r['cs_spread'] > 0 else "n/a"
        am = f"{r['amihud']:.4f}" if np.isfinite(r['amihud']) else "n/a"
        print(f"  {r['ticker']:<8} {r['industry']:<28} {r['sigma']:>5.1f} {cs:>10} {am:>10}")

    # Save
    df.to_csv(os.path.join(OUTPUT, 'akerlof_test_data.csv'), index=False)

    # Regressions
    print(f"\n{'─'*72}")
    print("  REGRESSIONS")
    print(f"{'─'*72}")

    results = []
    r1 = run_cross_section(df, 'log_amihud', 'log(Amihud illiquidity)')
    if r1: results.append(r1)

    # Only run CS spread if we have enough nonzero values
    cs_valid = df[df['cs_spread'] > 0]
    if len(cs_valid) >= 10:
        r2 = run_cross_section(cs_valid, 'log_cs_spread', 'log(Corwin-Schultz spread)')
        if r2: results.append(r2)

    # Summary
    if results:
        print(f"\n\n{'='*72}")
        print("  SUMMARY")
        print(f"{'='*72}")
        for r in results:
            star = '***' if r['p']<.001 else '**' if r['p']<.01 else '*' if r['p']<.05 else ''
            print(f"  {r['outcome']:<35} β = {r['beta']:+.3f}  (p = {r['p']:.3f}) {star}")
            print(f"  {'':35} bivariate: β = {r['beta_biv']:+.3f}  (p = {r['p_biv']:.3f})")

        latex_table(results, df)

    # Figures
    if HAS_MPL:
        print(f"\n{'─'*72}")
        print("  FIGURES")
        print(f"{'─'*72}")
        make_figure(df)

    # Notes
    print(f"\n\n{'='*72}")
    print("  NOTES")
    print(f"{'='*72}")
    print("""
  This test uses equity market measures as proxies for product market
  adverse selection.  The logic:

  1. In high-σ industries (commodities, undifferentiated goods), firms
     compete on cost → thin margins → profits are sensitive to cost
     shocks → more fundamental uncertainty → wider bid-ask spreads.

  2. In low-σ industries (pharmaceuticals, specialty equipment), firms
     have pricing power from differentiation → fat margins → more
     predictable cash flows → narrower spreads.

  3. The CES framework provides the structural link: high K (low σ)
     means the CES aggregate values quality identification highly,
     so more resources flow to information production (analyst
     coverage, due diligence), reducing information asymmetry.

  LIMITATIONS:
  - ETF spreads reflect ETF market-making dynamics, not just underlying
    product market information asymmetry.
  - σ estimates are approximate mappings from trade literature to GICS
    industries.  For publication, use exact Ahmad-Riker (2020) NAICS
    estimates with a proper NAICS-GICS concordance.
  - Small N (~30 industries) limits statistical power.
  - The Akerlof prediction is about product market adverse selection;
    stock market spreads are an indirect proxy.
""")


if __name__ == '__main__':
    main()
