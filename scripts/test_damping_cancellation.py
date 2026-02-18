#!/usr/bin/env python3
"""
Damping Cancellation Test — Theorem E.3 of Complementary Heterogeneity
========================================================================

Theory: The tree coefficients cₙ satisfy cₙσₙ = P_cycle / k_{n,n-1},
independent of σₙ. Doubling damping at one layer halves that layer's
tree coefficient, leaving the product cₙσₙ unchanged. Therefore:
  - Regulatory tightening at one layer has TRANSIENT but not PERSISTENT
    effect on aggregate financial development.
  - The LAYER of regulatory change should not matter for long-run persistence.

Test: Local projection (Jordà 2005) of IMF Financial Development Index
on lagged Barth-Caprio-Levine regulatory changes, with country FE.

  - Prediction: impulse response significant at h=1-2 years, insignificant by h=4-6
  - Falsification: significant at h=8-10 years (persistent)

Secondary: Basel III as natural experiment (staggered capital tightening).

Inputs: thesis_data/bank_regulation_panel.csv,
        thesis_data/imf_financial_development.csv,
        thesis_data/fraser_regulation.csv
Output: thesis_data/damping_test_results.txt + figures

Requires: pip install pandas numpy statsmodels scipy matplotlib
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    from scipy import stats
    HAS_STATS = True
except ImportError:
    HAS_STATS = False
    print("WARNING: statsmodels/scipy not installed.")
    print("  pip install statsmodels scipy\n")


DATA_DIR = "/home/jonsmirl/thesis/thesis_data"
OUT_DIR = "/home/jonsmirl/thesis/thesis_data"
FIG_DIR = "/home/jonsmirl/thesis/figures/complementary_heterogeneity"
LOG_LINES = []

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    LOG_LINES.append(line)


def load_and_merge():
    """Load BCL, IMF FDI, and Fraser; merge into panel."""
    # BCL regulation
    bcl_path = f"{DATA_DIR}/bank_regulation_panel.csv"
    if not os.path.exists(bcl_path):
        bcl_path = "bank_regulation_panel.csv"
    bcl = pd.read_csv(bcl_path)
    log(f"BCL regulation: {len(bcl)} obs, {bcl['country_code'].nunique()} countries, "
        f"waves: {sorted(bcl['year'].unique())}")

    # IMF Financial Development
    imf_path = f"{DATA_DIR}/imf_financial_development.csv"
    if not os.path.exists(imf_path):
        imf_path = "imf_financial_development.csv"
    imf = pd.read_csv(imf_path)
    log(f"IMF FDI: {len(imf)} obs, {imf['country_code'].nunique()} countries, "
        f"{imf['year'].min()}-{imf['year'].max()}")

    # Fraser regulation
    fraser_path = f"{DATA_DIR}/fraser_regulation.csv"
    if not os.path.exists(fraser_path):
        fraser_path = "fraser_regulation.csv"
    fraser = pd.read_csv(fraser_path)
    log(f"Fraser: {len(fraser)} obs, {fraser['country_code'].nunique()} countries")

    # Merge BCL + IMF on country_code × year
    panel = bcl.merge(imf, on=["country_code", "year"], how="inner", suffixes=("_bcl", "_imf"))
    log(f"Merged BCL+IMF: {len(panel)} obs")

    # Merge Fraser (where available)
    panel = panel.merge(fraser, on=["country_code", "year"], how="left", suffixes=("", "_fraser"))
    log(f"Merged +Fraser: {len(panel)} obs")

    return panel, bcl, imf, fraser


def descriptive_regulation(panel):
    """Descriptive statistics on regulatory changes."""
    log("\n" + "=" * 60)
    log("1. DESCRIPTIVE: REGULATORY CHANGES")
    log("=" * 60)

    reg_cols = ["capital_stringency_idx", "activity_restrictions_idx",
                "supervisory_power_idx", "entry_barriers_idx"]

    for col in reg_cols:
        d_col = f"d_{col}"
        if d_col in panel.columns:
            changes = panel[d_col].dropna()
            if len(changes) > 0:
                log(f"\n  {col}:")
                log(f"    Mean Δ:  {changes.mean():+.2f}")
                log(f"    Std Δ:   {changes.std():.2f}")
                log(f"    Tightened (Δ>0): {(changes > 0).sum()}")
                log(f"    Loosened  (Δ<0): {(changes < 0).sum()}")
                log(f"    Unchanged (Δ=0): {(changes == 0).sum()}")

    # Basel III indicator
    if "basel3_capital_tightening" in panel.columns:
        treated = panel[panel["basel3_capital_tightening"] == 1]["country_code"].unique()
        control = panel[panel["basel3_capital_tightening"] == 0]["country_code"].unique()
        log(f"\n  Basel III capital tightening:")
        log(f"    Treated countries:  {list(treated)}")
        log(f"    Control countries:  {list(set(control) - set(treated))}")


def local_projection(panel, imf_full):
    """
    Jordà (2005) local projection:
      FDI_{t+h} - FDI_t = α_i + β_h × ΔRegulation_t + γ × X_t + ε_{it+h}

    where h = 0, 1, 2, ..., max_h years.

    The coefficient β_h is the impulse response at horizon h.
    Damping cancellation predicts: β_h ≠ 0 for small h, β_h → 0 for large h.
    """
    if not HAS_STATS:
        log("\nSkipping local projection — statsmodels required")
        return None

    log("\n" + "=" * 60)
    log("2. LOCAL PROJECTION — IMPULSE RESPONSE")
    log("=" * 60)

    # We need to construct a panel with different horizons
    # For each BCL wave transition, compute FDI change over next h years

    countries = sorted(panel["country_code"].unique())
    bcl_years = sorted(panel["year"].unique())

    # Regulatory shock variables to test
    shock_vars = [
        ("d_capital_stringency_idx", "Capital Stringency"),
        ("d_activity_restrictions_idx", "Activity Restrictions"),
        ("d_supervisory_power_idx", "Supervisory Power"),
        ("d_overall_restrictiveness_idx", "Overall Restrictiveness"),
    ]

    # Maximum horizon: limited by BCL wave spacing (roughly 5-7 years between waves)
    # and IMF FDI availability
    max_h = 8  # years

    all_results = {}

    for shock_col, shock_label in shock_vars:
        if shock_col not in panel.columns:
            continue

        log(f"\n  ─── Shock: {shock_label} ───")
        log(f"  {'h':>3s}  {'β_h':>8s}  {'SE':>8s}  {'t-stat':>8s}  {'p':>8s}  {'N':>4s}  {'Sig':>5s}")
        log(f"  {'─'*3}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*4}  {'─'*5}")

        irfs = {}

        for h in range(0, max_h + 1):
            # For each country-wave pair, compute:
            # y = FDI(wave_year + h) - FDI(wave_year)
            # x = Δregulation at this wave

            rows = []
            for _, row in panel.iterrows():
                cc = row["country_code"]
                yr = row["year"]
                shock = row.get(shock_col, np.nan)
                fdi_now = row.get("financial_development_overall", np.nan)

                if pd.isna(shock) or pd.isna(fdi_now):
                    continue

                # Find FDI at year + h
                fdi_future = imf_full.loc[
                    (imf_full["country_code"] == cc) &
                    (imf_full["year"] == yr + h),
                    "financial_development_overall"
                ]

                if len(fdi_future) > 0:
                    y = fdi_future.iloc[0] - fdi_now
                    rows.append({
                        "country_code": cc,
                        "year": yr,
                        "y": y,
                        "shock": shock,
                        "fdi_initial": fdi_now,
                    })

            if len(rows) < 5:
                log(f"  {h:3d}  {'--':>8s}  {'--':>8s}  {'--':>8s}  {'--':>8s}  {len(rows):4d}")
                irfs[h] = {"beta": np.nan, "se": np.nan, "tstat": np.nan, "pval": np.nan, "n": len(rows)}
                continue

            reg_df = pd.DataFrame(rows)

            # OLS with initial FDI as control
            # Ideally we'd include country FE, but with N~16 this eats all DOF
            y = reg_df["y"]
            X = add_constant(reg_df[["shock", "fdi_initial"]])

            try:
                model = OLS(y, X).fit(cov_type='HC1')  # heteroskedasticity-robust
                beta = model.params["shock"]
                se = model.bse["shock"]
                tstat = model.tvalues["shock"]
                pval = model.pvalues["shock"]
                sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                log(f"  {h:3d}  {beta:8.4f}  {se:8.4f}  {tstat:8.3f}  {pval:8.4f}  {len(rows):4d}  {sig}")
                irfs[h] = {"beta": beta, "se": se, "tstat": tstat, "pval": pval, "n": len(rows)}
            except Exception as e:
                log(f"  {h:3d}  Error: {e}")
                irfs[h] = {"beta": np.nan, "se": np.nan, "tstat": np.nan, "pval": np.nan, "n": len(rows)}

        all_results[shock_col] = irfs

        # Interpretation
        sig_horizons = [h for h, r in irfs.items() if r.get("pval", 1) < 0.10]
        if sig_horizons:
            max_sig = max(sig_horizons)
            log(f"\n  Significant at horizons: {sig_horizons}")
            if max_sig <= 3:
                log(f"  → CONSISTENT with damping cancellation (transient, dies by h={max_sig})")
            elif max_sig <= 6:
                log(f"  → AMBIGUOUS (significant to h={max_sig}, borderline persistence)")
            else:
                log(f"  → INCONSISTENT with damping cancellation (persistent to h={max_sig})")
        else:
            log(f"  No significant effects at any horizon (may lack power with N={len(rows)})")

    return all_results


def basel3_did(panel, imf_full):
    """Difference-in-differences around Basel III implementation."""
    log("\n" + "=" * 60)
    log("3. BASEL III DIFFERENCE-IN-DIFFERENCES")
    log("=" * 60)

    if "basel3_capital_tightening" not in panel.columns:
        log("  Basel III indicator not found — skipping")
        return

    # Treatment: countries that tightened capital stringency by 2+ points (2007→2019)
    # Treatment date: ~2013 (Basel III phase-in began)
    # Pre-period: 2007 (wave 3)
    # Post-periods: 2012 (wave 4), 2015, 2019, 2022

    countries = panel["country_code"].unique()

    # Build DID panel from IMF data
    rows = []
    for cc in countries:
        # Is this country treated?
        treated = panel.loc[
            (panel["country_code"] == cc) & (panel["basel3_capital_tightening"] == 1)
        ]
        is_treated = len(treated) > 0

        for yr in [2000, 2005, 2007, 2010, 2012, 2015, 2019, 2022]:
            fdi_row = imf_full.loc[
                (imf_full["country_code"] == cc) & (imf_full["year"] == yr),
                "financial_development_overall"
            ]
            if len(fdi_row) > 0:
                rows.append({
                    "country_code": cc,
                    "year": yr,
                    "fdi": fdi_row.iloc[0],
                    "treated": int(is_treated),
                    "post": int(yr >= 2013),
                    "did": int(is_treated) * int(yr >= 2013),
                })

    did_df = pd.DataFrame(rows)
    log(f"  DID panel: {len(did_df)} obs, "
        f"{did_df['country_code'].nunique()} countries, "
        f"{did_df[did_df['treated']==1]['country_code'].nunique()} treated")

    if not HAS_STATS or len(did_df) < 10:
        log("  Insufficient data or no statsmodels — skipping regression")
        return

    # Simple DID
    y = did_df["fdi"]
    X = add_constant(did_df[["treated", "post", "did"]])

    try:
        model = OLS(y, X).fit(cov_type='HC1')
        log(f"\n  FDI = α + β₁×Treated + β₂×Post + β₃×(Treated×Post) + ε")
        log(f"\n  {'Variable':>20s}  {'Coef':>8s}  {'SE':>8s}  {'t':>8s}  {'p':>8s}")
        log(f"  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
        for name in model.params.index:
            coef = model.params[name]
            se = model.bse[name]
            t = model.tvalues[name]
            p = model.pvalues[name]
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            log(f"  {name:>20s}  {coef:8.4f}  {se:8.4f}  {t:8.3f}  {p:8.4f} {sig}")

        log(f"\n  R² = {model.rsquared:.4f}, N = {len(did_df)}")

        beta_did = model.params["did"]
        p_did = model.pvalues["did"]

        log(f"\n  ══════════════════════════════════════════")
        log(f"  INTERPRETATION:")
        if p_did < 0.05:
            log(f"  β₃ = {beta_did:.4f} (p = {p_did:.4f}) — SIGNIFICANT")
            if abs(beta_did) < 0.02:
                log(f"  Effect is statistically significant but economically small")
                log(f"  → Consistent with transient effect (damping cancellation)")
            else:
                log(f"  Effect is both statistically and economically significant")
                log(f"  → May indicate PERSISTENT effect (challenges damping cancellation)")
        else:
            log(f"  β₃ = {beta_did:.4f} (p = {p_did:.4f}) — NOT SIGNIFICANT")
            log(f"  → Consistent with damping cancellation")
            log(f"  Caveat: may lack power with N={len(did_df)}")

    except Exception as e:
        log(f"  DID regression failed: {e}")


def layer_comparison(panel, imf_full):
    """Test whether different regulatory layers have different persistence."""
    if not HAS_STATS:
        return

    log("\n" + "=" * 60)
    log("4. LAYER COMPARISON — DOES THE LAYER MATTER?")
    log("=" * 60)
    log("  Theory: cₙσₙ independent of n → the layer of regulatory")
    log("  change should NOT matter for long-run persistence.")
    log("  All layers should show same transient-then-zero pattern.")

    # Compare capital tighteners vs activity restriction tighteners
    layers = [
        ("d_capital_stringency_idx", "Capital (Layer 4: Finance)"),
        ("d_activity_restrictions_idx", "Activity (Layer 3: Operations)"),
        ("d_supervisory_power_idx", "Supervision (Layer 2: Oversight)"),
    ]

    h_test = 5  # test horizon

    betas = {}
    for shock_col, label in layers:
        if shock_col not in panel.columns:
            continue

        rows = []
        for _, row in panel.iterrows():
            cc = row["country_code"]
            yr = row["year"]
            shock = row.get(shock_col, np.nan)
            fdi_now = row.get("financial_development_overall", np.nan)

            if pd.isna(shock) or pd.isna(fdi_now) or shock == 0:
                continue

            fdi_future = imf_full.loc[
                (imf_full["country_code"] == cc) &
                (imf_full["year"] == yr + h_test),
                "financial_development_overall"
            ]
            if len(fdi_future) > 0:
                rows.append({
                    "y": fdi_future.iloc[0] - fdi_now,
                    "shock": shock,
                    "fdi_initial": fdi_now,
                })

        if len(rows) < 5:
            log(f"  {label}: insufficient obs (N={len(rows)})")
            continue

        reg_df = pd.DataFrame(rows)
        y = reg_df["y"]
        X = add_constant(reg_df[["shock", "fdi_initial"]])

        try:
            model = OLS(y, X).fit(cov_type='HC1')
            beta = model.params["shock"]
            pval = model.pvalues["shock"]
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
            log(f"  {label:40s}: β_{h_test} = {beta:+.4f} (p = {pval:.4f}) N={len(rows)} {sig}")
            betas[shock_col] = beta
        except Exception as e:
            log(f"  {label}: regression failed — {e}")

    if len(betas) >= 2:
        vals = list(betas.values())
        spread = max(vals) - min(vals)
        log(f"\n  Cross-layer β spread: {spread:.4f}")
        if spread < 0.01:
            log(f"  → CONSISTENT with damping cancellation (layers have ~equal persistence)")
        else:
            log(f"  → POSSIBLE violation: layers differ in persistence")
            log(f"  (Caveat: low power with small N; spread may not be significant)")


def plot_irf(all_results):
    """Plot impulse response functions."""
    if not HAS_MPL or not all_results:
        return

    log("\n" + "=" * 60)
    log("5. GENERATING FIGURES")
    log("=" * 60)

    n_panels = len(all_results)
    fig, axes = plt.subplots(1, min(n_panels, 4), figsize=(4 * min(n_panels, 4), 5),
                              squeeze=False)

    labels = {
        "d_capital_stringency_idx": "Capital Stringency",
        "d_activity_restrictions_idx": "Activity Restrictions",
        "d_supervisory_power_idx": "Supervisory Power",
        "d_overall_restrictiveness_idx": "Overall Restrictiveness",
    }

    for i, (shock_col, irfs) in enumerate(all_results.items()):
        if i >= 4:
            break
        ax = axes[0, i]

        horizons = sorted(irfs.keys())
        betas = [irfs[h].get("beta", np.nan) for h in horizons]
        ses = [irfs[h].get("se", np.nan) for h in horizons]

        betas = np.array(betas)
        ses = np.array(ses)

        # Plot IRF with confidence bands
        valid = ~np.isnan(betas)
        h_valid = np.array(horizons)[valid]
        b_valid = betas[valid]
        se_valid = ses[valid]

        ax.plot(h_valid, b_valid, 'b-o', markersize=4, linewidth=1.5)
        ax.fill_between(h_valid, b_valid - 1.96 * se_valid, b_valid + 1.96 * se_valid,
                         alpha=0.2, color='blue')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel("Horizon (years)")
        ax.set_ylabel("β_h")
        ax.set_title(labels.get(shock_col, shock_col), fontsize=10)
        ax.grid(True, alpha=0.3)

        # Mark the "theory predicts zero" region
        ax.axvspan(4, max(horizons) + 0.5, alpha=0.05, color='green')
        if i == 0:
            ax.text(6, ax.get_ylim()[1] * 0.8, "Theory:\nβ→0 here",
                    fontsize=7, ha='center', color='green', alpha=0.7)

    plt.suptitle("Impulse Response: Regulatory Shock → Financial Development\n"
                  "(Damping Cancellation predicts transient effect only)",
                  fontsize=11)
    plt.tight_layout()

    fig_path = f"{FIG_DIR}/figure_damping_cancellation_irf.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    log(f"  Saved: {fig_path}")
    plt.close()


def main():
    log("=" * 60)
    log("DAMPING CANCELLATION TEST — Theorem E.3")
    log("Complementary Heterogeneity (Smirl 2026)")
    log("=" * 60)

    # Load and merge data
    panel, bcl, imf, fraser = load_and_merge()

    # Run analyses
    descriptive_regulation(panel)
    all_results = local_projection(panel, imf)
    basel3_did(panel, imf)
    layer_comparison(panel, imf)
    plot_irf(all_results)

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY — FOR PAPER SECTION")
    log("=" * 60)
    log("The damping cancellation test examines whether regulatory")
    log("tightening at one institutional layer has persistent effects")
    log("on aggregate financial development, or whether the system")
    log("reroutes (cₙσₙ independent of σₙ, Theorem E.3).")
    log("")
    log("Key results to report:")
    log("  1. Local projection IRFs by regulatory dimension (Figure X)")
    log("  2. Basel III DID estimate (Table X)")
    log("  3. Cross-layer comparison at h=5 (Table X+1)")
    log("")
    log("Damping cancellation predicts:")
    log("  - β_h significant at h=1-2, insignificant by h=4-6")
    log("  - β_h at h=5 roughly equal across regulatory layers")
    log("")
    log("IMPORTANT CAVEAT:")
    log("  Panel is small (16 countries × 4 waves = 64 obs).")
    log("  Power is limited. Downloading full BCL survey (180+ countries)")
    log("  from worldbank.org/en/research/brief/BRSS would substantially")
    log("  improve precision.")

    # Save results
    results_path = f"{OUT_DIR}/damping_test_results.txt"
    with open(results_path, "w") as f:
        f.write("\n".join(LOG_LINES))
    log(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
