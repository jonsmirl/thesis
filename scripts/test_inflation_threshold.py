#!/usr/bin/env python3
"""
test_inflation_threshold.py
===========================
Empirical test: nonlinear inflation threshold for crypto/stablecoin adoption
and whether this threshold declines as the stablecoin ecosystem grows.

Paper 4 (Settlement Feedback), Prediction 3:
    Dollarization thresholds pi_bar(S) and pi_low(S) are both decreasing
    in S (stablecoin ecosystem size). Countries should adopt crypto at
    lower inflation rates as stablecoins become more accessible.

Tests:
    1. Logit: P(high_adoption) = f(inflation, GDP/cap, internet, stablecoin_mcap)
    2. Piecewise regression with grid-searched breakpoint (2%--50%)
    3. Interaction: inflation * stablecoin_mcap  (d > 0 => declining threshold)
    4. Time-varying threshold: year-by-year 50% adoption probability
    5. Quintile analysis: mean adoption by inflation quintile, monotonicity

Author: Connor Doll / Smirl (2026)
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Matplotlib: Agg backend (no display)
# --------------------------------------------------------------------------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --------------------------------------------------------------------------- #
# statsmodels
# --------------------------------------------------------------------------- #
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, Probit
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================================================================== #
# PATHS
# =========================================================================== #
BASE = "/home/jonsmirl/thesis"
WDI_PATH         = os.path.join(BASE, "thesis_data", "wdi_panel.csv")
CHAINALYSIS_PATH = os.path.join(BASE, "thesis_data", "chainalysis_adoption.csv")
FQI_PATH         = os.path.join(BASE, "data", "fqi_panel.csv")
STABLECOIN_PATH  = os.path.join(BASE, "thesis_data", "stablecoin_volumes.csv")

FIG_DIR      = os.path.join(BASE, "figures", "settlement_feedback")
RESULTS_PATH = os.path.join(BASE, "thesis_data", "inflation_threshold_results.txt")

os.makedirs(FIG_DIR, exist_ok=True)

# =========================================================================== #
# COUNTRY NAME HARMONISATION
# =========================================================================== #
CHAINALYSIS_TO_WDI = {
    "Russia":    "Russian Federation",
    "Turkey":    "Turkiye",
    "Venezuela": "Venezuela, RB",
    "Vietnam":   "Viet Nam",
}

# Stablecoin total market cap by year ($B) -- fallback values.
STABLECOIN_MCAP_FALLBACK = {
    2020:  28,
    2021: 150,
    2022: 145,
    2023: 130,
    2024: 200,
    2025: 235,
}


# =========================================================================== #
# 1.  LOAD AND MERGE
# =========================================================================== #
def load_data():
    """Load, harmonise, and merge all panels."""
    # ---- Chainalysis ----
    ch = pd.read_csv(CHAINALYSIS_PATH)
    ch = ch[["year", "country", "score"]].copy()
    ch["country_wdi"] = ch["country"].replace(CHAINALYSIS_TO_WDI)

    # ---- WDI ----
    wdi = pd.read_csv(WDI_PATH)
    wdi_cols = [
        "country_code", "country", "year",
        "inflation_cpi_annual_pct",
        "gdp_per_capita_ppp_constant",
        "internet_users_pct",
        "domestic_credit_private_pct_gdp",
        "broad_money_pct_gdp",
    ]
    wdi = wdi[[c for c in wdi_cols if c in wdi.columns]].copy()

    # ---- FQI (for fiat_quality_index as additional control) ----
    fqi = pd.read_csv(FQI_PATH)
    fqi_sub = fqi[["country", "year", "fiat_quality_index"]].copy()

    # ---- Stablecoin market cap ----
    try:
        sv = pd.read_csv(STABLECOIN_PATH)
        sv["year"] = pd.to_datetime(sv["quarter"], format="mixed").dt.year
        stable_annual = sv.groupby("year")["adjusted_volume_bn"].mean().reset_index()
        stable_annual.columns = ["year", "stablecoin_mcap_bn"]
    except Exception:
        stable_annual = pd.DataFrame(
            list(STABLECOIN_MCAP_FALLBACK.items()),
            columns=["year", "stablecoin_mcap_bn"]
        )

    # Fill any missing years with fallback
    for y, v in STABLECOIN_MCAP_FALLBACK.items():
        if y not in stable_annual["year"].values:
            stable_annual = pd.concat([
                stable_annual,
                pd.DataFrame({"year": [y], "stablecoin_mcap_bn": [v]})
            ], ignore_index=True)

    # ---- MERGE ----
    # Rename WDI country to avoid collision
    wdi = wdi.rename(columns={"country": "country_wdi_match"})
    panel = ch.merge(wdi, left_on=["country_wdi", "year"],
                     right_on=["country_wdi_match", "year"],
                     how="inner")
    panel = panel.drop(columns=["country_wdi_match"], errors="ignore")

    # Rename FQI country to avoid collision
    fqi_sub = fqi_sub.rename(columns={"country": "country_fqi_match"})
    panel = panel.merge(fqi_sub, left_on=["country_wdi", "year"],
                        right_on=["country_fqi_match", "year"],
                        how="left")
    panel = panel.drop(columns=["country_fqi_match"], errors="ignore")

    panel = panel.merge(stable_annual, on="year", how="left")

    # Clean up columns
    panel.rename(columns={
        "country_ch": "country",
        "inflation_cpi_annual_pct": "inflation",
        "gdp_per_capita_ppp_constant": "gdp_pc",
        "internet_users_pct": "internet",
        "domestic_credit_private_pct_gdp": "credit",
        "broad_money_pct_gdp": "broad_money",
        "fiat_quality_index": "fqi",
        "stablecoin_mcap_bn": "stable_mcap",
    }, inplace=True)

    # Derived variables
    panel["log_gdp_pc"] = np.log(panel["gdp_pc"].clip(lower=100))
    panel["log_inflation"] = np.log(panel["inflation"].clip(lower=0.01))
    panel["log_stable_mcap"] = np.log(panel["stable_mcap"].clip(lower=1))

    # Binary adoption: above overall median
    median_score = panel["score"].median()
    panel["high_adoption"] = (panel["score"] > median_score).astype(int)

    # Interaction
    panel["infl_x_stable"] = panel["inflation"] * panel["stable_mcap"]

    # Era dummy
    panel["post_2021"] = (panel["year"] >= 2021).astype(int)

    print(f"Panel constructed: {len(panel)} obs, {panel['country'].nunique()} countries, "
          f"years {panel['year'].min()}-{panel['year'].max()}")
    print(f"High-adoption median cutoff: {median_score:.3f}")
    print(f"Inflation range: {panel['inflation'].min():.2f} to {panel['inflation'].max():.2f}")
    print()

    return panel


# =========================================================================== #
# 2.  TEST 1 -- LOGIT / PROBIT
# =========================================================================== #
def test1_logit(panel, out):
    """Logit & Probit: P(high_adoption) = f(inflation, controls, stablecoin_mcap)."""
    out.append("=" * 72)
    out.append("TEST 1: Logit / Probit -- Inflation and Crypto Adoption")
    out.append("=" * 72)

    df = panel.dropna(subset=["high_adoption", "inflation", "log_gdp_pc",
                              "internet", "log_stable_mcap"]).copy()
    out.append(f"Sample size: {len(df)}")

    y = df["high_adoption"]
    X_vars = ["inflation", "log_gdp_pc", "internet", "log_stable_mcap"]
    X = sm.add_constant(df[X_vars])

    logit_model = None
    try:
        logit_model = Logit(y, X).fit(disp=0, maxiter=200)
        out.append("\n--- Logit Results ---")
        out.append(logit_model.summary().as_text())

        mfx = logit_model.get_margeff(at="mean")
        out.append("\n--- Marginal Effects at Means ---")
        out.append(mfx.summary().as_text())
    except Exception as e:
        out.append(f"Logit failed: {e}")

    try:
        probit_model = Probit(y, X).fit(disp=0, maxiter=200)
        out.append("\n--- Probit Results ---")
        out.append(probit_model.summary().as_text())
    except Exception as e:
        out.append(f"Probit failed: {e}")

    out.append("")
    return logit_model


# =========================================================================== #
# 3.  TEST 2 -- PIECEWISE REGRESSION (BREAKPOINT SEARCH)
# =========================================================================== #
def test2_piecewise(panel, out):
    """Grid search for inflation breakpoint: OLS with piecewise slope."""
    out.append("=" * 72)
    out.append("TEST 2: Piecewise Regression -- Inflation Breakpoint Search")
    out.append("=" * 72)

    df = panel.dropna(subset=["score", "inflation", "log_gdp_pc",
                              "internet"]).copy()
    df = df[df["inflation"] > 0].copy()
    out.append(f"Sample size: {len(df)}")

    breakpoints = np.arange(2, 51, 1)
    best_aic = np.inf
    best_bp = None
    results_grid = []

    y = df["score"]

    for bp in breakpoints:
        df["infl_below"] = np.minimum(df["inflation"], bp)
        df["infl_above"] = np.maximum(df["inflation"] - bp, 0)

        X_vars = ["infl_below", "infl_above", "log_gdp_pc", "internet"]
        X = sm.add_constant(df[X_vars])

        try:
            model = sm.OLS(y, X).fit()
            results_grid.append((bp, model.aic, model.rsquared_adj,
                                 model.params.get("infl_below", np.nan),
                                 model.params.get("infl_above", np.nan)))
            if model.aic < best_aic:
                best_aic = model.aic
                best_bp = bp
        except Exception:
            pass

    best_model = None
    if best_bp is not None:
        df["infl_below"] = np.minimum(df["inflation"], best_bp)
        df["infl_above"] = np.maximum(df["inflation"] - best_bp, 0)
        X_vars = ["infl_below", "infl_above", "log_gdp_pc", "internet"]
        X = sm.add_constant(df[X_vars])
        best_model = sm.OLS(y, X).fit()

        out.append(f"\nBest breakpoint: {best_bp}% inflation (AIC = {best_aic:.2f})")
        out.append(f"Slope below breakpoint: {best_model.params['infl_below']:.6f} "
                   f"(p = {best_model.pvalues['infl_below']:.4f})")
        out.append(f"Slope above breakpoint: {best_model.params['infl_above']:.6f} "
                   f"(p = {best_model.pvalues['infl_above']:.4f})")
        out.append(f"Adj R-squared: {best_model.rsquared_adj:.4f}")
        out.append("\n--- Full model at optimal breakpoint ---")
        out.append(best_model.summary().as_text())

        # Compare to linear-only
        X_lin = sm.add_constant(df[["inflation", "log_gdp_pc", "internet"]])
        linear_model = sm.OLS(y, X_lin).fit()
        out.append(f"\nLinear-only AIC: {linear_model.aic:.2f}")
        out.append(f"Piecewise AIC improvement: {linear_model.aic - best_aic:.2f}")
    else:
        out.append("No valid breakpoint found.")

    out.append("")
    return best_bp, results_grid


# =========================================================================== #
# 4.  TEST 3 -- INTERACTION: inflation * stablecoin_mcap
# =========================================================================== #
def test3_interaction(panel, out):
    """Test whether inflation effect on adoption is amplified by stablecoin ecosystem size."""
    out.append("=" * 72)
    out.append("TEST 3: Interaction -- Inflation x Stablecoin Ecosystem Size")
    out.append("=" * 72)

    df = panel.dropna(subset=["score", "inflation", "log_gdp_pc",
                              "internet", "stable_mcap"]).copy()
    out.append(f"Sample size: {len(df)}")

    # Standardize for interpretability
    df["inflation_z"] = (df["inflation"] - df["inflation"].mean()) / df["inflation"].std()
    df["stable_z"] = (df["stable_mcap"] - df["stable_mcap"].mean()) / df["stable_mcap"].std()
    df["infl_x_stable_z"] = df["inflation_z"] * df["stable_z"]

    # OLS on continuous score
    X_vars = ["inflation_z", "stable_z", "infl_x_stable_z", "log_gdp_pc", "internet"]
    X = sm.add_constant(df[X_vars])
    y = df["score"]

    model = sm.OLS(y, X).fit()
    out.append("\n--- OLS: adoption_score ~ inflation*stablecoin_mcap + controls ---")
    out.append(model.summary().as_text())

    d_coef = model.params["infl_x_stable_z"]
    d_pval = model.pvalues["infl_x_stable_z"]
    out.append(f"\nInteraction coefficient (d): {d_coef:.6f}")
    out.append(f"Interaction p-value:         {d_pval:.4f}")

    if d_coef > 0:
        out.append("INTERPRETATION: d > 0 => larger stablecoin ecosystem AMPLIFIES "
                   "the inflation-adoption link.")
        out.append("This is CONSISTENT with Paper 4 Prediction 3: the adoption "
                   "threshold is declining in S.")
    else:
        out.append("INTERPRETATION: d <= 0 => larger stablecoin ecosystem does NOT "
                   "amplify inflation-adoption link.")
        out.append("This would be INCONSISTENT with Paper 4 Prediction 3.")

    # Also try Logit on binary adoption
    out.append("\n--- Logit: P(high_adoption) ~ inflation*stablecoin_mcap + controls ---")
    y_bin = df["high_adoption"]
    try:
        logit_int = Logit(y_bin, X).fit(disp=0, maxiter=200)
        out.append(logit_int.summary().as_text())
        d_logit = logit_int.params["infl_x_stable_z"]
        d_logit_p = logit_int.pvalues["infl_x_stable_z"]
        out.append(f"\nLogit interaction coefficient: {d_logit:.6f} (p = {d_logit_p:.4f})")
    except Exception as e:
        out.append(f"Logit interaction failed: {e}")

    out.append("")
    return model


# =========================================================================== #
# 5.  TEST 4 -- TIME-VARYING THRESHOLD
# =========================================================================== #
def test4_time_varying(panel, out):
    """For each year, estimate the inflation level at which P(high_adoption) = 50%."""
    out.append("=" * 72)
    out.append("TEST 4: Time-Varying Inflation Threshold")
    out.append("=" * 72)

    thresholds = {}
    df_base = panel.dropna(subset=["high_adoption", "inflation", "log_gdp_pc",
                                    "internet"]).copy()

    for year in sorted(df_base["year"].unique()):
        df_yr = df_base[df_base["year"] == year].copy()
        if len(df_yr) < 8:
            out.append(f"  {year}: skipped (n={len(df_yr)})")
            continue

        y = df_yr["high_adoption"]

        # Check for variation in y
        if y.nunique() < 2:
            out.append(f"  {year}: skipped (no variation in adoption)")
            continue

        X = sm.add_constant(df_yr[["inflation", "log_gdp_pc", "internet"]])

        try:
            model = Logit(y, X).fit(disp=0, maxiter=200)
            b_const = model.params["const"]
            b_infl = model.params["inflation"]
            b_gdp = model.params["log_gdp_pc"]
            b_int = model.params["internet"]

            gdp_mean = df_yr["log_gdp_pc"].mean()
            int_mean = df_yr["internet"].mean()

            if abs(b_infl) > 1e-10:
                pi_50 = -(b_const + b_gdp * gdp_mean + b_int * int_mean) / b_infl
                # Only keep if threshold is in a reasonable range
                if -50 < pi_50 < 500:
                    thresholds[year] = pi_50
                    out.append(f"  {year}: threshold = {pi_50:.2f}% inflation "
                               f"(b_infl = {b_infl:.4f}, n = {len(df_yr)})")
                else:
                    out.append(f"  {year}: threshold = {pi_50:.2f}% (out of range, excluded)")
            else:
                out.append(f"  {year}: b_infl ~ 0, cannot compute threshold")
        except Exception as e:
            out.append(f"  {year}: estimation failed ({e})")

    if len(thresholds) >= 2:
        years_arr = np.array(list(thresholds.keys()))
        thresh_arr = np.array(list(thresholds.values()))
        rho_s, p_s = spearmanr(years_arr, thresh_arr)
        out.append(f"\nSpearman correlation (year vs threshold): rho = {rho_s:.4f}, p = {p_s:.4f}")
        if rho_s < 0:
            out.append("INTERPRETATION: Threshold is DECLINING over time.")
            out.append("CONSISTENT with Paper 4 Prediction 3.")
        else:
            out.append("INTERPRETATION: Threshold is NOT declining over time.")
    else:
        out.append("Insufficient yearly estimates for trend test.")

    out.append("")
    return thresholds


# =========================================================================== #
# 6.  TEST 5 -- QUINTILE ANALYSIS
# =========================================================================== #
def test5_quintiles(panel, out):
    """Split countries by inflation quintile; test monotonic adoption increase."""
    out.append("=" * 72)
    out.append("TEST 5: Inflation Quintile Analysis")
    out.append("=" * 72)

    df = panel.dropna(subset=["score", "inflation"]).copy()
    df = df[df["inflation"].notna() & np.isfinite(df["inflation"])].copy()
    out.append(f"Sample size: {len(df)}")

    # Create quintiles
    df["infl_quintile"] = pd.qcut(df["inflation"], q=5, labels=False,
                                   duplicates="drop") + 1

    quintile_stats = df.groupby("infl_quintile").agg(
        mean_inflation=("inflation", "mean"),
        median_inflation=("inflation", "median"),
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        pct_high=("high_adoption", "mean"),
        n=("score", "count"),
    ).reset_index()

    out.append("\nQuintile Summary:")
    out.append(f"{'Q':>3} {'Mean Infl':>10} {'Med Infl':>10} {'Mean Score':>11} "
               f"{'Med Score':>11} {'%High':>7} {'N':>5}")
    out.append("-" * 65)
    for _, row in quintile_stats.iterrows():
        out.append(f"{int(row['infl_quintile']):>3} {row['mean_inflation']:>10.2f} "
                   f"{row['median_inflation']:>10.2f} {row['mean_score']:>11.4f} "
                   f"{row['median_score']:>11.4f} {row['pct_high']:>7.1%} "
                   f"{int(row['n']):>5}")

    # Monotonicity test
    rho_s, p_s = spearmanr(quintile_stats["infl_quintile"],
                            quintile_stats["mean_score"])
    out.append(f"\nSpearman rank correlation (quintile vs mean_score): "
               f"rho = {rho_s:.4f}, p = {p_s:.4f}")
    if rho_s > 0:
        out.append("INTERPRETATION: Higher inflation quintiles have HIGHER adoption scores.")
        out.append("CONSISTENT with inflation driving crypto adoption.")
    elif rho_s < 0:
        out.append("INTERPRETATION: Higher inflation quintiles have LOWER adoption scores.")
        out.append("This may reflect confounding with income (richer = more internet/access).")

    out.append("")
    return quintile_stats


# =========================================================================== #
# 7.  FIGURES
# =========================================================================== #
def make_figures(panel, thresholds, quintile_stats, breakpoint):
    """Generate all four figures."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
    })

    # ------------------------------------------------------------------
    # Fig 1: Inflation vs Adoption Scatter with Nonlinear Fit
    # ------------------------------------------------------------------
    df = panel.dropna(subset=["inflation", "score"]).copy()
    df = df[(df["inflation"] > -5) & (df["inflation"] < 200)].copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df["inflation"], df["score"],
                         c=df["year"], cmap="viridis", alpha=0.7,
                         edgecolors="k", linewidths=0.3, s=50)
    fig.colorbar(scatter, ax=ax, label="Year")

    # LOWESS fit
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sorted_df = df.sort_values("inflation")
        lw = lowess(sorted_df["score"].values, sorted_df["inflation"].values, frac=0.5)
        ax.plot(lw[:, 0], lw[:, 1], "r-", linewidth=2.5, label="LOWESS fit")
        ax.legend(loc="upper right")
    except Exception:
        pass

    if breakpoint is not None:
        ax.axvline(breakpoint, color="gray", linestyle="--", alpha=0.7,
                   label=f"Breakpoint = {breakpoint}%")
        ax.legend(loc="upper right")

    ax.set_xlabel("CPI Inflation (%)")
    ax.set_ylabel("Chainalysis Adoption Score")
    ax.set_title("Inflation vs. Crypto Adoption (2020-2024)")
    ax.set_xlim(left=-2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_inflation_adoption_scatter.png"))
    plt.close(fig)
    print(f"  Saved fig_inflation_adoption_scatter.png")

    # ------------------------------------------------------------------
    # Fig 2: Adoption Probability Curve -- Pre-2021 vs Post-2021
    # ------------------------------------------------------------------
    df2 = panel.dropna(subset=["high_adoption", "inflation", "log_gdp_pc",
                                "internet"]).copy()
    df2 = df2[(df2["inflation"] > 0) & (df2["inflation"] < 200)].copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    colors_era = {"Pre-2021": "#2166ac", "Post-2021": "#b2182b"}

    for era_label, era_val in [("Pre-2021", 0), ("Post-2021", 1)]:
        df_era = df2[df2["post_2021"] == era_val].copy()
        if len(df_era) < 5:
            continue

        y = df_era["high_adoption"]
        X = sm.add_constant(df_era[["inflation", "log_gdp_pc", "internet"]])

        # Check for variation
        if y.nunique() < 2:
            continue

        try:
            m = Logit(y, X).fit(disp=0, maxiter=200)
            infl_max = min(80, df_era["inflation"].max())
            infl_grid = np.linspace(0.5, max(infl_max, 10), 200)
            X_pred = pd.DataFrame({
                "const": 1.0,
                "inflation": infl_grid,
                "log_gdp_pc": df_era["log_gdp_pc"].mean(),
                "internet": df_era["internet"].mean(),
            })
            prob = m.predict(X_pred)
            ax.plot(infl_grid, prob, color=colors_era[era_label], linewidth=2.5,
                    label=era_label)

            jitter = np.random.RandomState(42).uniform(-0.03, 0.03, len(df_era))
            ax.scatter(df_era["inflation"], df_era["high_adoption"] + jitter,
                       color=colors_era[era_label], alpha=0.25, s=30,
                       edgecolors="none")
        except Exception:
            pass

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="50% threshold")
    ax.set_xlabel("CPI Inflation (%)")
    ax.set_ylabel("P(High Crypto Adoption)")
    ax.set_title("Adoption Probability by Stablecoin Era")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_inflation_threshold_curve.png"))
    plt.close(fig)
    print(f"  Saved fig_inflation_threshold_curve.png")

    # ------------------------------------------------------------------
    # Fig 3: Declining Threshold Over Time
    # ------------------------------------------------------------------
    if len(thresholds) >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        years_list = sorted(thresholds.keys())
        thresh_list = [thresholds[y] for y in years_list]

        ax.plot(years_list, thresh_list, "o-", color="#d73027",
                linewidth=2, markersize=8, label="Estimated 50% threshold")

        # Linear trend
        z = np.polyfit(years_list, thresh_list, 1)
        trend_line = np.polyval(z, years_list)
        ax.plot(years_list, trend_line, "--", color="gray", linewidth=1.5,
                label=f"Linear trend (slope = {z[0]:.2f}%/yr)")

        ax.set_xlabel("Year")
        ax.set_ylabel("Inflation Threshold for 50% Adoption Probability (%)")
        ax.set_title("Declining Inflation Threshold for Crypto Adoption")
        ax.legend()
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig_inflation_threshold_declining.png"))
        plt.close(fig)
        print(f"  Saved fig_inflation_threshold_declining.png")
    else:
        print("  Skipped fig_inflation_threshold_declining.png (insufficient data)")

    # ------------------------------------------------------------------
    # Fig 4: Bar Chart -- Adoption by Inflation Quintile
    # ------------------------------------------------------------------
    if quintile_stats is not None and len(quintile_stats) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        qs = quintile_stats
        labels = [f"Q{int(q)}\n({mi:.1f}%)" for q, mi in
                  zip(qs["infl_quintile"], qs["median_inflation"])]

        # Left: Mean adoption score
        bars1 = ax1.bar(labels, qs["mean_score"], color="#4393c3",
                        edgecolor="k", linewidth=0.5)
        ax1.set_xlabel("Inflation Quintile (median inflation)")
        ax1.set_ylabel("Mean Chainalysis Adoption Score")
        ax1.set_title("Adoption Score by Inflation Quintile")
        for bar, n in zip(bars1, qs["n"]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"n={int(n)}", ha="center", va="bottom", fontsize=9)

        # Right: % high adoption
        bars2 = ax2.bar(labels, qs["pct_high"] * 100, color="#d6604d",
                        edgecolor="k", linewidth=0.5)
        ax2.set_xlabel("Inflation Quintile (median inflation)")
        ax2.set_ylabel("% Countries with High Adoption")
        ax2.set_title("High Adoption Rate by Inflation Quintile")
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
        for bar, n in zip(bars2, qs["n"]):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"n={int(n)}", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig_inflation_quintile_adoption.png"))
        plt.close(fig)
        print(f"  Saved fig_inflation_quintile_adoption.png")


# =========================================================================== #
# 8.  MAIN
# =========================================================================== #
def main():
    print("=" * 72)
    print("Inflation Threshold Test -- Paper 4 (Settlement Feedback), Prediction 3")
    print("=" * 72)
    print()

    panel = load_data()

    out = []
    out.append("INFLATION THRESHOLD ANALYSIS -- Paper 4, Prediction 3")
    out.append(f"Date: 2026-02-17")
    out.append(f"Panel: {len(panel)} obs, {panel['country'].nunique()} countries, "
               f"years {panel['year'].min()}-{panel['year'].max()}")
    out.append(f"Adoption median cutoff: {panel['score'].median():.4f}")
    out.append("")

    # --- Test 1 ---
    logit_model = test1_logit(panel, out)

    # --- Test 2 ---
    breakpoint, bp_grid = test2_piecewise(panel, out)

    # --- Test 3 ---
    interaction_model = test3_interaction(panel, out)

    # --- Test 4 ---
    thresholds = test4_time_varying(panel, out)

    # --- Test 5 ---
    quintile_stats = test5_quintiles(panel, out)

    # --- Summary ---
    out.append("=" * 72)
    out.append("SUMMARY")
    out.append("=" * 72)
    out.append("")
    out.append("Paper 4, Prediction 3: dollarization thresholds pi_bar(S) and")
    out.append("pi_low(S) are both decreasing in S (stablecoin ecosystem size).")
    out.append("")
    out.append("Evidence assessment:")
    out.append("")

    # Test 1 summary
    if logit_model is not None:
        try:
            b_infl = logit_model.params["inflation"]
            p_infl = logit_model.pvalues["inflation"]
            out.append(f"  Test 1 (Logit): inflation coeff = {b_infl:.4f} "
                       f"(p = {p_infl:.4f})")
            if b_infl > 0 and p_infl < 0.10:
                out.append("    => Inflation positively predicts crypto adoption (as expected)")
            else:
                out.append("    => Inflation effect not significant or wrong sign")
        except Exception:
            out.append("  Test 1: Could not extract inflation coefficient")

    # Test 2 summary
    if breakpoint is not None:
        out.append(f"  Test 2 (Piecewise): optimal breakpoint at {breakpoint}% inflation")
    else:
        out.append("  Test 2 (Piecewise): no clear breakpoint found")

    # Test 3 summary
    if interaction_model is not None:
        d = interaction_model.params.get("infl_x_stable_z", np.nan)
        dp = interaction_model.pvalues.get("infl_x_stable_z", np.nan)
        out.append(f"  Test 3 (Interaction): d = {d:.4f} (p = {dp:.4f})")
        if d > 0:
            out.append("    => Stablecoin ecosystem amplifies inflation effect (SUPPORTS prediction)")
        else:
            out.append("    => No amplification detected")

    # Test 4 summary
    if len(thresholds) >= 2:
        years_t = sorted(thresholds.keys())
        rho_s, _ = spearmanr(years_t, [thresholds[y] for y in years_t])
        out.append(f"  Test 4 (Time-varying): Spearman rho(year, threshold) = {rho_s:.4f}")
        if rho_s < 0:
            out.append("    => Threshold declining over time (SUPPORTS prediction)")
        else:
            out.append("    => Threshold not declining")

    # Test 5 summary
    if quintile_stats is not None:
        rho_q, p_q = spearmanr(quintile_stats["infl_quintile"],
                                quintile_stats["mean_score"])
        out.append(f"  Test 5 (Quintiles): Spearman rho = {rho_q:.4f} (p = {p_q:.4f})")
        if rho_q > 0:
            out.append("    => Monotonic increase: higher inflation => higher adoption")

    out.append("")
    out.append("Note: This is a cross-country panel with ~18 overlapping countries")
    out.append("and 5 years (2020-2024). Small sample sizes limit statistical power.")
    out.append("Results should be interpreted as suggestive, not definitive.")

    # Write results
    results_text = "\n".join(out)
    with open(RESULTS_PATH, "w") as f:
        f.write(results_text)
    print(f"\nResults saved to: {RESULTS_PATH}")

    # --- Figures ---
    print("\nGenerating figures...")
    make_figures(panel, thresholds, quintile_stats, breakpoint)
    print(f"\nFigures saved to: {FIG_DIR}")

    # Print summary to console
    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    summary_start = results_text.find("SUMMARY")
    if summary_start > 0:
        print(results_text[summary_start:])
    else:
        print(results_text[-1000:])


if __name__ == "__main__":
    main()
