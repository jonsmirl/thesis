#!/usr/bin/env python3
"""
test_fomc_impact.py
====================
Paper 4 (Settlement Feedback), Prediction 2:
    FOMC statement impact on Treasury yields has accelerated over time,
    consistent with faster information absorption as algorithmic/digital
    market infrastructure deepens.

Tests whether the adjustment ratio (immediate yield move / 5-day total move)
around FOMC announcements has increased over time, implying faster price
discovery and shorter absorption windows.

Data sources:
    - Hard-coded FOMC meeting dates 2000-2026 with action classification
    - FRED DGS2, DGS10, DGS1MO (Treasury yields, daily)
    - FRED VIXCLS (volatility control, daily)

Outputs:
    Figures -> /home/jonsmirl/thesis/figures/settlement_feedback/
    Results -> /home/jonsmirl/thesis/thesis_data/fomc_impact_results.txt

Requires: pandas numpy statsmodels scipy matplotlib requests
"""

import os
import sys
import warnings
from datetime import datetime
from io import StringIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ──────────────────────────────────────────────────────────────────
FIG_DIR = "/home/jonsmirl/thesis/figures/settlement_feedback"
DATA_DIR = "/home/jonsmirl/thesis/thesis_data"
RESULTS_PATH = os.path.join(DATA_DIR, "fomc_impact_results.txt")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    print("ERROR: FRED_API_KEY not set. Source env.sh first.")
    sys.exit(1)

# ── globals for results text ───────────────────────────────────────────────
results_buf = StringIO()


def log(msg=""):
    """Print to stdout and accumulate for results file."""
    print(msg)
    results_buf.write(msg + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  1.  FOMC MEETING DATES AND CLASSIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════
#
# Action codes: U = rate hike (up), C = rate cut, H = hold,
#               QE = QE announcement, QE_END = QE taper/end
#
# Sources: Federal Reserve Board press releases, FOMC calendars
# Coverage: 2000-01 through 2026-01 (~200 scheduled meetings)

FOMC_MEETINGS = [
    # ── 2000: tightening cycle ──
    ("2000-02-02", "U"), ("2000-03-21", "U"), ("2000-05-16", "U"),
    ("2000-06-28", "H"), ("2000-08-22", "H"), ("2000-10-03", "H"),
    ("2000-11-15", "H"), ("2000-12-19", "H"),

    # ── 2001: easing (recession + 9/11) ──
    ("2001-01-03", "C"), ("2001-01-31", "C"), ("2001-03-20", "C"),
    ("2001-04-18", "C"), ("2001-05-15", "H"), ("2001-06-27", "C"),
    ("2001-08-21", "C"), ("2001-09-17", "C"),  # emergency post-9/11
    ("2001-10-02", "C"), ("2001-11-06", "C"), ("2001-12-11", "C"),

    # ── 2002: hold at 1.75%, one cut ──
    ("2002-01-30", "H"), ("2002-03-19", "H"), ("2002-05-07", "H"),
    ("2002-06-26", "H"), ("2002-08-13", "H"), ("2002-09-24", "H"),
    ("2002-11-06", "C"), ("2002-12-10", "H"),

    # ── 2003: cut to 1.0% ──
    ("2003-01-29", "H"), ("2003-03-18", "H"), ("2003-05-06", "H"),
    ("2003-06-25", "C"), ("2003-08-12", "H"), ("2003-09-16", "H"),
    ("2003-10-28", "H"), ("2003-12-09", "H"),

    # ── 2004: begin tightening cycle ──
    ("2004-01-28", "H"), ("2004-03-16", "H"), ("2004-05-04", "H"),
    ("2004-06-30", "U"), ("2004-08-10", "U"), ("2004-09-21", "U"),
    ("2004-11-10", "U"), ("2004-12-14", "U"),

    # ── 2005: continued tightening ──
    ("2005-02-02", "U"), ("2005-03-22", "U"), ("2005-05-03", "U"),
    ("2005-06-30", "U"), ("2005-08-09", "U"), ("2005-09-20", "U"),
    ("2005-11-01", "U"), ("2005-12-13", "U"),

    # ── 2006: tightening to 5.25% then hold ──
    ("2006-01-31", "U"), ("2006-03-28", "U"), ("2006-05-10", "U"),
    ("2006-06-29", "U"), ("2006-08-08", "H"), ("2006-09-20", "H"),
    ("2006-10-25", "H"), ("2006-12-12", "H"),

    # ── 2007: hold then begin easing ──
    ("2007-01-31", "H"), ("2007-03-21", "H"), ("2007-05-09", "H"),
    ("2007-06-28", "H"), ("2007-08-07", "H"),
    ("2007-08-17", "C"),  # emergency discount rate cut
    ("2007-09-18", "C"), ("2007-10-31", "C"),
    ("2007-12-11", "C"),

    # ── 2008: aggressive easing + QE1 ──
    ("2008-01-22", "C"),  # emergency intermeeting cut
    ("2008-01-30", "C"), ("2008-03-18", "C"),
    ("2008-04-30", "C"), ("2008-06-25", "H"), ("2008-08-05", "H"),
    ("2008-09-16", "H"), ("2008-10-08", "C"),  # emergency intermeeting
    ("2008-10-29", "C"),
    ("2008-12-16", "C"),  # ZIRP begins (0-0.25%)

    # ── 2009: ZIRP + QE1 expansion ──
    ("2009-01-28", "H"), ("2009-03-18", "QE"),  # QE1 expansion announced
    ("2009-04-29", "H"), ("2009-06-24", "H"),
    ("2009-08-12", "H"), ("2009-09-23", "H"),
    ("2009-11-04", "H"), ("2009-12-16", "H"),

    # ── 2010: ZIRP + QE2 ──
    ("2010-01-27", "H"), ("2010-03-16", "H"), ("2010-04-28", "H"),
    ("2010-06-23", "H"), ("2010-08-10", "H"),
    ("2010-09-21", "H"),
    ("2010-11-03", "QE"),  # QE2 announced ($600B)
    ("2010-12-14", "H"),

    # ── 2011: QE2 ends, Operation Twist ──
    ("2011-01-26", "H"), ("2011-03-15", "H"), ("2011-04-27", "H"),
    ("2011-06-22", "H"),  # QE2 ends
    ("2011-08-09", "H"),  # forward guidance: "at least through mid-2013"
    ("2011-09-21", "QE"),  # Operation Twist announced
    ("2011-11-02", "H"), ("2011-12-13", "H"),

    # ── 2012: QE3 (open-ended) ──
    ("2012-01-25", "H"), ("2012-03-13", "H"), ("2012-04-25", "H"),
    ("2012-06-20", "H"),
    ("2012-08-01", "H"),
    ("2012-09-13", "QE"),  # QE3 announced ($40B/mo MBS, open-ended)
    ("2012-10-24", "H"),
    ("2012-12-12", "QE"),  # QE3 expanded (+$45B/mo Treasuries)

    # ── 2013: taper tantrum ──
    ("2013-01-30", "H"), ("2013-03-20", "H"),
    ("2013-05-01", "H"),
    ("2013-06-19", "H"),  # Bernanke taper signal -> taper tantrum
    ("2013-07-31", "H"), ("2013-09-18", "H"),  # surprise no-taper
    ("2013-10-30", "H"),
    ("2013-12-18", "QE_END"),  # taper begins ($10B/mo reduction)

    # ── 2014: taper continues, ends ──
    ("2014-01-29", "QE_END"), ("2014-03-19", "H"), ("2014-04-30", "H"),
    ("2014-06-18", "H"), ("2014-07-30", "H"), ("2014-09-17", "H"),
    ("2014-10-29", "QE_END"),  # QE3 ends
    ("2014-12-17", "H"),

    # ── 2015: liftoff ──
    ("2015-01-28", "H"), ("2015-03-18", "H"), ("2015-04-29", "H"),
    ("2015-06-17", "H"), ("2015-07-29", "H"), ("2015-09-17", "H"),
    ("2015-10-28", "H"),
    ("2015-12-16", "U"),  # first hike since 2006

    # ── 2016: one hike ──
    ("2016-01-27", "H"), ("2016-03-16", "H"), ("2016-04-27", "H"),
    ("2016-06-15", "H"), ("2016-07-27", "H"), ("2016-09-21", "H"),
    ("2016-11-02", "H"),
    ("2016-12-14", "U"),

    # ── 2017: three hikes ──
    ("2017-02-01", "H"),
    ("2017-03-15", "U"), ("2017-05-03", "H"),
    ("2017-06-14", "U"),  # + balance sheet normalization plan
    ("2017-07-26", "H"), ("2017-09-20", "H"),  # balance sheet runoff begins Oct
    ("2017-11-01", "H"),
    ("2017-12-13", "U"),

    # ── 2018: four hikes ──
    ("2018-01-31", "H"),
    ("2018-03-21", "U"), ("2018-05-02", "H"),
    ("2018-06-13", "U"), ("2018-08-01", "H"),
    ("2018-09-26", "U"), ("2018-11-08", "H"),
    ("2018-12-19", "U"),

    # ── 2019: mid-cycle cuts ──
    ("2019-01-30", "H"), ("2019-03-20", "H"), ("2019-05-01", "H"),
    ("2019-06-19", "H"),
    ("2019-07-31", "C"), ("2019-09-18", "C"),
    ("2019-10-30", "C"),
    ("2019-12-11", "H"),

    # ── 2020: COVID emergency ──
    ("2020-01-29", "H"),
    ("2020-03-03", "C"),  # emergency intermeeting cut (-50bp)
    ("2020-03-15", "C"),  # emergency cut to 0-0.25% + QE4
    ("2020-04-29", "H"), ("2020-06-10", "H"),
    ("2020-07-29", "H"), ("2020-08-27", "H"),  # new framework (AIT)
    ("2020-09-16", "H"), ("2020-11-05", "H"), ("2020-12-16", "H"),

    # ── 2021: hold, taper signal ──
    ("2021-01-27", "H"), ("2021-03-17", "H"), ("2021-04-28", "H"),
    ("2021-06-16", "H"),  # dot plot shift
    ("2021-07-28", "H"), ("2021-09-22", "H"),
    ("2021-11-03", "QE_END"),  # taper announced ($15B/mo reduction)
    ("2021-12-15", "QE_END"),  # accelerated taper ($30B/mo)

    # ── 2022: aggressive tightening cycle ──
    ("2022-01-26", "H"),
    ("2022-03-16", "U"),  # first hike (+25bp)
    ("2022-05-04", "U"),  # +50bp
    ("2022-06-15", "U"),  # +75bp
    ("2022-07-27", "U"),  # +75bp
    ("2022-09-21", "U"),  # +75bp
    ("2022-11-02", "U"),  # +75bp
    ("2022-12-14", "U"),  # +50bp

    # ── 2023: final hikes then hold ──
    ("2023-02-01", "U"),  # +25bp
    ("2023-03-22", "U"),  # +25bp (SVB crisis)
    ("2023-05-03", "U"),  # +25bp (last hike)
    ("2023-06-14", "H"), ("2023-07-26", "H"),
    ("2023-09-20", "H"), ("2023-11-01", "H"), ("2023-12-13", "H"),

    # ── 2024: hold then gradual easing ──
    ("2024-01-31", "H"), ("2024-03-20", "H"), ("2024-05-01", "H"),
    ("2024-06-12", "H"), ("2024-07-31", "H"),
    ("2024-09-18", "C"),  # first cut (-50bp)
    ("2024-11-07", "C"),  # -25bp
    ("2024-12-18", "C"),  # -25bp

    # ── 2025: gradual easing continues ──
    ("2025-01-29", "H"),
    ("2025-03-19", "H"), ("2025-05-07", "H"),
    ("2025-06-18", "H"), ("2025-07-30", "H"),
    ("2025-09-17", "C"), ("2025-10-29", "C"),
    ("2025-12-10", "C"),

    # ── 2026 ──
    ("2026-01-28", "H"),
]


def build_fomc_df() -> pd.DataFrame:
    """Convert FOMC list to DataFrame with date index and action column."""
    records = []
    for date_str, action in FOMC_MEETINGS:
        records.append({
            "fomc_date": pd.Timestamp(date_str),
            "action": action,
            "year": int(date_str[:4]),
        })
    df = pd.DataFrame(records)
    df = df.sort_values("fomc_date").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  2.  FETCH FRED DATA
# ═══════════════════════════════════════════════════════════════════════════

def fetch_fred_series(series_id: str, start: str = "2000-01-01",
                      end: str = "2026-02-17") -> pd.Series:
    """Fetch a single FRED series and return as a pandas Series indexed by date."""
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}"
        f"&api_key={FRED_API_KEY}"
        f"&file_type=json"
        f"&observation_start={start}"
        f"&observation_end={end}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    obs = resp.json()["observations"]
    records = []
    for o in obs:
        val = o["value"]
        if val == ".":
            continue
        records.append({"date": pd.Timestamp(o["date"]), "value": float(val)})
    s = pd.DataFrame(records).set_index("date")["value"]
    s.name = series_id
    return s


def fetch_all_fred() -> pd.DataFrame:
    """Fetch DGS2, DGS10, DGS1MO, VIXCLS from FRED and merge."""
    log("=" * 72)
    log("2. FETCHING TREASURY YIELD AND VIX DATA FROM FRED")
    log("=" * 72)

    series_ids = ["DGS2", "DGS10", "DGS1MO", "VIXCLS"]
    frames = {}
    for sid in series_ids:
        log(f"   Fetching {sid}...")
        s = fetch_fred_series(sid)
        frames[sid] = s
        log(f"     {len(s)} observations, {s.index.min().date()} to {s.index.max().date()}")

    df = pd.DataFrame(frames)
    df.index.name = "date"
    log(f"   Merged panel: {len(df)} trading days")
    log()
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  3.  COMPUTE FOMC IMPACT METRICS
# ═══════════════════════════════════════════════════════════════════════════

def find_nearest_trading_day(date: pd.Timestamp, dates_index: pd.DatetimeIndex,
                             direction: str = "forward") -> pd.Timestamp:
    """Find nearest trading day on or after/before a given date."""
    if direction == "forward":
        mask = dates_index >= date
        candidates = dates_index[mask]
        return candidates.min() if len(candidates) > 0 else None
    else:  # backward
        mask = dates_index <= date
        candidates = dates_index[mask]
        return candidates.max() if len(candidates) > 0 else None


def compute_fomc_impacts(fomc_df: pd.DataFrame, yields_df: pd.DataFrame,
                         yield_col: str = "DGS2") -> pd.DataFrame:
    """
    For each FOMC meeting, compute yield impact metrics.

    Returns DataFrame with one row per meeting and columns:
        immediate_move, sameday_move, fiveday_move, adj_ratio, overshoot
    """
    trading_dates = yields_df.index.sort_values()
    results = []

    for _, row in fomc_df.iterrows():
        fomc_date = row["fomc_date"]

        # Find t (FOMC day or next trading day), t-1, t+1, t+5
        t = find_nearest_trading_day(fomc_date, trading_dates, "forward")
        if t is None:
            continue

        # t-1: last trading day before FOMC
        t_prev = find_nearest_trading_day(fomc_date - pd.Timedelta(days=1),
                                          trading_dates, "backward")
        if t_prev is None:
            continue

        # t+1: next trading day after t
        idx_t = trading_dates.get_loc(t)
        if idx_t + 1 >= len(trading_dates):
            continue
        t_plus1 = trading_dates[idx_t + 1]

        # t+5: 5 trading days after t
        if idx_t + 5 >= len(trading_dates):
            continue
        t_plus5 = trading_dates[idx_t + 5]

        # Get yields (skip if any missing)
        y_prev = yields_df.loc[t_prev, yield_col]
        y_t = yields_df.loc[t, yield_col]
        y_plus1 = yields_df.loc[t_plus1, yield_col]
        y_plus5 = yields_df.loc[t_plus5, yield_col]

        if any(pd.isna([y_prev, y_t, y_plus1, y_plus5])):
            continue

        # Metrics
        sameday_move = abs(y_t - y_prev)
        immediate_move = abs(y_plus1 - y_prev)  # close of day after vs close before
        fiveday_move = abs(y_plus5 - y_prev)

        # Adjustment ratio: how much of 5-day move happens by t+1
        if fiveday_move > 0.001:  # avoid div by zero for tiny moves
            adj_ratio = immediate_move / fiveday_move
        else:
            adj_ratio = np.nan

        # Overshoot: max reversal in [t+1..t+5] relative to immediate direction
        direction = y_plus1 - y_prev  # signed
        max_reversal = 0.0
        for k in range(2, 6):
            if idx_t + k < len(trading_dates):
                y_k = yields_df.loc[trading_dates[idx_t + k], yield_col]
                if not pd.isna(y_k) and abs(direction) > 0.001:
                    reversal_k = y_k - y_plus1
                    # reversal is opposite sign to initial direction
                    if direction > 0 and reversal_k < 0:
                        max_reversal = max(max_reversal, abs(reversal_k))
                    elif direction < 0 and reversal_k > 0:
                        max_reversal = max(max_reversal, abs(reversal_k))

        # VIX on FOMC day (control)
        vix = yields_df.loc[t, "VIXCLS"] if "VIXCLS" in yields_df.columns else np.nan

        results.append({
            "fomc_date": fomc_date,
            "action": row["action"],
            "year": row["year"],
            "year_frac": row["year"] + (fomc_date.month - 1) / 12,
            "sameday_move": sameday_move,
            "immediate_move": immediate_move,
            "fiveday_move": fiveday_move,
            "adj_ratio": adj_ratio,
            "overshoot": max_reversal,
            "surprise_proxy": sameday_move,  # same-day move as surprise magnitude proxy
            "vix": vix,
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
#  4.  REGRESSIONS
# ═══════════════════════════════════════════════════════════════════════════

def run_regressions(impact_df: pd.DataFrame):
    """Run the main regression battery."""
    log("=" * 72)
    log("4. REGRESSION ANALYSIS")
    log("=" * 72)

    # Clean data: drop NaN adj_ratio and extreme outliers
    df = impact_df.dropna(subset=["adj_ratio", "vix"]).copy()
    # Winsorize adj_ratio at [0, 3] (ratios above 3 mean huge reversal)
    df["adj_ratio_w"] = df["adj_ratio"].clip(0, 3)
    log(f"   Observations with valid adj_ratio + VIX: {len(df)}")
    log()

    # ── Model 1: adj_ratio = a + b*year_frac ──
    log("-" * 50)
    log("Model 1: adj_ratio = a + b*year_frac")
    log("-" * 50)
    X1 = sm.add_constant(df["year_frac"])
    m1 = sm.OLS(df["adj_ratio_w"], X1).fit(cov_type="HC1")
    log(m1.summary().as_text())
    log()

    b_year = m1.params["year_frac"]
    p_year = m1.pvalues["year_frac"]
    log(f"   Trend coefficient: {b_year:.4f} (p = {p_year:.4f})")
    if b_year > 0 and p_year < 0.10:
        log("   => CONSISTENT with Prediction 2: adjustment speed increasing over time")
    elif b_year > 0:
        log("   => Positive trend but not statistically significant at 10%")
    else:
        log("   => Negative trend (inconsistent with prediction)")
    log()

    # ── Model 2: + controls ──
    log("-" * 50)
    log("Model 2: adj_ratio = a + b*year_frac + c*vix + d*surprise_proxy")
    log("-" * 50)
    df["log_vix"] = np.log(df["vix"].clip(lower=1))
    X2 = sm.add_constant(df[["year_frac", "log_vix", "surprise_proxy"]])
    m2 = sm.OLS(df["adj_ratio_w"], X2).fit(cov_type="HC1")
    log(m2.summary().as_text())
    log()

    # ── Split samples ──
    for split_year, label in [(2015, "pre/post-2015"), (2020, "pre/post-2020")]:
        log("-" * 50)
        log(f"Split sample: {label}")
        log("-" * 50)

        pre = df[df["year"] < split_year]
        post = df[df["year"] >= split_year]

        for sub_label, sub_df in [("Pre", pre), ("Post", post)]:
            if len(sub_df) < 10:
                log(f"   {sub_label}: too few observations ({len(sub_df)})")
                continue
            X_sub = sm.add_constant(sub_df["year_frac"])
            m_sub = sm.OLS(sub_df["adj_ratio_w"], X_sub).fit(cov_type="HC1")
            log(f"   {sub_label}-{split_year}: N={len(sub_df)}, "
                f"b_year={m_sub.params['year_frac']:.4f} "
                f"(p={m_sub.pvalues['year_frac']:.4f}), "
                f"mean adj_ratio={sub_df['adj_ratio_w'].mean():.3f}")
        log()

    # ── Era comparison ──
    log("-" * 50)
    log("Era comparison: mean adjustment ratio by era")
    log("-" * 50)

    era_bins = [1999, 2010, 2015, 2020, 2027]
    era_labels = ["2000-2009", "2010-2014", "2015-2019", "2020-2026"]
    df["era"] = pd.cut(df["year"], bins=era_bins, labels=era_labels, right=False)

    era_stats = df.groupby("era", observed=True)["adj_ratio_w"].agg(
        ["count", "mean", "median", "std"]
    )
    log(era_stats.to_string())
    log()

    # ANOVA across eras
    era_groups = [g["adj_ratio_w"].values for _, g in df.groupby("era", observed=True)]
    if len(era_groups) >= 2 and all(len(g) > 2 for g in era_groups):
        f_stat, p_anova = stats.f_oneway(*era_groups)
        log(f"   One-way ANOVA: F={f_stat:.3f}, p={p_anova:.4f}")
    log()

    # ── Kruskal-Wallis (non-parametric) ──
    if len(era_groups) >= 2:
        h_stat, p_kw = stats.kruskal(*era_groups)
        log(f"   Kruskal-Wallis: H={h_stat:.3f}, p={p_kw:.4f}")
    log()

    # ── Action type breakdown ──
    log("-" * 50)
    log("Mean adjustment ratio by action type")
    log("-" * 50)
    action_stats = df.groupby("action")["adj_ratio_w"].agg(["count", "mean", "median"])
    log(action_stats.to_string())
    log()

    # ── Rolling 5-year average ──
    df_sorted = df.sort_values("fomc_date")
    df_sorted["adj_ratio_rolling"] = df_sorted["adj_ratio_w"].rolling(
        window=40, min_periods=20, center=True
    ).mean()

    return df, df_sorted, m1, m2


# ═══════════════════════════════════════════════════════════════════════════
#  5.  YIELD PATH AROUND FOMC BY ERA (for fig 4)
# ═══════════════════════════════════════════════════════════════════════════

def compute_yield_paths(fomc_df: pd.DataFrame, yields_df: pd.DataFrame,
                        yield_col: str = "DGS2") -> dict:
    """
    Compute average normalized yield path around FOMC dates [-5, +10] trading days,
    grouped by era.
    """
    trading_dates = yields_df.index.sort_values()
    era_bins = [1999, 2010, 2015, 2020, 2027]
    era_labels = ["2000-2009", "2010-2014", "2015-2019", "2020-2026"]

    paths = {era: [] for era in era_labels}

    for _, row in fomc_df.iterrows():
        fomc_date = row["fomc_date"]
        t = find_nearest_trading_day(fomc_date, trading_dates, "forward")
        if t is None:
            continue

        idx_t = trading_dates.get_loc(t)
        # Need at least 5 days before and 10 after
        if idx_t < 5 or idx_t + 10 >= len(trading_dates):
            continue

        # Get yields from t-5 to t+10
        window_dates = trading_dates[idx_t - 5: idx_t + 11]
        window_yields = yields_df.loc[window_dates, yield_col].values

        if any(pd.isna(window_yields)):
            continue

        # Normalize: subtract yield at t-1 (index 4 in 0-based window)
        baseline = window_yields[4]
        if abs(window_yields[5] - baseline) < 0.001:
            continue  # skip trivial moves

        # Normalize direction so initial move is always positive
        sign = 1.0 if window_yields[5] >= baseline else -1.0
        normalized = sign * (window_yields - baseline)

        # Assign to era
        year = row["year"]
        for i, (lo, hi) in enumerate(zip(era_bins[:-1], era_bins[1:])):
            if lo <= year < hi:
                paths[era_labels[i]].append(normalized)
                break

    # Average within each era
    avg_paths = {}
    for era, era_paths in paths.items():
        if len(era_paths) >= 5:
            avg_paths[era] = np.mean(era_paths, axis=0)
        else:
            avg_paths[era] = None

    return avg_paths


# ═══════════════════════════════════════════════════════════════════════════
#  6.  FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def make_figures(df: pd.DataFrame, df_sorted: pd.DataFrame,
                 model1, avg_paths: dict):
    """Generate the 4 analysis figures."""
    log("=" * 72)
    log("6. GENERATING FIGURES")
    log("=" * 72)

    # ── Figure 1: Adjustment ratio over time with trend ──
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color by action type
    colors = {"U": "#d62728", "C": "#2ca02c", "H": "#7f7f7f",
              "QE": "#1f77b4", "QE_END": "#ff7f0e"}
    labels_used = set()
    for _, row in df.iterrows():
        c = colors.get(row["action"], "#7f7f7f")
        label_map = {"U": "Rate hike", "C": "Rate cut", "H": "Hold",
                     "QE": "QE announcement", "QE_END": "QE taper/end"}
        lbl = label_map.get(row["action"], row["action"])
        ax.scatter(row["fomc_date"], row["adj_ratio_w"],
                   color=c, alpha=0.5, s=30,
                   label=lbl if lbl not in labels_used else "")
        labels_used.add(lbl)

    # Trend line
    x_trend = df["year_frac"].values
    y_pred = model1.predict(sm.add_constant(x_trend))
    # Convert year_frac to dates for plotting
    trend_dates = pd.to_datetime([f"{int(y)}-{int((y % 1) * 12) + 1:02d}-15"
                                  for y in x_trend])
    sorted_idx = np.argsort(x_trend)
    ax.plot(trend_dates.values[sorted_idx], y_pred[sorted_idx],
            "k--", linewidth=2, label=f"Trend (b={model1.params['year_frac']:.4f})")

    ax.set_xlabel("Date")
    ax.set_ylabel("Adjustment Ratio (immediate / 5-day move)")
    ax.set_title("FOMC Impact: Speed of Treasury Yield Adjustment Over Time\n"
                 "(Higher = faster absorption of FOMC signal)")
    ax.legend(loc="upper left", fontsize=9)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="")
    ax.set_ylim(0, 3.1)
    fig.tight_layout()
    path1 = os.path.join(FIG_DIR, "fig_fomc_adjustment_speed.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    log(f"   Saved: {path1}")

    # ── Figure 2: Rolling average of adjustment ratio ──
    fig, ax = plt.subplots(figsize=(12, 6))
    valid = df_sorted.dropna(subset=["adj_ratio_rolling"])
    ax.plot(valid["fomc_date"], valid["adj_ratio_rolling"],
            "b-", linewidth=2, label="Rolling 40-meeting average")
    ax.fill_between(valid["fomc_date"],
                    valid["adj_ratio_rolling"] - valid["adj_ratio_w"].rolling(40, min_periods=20, center=True).std() * 0.5,
                    valid["adj_ratio_rolling"] + valid["adj_ratio_w"].rolling(40, min_periods=20, center=True).std() * 0.5,
                    alpha=0.2, color="blue", label="+-0.5 SD band")

    # Add vertical lines for key regime changes
    regime_dates = [
        ("2008-12-16", "ZIRP\nbegins"),
        ("2015-12-16", "Liftoff"),
        ("2020-03-15", "COVID\nZLB"),
        ("2022-03-16", "Hike\ncycle"),
    ]
    for rd, rl in regime_dates:
        ax.axvline(pd.Timestamp(rd), color="red", alpha=0.3, linestyle="--")
        ax.text(pd.Timestamp(rd), ax.get_ylim()[1] * 0.95, rl,
                fontsize=8, ha="center", color="red", alpha=0.7)

    ax.set_xlabel("Date")
    ax.set_ylabel("Adjustment Ratio (rolling average)")
    ax.set_title("FOMC Yield Absorption Speed: Rolling Average\n"
                 "(40-meeting centered window)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path2 = os.path.join(FIG_DIR, "fig_fomc_absorption_time.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    log(f"   Saved: {path2}")

    # ── Figure 3: Box plots by era ──
    fig, ax = plt.subplots(figsize=(10, 6))
    era_bins = [1999, 2010, 2015, 2020, 2027]
    era_labels = ["2000-2009", "2010-2014", "2015-2019", "2020-2026"]
    df["era"] = pd.cut(df["year"], bins=era_bins, labels=era_labels, right=False)

    era_data = [df[df["era"] == era]["adj_ratio_w"].dropna().values
                for era in era_labels]

    bp = ax.boxplot(era_data, tick_labels=era_labels, patch_artist=True,
                    showmeans=True, meanline=True,
                    meanprops=dict(color="red", linewidth=2))

    era_colors = ["#d4e6f1", "#a9cce3", "#7fb3d8", "#5499c7"]
    for patch, color in zip(bp["boxes"], era_colors):
        patch.set_facecolor(color)

    # Add observation counts
    for i, era in enumerate(era_labels):
        n = len(era_data[i])
        ax.text(i + 1, ax.get_ylim()[0] + 0.05, f"n={n}",
                ha="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Era")
    ax.set_ylabel("Adjustment Ratio")
    ax.set_title("FOMC Yield Adjustment Speed by Era\n"
                 "(Red line = mean; green line = median)")
    fig.tight_layout()
    path3 = os.path.join(FIG_DIR, "fig_fomc_era_comparison.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    log(f"   Saved: {path3}")

    # ── Figure 4: Average yield path around FOMC by era ──
    fig, ax = plt.subplots(figsize=(12, 6))
    x_ticks = np.arange(-5, 11)
    x_labels = [str(i) for i in x_ticks]

    era_colors_line = {"2000-2009": "#1f77b4", "2010-2014": "#ff7f0e",
                       "2015-2019": "#2ca02c", "2020-2026": "#d62728"}

    for era, path_data in avg_paths.items():
        if path_data is not None:
            ax.plot(x_ticks, path_data, "-o", markersize=4,
                    color=era_colors_line[era], linewidth=2, label=era)

    ax.axvline(0, color="black", linestyle="--", alpha=0.5, label="FOMC day")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Trading Days Relative to FOMC")
    ax.set_ylabel("Yield Change from t-1 (bp, direction-normalized)")
    ax.set_title("Average Yield Response Path Around FOMC Announcements\n"
                 "(DGS2, normalized so initial move is positive)")
    ax.legend()
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    fig.tight_layout()
    path4 = os.path.join(FIG_DIR, "fig_fomc_yield_response.png")
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    log(f"   Saved: {path4}")

    log()


# ═══════════════════════════════════════════════════════════════════════════
#  7.  ADDITIONAL ANALYSIS: DGS10 and yield spread
# ═══════════════════════════════════════════════════════════════════════════

def run_robustness(fomc_df: pd.DataFrame, yields_df: pd.DataFrame):
    """Run the same analysis on DGS10 and 2-10 spread for robustness."""
    log("=" * 72)
    log("7. ROBUSTNESS: DGS10 AND 2-10 SPREAD")
    log("=" * 72)

    for yield_col, label in [("DGS10", "10-year Treasury"),
                             ("DGS1MO", "1-month Treasury")]:
        log(f"\n--- {label} ({yield_col}) ---")
        impact_df = compute_fomc_impacts(fomc_df, yields_df, yield_col)
        valid = impact_df.dropna(subset=["adj_ratio", "vix"])
        valid = valid.copy()
        valid["adj_ratio_w"] = valid["adj_ratio"].clip(0, 3)

        if len(valid) < 20:
            log(f"   Too few valid observations ({len(valid)}), skipping")
            continue

        X = sm.add_constant(valid["year_frac"])
        m = sm.OLS(valid["adj_ratio_w"], X).fit(cov_type="HC1")
        log(f"   N={len(valid)}, b_year={m.params['year_frac']:.4f} "
            f"(p={m.pvalues['year_frac']:.4f}), "
            f"R2={m.rsquared:.4f}")

        era_bins = [1999, 2010, 2015, 2020, 2027]
        era_labels = ["2000-2009", "2010-2014", "2015-2019", "2020-2026"]
        valid["era"] = pd.cut(valid["year"], bins=era_bins, labels=era_labels, right=False)
        era_means = valid.groupby("era", observed=True)["adj_ratio_w"].mean()
        log(f"   Era means: {dict(era_means.round(3))}")

    log()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 72)
    log("FOMC IMPACT ON TREASURY YIELDS: SPEED OF ADJUSTMENT TEST")
    log("Paper 4 (Settlement Feedback), Prediction 2")
    log(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 72)
    log()

    # 1. Build FOMC meeting table
    log("=" * 72)
    log("1. FOMC MEETING DATABASE")
    log("=" * 72)
    fomc_df = build_fomc_df()
    log(f"   Total FOMC meetings: {len(fomc_df)}")
    log(f"   Date range: {fomc_df['fomc_date'].min().date()} to {fomc_df['fomc_date'].max().date()}")
    action_counts = fomc_df["action"].value_counts()
    for action, count in action_counts.items():
        label_map = {"U": "Rate hike", "C": "Rate cut", "H": "Hold",
                     "QE": "QE announcement", "QE_END": "QE taper/end"}
        log(f"     {label_map.get(action, action)}: {count}")
    log()

    # 2. Fetch FRED data
    yields_df = fetch_all_fred()

    # 3. Compute FOMC impact metrics (primary: DGS2)
    log("=" * 72)
    log("3. COMPUTING FOMC IMPACT METRICS (DGS2)")
    log("=" * 72)
    impact_df = compute_fomc_impacts(fomc_df, yields_df, "DGS2")
    log(f"   Computed impacts for {len(impact_df)} meetings")
    valid_ratio = impact_df["adj_ratio"].notna().sum()
    log(f"   Valid adjustment ratios: {valid_ratio}")
    log(f"   Mean immediate move: {impact_df['immediate_move'].mean():.3f} pp")
    log(f"   Mean 5-day move: {impact_df['fiveday_move'].mean():.3f} pp")
    log(f"   Mean adj ratio: {impact_df['adj_ratio'].dropna().mean():.3f}")
    log(f"   Median adj ratio: {impact_df['adj_ratio'].dropna().median():.3f}")
    log()

    # 4. Run regressions
    df_reg, df_sorted, m1, m2 = run_regressions(impact_df)

    # 5. Compute yield paths for figure 4
    avg_paths = compute_yield_paths(fomc_df, yields_df, "DGS2")

    # 6. Generate figures
    make_figures(df_reg, df_sorted, m1, avg_paths)

    # 7. Robustness checks
    run_robustness(fomc_df, yields_df)

    # ── Summary ──
    log("=" * 72)
    log("SUMMARY")
    log("=" * 72)

    b_coeff = m1.params["year_frac"]
    p_val = m1.pvalues["year_frac"]

    log(f"Main result (DGS2):")
    log(f"  Trend in adjustment ratio: b = {b_coeff:.4f} per year")
    log(f"  p-value (HC1): {p_val:.4f}")

    if b_coeff > 0 and p_val < 0.05:
        verdict = "SUPPORTED at 5% level"
    elif b_coeff > 0 and p_val < 0.10:
        verdict = "WEAKLY SUPPORTED at 10% level"
    elif b_coeff > 0:
        verdict = "DIRECTIONALLY CONSISTENT but not significant"
    else:
        verdict = "NOT SUPPORTED (negative trend)"

    log(f"  Prediction 2 verdict: {verdict}")
    log()
    log("Interpretation: A positive trend means Treasury yields absorb FOMC")
    log("information faster in more recent years, consistent with the hypothesis")
    log("that deepening digital/algorithmic market infrastructure accelerates")
    log("price discovery in sovereign debt markets.")
    log()

    # Save results
    with open(RESULTS_PATH, "w") as f:
        f.write(results_buf.getvalue())
    log(f"Results saved to: {RESULTS_PATH}")

    # Save impact data as CSV
    csv_path = os.path.join(DATA_DIR, "fomc_impact_panel.csv")
    impact_df.to_csv(csv_path, index=False)
    log(f"Impact data saved to: {csv_path}")


if __name__ == "__main__":
    main()
