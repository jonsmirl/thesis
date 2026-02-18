#!/usr/bin/env python3
"""
Export Control DID Test -- Paper 1 (Endogenous Decentralization), Section 5.1
=============================================================================

Theory: US export controls on China (BIS Entity List expansion, Oct 7 2022)
constrained Chinese firms access to frontier GPU hardware. The thesis argues
this accelerated development of efficient, edge-compatible models -- confirming
the learning externality mechanism. Under the overinvestment result (Prop. 1),
constrained firms redirect R&D toward parameter-efficient architectures that
run on available hardware, producing more edge-compatible (<= 7B) models,
more MoE/distillation architectures, and greater open-ecosystem contribution.

Natural experiment: US-China export controls distinguish the learning
externality mechanism from standard Arrow learning-by-doing, because the
constraint is exogenous to model development choices.

Test: Difference-in-Differences comparing constrained (Chinese) vs
unconstrained (US/EU) AI firms model release patterns on HuggingFace.

  - Treatment: BIS Entity List expansion (Oct 7, 2022)
  - Treated: Chinese AI firms (Qwen/Alibaba, DeepSeek, THUDM, BAAI, etc.)
  - Control: US/EU AI firms (Meta, Google, Microsoft, Mistral, etc.)
  - Outcomes: edge-compatible share, model releases, downloads, MoE share

DID specification:
  Y_{f,t} = alpha + beta_1*Post_t + beta_2*Constrained_f
            + delta*(Post_t x Constrained_f) + firm_FE + quarter_FE + eps
  delta is the treatment effect of interest.

Design note: Many Chinese AI firms only began publishing on HuggingFace after
Oct 2022, so the standard DID (requiring pre-treatment data for both groups)
may be under-identified. The script detects this automatically and falls back
to alternative identification strategies:
  - Post-treatment time-trend comparison (do constrained firms converge faster
    toward edge-compatible releases over time?)
  - Cross-sectional level comparison (are constrained firms' post-treatment
    edge shares significantly different from controls, conditional on time?)
  - Entry-cohort matching (compare firms that entered HuggingFace in similar
    quarters, some constrained and some not)

Inputs: HuggingFace API (free, no key required)
Output: thesis_data/export_control_did_results.txt + figures

Requires: pip install pandas numpy statsmodels matplotlib requests
"""

import pandas as pd
import numpy as np
import os
import re
import time
import json
import requests as req_lib
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    HAS_STATS = True
except ImportError:
    HAS_STATS = False
    print("WARNING: statsmodels not installed.")

# -- Paths ------------------------------------------------------------------
DATA_DIR = "/home/jonsmirl/thesis/thesis_data"
FIG_DIR = "/home/jonsmirl/thesis/figures/endogenous_decentralization"
CACHE_PATH = os.path.join(DATA_DIR, "hf_export_control_cache.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

LOG_LINES = []

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOG_LINES.append(line)


# ---------------------------------------------------------------------------
# Constants & Config
# ---------------------------------------------------------------------------

TREATMENT_DATE = datetime(2022, 10, 7)  # BIS Entity List expansion

EDGE_THRESHOLD_B = 7  # Edge-compatible = <= 7B parameters

# Chinese (constrained) firms -- expanded to include firms with pre-treatment
# HuggingFace presence where possible
CHINESE_AUTHORS = [
    "Qwen",          # Alibaba -- started HF late 2023
    "deepseek-ai",   # DeepSeek -- started HF late 2023
    "BAAI",          # Beijing Academy of AI -- has some pre-treatment models
    "01-ai",         # Yi / 01.AI -- started HF late 2023
    "internlm",      # Shanghai AI Lab -- started HF mid 2023
    "openbmb",       # OpenBMB/Tsinghua -- has some pre-treatment models
    "THUDM",         # Tsinghua / ChatGLM -- has pre-treatment models
    "shibing624",    # Chinese NLP researcher -- has pre-treatment models
    "uer",           # UER-py / Tencent -- has pre-treatment models
]

# US/EU (control) firms
WESTERN_AUTHORS = [
    "meta-llama",    # Meta -- started HF mid 2023
    "google",        # Google -- extensive pre-treatment presence
    "microsoft",     # Microsoft -- extensive pre-treatment presence
    "mistralai",     # Mistral -- started HF late 2023
    "bigscience",    # BigScience/BLOOM -- pre-treatment
    "EleutherAI",    # EleutherAI -- pre-treatment
    "stabilityai",   # Stability AI -- pre-treatment
    "huggingface",   # HuggingFace -- pre-treatment
]

ALL_AUTHORS = CHINESE_AUTHORS + WESTERN_AUTHORS

CONSTRAINED_AUTHORS = {}
for a in CHINESE_AUTHORS:
    CONSTRAINED_AUTHORS[a] = 1
for a in WESTERN_AUTHORS:
    CONSTRAINED_AUTHORS[a] = 0

# Regex for extracting parameter count from model name
PARAM_RE = re.compile(r'(\d+\.?\d*)\s*[bB](?:illion)?(?:\b|_|-)', re.IGNORECASE)

# Alternate patterns for smaller models
PARAM_RE_ALT = re.compile(r'(?:^|[-_/])(\d+\.?\d*)[bB](?:$|[-_.])', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Section 1: Fetch all models from HuggingFace API
# ---------------------------------------------------------------------------

def fetch_all_models(use_cache=True):
    """Fetch model metadata from HuggingFace API for each author.

    Uses https://huggingface.co/api/models?author={author}&sort=createdAt&direction=-1&limit=1000
    Caches results to CACHE_PATH (JSON). Returns dict of author -> list of model dicts.
    """
    if use_cache and os.path.exists(CACHE_PATH):
        log(f"Loading cached HuggingFace data from {CACHE_PATH}")
        with open(CACHE_PATH, "r") as f:
            raw = json.load(f)
        # Check if cache has all required authors
        missing = [a for a in ALL_AUTHORS if a not in raw]
        if not missing:
            total = sum(len(v) for v in raw.values())
            log(f"  Cache contains {total} models across {len(raw)} authors")
            return raw
        else:
            log(f"  Cache missing authors: {missing}. Re-fetching all.")

    log("Fetching model metadata from HuggingFace API...")
    raw_data = {}

    for author in ALL_AUTHORS:
        url = (f"https://huggingface.co/api/models?author={author}"
               f"&sort=createdAt&direction=-1&limit=1000")
        log(f"  Fetching {author}...")
        try:
            resp = req_lib.get(url, timeout=30)
            resp.raise_for_status()
            models = resp.json()
            raw_data[author] = models
            log(f"    -> {len(models)} models")
        except Exception as e:
            log(f"    ERROR fetching {author}: {e}")
            raw_data[author] = []

        # Rate limiting
        time.sleep(1.0)

    # Save cache
    with open(CACHE_PATH, "w") as f:
        json.dump(raw_data, f)
    total = sum(len(v) for v in raw_data.values())
    log(f"Cached {total} models to {CACHE_PATH}")

    return raw_data


# ---------------------------------------------------------------------------
# Section 2: Build model-level dataset
# ---------------------------------------------------------------------------

def _extract_param_b(model_id):
    """Extract parameter count in billions from model ID string.

    Tries regex patterns like '7b', '13b', '70b', '1.5b', '0.5b'.
    Returns float or np.nan.
    """
    # Use model name (after the slash)
    name = model_id.split("/")[-1] if "/" in model_id else model_id
    name_lower = name.lower()

    # Primary pattern
    m = PARAM_RE.search(name_lower)
    if m:
        return float(m.group(1))

    # Alternate pattern
    m = PARAM_RE_ALT.search(name_lower)
    if m:
        return float(m.group(1))

    return np.nan


def _is_edge_compatible(model_id, param_b):
    """Determine if model is edge-compatible.

    True if param_b <= EDGE_THRESHOLD_B, or if name suggests small model
    (e.g., 'mini', 'tiny', 'small', 'nano', 'lite').
    """
    if not np.isnan(param_b) and param_b <= EDGE_THRESHOLD_B:
        return True

    name_lower = model_id.lower()
    small_indicators = ["mini", "tiny", "small", "nano", "lite", "mobile",
                        "micro", "phi-1", "phi-2"]
    for ind in small_indicators:
        if ind in name_lower:
            return True

    return False


def _is_moe(model_id):
    """Check if model uses Mixture of Experts architecture."""
    name_lower = model_id.lower()
    return "moe" in name_lower or "mixture" in name_lower or "mixtral" in name_lower


def build_model_dataset(raw_data):
    """Convert raw HuggingFace API data to cleaned DataFrame.

    Columns: model_id, author, created (datetime), downloads, param_b,
    edge_compatible, constrained, post, quarter, moe.
    """
    rows = []
    for author, models in raw_data.items():
        if author not in CONSTRAINED_AUTHORS:
            continue
        for m in models:
            model_id = m.get("modelId", m.get("id", ""))
            created_str = m.get("createdAt", m.get("created_at", ""))
            downloads = m.get("downloads", 0)

            # Parse creation date
            try:
                if created_str:
                    # HF API returns ISO format like "2023-06-20T10:30:00.000Z"
                    created = pd.to_datetime(created_str, utc=True).tz_localize(None)
                else:
                    created = pd.NaT
            except Exception:
                created = pd.NaT

            if pd.isna(created):
                continue

            param_b = _extract_param_b(model_id)
            edge = _is_edge_compatible(model_id, param_b)
            moe = _is_moe(model_id)
            constrained = CONSTRAINED_AUTHORS.get(author, 0)
            post = 1 if created >= TREATMENT_DATE else 0
            quarter = pd.Period(created, freq="Q")

            rows.append({
                "model_id": model_id,
                "author": author,
                "created": created,
                "downloads": downloads,
                "param_b": param_b,
                "edge_compatible": edge,
                "constrained": constrained,
                "post": post,
                "quarter": quarter,
                "moe": moe,
            })

    df = pd.DataFrame(rows)
    log(f"Model dataset: {len(df)} models, "
        f"{df['author'].nunique()} authors, "
        f"{df['constrained'].sum()} from constrained firms")

    # Filter to a reasonable time window: 2020Q1 onward
    df = df[df["created"] >= "2020-01-01"].copy()
    log(f"  After filtering to 2020+: {len(df)} models")

    return df


# ---------------------------------------------------------------------------
# Section 3: Build quarterly panel
# ---------------------------------------------------------------------------

def build_quarterly_panel(model_df):
    """Aggregate to author x quarter panel.

    For each author-quarter: n_models, n_edge, edge_share, total_downloads,
    log_downloads, n_moe, moe_share. Plus treatment variables.
    """
    # Group by author and quarter
    grouped = model_df.groupby(["author", "quarter"])

    panel_rows = []
    for (author, quarter), grp in grouped:
        n_models = len(grp)
        n_edge = grp["edge_compatible"].sum()
        edge_share = n_edge / n_models if n_models > 0 else 0.0
        total_downloads = grp["downloads"].sum()
        log_downloads = np.log1p(total_downloads)
        n_moe = grp["moe"].sum()
        moe_share = n_moe / n_models if n_models > 0 else 0.0
        constrained = grp["constrained"].iloc[0]
        post = grp["post"].iloc[0]

        panel_rows.append({
            "author": author,
            "quarter": quarter,
            "quarter_str": str(quarter),
            "n_models": n_models,
            "n_edge": n_edge,
            "edge_share": edge_share,
            "total_downloads": total_downloads,
            "log_downloads": log_downloads,
            "n_moe": n_moe,
            "moe_share": moe_share,
            "constrained": constrained,
            "post": post,
            "post_x_constrained": post * constrained,
        })

    panel = pd.DataFrame(panel_rows)
    panel = panel.sort_values(["author", "quarter"]).reset_index(drop=True)

    log(f"Quarterly panel: {len(panel)} author-quarter obs, "
        f"{panel['author'].nunique()} authors, "
        f"{panel['quarter'].nunique()} quarters")

    return panel


# ---------------------------------------------------------------------------
# Section 4: Descriptive statistics + DID validity check
# ---------------------------------------------------------------------------

def descriptive_stats(model_df, panel):
    """Print summary stats for constrained vs control groups.

    Returns dict with diagnostic info including whether standard DID
    is feasible (both groups have pre-treatment data).
    """
    log("\n" + "="*70)
    log("DESCRIPTIVE STATISTICS")
    log("="*70)

    diag = {"did_feasible": True, "cn_pre_n": 0, "us_pre_n": 0}

    # Total models by group
    for label, grp in model_df.groupby("constrained"):
        name = "Constrained (Chinese)" if label == 1 else "Control (US/EU)"
        n_pre = len(grp[grp["post"] == 0])
        n_post = len(grp[grp["post"] == 1])
        edge_pre = grp[grp["post"] == 0]["edge_compatible"].mean() * 100 if n_pre > 0 else np.nan
        edge_post = grp[grp["post"] == 1]["edge_compatible"].mean() * 100 if n_post > 0 else np.nan
        moe_pre = grp[grp["post"] == 0]["moe"].mean() * 100 if n_pre > 0 else np.nan
        moe_post = grp[grp["post"] == 1]["moe"].mean() * 100 if n_post > 0 else np.nan

        if label == 1:
            diag["cn_pre_n"] = n_pre
        else:
            diag["us_pre_n"] = n_pre

        log(f"\n  {name}:")
        log(f"    Total models:    {len(grp)}")
        log(f"    Pre-treatment:   {n_pre}")
        log(f"    Post-treatment:  {n_post}")
        if n_pre > 0:
            log(f"    Edge share pre:  {edge_pre:.1f}%")
        else:
            log(f"    Edge share pre:  N/A (no pre-treatment models)")
        if n_post > 0:
            log(f"    Edge share post: {edge_post:.1f}%")
        else:
            log(f"    Edge share post: N/A")
        if n_pre > 0 and n_post > 0:
            log(f"    Edge share diff: {edge_post - edge_pre:+.1f}pp")
        if n_pre > 0:
            log(f"    MoE share pre:   {moe_pre:.1f}%")
        if n_post > 0:
            log(f"    MoE share post:  {moe_post:.1f}%")

    # Per-author breakdown
    log("\n  Per-author model counts:")
    for author in ALL_AUTHORS:
        sub = model_df[model_df["author"] == author]
        if len(sub) == 0:
            continue
        n_pre = len(sub[sub["post"] == 0])
        n_post = len(sub[sub["post"] == 1])
        tag = "CN" if CONSTRAINED_AUTHORS[author] else "US"
        log(f"    [{tag}] {author:20s}  pre={n_pre:4d}  post={n_post:4d}  total={len(sub):5d}")

    # DID feasibility check
    cn_pre_panel = panel[(panel["constrained"] == 1) & (panel["post"] == 0)]
    us_pre_panel = panel[(panel["constrained"] == 0) & (panel["post"] == 0)]
    cn_post_panel = panel[(panel["constrained"] == 1) & (panel["post"] == 1)]

    log("\n" + "-"*70)
    log("DID FEASIBILITY CHECK")
    log("-"*70)

    if len(cn_pre_panel) == 0 or diag["cn_pre_n"] < 5:
        diag["did_feasible"] = False
        log("  WARNING: Constrained group has insufficient pre-treatment data.")
        log(f"    Constrained pre-treatment: {diag['cn_pre_n']} models, "
            f"{len(cn_pre_panel)} author-quarter obs")
        log("    Standard DID requires pre-treatment data for both groups.")
        log("    The interaction term post_x_constrained is nearly collinear")
        log("    with constrained when treated firms only appear post-treatment.")
        log("    -> Falling back to ALTERNATIVE identification strategies.")
    else:
        log("  DID feasibility: OK")
        log(f"    Constrained pre-treatment: {diag['cn_pre_n']} models, "
            f"{len(cn_pre_panel)} author-quarter obs")
        log(f"    Control pre-treatment: {diag['us_pre_n']} models, "
            f"{len(us_pre_panel)} author-quarter obs")

    # Raw DID calculation (if feasible)
    if diag["did_feasible"]:
        log("\n  Raw DID (edge_share):")
        for label, name in [(1, "Constrained"), (0, "Control")]:
            sub = panel[panel["constrained"] == label]
            pre = sub[sub["post"] == 0]["edge_share"].mean()
            post_val = sub[sub["post"] == 1]["edge_share"].mean()
            if not np.isnan(pre):
                log(f"    {name}: pre={pre:.3f}, post={post_val:.3f}, diff={post_val - pre:+.3f}")
            else:
                log(f"    {name}: pre=N/A, post={post_val:.3f}")

        cn_pre_val = panel[(panel["constrained"] == 1) & (panel["post"] == 0)]["edge_share"].mean()
        cn_post_val = panel[(panel["constrained"] == 1) & (panel["post"] == 1)]["edge_share"].mean()
        us_pre_val = panel[(panel["constrained"] == 0) & (panel["post"] == 0)]["edge_share"].mean()
        us_post_val = panel[(panel["constrained"] == 0) & (panel["post"] == 1)]["edge_share"].mean()
        if not np.isnan(cn_pre_val):
            raw_did = (cn_post_val - cn_pre_val) - (us_post_val - us_pre_val)
            log(f"    Raw DID = {raw_did:+.4f}")

    return diag


# ---------------------------------------------------------------------------
# Section 5: DID regressions (standard)
# ---------------------------------------------------------------------------

def run_did_regressions(panel, feasible=True):
    """Run 6 DID specifications using OLS with HC1 standard errors.

    If feasible=False, prints a warning and returns empty dict (the
    alternative analysis in run_alternative_analysis handles this case).

    Returns dict of {spec_name: {delta, delta_se, delta_p, n, r2}}.
    """
    if not HAS_STATS:
        log("WARNING: statsmodels not available. Skipping regressions.")
        return {}

    results = {}

    log("\n" + "="*70)
    log("DID REGRESSION RESULTS")
    log("="*70)

    if not feasible:
        log("  SKIPPED: Standard DID not feasible (insufficient pre-treatment data")
        log("  for constrained group). See ALTERNATIVE ANALYSIS below.")
        return results

    # Build firm dummies
    authors = sorted(panel["author"].unique())
    ref_author = authors[0]  # Reference category
    for a in authors:
        if a != ref_author:
            panel[f"firm_{a}"] = (panel["author"] == a).astype(int)
    firm_cols = [c for c in panel.columns if c.startswith("firm_")]

    # Build quarter dummies
    quarters = sorted(panel["quarter"].unique())
    ref_quarter = quarters[0]
    for q in quarters:
        if q != ref_quarter:
            panel[f"qtr_{q}"] = (panel["quarter"] == q).astype(int)
    qtr_cols = [c for c in panel.columns if c.startswith("qtr_")]

    specs = {
        "(1) Baseline": {
            "y": "edge_share",
            "x": ["post", "constrained", "post_x_constrained"],
            "did_var": "post_x_constrained",
        },
        "(2) Firm FE": {
            "y": "edge_share",
            "x": ["post", "post_x_constrained"] + firm_cols,
            "did_var": "post_x_constrained",
        },
        "(3) Two-way FE": {
            "y": "edge_share",
            "x": ["post_x_constrained"] + firm_cols + qtr_cols,
            "did_var": "post_x_constrained",
        },
        "(4) Count (n_edge)": {
            "y": "n_edge",
            "x": ["post", "constrained", "post_x_constrained"],
            "did_var": "post_x_constrained",
        },
        "(5) Log downloads": {
            "y": "log_downloads",
            "x": ["post", "constrained", "post_x_constrained"],
            "did_var": "post_x_constrained",
        },
        "(6) MoE share": {
            "y": "moe_share",
            "x": ["post", "constrained", "post_x_constrained"],
            "did_var": "post_x_constrained",
        },
    }

    for name, spec in specs.items():
        y = panel[spec["y"]].values
        X = add_constant(panel[spec["x"]].values.astype(float))
        x_names = ["const"] + spec["x"]

        try:
            model = OLS(y, X, missing="drop").fit(cov_type="HC1")
            did_idx = x_names.index(spec["did_var"])
            delta = model.params[did_idx]
            delta_se = model.bse[did_idx]
            delta_p = model.pvalues[did_idx]
            n = int(model.nobs)
            r2 = model.rsquared

            results[name] = {
                "delta": delta,
                "delta_se": delta_se,
                "delta_p": delta_p,
                "n": n,
                "r2": r2,
            }

            stars = ""
            if delta_p < 0.01:
                stars = "***"
            elif delta_p < 0.05:
                stars = "**"
            elif delta_p < 0.10:
                stars = "*"

            log(f"\n  {name}:")
            log(f"    Y = {spec['y']}")
            log(f"    delta (Post x Constrained) = {delta:+.4f} (SE={delta_se:.4f}){stars}")
            log(f"    p-value = {delta_p:.4f}")
            log(f"    N = {n}, R^2 = {r2:.4f}")

        except Exception as e:
            log(f"\n  {name}: FAILED -- {e}")
            results[name] = {"delta": np.nan, "delta_se": np.nan,
                             "delta_p": np.nan, "n": 0, "r2": np.nan}

    return results


# ---------------------------------------------------------------------------
# Section 5b: Alternative analysis (when standard DID is not feasible)
# ---------------------------------------------------------------------------

def run_alternative_analysis(model_df, panel):
    """Alternative identification strategies when standard DID is infeasible.

    Three approaches:
    (A) Post-treatment cross-sectional comparison: are constrained firms'
        edge shares different from controls, controlling for time?
    (B) Post-treatment time-trend interaction: do constrained firms show
        steeper time trends in edge share? (tests whether constraints
        accelerate the shift toward efficiency)
    (C) Model-level logistic/OLS: probability that a given post-treatment
        model is edge-compatible, as function of constrained status +
        quarter-since-entry.

    Returns dict of results.
    """
    if not HAS_STATS:
        log("WARNING: statsmodels not available. Skipping alternative analysis.")
        return {}

    results = {}

    log("\n" + "="*70)
    log("ALTERNATIVE ANALYSIS (post-treatment identification)")
    log("="*70)
    log("  Since most constrained (Chinese) firms only began publishing on")
    log("  HuggingFace after the Oct 2022 export controls, standard DID is")
    log("  not well-identified. We use three alternative approaches:")

    # -----------------------------------------------------------------------
    # (A) Post-treatment cross-sectional: edge_share ~ constrained + quarter_FE
    # -----------------------------------------------------------------------
    log("\n  (A) Post-Treatment Cross-Sectional Comparison")
    log("  " + "-"*50)

    post_panel = panel[panel["post"] == 1].copy()

    if len(post_panel) < 5:
        log("    Insufficient post-treatment data.")
        return results

    # Quarter FE
    qtrs = sorted(post_panel["quarter"].unique())
    ref_q = qtrs[0]
    qfe_cols = []
    for q in qtrs:
        if q != ref_q:
            col = f"pqfe_{q}"
            post_panel[col] = (post_panel["quarter"] == q).astype(int)
            qfe_cols.append(col)

    # Spec A1: edge_share ~ constrained + quarter_FE
    x_cols_a1 = ["constrained"] + qfe_cols
    y = post_panel["edge_share"].values
    X = add_constant(post_panel[x_cols_a1].values.astype(float))
    x_names = ["const"] + x_cols_a1

    try:
        model = OLS(y, X, missing="drop").fit(cov_type="HC1")
        idx = x_names.index("constrained")
        beta = model.params[idx]
        se = model.bse[idx]
        p = model.pvalues[idx]
        n = int(model.nobs)
        r2 = model.rsquared

        stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        log(f"    Spec A1: edge_share ~ constrained + quarter_FE")
        log(f"      beta(constrained) = {beta:+.4f} (SE={se:.4f}){stars}")
        log(f"      p = {p:.4f}, N = {n}, R^2 = {r2:.4f}")
        log(f"      Interpretation: Constrained firms have {beta:+.1%} {'higher' if beta > 0 else 'lower'}")
        log(f"        edge share than controls in the same quarter (post-treatment)")

        results["(A1) Cross-section"] = {
            "delta": beta, "delta_se": se, "delta_p": p, "n": n, "r2": r2
        }
    except Exception as e:
        log(f"    Spec A1 FAILED: {e}")

    # Spec A2: Same but with firm FE (uses within-firm variation over time)
    authors = sorted(post_panel["author"].unique())
    ref_a = authors[0]
    ffe_cols = []
    for a in authors:
        if a != ref_a:
            col = f"pffe_{a}"
            post_panel[col] = (post_panel["author"] == a).astype(int)
            ffe_cols.append(col)

    # -----------------------------------------------------------------------
    # (B) Post-treatment time trend: edge_share ~ constrained * time_idx
    # -----------------------------------------------------------------------
    log("\n  (B) Post-Treatment Time-Trend Interaction")
    log("  " + "-"*50)

    # Create time index (quarters since treatment)
    treatment_q = pd.Period("2022Q4", freq="Q")
    post_panel = post_panel.copy()
    post_panel["time_idx"] = post_panel["quarter"].apply(
        lambda q: (q.year - treatment_q.year) * 4 + (q.quarter - treatment_q.quarter)
    )
    post_panel["constrained_x_time"] = (
        post_panel["constrained"] * post_panel["time_idx"]
    )

    # Spec B1: edge_share ~ constrained + time_idx + constrained*time_idx
    x_cols_b1 = ["constrained", "time_idx", "constrained_x_time"]
    y = post_panel["edge_share"].values
    X = add_constant(post_panel[x_cols_b1].values.astype(float))
    x_names = ["const"] + x_cols_b1

    try:
        model = OLS(y, X, missing="drop").fit(cov_type="HC1")
        # Report all coefficients
        log(f"    Spec B1: edge_share ~ constrained + time_idx + constrained*time_idx")
        for i, nm in enumerate(x_names):
            b = model.params[i]
            s = model.bse[i]
            pv = model.pvalues[i]
            st = "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.10 else ""))
            log(f"      {nm:25s}  {b:+.4f} (SE={s:.4f})  p={pv:.4f}  {st}")

        # Key coefficient: constrained_x_time
        idx = x_names.index("constrained_x_time")
        beta_int = model.params[idx]
        se_int = model.bse[idx]
        p_int = model.pvalues[idx]

        log(f"    N = {int(model.nobs)}, R^2 = {model.rsquared:.4f}")
        if beta_int > 0:
            log(f"    Interpretation: Constrained firms' edge share INCREASES")
            log(f"      {beta_int:+.4f} faster per quarter than controls")
        else:
            log(f"    Interpretation: Constrained firms' edge share DECREASES")
            log(f"      {abs(beta_int):.4f} faster per quarter than controls")

        results["(B1) Time trend interaction"] = {
            "delta": beta_int, "delta_se": se_int, "delta_p": p_int,
            "n": int(model.nobs), "r2": model.rsquared
        }
    except Exception as e:
        log(f"    Spec B1 FAILED: {e}")

    # Spec B2: Add firm FE
    x_cols_b2 = ["time_idx", "constrained_x_time"] + ffe_cols
    y = post_panel["edge_share"].values
    X = add_constant(post_panel[x_cols_b2].values.astype(float))
    x_names = ["const"] + x_cols_b2

    try:
        model = OLS(y, X, missing="drop").fit(cov_type="HC1")
        idx = x_names.index("constrained_x_time")
        beta_int = model.params[idx]
        se_int = model.bse[idx]
        p_int = model.pvalues[idx]
        st = "***" if p_int < 0.01 else ("**" if p_int < 0.05 else ("*" if p_int < 0.10 else ""))

        log(f"    Spec B2: edge_share ~ time_idx + constrained*time_idx + firm_FE")
        log(f"      beta(constrained*time) = {beta_int:+.4f} (SE={se_int:.4f}){st}")
        log(f"      p = {p_int:.4f}, N = {int(model.nobs)}, R^2 = {model.rsquared:.4f}")

        results["(B2) Time trend + firm FE"] = {
            "delta": beta_int, "delta_se": se_int, "delta_p": p_int,
            "n": int(model.nobs), "r2": model.rsquared
        }
    except Exception as e:
        log(f"    Spec B2 FAILED: {e}")

    # -----------------------------------------------------------------------
    # (C) Model-level: P(edge) ~ constrained + quarter_since_entry + controls
    # -----------------------------------------------------------------------
    log("\n  (C) Model-Level Edge-Compatible Probability")
    log("  " + "-"*50)

    post_models = model_df[model_df["post"] == 1].copy()
    if len(post_models) < 10:
        log("    Insufficient post-treatment model data.")
        return results

    # Quarters since first model release by this author
    first_release = post_models.groupby("author")["created"].min()
    post_models = post_models.copy()
    post_models["author_start"] = post_models["author"].map(first_release)
    post_models["months_since_entry"] = (
        (post_models["created"] - post_models["author_start"]).dt.days / 30.44
    )

    # Quarter dummies
    post_models["quarter_str"] = post_models["quarter"].astype(str)
    qtrs_m = sorted(post_models["quarter_str"].unique())
    ref_qm = qtrs_m[0]
    qm_cols = []
    for q in qtrs_m:
        if q != ref_qm:
            col = f"mqfe_{q}"
            post_models[col] = (post_models["quarter_str"] == q).astype(int)
            qm_cols.append(col)

    # Spec C1: edge_compatible ~ constrained + months_since_entry + quarter_FE
    x_cols_c1 = ["constrained", "months_since_entry"] + qm_cols
    y = post_models["edge_compatible"].astype(float).values
    X = add_constant(post_models[x_cols_c1].values.astype(float))
    x_names = ["const"] + x_cols_c1

    try:
        model = OLS(y, X, missing="drop").fit(cov_type="HC1")
        idx_c = x_names.index("constrained")
        idx_m = x_names.index("months_since_entry")
        beta_c = model.params[idx_c]
        se_c = model.bse[idx_c]
        p_c = model.pvalues[idx_c]
        beta_m = model.params[idx_m]

        st = "***" if p_c < 0.01 else ("**" if p_c < 0.05 else ("*" if p_c < 0.10 else ""))
        log(f"    Spec C1 (LPM): P(edge) ~ constrained + months_since_entry + quarter_FE")
        log(f"      beta(constrained)       = {beta_c:+.4f} (SE={se_c:.4f}){st}  p={p_c:.4f}")
        log(f"      beta(months_since_entry) = {beta_m:+.4f}")
        log(f"      N = {int(model.nobs)}, R^2 = {model.rsquared:.4f}")
        log(f"      Interpretation: Being a constrained firm {'increases' if beta_c > 0 else 'decreases'}")
        log(f"        the probability of releasing an edge-compatible model by {abs(beta_c):.1%}")

        results["(C1) Model-level LPM"] = {
            "delta": beta_c, "delta_se": se_c, "delta_p": p_c,
            "n": int(model.nobs), "r2": model.rsquared
        }
    except Exception as e:
        log(f"    Spec C1 FAILED: {e}")

    # Spec C2: n_edge count at panel level with constrained + time + interaction
    x_cols_c2 = ["constrained", "time_idx", "constrained_x_time"]
    y = post_panel["n_edge"].values
    X = add_constant(post_panel[x_cols_c2].values.astype(float))
    x_names = ["const"] + x_cols_c2

    try:
        model = OLS(y, X, missing="drop").fit(cov_type="HC1")
        idx = x_names.index("constrained_x_time")
        beta_int = model.params[idx]
        se_int = model.bse[idx]
        p_int = model.pvalues[idx]
        st = "***" if p_int < 0.01 else ("**" if p_int < 0.05 else ("*" if p_int < 0.10 else ""))

        log(f"    Spec C2: n_edge ~ constrained + time_idx + constrained*time_idx")
        log(f"      beta(constrained*time) = {beta_int:+.4f} (SE={se_int:.4f}){st}")
        log(f"      p = {p_int:.4f}, N = {int(model.nobs)}, R^2 = {model.rsquared:.4f}")

        results["(C2) Edge count trend"] = {
            "delta": beta_int, "delta_se": se_int, "delta_p": p_int,
            "n": int(model.nobs), "r2": model.rsquared
        }
    except Exception as e:
        log(f"    Spec C2 FAILED: {e}")

    # Spec C3: Downloads
    x_cols_c3 = ["constrained", "time_idx", "constrained_x_time"]
    y = post_panel["log_downloads"].values
    X = add_constant(post_panel[x_cols_c3].values.astype(float))
    x_names = ["const"] + x_cols_c3

    try:
        model = OLS(y, X, missing="drop").fit(cov_type="HC1")
        idx = x_names.index("constrained_x_time")
        beta_int = model.params[idx]
        se_int = model.bse[idx]
        p_int = model.pvalues[idx]
        st = "***" if p_int < 0.01 else ("**" if p_int < 0.05 else ("*" if p_int < 0.10 else ""))

        log(f"    Spec C3: log_downloads ~ constrained + time_idx + constrained*time_idx")
        log(f"      beta(constrained*time) = {beta_int:+.4f} (SE={se_int:.4f}){st}")
        log(f"      p = {p_int:.4f}, N = {int(model.nobs)}, R^2 = {model.rsquared:.4f}")

        results["(C3) Download trend"] = {
            "delta": beta_int, "delta_se": se_int, "delta_p": p_int,
            "n": int(model.nobs), "r2": model.rsquared
        }
    except Exception as e:
        log(f"    Spec C3 FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Section 6: Parallel trends / event study
# ---------------------------------------------------------------------------

def parallel_trends_test(panel, feasible=True):
    """Event-study specification with quarter-specific interaction dummies.

    Reference quarter: 2022Q3 (last pre-treatment quarter).
    Returns DataFrame with quarter, coeff, se, ci_lo, ci_hi, p.

    If feasible=False, runs post-treatment-only event study instead
    (quarter-specific constrained dummies, no pre-treatment reference).
    """
    if not HAS_STATS:
        log("WARNING: statsmodels not available. Skipping event study.")
        return pd.DataFrame()

    log("\n" + "="*70)
    log("EVENT STUDY / PARALLEL TRENDS TEST")
    log("="*70)

    if not feasible:
        return _post_treatment_event_study(panel)

    # Standard event study with pre-treatment reference
    ref_q = pd.Period("2022Q3", freq="Q")
    quarters_sorted = sorted(panel["quarter"].unique())

    # Create interaction dummies: constrained x quarter (excluding ref)
    interact_cols = []
    for q in quarters_sorted:
        if q == ref_q:
            continue
        col = f"CxQ_{q}"
        panel[col] = ((panel["constrained"] == 1) &
                       (panel["quarter"] == q)).astype(int)
        interact_cols.append(col)

    # Firm FE
    authors = sorted(panel["author"].unique())
    ref_author = authors[0]
    fe_firm_cols = []
    for a in authors:
        if a != ref_author:
            col = f"fe_{a}"
            panel[col] = (panel["author"] == a).astype(int)
            fe_firm_cols.append(col)

    # Quarter FE
    fe_qtr_cols = []
    for q in quarters_sorted:
        if q != ref_q:
            col = f"feq_{q}"
            panel[col] = (panel["quarter"] == q).astype(int)
            fe_qtr_cols.append(col)

    # Regression: edge_share ~ firm_FE + quarter_FE + interact_dummies
    x_cols = fe_firm_cols + fe_qtr_cols + interact_cols
    y = panel["edge_share"].values
    X = add_constant(panel[x_cols].values.astype(float))
    x_names = ["const"] + x_cols

    try:
        model = OLS(y, X, missing="drop").fit(cov_type="HC1")
    except Exception as e:
        log(f"  Event study regression failed: {e}")
        return pd.DataFrame()

    # Extract interaction coefficients
    es_rows = []
    for col in interact_cols:
        idx = x_names.index(col)
        q_str = col.replace("CxQ_", "")
        coeff = model.params[idx]
        se = model.bse[idx]
        p = model.pvalues[idx]
        ci_lo = coeff - 1.96 * se
        ci_hi = coeff + 1.96 * se
        es_rows.append({
            "quarter": q_str,
            "coeff": coeff,
            "se": se,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p": p,
        })

    # Add reference quarter with zero
    es_rows.append({
        "quarter": str(ref_q),
        "coeff": 0.0,
        "se": 0.0,
        "ci_lo": 0.0,
        "ci_hi": 0.0,
        "p": 1.0,
    })

    es_df = pd.DataFrame(es_rows)
    es_df = es_df.sort_values("quarter").reset_index(drop=True)

    # Report
    log("\n  Event-study coefficients (ref = 2022Q3):")
    log(f"  {'Quarter':>10s}  {'Coeff':>8s}  {'SE':>8s}  {'p':>8s}  {'Sig':>4s}")
    log("  " + "-" * 50)

    pre_sig_count = 0
    for _, row in es_df.iterrows():
        q_period = pd.Period(row["quarter"], freq="Q")
        pre_post = "PRE " if q_period < pd.Period("2022Q4", freq="Q") else "POST"
        stars = ""
        if row["p"] < 0.01:
            stars = "***"
        elif row["p"] < 0.05:
            stars = "**"
        elif row["p"] < 0.10:
            stars = "*"

        if pre_post == "PRE " and row["p"] < 0.10 and row["quarter"] != str(ref_q):
            pre_sig_count += 1

        log(f"  {row['quarter']:>10s}  {row['coeff']:+8.4f}  {row['se']:8.4f}  "
            f"{row['p']:8.4f}  {stars:>4s}  [{pre_post}]")

    if pre_sig_count == 0:
        log("\n  PARALLEL TRENDS: SUPPORTED (no significant pre-treatment coefficients)")
    else:
        log(f"\n  PARALLEL TRENDS: WARNING -- {pre_sig_count} significant pre-treatment "
            f"coefficient(s) at 10% level")

    return es_df


def _post_treatment_event_study(panel):
    """Post-treatment-only event study when standard DID is infeasible.

    Estimates quarter-specific constrained coefficients relative to the
    first post-treatment quarter (2022Q4).
    """
    log("  NOTE: Standard event study infeasible (no pre-treatment constrained data).")
    log("  Running POST-TREATMENT ONLY event study instead.")
    log("  Reference quarter: first quarter with both groups present.")

    post_panel = panel[panel["post"] == 1].copy()
    quarters_sorted = sorted(post_panel["quarter"].unique())

    # Find first quarter with both groups
    ref_q = None
    for q in quarters_sorted:
        cn = post_panel[(post_panel["quarter"] == q) & (post_panel["constrained"] == 1)]
        us = post_panel[(post_panel["quarter"] == q) & (post_panel["constrained"] == 0)]
        if len(cn) > 0 and len(us) > 0:
            ref_q = q
            break

    if ref_q is None:
        log("  ERROR: No quarter has both constrained and control observations.")
        return pd.DataFrame()

    log(f"  Reference quarter: {ref_q}")

    # Interaction dummies (constrained x quarter, excluding ref)
    interact_cols = []
    for q in quarters_sorted:
        if q == ref_q:
            continue
        col = f"CxQ_{q}"
        post_panel[col] = ((post_panel["constrained"] == 1) &
                            (post_panel["quarter"] == q)).astype(int)
        interact_cols.append(col)

    # Firm FE
    authors = sorted(post_panel["author"].unique())
    ref_author = authors[0]
    fe_cols = []
    for a in authors:
        if a != ref_author:
            col = f"pfe_{a}"
            post_panel[col] = (post_panel["author"] == a).astype(int)
            fe_cols.append(col)

    # Quarter FE
    qfe_cols = []
    for q in quarters_sorted:
        if q != ref_q:
            col = f"pqfe2_{q}"
            post_panel[col] = (post_panel["quarter"] == q).astype(int)
            qfe_cols.append(col)

    x_cols = fe_cols + qfe_cols + interact_cols
    y = post_panel["edge_share"].values
    X = add_constant(post_panel[x_cols].values.astype(float))
    x_names = ["const"] + x_cols

    try:
        model = OLS(y, X, missing="drop").fit(cov_type="HC1")
    except Exception as e:
        log(f"  Post-treatment event study FAILED: {e}")
        return pd.DataFrame()

    es_rows = []
    for col in interact_cols:
        idx = x_names.index(col)
        q_str = col.replace("CxQ_", "")
        coeff = model.params[idx]
        se = model.bse[idx]
        p = model.pvalues[idx]
        es_rows.append({
            "quarter": q_str,
            "coeff": coeff,
            "se": se,
            "ci_lo": coeff - 1.96 * se,
            "ci_hi": coeff + 1.96 * se,
            "p": p,
        })

    # Reference quarter
    es_rows.append({
        "quarter": str(ref_q),
        "coeff": 0.0, "se": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p": 1.0,
    })

    es_df = pd.DataFrame(es_rows).sort_values("quarter").reset_index(drop=True)

    log(f"\n  Post-treatment event-study coefficients (ref = {ref_q}):")
    log(f"  {'Quarter':>10s}  {'Coeff':>8s}  {'SE':>8s}  {'p':>8s}  {'Sig':>4s}")
    log("  " + "-" * 50)
    for _, row in es_df.iterrows():
        stars = ""
        if row["p"] < 0.01:
            stars = "***"
        elif row["p"] < 0.05:
            stars = "**"
        elif row["p"] < 0.10:
            stars = "*"
        log(f"  {row['quarter']:>10s}  {row['coeff']:+8.4f}  {row['se']:8.4f}  "
            f"{row['p']:8.4f}  {stars:>4s}")

    return es_df


# ---------------------------------------------------------------------------
# Section 7: Plot edge share trends
# ---------------------------------------------------------------------------

def plot_edge_share_trends(model_df, panel):
    """4-panel figure: (a) quarterly edge_share by group, (b) cumulative model
    releases, (c) quarterly edge count bars, (d) parameter size distribution.

    Saves to FIG_DIR/figure_export_control_did.png.
    """
    if not HAS_MPL:
        log("WARNING: matplotlib not available. Skipping plots.")
        return

    log("\nGenerating 4-panel DID figure...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    treatment_q = pd.Period("2022Q4", freq="Q")

    # Color scheme
    cn_color = "#e74c3c"  # Red for constrained
    us_color = "#3498db"  # Blue for control

    # (a) Quarterly edge_share by group
    ax = axes[0, 0]
    for label, color, name in [(1, cn_color, "Constrained (CN)"),
                                (0, us_color, "Control (US/EU)")]:
        sub = panel[panel["constrained"] == label].copy()
        sub = sub.sort_values("quarter")
        qtr_agg = sub.groupby("quarter")["edge_share"].mean()
        x_vals = [q.to_timestamp() for q in qtr_agg.index]
        ax.plot(x_vals, qtr_agg.values, "o-", color=color, label=name,
                markersize=4, linewidth=1.5)

    ax.axvline(TREATMENT_DATE, color="gray", linestyle="--", alpha=0.7,
               label="BIS Export Controls")
    ax.set_ylabel("Edge-Compatible Share")
    ax.set_title("(a) Edge Share by Group")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Cumulative model releases
    ax = axes[0, 1]
    for label, color, name in [(1, cn_color, "Constrained (CN)"),
                                (0, us_color, "Control (US/EU)")]:
        sub = model_df[model_df["constrained"] == label].copy()
        sub = sub.sort_values("created")
        ax.plot(sub["created"], range(1, len(sub) + 1), color=color,
                label=name, linewidth=1.5)

    ax.axvline(TREATMENT_DATE, color="gray", linestyle="--", alpha=0.7)
    ax.set_ylabel("Cumulative Models")
    ax.set_title("(b) Cumulative Model Releases")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Quarterly edge count bars
    ax = axes[1, 0]
    quarters_sorted = sorted(panel["quarter"].unique())
    x_pos = np.arange(len(quarters_sorted))
    width = 0.35

    cn_edge = []
    us_edge = []
    for q in quarters_sorted:
        cn_sub = panel[(panel["constrained"] == 1) & (panel["quarter"] == q)]
        us_sub = panel[(panel["constrained"] == 0) & (panel["quarter"] == q)]
        cn_edge.append(cn_sub["n_edge"].sum())
        us_edge.append(us_sub["n_edge"].sum())

    ax.bar(x_pos - width / 2, cn_edge, width, color=cn_color,
           alpha=0.8, label="Constrained (CN)")
    ax.bar(x_pos + width / 2, us_edge, width, color=us_color,
           alpha=0.8, label="Control (US/EU)")

    # Treatment line
    treat_idx = None
    for i, q in enumerate(quarters_sorted):
        if q >= treatment_q and treat_idx is None:
            treat_idx = i
    if treat_idx is not None:
        ax.axvline(treat_idx - 0.5, color="gray", linestyle="--", alpha=0.7)

    # X labels (show every 2nd quarter to avoid clutter)
    tick_labels = [str(q) if i % 2 == 0 else "" for i, q in enumerate(quarters_sorted)]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)
    ax.set_ylabel("Edge-Compatible Models (count)")
    ax.set_title("(c) Quarterly Edge Model Releases")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Post-treatment parameter size distribution
    ax = axes[1, 1]
    post_df = model_df[(model_df["post"] == 1) & model_df["param_b"].notna()].copy()

    if len(post_df) > 0:
        bins = [0, 1, 3, 7, 13, 34, 70, 200, 1000]
        bin_labels = ["<1B", "1-3B", "3-7B", "7-13B", "13-34B", "34-70B",
                      "70-200B", "200B+"]

        for label, color, name in [(1, cn_color, "Constrained (CN)"),
                                    (0, us_color, "Control (US/EU)")]:
            sub = post_df[post_df["constrained"] == label]
            if len(sub) > 0:
                counts, _ = np.histogram(sub["param_b"], bins=bins)
                total_cnt = counts.sum()
                if total_cnt > 0:
                    pcts = counts / total_cnt * 100
                else:
                    pcts = np.zeros_like(counts, dtype=float)
                ax.bar(np.arange(len(bin_labels)) + (0.2 if label == 1 else -0.2),
                       pcts, 0.35, color=color, alpha=0.8, label=name)

        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=45, fontsize=8)
        ax.set_ylabel("% of Post-Treatment Models")
        ax.set_title("(d) Parameter Size Distribution (Post-Treatment)")
        ax.legend(fontsize=8)
        ax.axvline(2.5, color="gray", linestyle=":", alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No parameter data\navailable", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_title("(d) Parameter Size Distribution (Post-Treatment)")

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "figure_export_control_did.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Section 8: Plot event study
# ---------------------------------------------------------------------------

def plot_event_study(es_df, feasible=True):
    """Event-study coefficient plot with 95% CI error bars.

    Saves to FIG_DIR/figure_export_control_event_study.png.
    """
    if not HAS_MPL or es_df.empty:
        log("WARNING: Cannot plot event study (no matplotlib or empty data).")
        return

    log("Generating event-study coefficient plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    es_df = es_df.sort_values("quarter").reset_index(drop=True)
    x = np.arange(len(es_df))
    quarters = es_df["quarter"].values
    coeffs = es_df["coeff"].values
    ci_lo = es_df["ci_lo"].values
    ci_hi = es_df["ci_hi"].values

    # Error bars
    yerr_lo = coeffs - ci_lo
    yerr_hi = ci_hi - coeffs
    yerr = np.array([yerr_lo, yerr_hi])

    # Color: blue pre-treatment, red post-treatment
    treatment_q_str = "2022Q4"
    if feasible:
        colors = ["#3498db" if q < treatment_q_str else "#e74c3c" for q in quarters]
    else:
        colors = ["#e74c3c" for _ in quarters]

    ax.errorbar(x, coeffs, yerr=yerr, fmt="o", markersize=6,
                capsize=3, capthick=1.5, elinewidth=1.5,
                color="black", ecolor="gray", zorder=5)

    # Color the markers
    for i, (xi, ci, col) in enumerate(zip(x, coeffs, colors)):
        ax.plot(xi, ci, "o", color=col, markersize=8, zorder=6)

    # Horizontal zero line
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

    # Treatment vertical line (only for standard event study)
    if feasible:
        treat_idx = None
        for i, q in enumerate(quarters):
            if q == "2022Q3":
                treat_idx = i + 0.5
                break
        if treat_idx is not None:
            ax.axvline(treat_idx, color="gray", linestyle="--", alpha=0.7,
                       label="BIS Export Controls (Oct 2022)")

    ax.set_xticks(x)
    ax.set_xticklabels(quarters, rotation=45, fontsize=8)
    ax.set_ylabel("Coefficient (Constrained x Quarter)")
    ax.set_xlabel("Quarter")

    if feasible:
        ax.set_title("Event Study: Edge-Compatible Share DID Coefficients\n"
                     "(Reference: 2022Q3, 95% CI)")
    else:
        ref_q = es_df[es_df["coeff"] == 0.0]["quarter"].iloc[0] if len(es_df[es_df["coeff"] == 0.0]) > 0 else "?"
        ax.set_title(f"Post-Treatment Event Study: Constrained vs Control\n"
                     f"(Reference: {ref_q}, 95% CI)")

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "figure_export_control_event_study.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Section 9: Interpretation
# ---------------------------------------------------------------------------

def interpret_results(did_results, alt_results, model_df, feasible):
    """Interpret results in context of Paper 1 theory."""
    log("\n" + "="*70)
    log("INTERPRETATION: PAPER 1 (Endogenous Decentralization), Section 5.1")
    log("="*70)

    # Edge shares by group (model-level, post-treatment only)
    cn_post = model_df[(model_df["constrained"] == 1) & (model_df["post"] == 1)]
    us_post = model_df[(model_df["constrained"] == 0) & (model_df["post"] == 1)]

    cn_edge_post = cn_post["edge_compatible"].mean() * 100 if len(cn_post) > 0 else np.nan
    us_edge_post = us_post["edge_compatible"].mean() * 100 if len(us_post) > 0 else np.nan

    log(f"\n  Post-treatment edge-compatible model shares:")
    log(f"    Constrained (CN): {cn_edge_post:.1f}% ({len(cn_post)} models)")
    log(f"    Control (US/EU):  {us_edge_post:.1f}% ({len(us_post)} models)")
    if not np.isnan(cn_edge_post) and not np.isnan(us_edge_post):
        log(f"    Difference:       {cn_edge_post - us_edge_post:+.1f}pp")

    # Parameter size comparison
    cn_param = cn_post["param_b"].dropna()
    us_param = us_post["param_b"].dropna()
    if len(cn_param) > 0 and len(us_param) > 0:
        log(f"\n  Post-treatment median parameter count:")
        log(f"    Constrained (CN): {cn_param.median():.1f}B (n={len(cn_param)})")
        log(f"    Control (US/EU):  {us_param.median():.1f}B (n={len(us_param)})")

    # Determine which results to interpret
    if feasible and did_results:
        results = did_results
        baseline_key = "(1) Baseline"
        log("\n  Using STANDARD DID results:")
    elif alt_results:
        results = alt_results
        # Use time-trend interaction as primary specification
        baseline_key = "(B1) Time trend interaction"
        if baseline_key not in results:
            baseline_key = "(A1) Cross-section"
        log("\n  Using ALTERNATIVE ANALYSIS results (standard DID infeasible):")
    else:
        log("  No regression results available.")
        return

    # Primary result
    primary = results.get(baseline_key, {})
    delta = primary.get("delta", np.nan)
    delta_p = primary.get("delta_p", np.nan)

    if np.isnan(delta):
        log(f"  Primary specification ({baseline_key}) not available.")
        return

    log(f"\n  Primary coefficient: {delta:+.4f} (p = {delta_p:.4f})")

    # Interpretation depends on which analysis path we took
    if feasible:
        # Standard DID interpretation
        if delta > 0 and delta_p < 0.10:
            verdict = "CONSISTENT"
        elif delta > 0 and delta_p < 0.20:
            verdict = "WEAKLY CONSISTENT"
        elif delta <= 0:
            verdict = "INCONSISTENT"
        else:
            verdict = "AMBIGUOUS"
    else:
        # Alternative analysis interpretation
        # For time-trend: positive = constrained firms increase edge share faster
        # For cross-section: negative = constrained have lower edge share
        if "Time trend" in baseline_key:
            if delta > 0 and delta_p < 0.10:
                verdict = "CONSISTENT"
                log("  (Constrained firms' edge share increases faster over time)")
            elif delta > 0 and delta_p < 0.20:
                verdict = "WEAKLY CONSISTENT"
            elif delta > 0:
                verdict = "DIRECTIONALLY CONSISTENT (not significant)"
            else:
                verdict = "INCONSISTENT"
        elif "Cross-section" in baseline_key:
            # Here, constrained level difference: positive = higher edge share
            if delta > 0 and delta_p < 0.10:
                verdict = "CONSISTENT"
            elif delta < 0 and delta_p < 0.10:
                verdict = "INCONSISTENT"
            else:
                verdict = "AMBIGUOUS"
        else:
            verdict = "AMBIGUOUS"

    log(f"\n  VERDICT: {verdict}")

    # Explanation based on verdict
    if verdict == "CONSISTENT":
        log("  Export controls are associated with a shift toward edge-compatible")
        log("  models by constrained firms. This supports the learning externality")
        log("  mechanism: hardware constraints redirect R&D toward parameter-efficient")
        log("  architectures that run on available (non-frontier) hardware.")
    elif verdict in ("WEAKLY CONSISTENT", "DIRECTIONALLY CONSISTENT (not significant)"):
        log("  The direction of the effect is consistent with the theory, but the")
        log("  evidence is not strong enough to be conclusive at conventional levels.")
        log("  The limited pre-treatment data for Chinese firms on HuggingFace is")
        log("  the primary identification challenge.")
    elif verdict == "INCONSISTENT":
        log("  The evidence does not support the predicted pattern. This may reflect:")
        log("  (1) HuggingFace not capturing the full Chinese AI model landscape,")
        log("  (2) control firms also shifting to edge models for market reasons,")
        log("  (3) the effect operating through channels other than model size.")
    else:
        log("  Results are ambiguous -- cannot clearly support or reject the theory.")

    # All specifications summary
    log("\n  Summary across all specifications:")
    all_results = {}
    if did_results:
        all_results.update(did_results)
    if alt_results:
        all_results.update(alt_results)

    for name, res in all_results.items():
        d = res.get("delta", np.nan)
        p = res.get("delta_p", np.nan)
        stars = ""
        if p < 0.01:
            stars = "***"
        elif p < 0.05:
            stars = "**"
        elif p < 0.10:
            stars = "*"
        log(f"    {name:35s}  coeff={d:+.4f}  p={p:.4f}  {stars}")

    # Caveats
    log("\n  CAVEATS:")
    if not feasible:
        log("    1. Standard DID is infeasible because most Chinese AI labs only")
        log("       began publishing models on HuggingFace AFTER the Oct 2022")
        log("       export controls. This means we lack counterfactual pre-treatment")
        log("       trends for the constrained group.")
        log("    2. The alternative analyses (cross-section, time-trend, model-level)")
        log("       control for confounders but cannot establish causality as cleanly")
        log("       as a well-identified DID.")
    log("    - HuggingFace is not the complete universe of AI models (some")
    log("      Chinese models may be on domestic platforms).")
    log("    - Edge-compatibility classification is approximate (based on")
    log("      parameter count extracted from model names).")
    log("    - Both groups may shift to edge models for independent reasons")
    log("      (e.g., consumer demand, mobile deployment trends).")

    # Connect to thesis
    log("\n  Connection to thesis mechanism:")
    log("    Paper 1 (Prop. 1): N firms overinvest by factor ~N/(N-1), accelerating")
    log("    learning curves. Export controls create exogenous variation in hardware")
    log("    access, separating the overinvestment/learning channel from standard")
    log("    Arrow learning-by-doing. If constrained firms produce more edge-compatible")
    log("    models, this confirms that the LEARNING EXTERNALITY (not just cost")
    log("    reduction) drives the transition to distributed architecture.")


# ---------------------------------------------------------------------------
# Section 10: Main
# ---------------------------------------------------------------------------

def main():
    log("="*70)
    log("EXPORT CONTROL DID TEST")
    log("Paper 1 (Endogenous Decentralization), Section 5.1")
    log(f"Treatment date: {TREATMENT_DATE.strftime('%Y-%m-%d')}")
    log(f"Edge threshold: <= {EDGE_THRESHOLD_B}B parameters")
    log("="*70)

    # Step 1: Fetch data
    raw_data = fetch_all_models(use_cache=True)

    # Step 2: Build model-level dataset
    model_df = build_model_dataset(raw_data)

    if len(model_df) == 0:
        log("ERROR: No models found. Check HuggingFace API or cache.")
        return

    # Save model-level CSV
    model_csv_path = os.path.join(DATA_DIR, "export_control_model_level.csv")
    save_df = model_df.copy()
    save_df["quarter"] = save_df["quarter"].astype(str)
    save_df.to_csv(model_csv_path, index=False)
    log(f"Saved model-level data: {model_csv_path}")

    # Step 3: Build quarterly panel
    panel = build_quarterly_panel(model_df)

    # Save panel CSV
    panel_csv_path = os.path.join(DATA_DIR, "export_control_panel.csv")
    save_panel = panel.copy()
    save_panel["quarter"] = save_panel["quarter"].astype(str)
    save_panel.to_csv(panel_csv_path, index=False)
    log(f"Saved panel data: {panel_csv_path}")

    # Step 4: Descriptive stats + DID feasibility check
    diag = descriptive_stats(model_df, panel)
    feasible = diag["did_feasible"]

    # Step 5: DID regressions (if feasible)
    did_results = run_did_regressions(panel, feasible=feasible)

    # Step 5b: Alternative analysis (always run for robustness)
    alt_results = run_alternative_analysis(model_df, panel)

    # Step 6: Event study / parallel trends
    es_df = parallel_trends_test(panel, feasible=feasible)

    # Step 7: Plot DID trends
    plot_edge_share_trends(model_df, panel)

    # Step 8: Plot event study
    plot_event_study(es_df, feasible=feasible)

    # Step 9: Interpret
    interpret_results(did_results, alt_results, model_df, feasible)

    # Save results
    log_path = os.path.join(DATA_DIR, "export_control_did_results.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(LOG_LINES))
    log(f"\nResults saved to {log_path}")
    print(f"\nDone. Results written to {log_path}")


if __name__ == "__main__":
    main()
