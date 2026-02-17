# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains data collection scripts, analysis code, paper drafts, and research materials for a multi-paper academic project by Connor Doll / Smirl (2026). The project builds a unified mathematical framework showing how concentrated investment in centralized AI infrastructure finances the learning curves that enable distributed alternatives—and how this self-undermining dynamic propagates through a four-level economic hierarchy.

### The Mathematical Spine: CES Generating Function

A single CES (Constant Elasticity of Substitution) aggregate `F_n = (1/J * sum x_nj^rho)^(1/rho)` with curvature parameter `K = (1-rho)(J-1)/J` unifies papers 1-5. The CES free energy `Phi = -sum log F_n` serves as the Hamiltonian of a port-Hamiltonian system. The parameter rho (equivalently sigma = 1/(1-rho)) simultaneously controls:

- **Superadditivity**: complementary heterogeneous agents produce more together than the sum of parts (gap proportional to K)
- **Correlation robustness**: CES nonlinearity extracts idiosyncratic variation that linear aggregates miss (bonus proportional to K^2), preventing model collapse in self-referential training
- **Strategic independence**: balanced allocation is a Nash equilibrium; coalitions cannot profitably redistribute (penalty proportional to K)

These three roles are not separate assumptions—they are three views of the same geometric fact: the curvature of CES isoquants at symmetric equilibrium.

### The Four-Level Hierarchy

The framework operates on four levels with strict timescale separation (epsilon_1 >> epsilon_2 >> ... >> epsilon_4):

| Level | Paper | State Variable | Gain Function phi_n | Timescale |
|-------|-------|---------------|---------------------|-----------|
| 1 (slowest) | Endogenous Decentralization | Hardware/semiconductor cost | Wright's Law learning curve (alpha ~0.23) | Decades |
| 2 | Mesh Equilibrium | Heterogeneous AI agent density | Recruitment/adoption dynamics | Years |
| 3 | Autocatalytic Mesh | Training effectiveness/capability | Autocatalytic training feedback | Months |
| 4 (fastest) | Settlement Feedback | Stablecoin infrastructure | Settlement demand | Days-weeks |

Each level's equilibrium is bounded by the level below it (hierarchical ceiling). The Baumol bottleneck (Paper 3) and Triffin contradiction (Paper 4) are mathematically the same object: slow-manifold constraints at adjacent layers.

### The Feedback Loop (Papers 1-4)

The papers form a cycle:
1. **Endogenous Decentralization**: Concentrated investment finances learning curves -> hardware crossing at cost parity
2. **Mesh Equilibrium**: Above critical mass N*, distributed mesh forms via first-order phase transition (Potts crystallization, not gradual adoption)
3. **Autocatalytic Mesh**: Mesh capability grows endogenously but converges to frontier training rate g_Z (Baumol bottleneck)
4. **Settlement Feedback**: Mesh enters capital markets -> stablecoin demand -> Treasury absorption -> monetary policy degradation -> feeds back to constrain/enable centralized investment

The **master reproduction number R_0** governs each transition: nontrivial equilibrium exists iff rho(K) > 1 (spectral radius of next-generation matrix exceeds 1). Cross-level amplification can activate the entire system even when all individual levels are sub-threshold.

### The Eigenstructure Bridge (Paper 5)

The deepest result connects technology to welfare: `nabla^2 Phi|_slow = W^{-1} * nabla^2 V`, where Phi is the CES free energy (technology), V is the Lyapunov function (welfare loss), and W is the institutional supply-rate matrix (how efficiently adjustment occurs). This yields:

- **Damping cancellation theorem**: increasing local regulation sigma_n speeds convergence but lowers equilibrium output; effects exactly cancel. To accelerate adjustment at level n, reform upstream (reduce sigma_{n-1} or increase gain elasticity beta_n).
- **Crisis duration**: ~8 years at Wright's Law semiconductor rates, via canard bifurcation dynamics. Duration scales as O(1/sqrt(epsilon_drift)).

### Papers

**Working papers (unified by CES framework):**
1. **Endogenous Decentralization** — Continuous-time differential game with exact closed-form Nash MPE. Key result: N firms overinvest 3-4x (Proposition 1), accelerating crossing 79%. Generalized R_0 crossing condition extends pure cost parity to include coordination dynamics. Training-inference bifurcation explains partial decentralization. Natural experiment: US-China export controls distinguish mechanism from standard Arrow learning. Predicts hardware crossing ~2028, self-sustaining distributed adoption ~2030-2032.
2. **The Mesh Equilibrium** — Post-crossing dynamics. Giant component via percolation (R_0^mesh > 1), CES diversity premium from complementary specialization (rho < 1), knowledge diffusion via graph Laplacian. Key result: mesh dominance above N* (Theorem 1), with first-order phase transition (Fortuin-Kasteleyn/Potts, not smooth logistic). Inverse Bose-Einstein condensation at crossing point.
3. **The Autocatalytic Mesh** — Endogenous capability growth via RAF (reflexively autocatalytic and food-generated) sets. Effective training productivity phi_eff = phi_0/(1 - beta_auto * phi_0) can exceed 1 with sufficient automation. Three growth regimes (convergence/exponential/singularity); convergence most likely due to Lotka-Volterra saturation + Baumol bottleneck. CES heterogeneity prevents model collapse (Theorem 2): effective external data fraction alpha_eff > alpha_crit even when actual external data is below threshold.
4. **The Settlement Feedback** — Coupled 4-ODE system (mesh participation phi, stablecoin size S, debt ratio b, financial capitalization eta). Monetary policy degrades in sequence: forward guidance first, then QE, then financial repression. Bistable equilibrium (low-mesh vs high-mesh) with transcritical bifurcation at R_settle > 1. High-mesh equilibrium = synthetic gold standard with continuous market discipline. Kyle's lambda non-monotonic in phi.
5. **Complementary Heterogeneity** (unifying paper) — Proves CES triple role theorem, derives network architecture from CES geometry (port topology theorem: aggregate coupling, directed feed-forward, nearest-neighbor forced by timescale separation). Eigenstructure bridge connects technology Hessian to welfare Hessian. Damping cancellation and upstream reform principle. Moduli space theorem: rho determines qualitative dynamics; timescales, damping, and gain functions are free parameters.

**Empirical thesis:**
6. **Monetary Productivity Gap** (EC118 thesis) — Constructs Fiat Quality Index (FQI: inflation stability, banking access, ATM density, government effectiveness, internet penetration). Key finding: 13:1 remittance cost ratio fiat vs stablecoin in Sub-Saharan Africa. India 2022 tax natural experiment shows -86% domestic volume but 72% displacement offshore (not suppression). Yield Access Gap (YAG) regression: within-country beta = +0.248 (p < 0.001), sign flip from cross-section confirms precautionary savings motive. Provides micro-level evidence for settlement layer demand (Paper 4). 41-country panel, data from World Bank WDI/Findex/RPW, CBECI, Chainalysis.

**Policy paper:**
7. **Fair Inheritance** — Recipient-based inheritance tax replacing estate tax. Treats inheritance as ordinary income; eliminates trust recognition. Creates binary choice: pay tax (concentrated transfer) or disperse widely (zero-tax pathway). Revenue projection $85-135B/year (5x current estate tax). Addresses automation-era wealth concentration. Connected to Papers 1-5 via the distributional consequences of technological transitions.

## Environment Setup

```bash
source .venv/bin/activate
source env.sh  # loads FRED_API_KEY
```

Python 3.12 venv with: `requests`, `pandas`, `numpy`, `openpyxl`, `xlrd`. Optional: `statsmodels`, `scipy`, `matplotlib` (for empirical tests in sections 29-30).

## Running Scripts

```bash
# Main data fetcher — idempotent, skips already-downloaded data, re-run until complete
python fetch_thesis_data.py

# BCL index construction from BRSS Excel files (requires brss_data/ populated)
python build_bcl_indices.py

# RPW data helpers (run after fetch_thesis_data.py if RPW parsing needs fixing)
python fix_rpw.py
python filter_rpw.py
```

## Architecture

### fetch_thesis_data.py (~6000 lines)

The monolithic data pipeline. Contains 40+ functions organized into numbered sections, serving multiple papers:

- **Sections 1-4**: API-fetched data (World Bank WDI, Findex, Remittance Prices Worldwide, CBECI mining map) — **Paper 6 (MPG)**
- **Sections 5-13**: Constructed/compiled MPG thesis data (India volumes, Chainalysis adoption, regulatory panel, yield access gap, sovereign risk, stablecoin reserves, FQI improvement) — **Paper 6 (MPG)**
- **Sections 14-21**: Learning curve and technology data (DRAM/HBM pricing, hyperscaler capex, consumer silicon, stablecoin volumes, tokenized securities, historical adoption curves, learning curve literature) — **Paper 1 (ED)**, feeds alpha estimates into Papers 2-5
- **Sections 22-22b**: Publishability upgrades (DRAM diagnostics, alpha resolution with piecewise/rolling/sensitivity analysis) — **Paper 1 (ED)**
- **Sections 23-28**: Cross-layer heterogeneity data (FRED semiconductor IP, WSTS cross-segment, semiconductor dispersion, BCL bank regulation, IMF financial development, Fraser regulation) — **Paper 5 (CH)**, tests damping cancellation and dispersion predictions
- **Sections 29-30**: Empirical tests (dispersion indicator test, damping cancellation test) — **Paper 5 (CH)**, require `statsmodels`/`scipy`

Key patterns:
- Each `fetch_*` or `construct_*` function writes a CSV to `thesis_data/` and returns a DataFrame
- `main()` orchestrates all sections sequentially, then packages results into `thesis_data/thesis_data_package.xlsx`
- Incremental: checks for existing CSVs/indicators before re-fetching; `existing_indicators()` and `load_unavailable()` track API state
- Country set: 41 countries defined in `COUNTRIES` dict (top crypto-adopting + controls)

### build_bcl_indices.py (~1640 lines)

Constructs 5 Barth-Caprio-Levine regulatory indices from raw BRSS survey Excel files across 5 waves (2001-2019). Outputs `bcl_indices_panel.csv` with columns: country_code, country_name, wave, year, and 5 index scores (capital_stringency_idx, activity_restrictions_idx, supervisory_power_idx, entry_barriers_idx, private_monitoring_idx).

Internal sections:
- **Section 1**: BCL index definitions — component scoring rules for each of the 5 indices
- **Section 1B**: `WAVE5_QUESTION_MAP` — curated mapping of BCL component IDs to Wave 5 Q-format codes (see below)
- **Section 2**: Response parsing (`normalize_response`, `parse_yes_no`, `parse_activity_score`, `score_component`)
- **Section 3**: `BRSSParser` class — adaptive Excel parser with wave-specific handling
- **Section 4**: `compute_indices()` — sums component scores with `min_count` thresholds (CSI≥5, SPI≥7, EBI≥4, PMI≥6)
- **Section 5-7**: Panel construction, validation against existing data, download/main

**Wave 5 special handling** (critical for modifications):
- Wave 5 (2019) restructured the BRSS questionnaire. Data is split across 15 numbered sheets (`01`-`15`) instead of a single sheet. `BRSSParser.load()` detects and concatenates these sheets.
- `WAVE5_QUESTION_MAP` provides explicit Q-format code mappings (e.g., `cap_credit_risk` → `Q3_2a_2016`) because the dot-notation IDs from waves 1-4 don't match wave 5's restructured numbering.
- Wave 5 uses `"X"` checkmarks for multi-select questions and categorical codes (`"BS"`, `"C"`, `"OTH"`) instead of `"Yes"`/`"No"`. `parse_yes_no()` handles these.
- 3 components have no wave 5 equivalent: `cap_deduct_fx`, `sup_suspend_mgmt_fees`, `entry_deny_domestic`.
- BRSS files auto-download from World Bank via `download_brss_files()` if not already in `brss_data/`.

### Directory Structure

```
thesis/
├── fetch_thesis_data.py          # Main data pipeline (~6000 lines)
├── build_bcl_indices.py          # BCL index construction (~1640 lines)
├── fix_rpw.py, filter_rpw.py    # RPW data helpers
├── Complementary_Heterogeneity_v3.tex  # Latest CH paper source
├── unified_ces_framework.md      # CES framework writeup
├── ces_three_extensions.md       # CES extensions
├── bcl_indices_panel.csv         # BCL output panel
│
├── thesis_data/                  # Output CSVs (70+ files), packaged xlsx, test results
├── brss_data/                    # Input BRSS Excel files (waves 1-5)
│
├── papers/                       # Paper drafts organized by paper
│   ├── 1_endogenous_decentralization/  # ED paper (latest: v7_4.tex, v7.3.pdf, v6.md)
│   │   └── archive/                    # Older versions (v7-v7.3 .tex/.pdf, v3-v4 .md)
│   ├── 2_mesh_equilibrium/             # ME paper (v1.tex)
│   ├── 3_autocatalytic_mesh/           # AM paper (.tex, .pdf)
│   ├── 4_settlement_feedback/          # SF paper (.tex, .pdf)
│   ├── 5_complementary_heterogeneity/  # CH compiled PDF + archive
│   ├── 6_mpg_thesis/                   # MPG thesis docs, tables, appendices
│   └── 7_fair_inheritance/             # FI paper (v71.docx, act_v2.docx)
│
├── scripts/                      # Analysis scripts (not main pipeline)
│   ├── solve_model.py            # Paper 1: ED differential game solver (Nash MPE + cooperative)
│   ├── r0_analysis.py            # Paper 1: R0 open-weight adoption dynamics
│   ├── monetary_schism_model.py  # Paper 4: monetary schism simulation
│   ├── test_damping_cancellation.py  # Paper 5: CH damping cancellation empirical test
│   ├── test_dispersion_indicator.py  # Paper 5: CH dispersion indicator empirical test
│   ├── empirical_analysis.py     # Paper 6: MPG cross-country + event study regressions
│   ├── yield_gap_and_synth.py    # Paper 6: synthetic control analysis
│   ├── india_did.py              # Paper 6: India DID analysis
│   └── AI_Crypto_Model_v2.py     # Paper 6: AI/crypto adoption simulation
│
├── figures/                      # All generated figures
│   ├── mpg/                      # MPG thesis figures (fig1-fig7, SCM, DID, etc.)
│   ├── endogenous_decentralization/  # ED figures (value functions, shadow costs, etc.)
│   ├── complementary_heterogeneity/  # CH figures (damping, dispersion, leadlag)
│   └── monetary_schism/          # Monetary schism dashboard/sensitivity
│
├── research/                     # Research materials
│   ├── prompts/                  # Writing and search prompts
│   ├── field_searches/           # Literature field search results
│   ├── analysis/                 # Research analysis documents
│   └── ces_development/          # CES triple role development (+ archive/)
│
├── tools/                        # Interactive tools
│   ├── debate_v2.jsx             # Debate simulator
│   ├── analysis.jsx              # Chart visualization
│   └── debate.md                 # Debate transcript
│
└── data/                         # Supplementary data files
    ├── GlobalFindexDatabase2025.csv  # 15MB Findex raw data
    ├── fqi_panel.csv                 # FQI panel data
    ├── consumer_silicon_trajectory_corrected.csv
    └── yag_risk_decomposition.csv
```

### API Dependencies

- **World Bank API v2**: WDI indicators, Findex, Remittance Prices Worldwide bulk xlsx
- **FRED API**: semiconductor production indices (key set via `env.sh` or `FRED_API_KEY` env var)
- **HuggingFace API**: model download statistics (no key needed)
- **PCDB (Penn Capital and Development)**: optional DRAM data fetch
