# Thesis Restructuring Orchestrator Prompt

## Overview

This prompt restructures the thesis around two published foundation papers (Paper A and Paper B). The application papers (old Papers 1-4) are restructured to **cite** the foundation papers rather than re-derive CES results.

**Thesis structure:**

| Chapter | Directory | Source | Action |
|---------|-----------|--------|--------|
| 1. Introduction | `papers/thesis/1_introduction/` | New | Generate |
| 2. The CES Triple Role | `papers/thesis/2_ces_triple_role/` | Paper A (published) | Already copied, unchanged |
| 3. Complementary Heterogeneity | `papers/thesis/3_complementary_heterogeneity/` | Paper B (published) | Already copied, unchanged |
| 4. Endogenous Decentralization | `papers/thesis/4_endogenous_decentralization/` | Old Paper 1 | Restructure |
| 5. The Mesh Economy | `papers/thesis/5_mesh_economy/` | Old Papers 2+3 merged | Restructure + merge |
| 6. The Settlement Feedback | `papers/thesis/6_settlement_feedback/` | Old Paper 4 | Restructure |
| 7. Monetary Productivity Gap | `papers/thesis/7_monetary_productivity_gap/` | Paper 6 (empirical) | Already copied, unchanged |
| 8. Fair Inheritance | `papers/thesis/8_fair_inheritance/` | Paper 7 (policy) | Already copied, unchanged |

---

## Instructions for Claude Code

Execute this prompt by launching **4 subagents in parallel** using the Task tool. Each subagent reads the source files, generates a restructured .tex file, and writes it to the correct output directory.

**Before launching subagents**, read these two foundation papers to understand what results are available for citation:
- `papers/thesis/2_ces_triple_role/CES_Triple_Role.tex` (Paper A)
- `papers/thesis/3_complementary_heterogeneity/Complementary_Heterogeneity.tex` (Paper B)

Then launch 4 subagents simultaneously:

```
Subagent 1: Generate Chapter 1 (Introduction)
Subagent 2: Restructure Chapter 4 (Endogenous Decentralization)
Subagent 3: Generate Chapter 5 (The Mesh Economy — merge Papers 2+3)
Subagent 4: Restructure Chapter 6 (Settlement Feedback)
```

After all 4 subagents complete, compile each .tex file:
```bash
cd papers/thesis/1_introduction && pdflatex Introduction.tex && pdflatex Introduction.tex
cd papers/thesis/4_endogenous_decentralization && pdflatex Endogenous_Decentralization.tex && pdflatex Endogenous_Decentralization.tex
cd papers/thesis/5_mesh_economy && pdflatex Mesh_Economy.tex && pdflatex Mesh_Economy.tex
cd papers/thesis/6_settlement_feedback && pdflatex Settlement_Feedback.tex && pdflatex Settlement_Feedback.tex
```

---

## Global Rules (Apply to ALL Subagents)

### Citation Rules
1. **Never re-derive CES results.** Any CES setup, curvature parameter $K$, superadditivity, correlation robustness, or strategic independence result must cite Paper A (Smirl 2026a, "The CES Triple Role").
2. **Never re-derive hierarchical architecture.** Port topology theorem, spectral activation threshold, eigenstructure bridge, damping cancellation, or welfare decomposition must cite Paper B (Smirl 2026b, "Complementary Heterogeneity").
3. **Cite specifically.** Not "as shown in Paper A" but "by the Curvature Lemma (Smirl 2026a, Theorem 3.1)" or "the Port Topology Theorem (Smirl 2026b, Theorem 3.1)".
4. **State results before using them.** When citing a foundation paper result, state the result in italics as a recalled fact, then proceed to apply it. Example: "*By the CES Triple Role (Smirl 2026a, Theorem 7.1), the curvature parameter $K=(1-\rho)(J-1)/J$ simultaneously controls superadditivity, correlation robustness, and strategic independence.* We now apply this to the semiconductor learning curve setting..."

### LaTeX Style Rules
1. Use `\documentclass[12pt,letterpaper]{article}` with `[margin=1in]{geometry}` and `\onehalfspacing`.
2. Use `amsmath,amssymb,amsthm` for math. Use `booktabs` for tables.
3. Use `\newtheorem` for theorem environments numbered within sections.
4. Use `hyperref` with `colorlinks=true,linkcolor=blue,citecolor=blue`.
5. Include a `\begin{titlepage}` with title, author ("Connor Smirl, Department of Economics, Tufts University"), date ("February 2026"), and abstract.
6. Include JEL codes and keywords after the abstract.
7. All references go in a `thebibliography` environment at the end (no BibTeX).
8. Papers A and B are cited as:
   - `\bibitem{smirl2026a} Smirl, C. (2026a). The CES triple role: Superadditivity, correlation robustness, and strategic independence as three views of isoquant curvature. Working Paper, Tufts University.`
   - `\bibitem{smirl2026b} Smirl, C. (2026b). Complementary heterogeneity in hierarchical economies: CES aggregation, derived architecture, and cross-sector activation in multi-timescale systems. Working Paper, Tufts University.`

### Content Rules
1. **Preserve all original theorems, propositions, and proofs** from the source papers — only remove CES re-derivations and replace with citations.
2. **Preserve all empirical content** — data tables, calibration, natural experiments, predictions.
3. **Add a "Relation to the Thesis Framework" subsection** (1-2 paragraphs) at the end of the Introduction of each application chapter, explaining how this chapter maps to one level of the 4-level hierarchy from Paper B.
4. **Remove self-referential forward/backward references** between the old papers (e.g., "Smirl 2026, forthcoming"). Replace with chapter cross-references (e.g., "Chapter 5" or "Chapter 6").
5. Each application chapter should be self-contained enough to read independently, while making clear it builds on Chapters 2-3.

---

## Subagent 1: Chapter 1 — Introduction

**Output:** `papers/thesis/1_introduction/Introduction.tex`

**Read before writing:**
- `papers/thesis/2_ces_triple_role/CES_Triple_Role.tex` (Paper A abstract + intro)
- `papers/thesis/3_complementary_heterogeneity/Complementary_Heterogeneity.tex` (Paper B abstract + intro)
- `papers/1_endogenous_decentralization/Endogenous_Decentralization_v7_4.tex` (abstract only)
- `papers/2_mesh_equilibrium/Mesh_Equilibrium_v1.tex` (abstract only)
- `papers/3_autocatalytic_mesh/Autocatalytic_Mesh.tex` (abstract only)
- `papers/4_settlement_feedback/Settlement_Feedback.tex` (abstract only)
- `CLAUDE.md` (for the full thesis framework description)

**Structure (~15-20 pages):**

### 1.1 The Economic Question
Open with the central phenomenon: between 2018 and 2025, the five largest US technology companies committed ~$1.3 trillion in cumulative capex to centralized AI infrastructure. This paper asks: does this investment finance its own disruption?

### 1.2 The Thesis Argument in One Page
State the four-step feedback loop:
1. Concentrated investment finances learning curves (Chapter 4)
2. Learning curves enable distributed alternatives that self-organize into a mesh economy (Chapter 5)
3. The mesh economy requires programmable settlement, creating stablecoin demand (Chapter 6)
4. Stablecoin demand transforms monetary infrastructure, feeding back to constrain/enable centralized investment (Chapter 6)

### 1.3 The Mathematical Spine: CES Geometry
Explain that a single CES aggregate with curvature parameter $K$ unifies the entire thesis. State (citing Paper A) the triple role. State (citing Paper B) that CES geometry derives the hierarchical architecture, activation threshold, and welfare decomposition. This section should be ~2-3 pages and cite heavily from Papers A and B without re-proving anything.

### 1.4 The Four-Level Hierarchy
Present the hierarchy table from CLAUDE.md:

| Level | Chapter | State Variable | Gain Function | Timescale |
|-------|---------|---------------|---------------|-----------|
| 1 (slowest) | 4. Endogenous Decentralization | Hardware/semiconductor cost | Wright's Law ($\alpha \approx 0.23$) | Decades |
| 2 | 5. The Mesh Economy | Agent density | Recruitment dynamics | Years |
| 3 | 5. The Mesh Economy | Training effectiveness | Autocatalytic feedback | Months |
| 4 (fastest) | 6. Settlement Feedback | Stablecoin infrastructure | Settlement demand | Days-weeks |

Explain timescale separation and the hierarchical ceiling cascade (citing Paper B).

### 1.5 The Master Reproduction Number
Explain that each transition requires $R_0 > 1$ (spectral radius of next-generation matrix). Individual levels can be sub-threshold while the system as a whole is super-threshold through cross-level amplification.

### 1.6 Summary of Contributions by Chapter
Brief (1 paragraph each) summary of what each chapter contributes:
- **Chapter 2** (Paper A): The CES triple role — within-level geometry
- **Chapter 3** (Paper B): Complementary heterogeneity — between-level architecture and dynamics
- **Chapter 4**: Endogenous decentralization — Level 1 (hardware learning curves)
- **Chapter 5**: The mesh economy — Levels 2-3 (network formation + capability growth)
- **Chapter 6**: Settlement feedback — Level 4 (monetary/financial)
- **Chapter 7**: Monetary productivity gap — Empirical evidence for settlement layer demand
- **Chapter 8**: Fair inheritance — Policy implications of automation-era wealth concentration

### 1.7 Falsifiable Predictions
Collect the key predictions from Chapters 4-6 into a summary table with timing and failure conditions.

### 1.8 Roadmap
One paragraph per chapter describing the logical flow.

**No bibliography needed** — the introduction cites chapters, not external papers.

---

## Subagent 2: Chapter 4 — Endogenous Decentralization

**Output:** `papers/thesis/4_endogenous_decentralization/Endogenous_Decentralization.tex`

**Read before writing:**
- `papers/1_endogenous_decentralization/Endogenous_Decentralization_v7_4.tex` (FULL — this is the source)
- `papers/thesis/2_ces_triple_role/CES_Triple_Role.tex` (Sections 1-3 for citation targets)
- `papers/thesis/3_complementary_heterogeneity/Complementary_Heterogeneity.tex` (Section 4.1, "Hardware Level" for how Paper B describes this level)

**What to change:**
1. **Remove CES re-derivations.** The source paper derives CES-related properties inline (especially in the $R_0$ crossing condition section). Replace with citations to Paper A.
2. **Replace forward references.** The source says "Smirl (2026)" and "Smirl (2026b, forthcoming)" for the mesh and settlement papers. Replace with "Chapter 5" and "Chapter 6" respectively.
3. **Add "Relation to the Thesis Framework" subsection** after the Introduction. This chapter instantiates Level 1 (Hardware, slowest timescale) of the 4-level hierarchy from Paper B. The state variable is semiconductor cost, the gain function is Wright's Law ($\alpha \approx 0.23$), and the timescale is decades. The crossing condition ($R_0 > 1$) is a special case of the spectral activation threshold from Paper B (Theorem 4.1).
4. **Preserve everything else.** The differential game (Section 3), training-inference bifurcation (Section 4), all empirical evidence (Section 5), historical validation (Section 6), predictions (Section 7), and appendices must be preserved verbatim or with minimal rewording.

**Section structure (preserve from source, with additions marked *):**

1. Introduction
   - 1.1* Relation to the Thesis Framework (NEW — 1-2 paragraphs)
2. The Endogenous Decentralization Mechanism
   - 2.1 Three-Stage Structure
   - 2.2 The Self-Undermining Investment Property
   - 2.3 Dual Convergence
   - 2.4 Distinction from Adjacent Theory
3. Formal Model
   - 3.1 Environment
   - 3.2 Markov Perfect Equilibrium
   - 3.3 Cooperative Benchmark
   - 3.4 Analytical Solutions
   - 3.5 The Overinvestment Result
   - 3.6 Comparative Statics
   - 3.7 Calibration
   - 3.8 Note on Identification
   - 3.9 Generalized Crossing Condition (MODIFY: cite Paper A for CES properties, cite Paper B for spectral threshold)
4. The Training-Inference Structural Distinction
5. Empirical Evidence: Dual Convergence
6. Historical Validation and Parameter Consistency
7. Falsifiable Predictions
8. Conclusion
Appendices A-D (preserve)

---

## Subagent 3: Chapter 5 — The Mesh Economy

**Output:** `papers/thesis/5_mesh_economy/Mesh_Economy.tex`

**Read before writing:**
- `papers/2_mesh_equilibrium/Mesh_Equilibrium_v1.tex` (FULL — first source)
- `papers/3_autocatalytic_mesh/Autocatalytic_Mesh.tex` (FULL — second source)
- `papers/thesis/2_ces_triple_role/CES_Triple_Role.tex` (for citation targets)
- `papers/thesis/3_complementary_heterogeneity/Complementary_Heterogeneity.tex` (Sections 4.2-4.3 for how Paper B describes Levels 2-3)

**This is the hardest chapter — it merges two papers into one.**

The old Paper 2 (Mesh Equilibrium) covers Level 2 (network formation, years timescale) and the old Paper 3 (Autocatalytic Mesh) covers Level 3 (capability growth, months timescale). They share substantial CES machinery that was derived independently in each paper. In the thesis, this machinery comes from Papers A and B.

**What to change:**
1. **Merge into a single paper** with two major parts: Part I (Network Formation) from old Paper 2 and Part II (Capability Growth) from old Paper 3.
2. **Remove all CES re-derivations.** Both papers derive CES aggregation properties. Replace with citations to Paper A (triple role, curvature parameter) and Paper B (hierarchical architecture, activation threshold).
3. **Remove the "Terminal Conditions" sections** from both papers. Old Paper 2 has "Terminal Conditions from the Predecessor Paper" and old Paper 3 has "Terminal Conditions from the Mesh Paper." In the thesis, these terminal conditions are just earlier sections of the same chapter or references to Chapter 4.
4. **Unify notation.** Both papers use $\Rz$, $\Ceff$, $\Cmesh$, etc. Consolidate into a single notation section.
5. **Replace forward/backward references.** References to "Smirl (2026a)" become "Chapter 4", references to "Smirl (2026b)" or settlement paper become "Chapter 6".
6. **Merge the "Settlement Layer" / "Connection to Settlement Infrastructure" sections** from both papers into a single brief transition section pointing to Chapter 6.
7. **Merge the "Frameworks Considered and Rejected" sections** (deduplicate).
8. **Merge and deduplicate the prediction sets** from both papers.
9. **Preserve all theorems and proofs.** Theorem 1 (mesh equilibrium existence/uniqueness/stability), the Potts model phase transition, inverse Bose-Einstein condensation, RAF existence threshold, three growth regimes, model collapse protection — all preserved.

**Proposed section structure:**

1. Introduction
   - 1.1* Relation to the Thesis Framework (NEW)
2. Setup
   - 2.1 The CES Aggregate (brief — cite Paper A)
   - 2.2 Notation
3. **Part I: Network Formation (from old Paper 2)**
   - 3.1 The $R_0^{\text{mesh}}$ Framework (cite Paper B for spectral threshold)
   - 3.2 Giant Component Existence
   - 3.3 The Fortuin-Kasteleyn Unification (Potts model, first-order phase transition)
   - 3.4 Inverse Bose-Einstein Condensation
   - 3.5 Heterogeneous Specialization (CES aggregation — cite Paper A for diversity premium)
   - 3.6 Specialization Dynamics (fixed response threshold model)
   - 3.7 Knowledge Diffusion (Laplacian dynamics, vanishing epidemic threshold)
   - 3.8 The Central Theorem: Mesh Equilibrium (Theorem 1)
   - 3.9 Post-Crossing Dynamics (nucleation, rapid growth, maturity)
4. **Part II: Capability Growth (from old Paper 3)**
   - 4.1 The Fixed-Capability Assumption Relaxed
   - 4.2 The Autocatalytic Existence Threshold (RAF sets)
   - 4.3 Growth Dynamics (improvement function, semi-endogenous formulation)
   - 4.4 The Three Regimes (convergence, exponential, singularity)
   - 4.5 Diversity as Collapse Protection (cite Paper A for correlation robustness)
   - 4.6 The Baumol Bottleneck (cite Paper B for hierarchical ceiling)
5. Transition to Settlement (brief — points to Chapter 6)
6. Frameworks Considered and Rejected (merged, deduplicated)
7. Falsifiable Predictions (merged, deduplicated)
8. Conclusion
Appendices (from both papers, consolidated)

---

## Subagent 4: Chapter 6 — The Settlement Feedback

**Output:** `papers/thesis/6_settlement_feedback/Settlement_Feedback.tex`

**Read before writing:**
- `papers/4_settlement_feedback/Settlement_Feedback.tex` (FULL — this is the source)
- `papers/thesis/2_ces_triple_role/CES_Triple_Role.tex` (for citation targets)
- `papers/thesis/3_complementary_heterogeneity/Complementary_Heterogeneity.tex` (Section 4.4 for how Paper B describes Level 4)

**What to change:**
1. **Remove CES re-derivations.** Any CES aggregation properties cited from earlier papers in the sequence should now cite Papers A and B.
2. **Replace the "Terminal Conditions from the Prior Papers" section.** This section references Smirl (2026), Smirl (2026b), Smirl (2026c), and Smirl (2026a). Replace with chapter cross-references:
   - "From the Mesh Equilibrium (Smirl 2026)" → "From Chapter 5 (The Mesh Economy)"
   - "From the Monetary Productivity Gap (Smirl 2026b)" → "From Chapter 7 (The Monetary Productivity Gap)"
   - "From the Autocatalytic Mesh (Smirl 2026c)" → "From Chapter 5, Part II"
   - "From Endogenous Decentralization (Smirl 2026a)" → "From Chapter 4"
3. **Add "Relation to the Thesis Framework" subsection** after the Introduction. This chapter instantiates Level 4 (Settlement, fastest timescale) of the 4-level hierarchy from Paper B. The state variable is stablecoin infrastructure, the gain function is settlement demand, the timescale is days-weeks. The Triffin contradiction identified here is mathematically the same object as the Baumol bottleneck in Chapter 5 — both are slow-manifold constraints at adjacent layers (cite Paper B, Section 5 or the hierarchical ceiling cascade).
4. **Preserve everything else.** Market microstructure (Section 3), monetary policy effectiveness (Section 4), dollarization spiral (Section 5), Triffin contradiction (Section 6), coupled ODE system (Section 7), sovereign fiscal implications (Section 8), predictions (Section 9) — all preserved.

**Section structure (preserve from source, with modifications marked *):**

1. Introduction
   - 1.1* Relation to the Thesis Framework (NEW)
2. Terminal Conditions* (REWRITTEN — chapter cross-references instead of paper references)
3. Market Microstructure Transition
4. Monetary Policy Effectiveness as a Function of $\phi$
5. The Dollarization Spiral
6. The Triffin Contradiction (MODIFY: cite Paper B for the Baumol-Triffin equivalence)
7. The Coupled System — Equilibrium Characterization
8. Implications for Sovereign Fiscal Policy
9. Frameworks Considered and Rejected
10. Falsifiable Predictions
11. Conclusion

---

## Post-Compilation Verification

After all 4 .tex files are compiled, verify:

1. **All 8 chapter directories populated** — check that each has at least one .tex or .docx file.
2. **Zero LaTeX errors** — each pdflatex run should complete without errors (warnings are acceptable).
3. **Foundation paper citations present** — grep each application chapter for `smirl2026a` and `smirl2026b` to confirm they cite both foundation papers.
4. **No CES re-derivations** — grep for `curvature parameter` in Chapters 4-6; each occurrence should be accompanied by a `\cite{smirl2026a}` or reference to "Paper A" / "Chapter 2".
5. **No orphan forward references** — grep for "forthcoming" or "Smirl (2026" in Chapters 4-6; there should be none.
