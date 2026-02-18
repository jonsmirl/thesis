"""
THE MONETARY SCHISM: Comprehensive Simulation Model
=====================================================
Integrates all theoretical frameworks from v5 of the paper:

1. AI Sector Growth & Crypto S-Curve (Sections 2-3)
2. Metabolic Rift: Non-metabolic vs metabolic labor cost dynamics (Mostaque 2025)
3. Coasean Firm Dissolution: Transaction cost collapse → organizational physics (Coase 1937)
4. Three Flows Decomposition: Gradient/Circular/Harmonic (Hodge, Mostaque 2025)
5. Abundance Paradox: GDP vs true welfare divergence (Kuznets 1934, Mostaque 2025)
6. Government Response Scenarios with Lucas Critique (Lucas 1976)
7. Game Theory: Symmetric, Asymmetric, Sequential (Nash 1950)
8. Tipping Point & Nucleation Dynamics (Mostaque 2025)
9. Perez Cycle: Installation → Turning Point → Deployment (Perez 2002)
10. Dual-Currency Extended Solow Model (Section 6)
11. Piketty r>g reinterpreted: Circular Flow vs Gradient Flow (Piketty 2014, Mostaque 2025)
12. Developing vs Advanced Economy Divergence Paths

Connor Smirl | EC 118 Quantitative Economic Growth | Tufts University | Spring 2026
"""

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'legend.fontsize': 7.5,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Color palette
C = {
    'fiat': '#2E4057',       # dark navy
    'crypto': '#E8963E',     # amber
    'ai': '#4A90D9',         # blue
    'growth': '#27AE60',     # green
    'danger': '#C0392B',     # red
    'purple': '#8E44AD',     # purple
    'teal': '#16A085',       # teal
    'gradient': '#E74C3C',   # red-orange (Gradient Flow)
    'circular': '#3498DB',   # blue (Circular Flow)
    'harmonic': '#2ECC71',   # green (Harmonic Flow)
    'metabolic': '#D35400',  # burnt orange
    'nonmetabolic': '#2980B9', # steel blue
    'gdp': '#7F8C8D',       # grey
    'welfare': '#27AE60',    # green
}

YEARS = np.arange(2020, 2051)
T = YEARS - 2025  # centered on 2025

# ══════════════════════════════════════════════════════════════════════════════
# 1. CORE MODELS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WorldEconomy:
    """Master simulation state."""
    years: np.ndarray = field(default_factory=lambda: YEARS)

    # --- AI Sector ---
    ai_base_2025: float = 391e9        # $391B
    ai_cagr: float = 0.306             # 30.6%
    ai_saturation: float = 0.35        # max GDP share

    # --- Crypto S-Curve ---
    theta_0: float = 0.02              # initial crypto share of AI
    theta_max: float = 0.65            # asymptotic ceiling
    k_scurve: float = 0.8              # steepness
    t_star: int = 2032                 # inflection year

    # --- World GDP ---
    gdp_2025: float = 105e12           # $105T
    gdp_growth_base: float = 0.03      # 3% baseline

    # --- Solow Parameters ---
    alpha: float = 0.33                # capital share
    s_base: float = 0.22              # savings rate
    delta: float = 0.05               # depreciation
    g0: float = 0.02                  # baseline TFP growth

    # --- Metabolic Rift ---
    human_labor_cost_2025: float = 35.0   # $/hr average
    ai_compute_cost_2025: float = 2.0     # $/hr equivalent
    ai_cost_deflation: float = 0.40       # 40x/yr → ~97.5% annual decline
    metabolic_overhead: float = 0.72      # fraction of human cost that is "metabolic"

    # --- Coasean Parameters ---
    txn_cost_market_2025: float = 1.0     # normalized
    txn_cost_internal_2025: float = 0.4   # normalized
    smart_contract_deflation: float = 0.30 # annual reduction in market txn costs

    # --- Perez Cycle ---
    installation_start: int = 2020
    turning_point_base: int = 2031        # without compression
    deployment_end: int = 2045
    compression_factor: float = 0.6       # Mostaque: Intelligence Inversion compresses cycle

    def __post_init__(self):
        self.n = len(self.years)
        self._compute_all()

    def _compute_all(self):
        self._compute_gdp()
        self._compute_ai_sector()
        self._compute_crypto_scurve()
        self._compute_monetary_displacement()
        self._compute_metabolic_rift()
        self._compute_coasean_dissolution()
        self._compute_three_flows()
        self._compute_abundance_paradox()
        self._compute_solow_dual_currency()
        self._compute_tipping_point()
        self._compute_government_scenarios()
        self._compute_piketty_flows()
        self._compute_perez_cycle()
        self._compute_developing_advanced()
        self._compute_lucas_critique()
        self._compute_nucleation()

    # ── GDP Projections ──
    def _compute_gdp(self):
        self.gdp = np.zeros(self.n)
        self.gdp[0] = self.gdp_2025 * (1 + self.gdp_growth_base) ** (self.years[0] - 2025)
        for i in range(1, self.n):
            self.gdp[i] = self.gdp[i-1] * (1 + self.gdp_growth_base)

    # ── AI Sector ──
    def _compute_ai_sector(self):
        self.ai_sector = np.zeros(self.n)
        for i, y in enumerate(self.years):
            raw = self.ai_base_2025 * (1 + self.ai_cagr) ** (y - 2025)
            # logistic dampening as share of GDP
            share = raw / self.gdp[i]
            dampened_share = self.ai_saturation * share / (self.ai_saturation + share * (1 - self.ai_saturation / 0.5))
            self.ai_sector[i] = dampened_share * self.gdp[i]
        self.ai_gdp_share = self.ai_sector / self.gdp

    # ── Crypto S-Curve ──
    def _compute_crypto_scurve(self):
        self.theta = np.array([
            self.theta_0 + (self.theta_max - self.theta_0) / (1 + np.exp(-self.k_scurve * (y - self.t_star)))
            for y in self.years
        ])

    # ── Monetary Displacement ──
    def _compute_monetary_displacement(self):
        self.ai_crypto_volume = self.ai_sector * self.theta
        self.fiat_exit_ratio = self.ai_crypto_volume / self.gdp
        money_multiplier = 8.5
        self.seigniorage_loss = self.ai_crypto_volume * 0.002 * money_multiplier
        self.policy_effectiveness = 1 - self.fiat_exit_ratio

    # ── Metabolic Rift (Mostaque 2025) ──
    def _compute_metabolic_rift(self):
        """Model the divergence between metabolic (human) and non-metabolic (AI) labor costs."""
        self.human_cost = np.zeros(self.n)
        self.ai_cost = np.zeros(self.n)
        self.metabolic_ratio = np.zeros(self.n)

        for i, y in enumerate(self.years):
            dt = y - 2025
            # Human labor cost rises ~3% per year (wage inflation)
            self.human_cost[i] = self.human_labor_cost_2025 * (1.03 ** dt)
            # AI cost deflates ~97.5% per year (40x annual = 1/40 remaining)
            # But use a more conservative 60% annual decline for sustained modeling
            self.ai_cost[i] = max(self.ai_compute_cost_2025 * (0.40 ** dt), 0.001)
            # Ratio: how many times cheaper is AI than human labor
            self.metabolic_ratio[i] = self.human_cost[i] / self.ai_cost[i]

        # Metabolic overhead decomposition
        self.human_metabolic = self.human_cost * self.metabolic_overhead
        self.human_cognitive = self.human_cost * (1 - self.metabolic_overhead)

    # ── Coasean Firm Dissolution (Coase 1937) ──
    def _compute_coasean_dissolution(self):
        """Model transaction cost collapse and optimal firm size."""
        self.txn_cost_market = np.zeros(self.n)
        self.txn_cost_internal = np.zeros(self.n)
        self.optimal_firm_size = np.zeros(self.n)
        self.pct_activity_in_firms = np.zeros(self.n)

        for i, y in enumerate(self.years):
            dt = max(y - 2025, 0)
            # Market transaction costs collapse due to smart contracts + AI agents
            self.txn_cost_market[i] = self.txn_cost_market_2025 * ((1 - self.smart_contract_deflation) ** dt)
            # Internal costs decline more slowly (organizational inertia)
            self.txn_cost_internal[i] = self.txn_cost_internal_2025 * ((1 - 0.08) ** dt)

            # Coasean equilibrium: firm expands until internal cost = market cost
            # When market < internal, activity shifts to market (Second Economy)
            if self.txn_cost_market[i] < self.txn_cost_internal[i]:
                ratio = self.txn_cost_market[i] / self.txn_cost_internal[i]
                self.optimal_firm_size[i] = ratio  # shrinks as market gets cheaper
                self.pct_activity_in_firms[i] = 0.3 + 0.7 * ratio
            else:
                self.optimal_firm_size[i] = 1.0
                self.pct_activity_in_firms[i] = 1.0

        # Normalize firm size to 2025 = 100
        idx_2025 = np.where(self.years == 2025)[0][0]
        self.optimal_firm_size = (self.optimal_firm_size / self.optimal_firm_size[idx_2025]) * 100

    # ── Three Flows Decomposition (Hodge/Mostaque 2025) ──
    def _compute_three_flows(self):
        """Decompose economic activity into Gradient, Circular, and Harmonic flows."""
        self.gradient_share = np.zeros(self.n)  # Scarcity-driven, rivalrous (fiat domain)
        self.circular_share = np.zeros(self.n)  # Abundance-driven, non-rival (crypto domain)
        self.harmonic_share = np.zeros(self.n)  # Institutional channels (both domains)

        for i, y in enumerate(self.years):
            dt = y - 2025
            # Gradient flow (Atomic Economy) shrinks as AI automates rivalrous production
            self.gradient_share[i] = max(0.55 * np.exp(-0.02 * max(dt, 0)), 0.15)
            # Circular flow (Bit Economy) grows as non-rival AI economy expands
            self.circular_share[i] = min(0.25 + 0.015 * max(dt, 0) + 0.001 * max(dt, 0)**2, 0.65)
            # Harmonic flow (institutional) is the residual, slowly adapting
            self.harmonic_share[i] = 1.0 - self.gradient_share[i] - self.circular_share[i]
            self.harmonic_share[i] = max(self.harmonic_share[i], 0.10)
            # Renormalize
            total = self.gradient_share[i] + self.circular_share[i] + self.harmonic_share[i]
            self.gradient_share[i] /= total
            self.circular_share[i] /= total
            self.harmonic_share[i] /= total

    # ── Abundance Paradox: GDP vs Welfare (Kuznets 1934, Mostaque 2025) ──
    def _compute_abundance_paradox(self):
        """Model the divergence between GDP (scarcity metric) and true welfare."""
        self.measured_gdp_growth = np.zeros(self.n)
        self.true_welfare_growth = np.zeros(self.n)
        self.abundance_gap = np.zeros(self.n)

        for i, y in enumerate(self.years):
            dt = max(y - 2025, 0)
            # GDP growth slows as AI makes things free (abundance appears as contraction)
            deflation_drag = 0.005 * self.circular_share[i]  # more Circular Flow = more mismeasurement
            self.measured_gdp_growth[i] = self.gdp_growth_base - deflation_drag * dt * 0.1

            # True welfare grows faster due to consumer surplus from free AI services
            ai_welfare_boost = 0.02 * self.ai_gdp_share[i] * 3  # AI welfare 3x its GDP contribution
            self.true_welfare_growth[i] = self.gdp_growth_base + ai_welfare_boost

            # Cumulative gap
            if i > 0:
                self.abundance_gap[i] = self.abundance_gap[i-1] + (
                    self.true_welfare_growth[i] - self.measured_gdp_growth[i]
                )

    # ── Dual-Currency Solow Model (Section 6) ──
    def _compute_solow_dual_currency(self):
        """Extended Solow with dual-currency dynamics."""
        self.y_capita = np.zeros(self.n)
        self.growth_rate = np.zeros(self.n)
        self.s_eff = np.zeros(self.n)
        self.tech_A = np.zeros(self.n)
        self.y_capita_fiat_only = np.zeros(self.n)

        # Initial conditions
        self.y_capita[0] = 2.0
        self.y_capita_fiat_only[0] = 2.0
        self.tech_A[0] = 1.0
        tech_A_fiat = 1.0

        for i in range(self.n):
            theta_i = self.theta[i]
            # Effective savings: crypto adds transaction efficiency, loses some monetary stability
            eps_x = 0.02  # transaction cost savings
            eps_m = 0.003  # monetary policy loss
            self.s_eff[i] = self.s_base + (eps_x - eps_m) * theta_i

            # Technology augmented by crypto infrastructure
            self.tech_A[i] = self.tech_A[0] * (1 + self.g0 * (1 + 0.5 * theta_i)) ** max(self.years[i] - 2020, 0)
            tech_A_fiat = self.tech_A[0] * (1 + self.g0) ** max(self.years[i] - 2020, 0)

            # Steady-state output per effective worker
            k_star = (self.s_eff[i] / (self.g0 + self.delta)) ** (1 / (1 - self.alpha))
            self.y_capita[i] = self.tech_A[i] * k_star ** self.alpha

            k_star_fiat = (self.s_base / (self.g0 + self.delta)) ** (1 / (1 - self.alpha))
            self.y_capita_fiat_only[i] = tech_A_fiat * k_star_fiat ** self.alpha

            if i > 0:
                self.growth_rate[i] = (self.y_capita[i] - self.y_capita[i-1]) / self.y_capita[i-1]

        # Normalize to 2025 = 100
        idx = np.where(self.years == 2025)[0][0]
        norm = self.y_capita[idx]
        norm_fiat = self.y_capita_fiat_only[idx]
        self.y_capita = self.y_capita / norm * 100
        self.y_capita_fiat_only = self.y_capita_fiat_only / norm_fiat * 100

    # ── Tipping Point Dynamics (Section 5.4) ──
    def _compute_tipping_point(self):
        """Network effects model: benefit vs cost as function of crypto share."""
        self.tp_shares = np.linspace(0, 1, 101)
        eta = 40
        b0 = 5
        c_base = 22
        rho = 8

        self.tp_benefit = b0 + eta * self.tp_shares
        self.tp_cost = c_base * (1 - self.tp_shares) + rho * (1 - self.tp_shares)**2
        self.tp_net = self.tp_benefit - self.tp_cost

        # Find tipping point (where net crosses zero)
        crossings = np.where(np.diff(np.sign(self.tp_net)))[0]
        self.tipping_share = self.tp_shares[crossings[0]] if len(crossings) > 0 else 0.30

    # ── Government Response Scenarios ──
    def _compute_government_scenarios(self):
        """Three scenarios: Accommodate, Resist, Adapt."""
        self.scen_accommodate = {'tax_rev': np.zeros(self.n), 'growth_loss': np.zeros(self.n), 'sovereignty': np.zeros(self.n)}
        self.scen_resist = {'tax_rev': np.zeros(self.n), 'growth_loss': np.zeros(self.n), 'sovereignty': np.zeros(self.n)}
        self.scen_adapt = {'tax_rev': np.zeros(self.n), 'growth_loss': np.zeros(self.n), 'sovereignty': np.zeros(self.n)}

        for i, y in enumerate(self.years):
            theta_i = self.theta[i]
            ai_vol = self.ai_crypto_volume[i]

            # Accommodate: tax crypto at 15%, lose some sovereignty
            self.scen_accommodate['tax_rev'][i] = ai_vol * 0.15
            self.scen_accommodate['growth_loss'][i] = 0.001 * theta_i  # minimal friction
            self.scen_accommodate['sovereignty'][i] = 1.0 - 0.6 * theta_i

            # Resist: no tax revenue, massive growth loss from innovation flight
            self.scen_resist['tax_rev'][i] = 0
            cum_loss = 0.005 * (max(y - 2025, 0))**2 * theta_i
            self.scen_resist['growth_loss'][i] = min(cum_loss, 0.30)
            self.scen_resist['sovereignty'][i] = min(1.0 - 0.15 * theta_i, 1.0)

            # Adapt (CBDC): moderate tax, moderate sovereignty preservation
            self.scen_adapt['tax_rev'][i] = ai_vol * 0.10
            self.scen_adapt['growth_loss'][i] = 0.0005 * theta_i
            self.scen_adapt['sovereignty'][i] = 1.0 - 0.35 * theta_i

    # ── Piketty r > g Through Three Flows (Piketty 2014, Mostaque 2025) ──
    def _compute_piketty_flows(self):
        """r (Circular Flow returns) vs g (Gradient Flow growth)."""
        self.r_circular = np.zeros(self.n)  # returns on non-rival capital
        self.g_gradient = np.zeros(self.n)  # growth of rivalrous economy

        for i, y in enumerate(self.years):
            dt = max(y - 2025, 0)
            # r accelerates as AI amplifies non-rival capital returns
            self.r_circular[i] = 0.05 + 0.003 * dt + 0.08 * self.circular_share[i]
            # g decelerates as AI displaces labor in rivalrous sectors
            self.g_gradient[i] = max(0.03 - 0.001 * dt * self.ai_gdp_share[i] * 5, 0.005)

        self.rg_gap = self.r_circular - self.g_gradient

    # ── Perez Cycle (Perez 2002) ──
    def _compute_perez_cycle(self):
        """Installation → Turning Point → Deployment with optional compression."""
        self.perez_phase = np.zeros(self.n)  # 0=pre, 1=installation, 2=turning, 3=deployment
        self.perez_intensity = np.zeros(self.n)  # speculative intensity

        compressed_turning = self.installation_start + int(
            (self.turning_point_base - self.installation_start) * self.compression_factor
        )

        for i, y in enumerate(self.years):
            if y < self.installation_start:
                self.perez_phase[i] = 0
                self.perez_intensity[i] = 0.1
            elif y < compressed_turning:
                self.perez_phase[i] = 1  # Installation
                progress = (y - self.installation_start) / (compressed_turning - self.installation_start)
                # Speculative intensity rises then peaks
                self.perez_intensity[i] = 0.3 + 0.7 * np.sin(progress * np.pi)
            elif y < compressed_turning + 3:
                self.perez_phase[i] = 2  # Turning point (crisis + reform)
                self.perez_intensity[i] = 0.8 - 0.6 * (y - compressed_turning) / 3
            else:
                self.perez_phase[i] = 3  # Deployment (golden age)
                progress = min((y - compressed_turning - 3) / 15, 1.0)
                self.perez_intensity[i] = 0.2 + 0.5 * progress

    # ── Developing vs Advanced Economy Divergence ──
    def _compute_developing_advanced(self):
        """Two-track simulation: advanced (strong fiat) vs developing (weak fiat)."""
        self.advanced_welfare = np.zeros(self.n)
        self.developing_accommodate = np.zeros(self.n)
        self.developing_resist = np.zeros(self.n)

        self.advanced_welfare[0] = 100
        self.developing_accommodate[0] = 30  # start at 30% of advanced
        self.developing_resist[0] = 30

        for i in range(1, self.n):
            theta_i = self.theta[i]

            # Advanced economy: steady growth + modest crypto boost
            adv_growth = 0.02 + 0.005 * theta_i
            self.advanced_welfare[i] = self.advanced_welfare[i-1] * (1 + adv_growth)

            # Developing + accommodate: higher growth from crypto stability + leapfrogging
            dev_acc_growth = 0.04 + 0.02 * theta_i  # larger crypto premium (fiat quality gap)
            self.developing_accommodate[i] = self.developing_accommodate[i-1] * (1 + dev_acc_growth)

            # Developing + resist: stagnation, brain drain, fiat deterioration
            dev_res_growth = max(0.02 - 0.01 * theta_i, -0.01)  # can go negative
            self.developing_resist[i] = self.developing_resist[i-1] * (1 + dev_res_growth)

    # ── Lucas Critique: Policy Effectiveness Decay ──
    def _compute_lucas_critique(self):
        """Model how monetary policy effectiveness degrades as agents choose their monetary regime."""
        self.policy_eff_naive = np.zeros(self.n)      # what central banks expect
        self.policy_eff_actual = np.zeros(self.n)      # what actually happens
        self.lucas_gap = np.zeros(self.n)               # the surprise

        for i, y in enumerate(self.years):
            theta_i = self.theta[i]
            # Naive model: linear degradation based on fiat exit
            self.policy_eff_naive[i] = 1.0 - 0.5 * self.fiat_exit_ratio[i]
            # Actual: nonlinear because agents actively optimize around policy
            # AI agents respond to rate changes by shifting MORE to crypto
            feedback = theta_i * 0.3  # endogenous response
            self.policy_eff_actual[i] = (1.0 - self.fiat_exit_ratio[i]) * (1.0 - feedback)
            self.lucas_gap[i] = self.policy_eff_naive[i] - self.policy_eff_actual[i]

    # ── Nucleation Dynamics (Mostaque 2025) ──
    def _compute_nucleation(self):
        """Phase transition: nucleation sites accumulate until critical mass triggers cascade."""
        self.nucleation_sites = np.zeros(self.n)
        self.phase_state = np.zeros(self.n)  # 0=liquid (fiat), 1=crystallizing, 2=solid (crypto)

        for i, y in enumerate(self.years):
            dt = max(y - 2025, 0)
            # Nucleation sites: each institutional adoption is a seed crystal
            # Exponential: MicroStrategy, GENIUS Act, AgentPay, UCP, etc.
            self.nucleation_sites[i] = 5 * np.exp(0.25 * dt)  # ~28% annual growth in sites

            # Phase state depends on nucleation density reaching critical threshold
            if self.theta[i] < 0.10:
                self.phase_state[i] = 0  # Supercooled (fiat dominant, but unstable)
            elif self.theta[i] < self.tipping_share:
                self.phase_state[i] = 1  # Crystallizing (nucleation spreading)
            else:
                self.phase_state[i] = 2  # Phase transition complete


# ══════════════════════════════════════════════════════════════════════════════
# 2. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_dashboard(econ: WorldEconomy, savepath: str = None):
    """Generate the comprehensive 12-panel dashboard."""
    fig = plt.figure(figsize=(24, 30))
    fig.suptitle(
        'The Monetary Schism: Comprehensive Simulation Dashboard',
        fontsize=18, fontweight='bold', y=0.98
    )
    fig.text(0.5, 0.965,
        'AI Migration to Cryptocurrency and the Crisis of Fiat Sovereignty  |  EC 118 Quantitative Economic Growth  |  Spring 2026',
        ha='center', fontsize=10, style='italic', color='#555555'
    )

    gs = gridspec.GridSpec(4, 3, hspace=0.35, wspace=0.30, top=0.94, bottom=0.03, left=0.06, right=0.97)
    yrs = econ.years

    # ── Panel 1: AI Sector & Crypto S-Curve ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1b = ax1.twinx()
    ax1.fill_between(yrs, econ.ai_sector / 1e12, alpha=0.3, color=C['ai'])
    ax1.plot(yrs, econ.ai_sector / 1e12, color=C['ai'], linewidth=2, label='AI Sector ($T)')
    ax1b.plot(yrs, econ.theta * 100, color=C['crypto'], linewidth=2, linestyle='--', label='Crypto Share (%)')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('AI Sector Size ($T)', color=C['ai'])
    ax1b.set_ylabel('Crypto Share of AI (%)', color=C['crypto'])
    ax1.set_title('1. AI Sector Growth & Crypto Adoption S-Curve')
    ax1.axvline(x=2032, color='grey', linestyle=':', alpha=0.5)
    ax1.text(2032.5, ax1.get_ylim()[1]*0.85, 't* inflection', fontsize=7, color='grey')
    lines1 = ax1.get_lines() + ax1b.get_lines()
    ax1.legend(lines1, [l.get_label() for l in lines1], loc='upper left', fontsize=7)

    # ── Panel 2: Metabolic Rift (Mostaque 2025) ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(yrs, econ.human_cost, color=C['metabolic'], linewidth=2, label='Human Labor ($/hr)')
    ax2.semilogy(yrs, econ.ai_cost, color=C['nonmetabolic'], linewidth=2, label='AI Compute ($/hr equiv.)')
    ax2.fill_between(yrs, econ.ai_cost, econ.human_cost, alpha=0.15, color=C['danger'])
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Cost ($/hr, log scale)')
    ax2.set_title('2. The Metabolic Rift: Human vs AI Labor Cost')
    ax2.legend(loc='center left', fontsize=7)
    # Annotate the ratio
    idx_2035 = np.where(yrs == 2035)[0][0]
    ratio_2035 = econ.metabolic_ratio[idx_2035]
    ax2.annotate(f'2035: AI is {ratio_2035:,.0f}x cheaper',
        xy=(2035, econ.ai_cost[idx_2035]), xytext=(2028, 0.01),
        fontsize=7, color=C['danger'],
        arrowprops=dict(arrowstyle='->', color=C['danger'], lw=0.8))

    # ── Panel 3: Coasean Firm Dissolution ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(yrs, econ.txn_cost_market, color=C['crypto'], linewidth=2, label='Market Txn Cost (smart contracts)')
    ax3.plot(yrs, econ.txn_cost_internal, color=C['fiat'], linewidth=2, label='Internal Org Cost (firms)')
    ax3.fill_between(yrs, econ.txn_cost_market, econ.txn_cost_internal,
        where=econ.txn_cost_market < econ.txn_cost_internal,
        alpha=0.2, color=C['crypto'], label='Coasean reversal zone')
    # Find crossover
    crossover_idx = np.where(np.diff(np.sign(econ.txn_cost_market - econ.txn_cost_internal)))[0]
    if len(crossover_idx) > 0:
        cx = yrs[crossover_idx[0]]
        ax3.axvline(x=cx, color=C['danger'], linestyle=':', alpha=0.7)
        ax3.text(cx + 0.5, 0.5, f'Crossover\n{cx}', fontsize=7, color=C['danger'])
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Normalized Transaction Cost')
    ax3.set_title('3. Coasean Dissolution: Transaction Costs (Coase 1937)')
    ax3.legend(loc='upper right', fontsize=7)

    # ── Panel 4: Three Flows Decomposition (Hodge/Mostaque) ──
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.stackplot(yrs,
        econ.gradient_share * 100, econ.circular_share * 100, econ.harmonic_share * 100,
        labels=['Gradient (Atomic/Fiat)', 'Circular (Bit/Crypto)', 'Harmonic (Institutional)'],
        colors=[C['gradient'], C['circular'], C['harmonic']], alpha=0.7)
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Share of Economic Activity (%)')
    ax4.set_title('4. Three Flows Decomposition (Hodge/Mostaque 2025)')
    ax4.legend(loc='center right', fontsize=7)
    ax4.set_ylim(0, 100)

    # ── Panel 5: Abundance Paradox (Kuznets 1934, Mostaque 2025) ──
    ax5 = fig.add_subplot(gs[1, 1])
    idx_2025 = np.where(yrs == 2025)[0][0]
    # Compute cumulative indices
    gdp_index = np.ones(econ.n)
    welfare_index = np.ones(econ.n)
    for i in range(1, econ.n):
        gdp_index[i] = gdp_index[i-1] * (1 + econ.measured_gdp_growth[i])
        welfare_index[i] = welfare_index[i-1] * (1 + econ.true_welfare_growth[i])
    gdp_index = gdp_index / gdp_index[idx_2025] * 100
    welfare_index = welfare_index / welfare_index[idx_2025] * 100
    ax5.plot(yrs, gdp_index, color=C['gdp'], linewidth=2, label='Measured GDP (scarcity metric)')
    ax5.plot(yrs, welfare_index, color=C['welfare'], linewidth=2, label='True Welfare (abundance-adjusted)')
    ax5.fill_between(yrs, gdp_index, welfare_index,
        where=welfare_index > gdp_index, alpha=0.2, color=C['welfare'])
    ax5.set_xlabel('Year')
    ax5.set_ylabel('Index (2025 = 100)')
    ax5.set_title('5. The Abundance Paradox: GDP vs Welfare (Kuznets 1934)')
    ax5.legend(loc='upper left', fontsize=7)
    # Annotate gap
    idx_2040 = np.where(yrs == 2040)[0][0]
    gap_2040 = welfare_index[idx_2040] - gdp_index[idx_2040]
    ax5.annotate(f'{gap_2040:.0f}pt gap by 2040',
        xy=(2040, (welfare_index[idx_2040] + gdp_index[idx_2040])/2),
        fontsize=8, fontweight='bold', color=C['welfare'], ha='center')

    # ── Panel 6: Tipping Point & Nucleation ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(econ.tp_shares * 100, econ.tp_benefit, color=C['growth'], linewidth=2, label='Benefit B(φ)')
    ax6.plot(econ.tp_shares * 100, econ.tp_cost, color=C['danger'], linewidth=2, label='Cost C(φ)')
    ax6.fill_between(econ.tp_shares * 100, econ.tp_benefit, econ.tp_cost,
        where=econ.tp_benefit > econ.tp_cost, alpha=0.15, color=C['growth'])
    ax6.fill_between(econ.tp_shares * 100, econ.tp_benefit, econ.tp_cost,
        where=econ.tp_benefit < econ.tp_cost, alpha=0.15, color=C['danger'])
    ax6.axvline(x=econ.tipping_share * 100, color=C['purple'], linestyle='--', linewidth=1.5)
    ax6.text(econ.tipping_share * 100 + 2, ax6.get_ylim()[1]*0.9,
        f'Tipping Point\n~{econ.tipping_share*100:.0f}% (~2031)',
        fontsize=8, color=C['purple'], fontweight='bold')
    ax6.set_xlabel('Crypto Share of AI Economy (%)')
    ax6.set_ylabel('Welfare Units')
    ax6.set_title('6. Tipping Point: Network Effects vs Switching Costs')
    ax6.legend(loc='center left', fontsize=7)

    # ── Panel 7: Dual-Currency Solow Model ──
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(yrs, econ.y_capita, color=C['crypto'], linewidth=2.5, label='Dual-Currency Path')
    ax7.plot(yrs, econ.y_capita_fiat_only, color=C['fiat'], linewidth=2, linestyle='--', label='Fiat-Only Counterfactual')
    ax7.fill_between(yrs, econ.y_capita_fiat_only, econ.y_capita,
        where=econ.y_capita > econ.y_capita_fiat_only, alpha=0.2, color=C['crypto'])
    ax7.set_xlabel('Year')
    ax7.set_ylabel('Output per Capita (2025 = 100)')
    ax7.set_title('7. Dual-Currency Extended Solow Model (Section 6)')
    ax7.legend(loc='upper left', fontsize=7)
    # Annotate divergence
    idx_2045 = np.where(yrs == 2045)[0][0]
    div = econ.y_capita[idx_2045] - econ.y_capita_fiat_only[idx_2045]
    ax7.annotate(f'+{div:.1f}pts by 2045',
        xy=(2045, econ.y_capita[idx_2045]), xytext=(2038, econ.y_capita[idx_2045] + 5),
        fontsize=8, fontweight='bold', color=C['crypto'],
        arrowprops=dict(arrowstyle='->', color=C['crypto']))

    # ── Panel 8: Piketty r > g via Three Flows ──
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(yrs, econ.r_circular * 100, color=C['circular'], linewidth=2, label='r (Circular Flow returns)')
    ax8.plot(yrs, econ.g_gradient * 100, color=C['gradient'], linewidth=2, label='g (Gradient Flow growth)')
    ax8.fill_between(yrs, econ.g_gradient * 100, econ.r_circular * 100,
        alpha=0.15, color=C['purple'])
    ax8.set_xlabel('Year')
    ax8.set_ylabel('Rate (%)')
    ax8.set_title('8. Piketty r > g Reinterpreted (Piketty 2014/Mostaque 2025)')
    ax8.legend(loc='center right', fontsize=7)
    # Annotate gap
    idx_2040 = np.where(yrs == 2040)[0][0]
    ax8.annotate(f'r−g = {econ.rg_gap[idx_2040]*100:.1f}pp',
        xy=(2040, (econ.r_circular[idx_2040] + econ.g_gradient[idx_2040]) / 2 * 100),
        fontsize=8, fontweight='bold', color=C['purple'], ha='center')

    # ── Panel 9: Government Scenarios ──
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(yrs, econ.scen_accommodate['tax_rev'] / 1e9, color=C['growth'], linewidth=2, label='Accommodate (tax rev $B)')
    ax9.plot(yrs, econ.scen_adapt['tax_rev'] / 1e9, color=C['ai'], linewidth=2, linestyle='--', label='Adapt/CBDC (tax rev $B)')
    ax9.plot(yrs, econ.scen_resist['growth_loss'] * 100 * 50, color=C['danger'], linewidth=2, linestyle=':', label='Resist (growth loss × 50)')
    ax9.set_xlabel('Year')
    ax9.set_ylabel('$B / Index')
    ax9.set_title('9. Government Response Scenarios (Section 4)')
    ax9.legend(loc='upper left', fontsize=7)

    # ── Panel 10: Lucas Critique — Policy Effectiveness ──
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.plot(yrs, econ.policy_eff_naive * 100, color=C['ai'], linewidth=2, linestyle='--', label='Central Bank Expectation')
    ax10.plot(yrs, econ.policy_eff_actual * 100, color=C['danger'], linewidth=2, label='Actual Effectiveness')
    ax10.fill_between(yrs, econ.policy_eff_actual * 100, econ.policy_eff_naive * 100,
        alpha=0.2, color=C['danger'], label='Lucas Gap')
    ax10.set_xlabel('Year')
    ax10.set_ylabel('Policy Effectiveness (%)')
    ax10.set_title('10. The Lucas Critique: Monetary Policy Blindspot (Lucas 1976)')
    ax10.legend(loc='lower left', fontsize=7)
    ax10.set_ylim(0, 105)

    # ── Panel 11: Perez Cycle ──
    ax11 = fig.add_subplot(gs[3, 1])
    phase_colors = {0: '#CCCCCC', 1: C['danger'], 2: C['purple'], 3: C['growth']}
    phase_labels = {0: 'Pre-revolution', 1: 'Installation\n(speculation)', 2: 'Turning\nPoint', 3: 'Deployment\n(golden age)'}
    for phase_val in [0, 1, 2, 3]:
        mask = econ.perez_phase == phase_val
        if mask.any():
            ax11.fill_between(yrs, 0, econ.perez_intensity, where=mask,
                alpha=0.4, color=phase_colors[phase_val], label=phase_labels[phase_val])
    ax11.plot(yrs, econ.perez_intensity, color='black', linewidth=1.5)
    ax11.plot(yrs, econ.theta, color=C['crypto'], linewidth=2, linestyle='--', label='Crypto adoption θ(t)')
    ax11.set_xlabel('Year')
    ax11.set_ylabel('Intensity / Share')
    ax11.set_title('11. Perez Cycle with Compression (Perez 2002/Mostaque 2025)')
    ax11.legend(loc='upper left', fontsize=6.5, ncol=2)

    # ── Panel 12: Developing vs Advanced Economy Paths ──
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.plot(yrs, econ.advanced_welfare, color=C['ai'], linewidth=2, label='Advanced Economy')
    ax12.plot(yrs, econ.developing_accommodate, color=C['growth'], linewidth=2.5, label='Developing + Accommodate')
    ax12.plot(yrs, econ.developing_resist, color=C['danger'], linewidth=2, linestyle='--', label='Developing + Resist')
    ax12.fill_between(yrs, econ.developing_resist, econ.developing_accommodate,
        alpha=0.1, color=C['growth'])
    ax12.set_xlabel('Year')
    ax12.set_ylabel('Welfare Index (Advanced 2020 = 100)')
    ax12.set_title('12. The Fiat Quality Gap: Developing Economy Paths (Section 5.2)')
    ax12.legend(loc='upper left', fontsize=7)
    # Annotate divergence
    idx_2045 = np.where(yrs == 2045)[0][0]
    ax12.annotate(f'Accommodate: {econ.developing_accommodate[idx_2045]:.0f}\nResist: {econ.developing_resist[idx_2045]:.0f}',
        xy=(2045, econ.developing_accommodate[idx_2045]),
        xytext=(2037, econ.developing_accommodate[idx_2045] * 0.7),
        fontsize=7, color=C['growth'],
        arrowprops=dict(arrowstyle='->', color=C['growth'], lw=0.8))

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Dashboard saved: {savepath}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 3. SUMMARY TABLES (Paper Tables 1-8)
# ══════════════════════════════════════════════════════════════════════════════

def print_tables(econ: WorldEconomy):
    """Print all paper tables from the simulation."""
    milestone_years = [2025, 2030, 2035, 2040, 2045, 2050]
    indices = [np.where(econ.years == y)[0][0] for y in milestone_years if y <= econ.years[-1]]
    yrs_avail = [econ.years[i] for i in indices]

    def divider():
        print("=" * 90)

    divider()
    print("TABLE 1: AI Sector Growth Projections")
    divider()
    print(f"{'Year':<8} {'World GDP ($T)':<18} {'AI Sector ($B)':<18} {'AI % of GDP':<14}")
    for i, y in zip(indices, yrs_avail):
        print(f"{y:<8} {econ.gdp[i]/1e12:<18.1f} {econ.ai_sector[i]/1e9:<18,.0f} {econ.ai_gdp_share[i]*100:<14.2f}%")

    divider()
    print("TABLE 2: AI-Crypto Adoption S-Curve")
    divider()
    print(f"{'Year':<8} {'Crypto Share':<14}")
    for i, y in zip(indices, yrs_avail):
        print(f"{y:<8} {econ.theta[i]*100:<14.1f}%")

    divider()
    print("TABLE 3: Monetary Policy Impact")
    divider()
    print(f"{'Year':<8} {'AI-Crypto ($T)':<16} {'Fiat Exit (%GDP)':<18} {'Seigniorage Loss ($B)':<22} {'Policy Eff.':<12}")
    for i, y in zip(indices, yrs_avail):
        print(f"{y:<8} ${econ.ai_crypto_volume[i]/1e12:<15.1f} {econ.fiat_exit_ratio[i]*100:<18.1f}% ${econ.seigniorage_loss[i]/1e9:<21.0f} {econ.policy_effectiveness[i]:<12.3f}")

    divider()
    print("METABOLIC RIFT: Human vs AI Labor Cost")
    divider()
    print(f"{'Year':<8} {'Human ($/hr)':<14} {'AI ($/hr)':<14} {'Ratio (Human/AI)':<18}")
    for i, y in zip(indices, yrs_avail):
        print(f"{y:<8} ${econ.human_cost[i]:<13.2f} ${econ.ai_cost[i]:<13.6f} {econ.metabolic_ratio[i]:<18,.0f}x")

    divider()
    print("THREE FLOWS DECOMPOSITION (%)")
    divider()
    print(f"{'Year':<8} {'Gradient (Fiat)':<18} {'Circular (Crypto)':<20} {'Harmonic (Instit.)':<20}")
    for i, y in zip(indices, yrs_avail):
        print(f"{y:<8} {econ.gradient_share[i]*100:<18.1f} {econ.circular_share[i]*100:<20.1f} {econ.harmonic_share[i]*100:<20.1f}")

    divider()
    print("PIKETTY r > g VIA THREE FLOWS")
    divider()
    print(f"{'Year':<8} {'r (Circular)':<14} {'g (Gradient)':<14} {'r - g':<10}")
    for i, y in zip(indices, yrs_avail):
        print(f"{y:<8} {econ.r_circular[i]*100:<14.1f}% {econ.g_gradient[i]*100:<14.1f}% {econ.rg_gap[i]*100:<10.1f}pp")

    divider()
    print("LUCAS CRITIQUE: Policy Effectiveness Gap")
    divider()
    print(f"{'Year':<8} {'CB Expectation':<16} {'Actual':<14} {'Lucas Gap':<12}")
    for i, y in zip(indices, yrs_avail):
        print(f"{y:<8} {econ.policy_eff_naive[i]*100:<16.1f}% {econ.policy_eff_actual[i]*100:<14.1f}% {econ.lucas_gap[i]*100:<12.1f}pp")

    divider()
    print("DUAL-CURRENCY SOLOW MODEL")
    divider()
    print(f"{'Year':<8} {'θ (Share)':<12} {'Y/capita':<12} {'Y fiat-only':<14} {'Divergence':<12} {'s_eff':<10}")
    for i, y in zip(indices, yrs_avail):
        div = econ.y_capita[i] - econ.y_capita_fiat_only[i]
        print(f"{y:<8} {econ.theta[i]*100:<12.1f}% {econ.y_capita[i]:<12.1f} {econ.y_capita_fiat_only[i]:<14.1f} +{div:<12.1f} {econ.s_eff[i]:<10.4f}")

    divider()
    print("DEVELOPING ECONOMY DIVERGENCE")
    divider()
    print(f"{'Year':<8} {'Advanced':<12} {'Dev+Accommodate':<18} {'Dev+Resist':<14} {'Gap (A-R)':<12}")
    for i, y in zip(indices, yrs_avail):
        gap = econ.developing_accommodate[i] - econ.developing_resist[i]
        print(f"{y:<8} {econ.advanced_welfare[i]:<12.1f} {econ.developing_accommodate[i]:<18.1f} {econ.developing_resist[i]:<14.1f} {gap:<12.1f}")

    divider()
    print(f"\nCRITICAL THRESHOLDS:")
    # Find key years
    idx_50pct = np.where(econ.theta >= 0.50)[0]
    idx_5pct_exit = np.where(econ.fiat_exit_ratio >= 0.05)[0]
    idx_10pct_exit = np.where(econ.fiat_exit_ratio >= 0.10)[0]
    print(f"  Tipping point (~30% adoption):  ~{int(econ.tipping_share * 100)}% → ~2031")
    print(f"  Crypto majority in AI:          {econ.years[idx_50pct[0]] if len(idx_50pct) else 'N/A'}")
    print(f"  5% fiat exit:                   {econ.years[idx_5pct_exit[0]] if len(idx_5pct_exit) else 'N/A'}")
    print(f"  10% fiat exit:                  {econ.years[idx_10pct_exit[0]] if len(idx_10pct_exit) else 'N/A'}")
    divider()


# ══════════════════════════════════════════════════════════════════════════════
# 4. SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def sensitivity_analysis(savepath: str = None):
    """Run sensitivity analysis across key parameters."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Sensitivity Analysis: Key Parameter Variations', fontsize=14, fontweight='bold')

    # --- S-curve steepness k ---
    ax = axes[0, 0]
    for k_val, ls in [(0.5, ':'), (0.8, '-'), (1.2, '--')]:
        e = WorldEconomy(k_scurve=k_val)
        ax.plot(e.years, e.theta * 100, linewidth=2, linestyle=ls, label=f'k = {k_val}')
    ax.set_title('S-Curve Steepness (k)')
    ax.set_ylabel('Crypto Share (%)')
    ax.legend()

    # --- Inflection year t* ---
    ax = axes[0, 1]
    for ts, ls in [(2029, ':'), (2032, '-'), (2035, '--')]:
        e = WorldEconomy(t_star=ts)
        ax.plot(e.years, e.fiat_exit_ratio * 100, linewidth=2, linestyle=ls, label=f't* = {ts}')
    ax.set_title('Inflection Year (t*)')
    ax.set_ylabel('Fiat Exit Ratio (%GDP)')
    ax.legend()

    # --- AI CAGR ---
    ax = axes[0, 2]
    for cagr, ls in [(0.20, ':'), (0.306, '-'), (0.40, '--')]:
        e = WorldEconomy(ai_cagr=cagr)
        ax.plot(e.years, e.ai_sector / 1e12, linewidth=2, linestyle=ls, label=f'CAGR = {cagr*100:.0f}%')
    ax.set_title('AI Sector CAGR')
    ax.set_ylabel('AI Sector ($T)')
    ax.legend()

    # --- θ_max (asymptotic ceiling) ---
    ax = axes[1, 0]
    for tm, ls in [(0.45, ':'), (0.65, '-'), (0.85, '--')]:
        e = WorldEconomy(theta_max=tm)
        ax.plot(e.years, e.y_capita, linewidth=2, linestyle=ls, label=f'θ_max = {tm}')
    ax.set_title('Crypto Ceiling θ_max → Solow Output')
    ax.set_ylabel('Y/capita (2025=100)')
    ax.legend()

    # --- Smart contract deflation (Coasean) ---
    ax = axes[1, 1]
    for sc, ls in [(0.15, ':'), (0.30, '-'), (0.50, '--')]:
        e = WorldEconomy(smart_contract_deflation=sc)
        ax.plot(e.years, e.optimal_firm_size, linewidth=2, linestyle=ls, label=f'SC deflation = {sc*100:.0f}%/yr')
    ax.set_title('Coasean Firm Size vs Smart Contract Deflation')
    ax.set_ylabel('Optimal Firm Size (2025=100)')
    ax.legend()

    # --- Perez compression factor ---
    ax = axes[1, 2]
    for cf, ls in [(1.0, ':'), (0.6, '-'), (0.3, '--')]:
        e = WorldEconomy(compression_factor=cf)
        ax.plot(e.years, e.perez_intensity, linewidth=2, linestyle=ls, label=f'Compression = {cf}')
    ax.set_title('Perez Cycle Compression (Mostaque 2025)')
    ax.set_ylabel('Speculative Intensity')
    ax.legend()

    for ax in axes.flat:
        ax.set_xlabel('Year')

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Sensitivity analysis saved: {savepath}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 5. GAME THEORY VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_game_theory(savepath: str = None):
    """Visualize game theory matrices and sequential game."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Game Theory of Regulatory Competition (Section 5)', fontsize=14, fontweight='bold')

    # --- Symmetric Game ---
    ax = axes[0]
    data = np.array([[35, 50], [-30, 25]])
    im = ax.imshow(data, cmap='RdYlGn', vmin=-40, vmax=60, aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['B: Accommodate', 'B: Resist'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['A: Accommodate', 'A: Resist'])
    for i in range(2):
        for j in range(2):
            payoffs = [(35,35),(50,-30)][j] if i == 0 else [(-30,50),(25,25)][j]
            ax.text(j, i, f'({payoffs[0]}, {payoffs[1]})', ha='center', va='center',
                fontsize=12, fontweight='bold', color='black')
    ax.set_title('Symmetric Game\n(Nash Eq: A,A)', fontweight='bold')
    # Highlight Nash
    rect = plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor='blue', linewidth=3)
    ax.add_patch(rect)

    # --- Asymmetric Game ---
    ax = axes[1]
    data = np.array([[35, 45], [-30, 25]])
    im = ax.imshow(data, cmap='RdYlGn', vmin=-40, vmax=80, aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Dev: Accommodate', 'Dev: Resist'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Adv: Accommodate', 'Adv: Resist'])
    asym_payoffs = [[(35,55),(45,-5)],[(-30,70),(25,-15)]]
    for i in range(2):
        for j in range(2):
            p = asym_payoffs[i][j]
            ax.text(j, i, f'({p[0]}, {p[1]})', ha='center', va='center',
                fontsize=12, fontweight='bold', color='black')
    ax.set_title('Asymmetric Game\n(Dev benefits more)', fontweight='bold')
    rect = plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor='blue', linewidth=3)
    ax.add_patch(rect)

    # --- Sequential Payoff Comparison ---
    ax = axes[2]
    strategies = ['First Mover\n(US)', 'Fast Follower', 'Late Adopter', 'Resistor']
    payoffs = [50, 35, 15, -30]
    colors = [C['growth'], C['ai'], C['crypto'], C['danger']]
    bars = ax.bar(strategies, payoffs, color=colors, edgecolor='white', linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Welfare Payoff')
    ax.set_title('Sequential Game:\nStackelberg First-Mover Advantage', fontweight='bold')
    for bar, val in zip(bars, payoffs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:+d}', ha='center', fontweight='bold', fontsize=11)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Game theory saved: {savepath}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  THE MONETARY SCHISM: Comprehensive Simulation Model        ║")
    print("║  EC 118 Quantitative Economic Growth | Tufts University     ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Run simulation
    econ = WorldEconomy()

    # Print all tables
    print_tables(econ)

    # Generate visualizations
    output_dir = "/home/jonsmirl/thesis/figures/monetary_schism"
    os.makedirs(output_dir, exist_ok=True)

    plot_dashboard(econ, f"{output_dir}/monetary_schism_dashboard.png")
    sensitivity_analysis(f"{output_dir}/monetary_schism_sensitivity.png")
    plot_game_theory(f"{output_dir}/monetary_schism_game_theory.png")

    print(f"\n✓ All outputs saved to {output_dir}/")
    print("  - monetary_schism_dashboard.png      (12-panel master dashboard)")
    print("  - monetary_schism_sensitivity.png     (6-panel sensitivity analysis)")
    print("  - monetary_schism_game_theory.png     (game theory matrices)")
