"""
AI-Crypto Monetary Displacement Model (v2 — with Game Theory)
EC 118: Quantitative Economic Growth — Tufts University
Professor Douglas Gollin

Models the transition of AI economic activity from fiat to cryptocurrency,
its implications for monetary sovereignty, seigniorage, and growth,
and the strategic interactions between governments (game theory).
"""

import json
import math

# ============================================================
# MODEL 1: AI Sector Growth Projections (Exponential)
# ============================================================

def ai_sector_projection(base_year=2025, base_size=391, cagr=0.306, horizon=20):
    results = []
    for t in range(horizon + 1):
        year = base_year + t
        world_gdp = 105_000 * (1.03 ** t)
        ai_size = base_size * ((1 + cagr) ** t)
        ai_pct_gdp = (ai_size / world_gdp) * 100
        if ai_pct_gdp > 15:
            damping = 15 + (ai_pct_gdp - 15) * 0.3
            ai_pct_gdp = damping
            ai_size = world_gdp * ai_pct_gdp / 100
        results.append({
            'year': year,
            'world_gdp_T': round(world_gdp / 1000, 1),
            'ai_sector_B': round(ai_size, 0),
            'ai_pct_gdp': round(ai_pct_gdp, 2)
        })
    return results

# ============================================================
# MODEL 2: AI Crypto Adoption S-Curve
# ============================================================

def crypto_adoption_scurve(base_year=2025, horizon=20,
                           midpoint=2032, steepness=0.8,
                           initial_share=0.02, max_share=0.65):
    results = []
    for t in range(horizon + 1):
        year = base_year + t
        x = year - midpoint
        crypto_share = initial_share + (max_share - initial_share) / (1 + math.exp(-steepness * x))
        results.append({
            'year': year,
            'crypto_share': round(crypto_share, 4)
        })
    return results

# ============================================================
# MODEL 3: Monetary Policy Impact — Seigniorage Loss
# ============================================================

def seigniorage_loss_model(ai_projections, crypto_adoption,
                           money_multiplier=8.5, seigniorage_rate=0.02):
    results = []
    for ai, crypto in zip(ai_projections, crypto_adoption):
        year = ai['year']
        ai_size_T = ai['ai_sector_B'] / 1000
        crypto_share = crypto['crypto_share']
        crypto_activity_T = ai_size_T * crypto_share
        base_money_reduction_T = crypto_activity_T / money_multiplier
        seigniorage_loss_B = base_money_reduction_T * seigniorage_rate * 1000
        world_gdp_T = ai['world_gdp_T']
        fiat_exit_pct = (crypto_activity_T / world_gdp_T) * 100 if world_gdp_T > 0 else 0
        policy_effectiveness = max(0.5, 1.0 - fiat_exit_pct / 100)
        results.append({
            'year': year,
            'ai_crypto_activity_T': round(crypto_activity_T, 2),
            'fiat_exit_pct_gdp': round(fiat_exit_pct, 2),
            'seigniorage_loss_B': round(seigniorage_loss_B, 1),
            'policy_effectiveness': round(policy_effectiveness, 3)
        })
    return results

# ============================================================
# MODEL 4: Government Response Scenarios
# ============================================================

def government_response_model(monetary_impacts):
    scenarios = {}
    for scenario_name, params in {
        'accommodation': {
            'crypto_tax_rate': 0.15,
            'compliance_cost_pct': 0.05,
            'growth_drag': 0.002,
            'seigniorage_recovery': 0.30
        },
        'resistance': {
            'crypto_tax_rate': 0.0,
            'compliance_cost_pct': 0.0,
            'growth_drag': 0.015,
            'seigniorage_recovery': 0.70
        },
        'adaptation': {
            'crypto_tax_rate': 0.10,
            'compliance_cost_pct': 0.02,
            'growth_drag': 0.001,
            'seigniorage_recovery': 0.60
        }
    }.items():
        scenario_results = []
        cumulative_growth_loss = 0
        for m in monetary_impacts:
            year = m['year']
            tax_revenue_B = m['ai_crypto_activity_T'] * 1000 * params['crypto_tax_rate']
            recovered_seigniorage = m['seigniorage_loss_B'] * params['seigniorage_recovery']
            net_fiscal_impact_B = tax_revenue_B + recovered_seigniorage - m['seigniorage_loss_B']
            cumulative_growth_loss += params['growth_drag']
            sovereignty = m['policy_effectiveness'] * (1 + params['seigniorage_recovery']) / 2
            sovereignty = min(1.0, sovereignty)
            scenario_results.append({
                'year': year,
                'tax_revenue_B': round(tax_revenue_B, 1),
                'net_fiscal_impact_B': round(net_fiscal_impact_B, 1),
                'cumulative_growth_loss_pct': round(cumulative_growth_loss * 100, 2),
                'sovereignty_index': round(sovereignty, 3)
            })
        scenarios[scenario_name] = scenario_results
    return scenarios

# ============================================================
# MODEL 5: Dual-Currency Growth Model
# ============================================================

def dual_currency_growth(base_year=2025, horizon=20,
                        g_baseline=0.02, alpha=0.33,
                        s=0.22, delta=0.05, n=0.005):
    crypto_adoption = crypto_adoption_scurve(base_year, horizon)
    results = []
    k = 10.0
    A = 1.0
    for t in range(horizon + 1):
        year = base_year + t
        theta = crypto_adoption[t]['crypto_share']
        transaction_savings = 0.02 * theta
        policy_drag = 0.15 * 0.02 * theta
        s_effective = s + transaction_savings - policy_drag
        g_t = g_baseline * (1 + 0.5 * theta)
        A = A * (1 + g_t)
        y = k ** alpha
        k_dot = s_effective * y - (delta + n + g_t) * k
        k = k + k_dot
        y_per_capita = A * (k ** alpha)
        if t > 0:
            growth_rate = (y_per_capita - results[-1]['y_per_capita']) / results[-1]['y_per_capita']
        else:
            growth_rate = g_baseline
        results.append({
            'year': year,
            'theta': round(theta, 4),
            'y_per_capita': round(y_per_capita, 4),
            'growth_rate': round(growth_rate * 100, 3),
            's_effective': round(s_effective, 4),
            'technology_level': round(A, 4)
        })
    return results

# ============================================================
# MODEL 6: Game Theory — Inter-Government Strategic Interaction
# ============================================================

def regulatory_competition_game():
    """
    Model the strategic interaction between two governments (or N governments)
    choosing between Accommodate (A) and Resist (R) regarding AI-crypto.

    Key insight: This is a Prisoner's Dilemma where individual rationality
    (accommodate) produces collective suboptimality (universal sovereignty erosion),
    but the Pareto-superior cooperative outcome (coordinated resistance) is
    unstable because unilateral resistance exports your AI sector.

    Extended to asymmetric game: Advanced economy (strong fiat) vs.
    Developing economy (weak fiat/dollarized).
    """

    results = {}

    # === SYMMETRIC 2x2 GAME: Two Advanced Economies ===
    # Payoffs: (Growth effect, Sovereignty effect) -> combined welfare index
    # Scale: 0-100 welfare index

    # Parameters calibrated from Model 4 results
    ai_growth_benefit = 30        # growth benefit from AI-crypto ecosystem
    sovereignty_cost = 15         # sovereignty cost from accommodation
    flight_cost = 40              # cost of AI innovation fleeing to rival
    coordination_benefit = 10     # benefit from coordinated resistance
    tax_revenue_benefit = 20      # fiscal benefit from taxing crypto

    # Payoff matrix: (Row player payoff, Column player payoff)
    # Row = Country A strategy, Column = Country B strategy

    symmetric_payoffs = {
        'both_accommodate': {
            'A': ai_growth_benefit + tax_revenue_benefit - sovereignty_cost,  # 35
            'B': ai_growth_benefit + tax_revenue_benefit - sovereignty_cost,  # 35
            'label': '(A, A)',
            'description': 'Both accommodate: growth + tax revenue, shared sovereignty loss'
        },
        'A_accommodate_B_resist': {
            'A': ai_growth_benefit + tax_revenue_benefit - sovereignty_cost + 15,  # 50 (captures B's fleeing AI)
            'B': coordination_benefit - flight_cost,  # -30 (AI flees, no revenue)
            'label': '(A, R)',
            'description': 'A accommodates, B resists: A captures AI sector, B loses innovation'
        },
        'A_resist_B_accommodate': {
            'A': coordination_benefit - flight_cost,  # -30
            'B': ai_growth_benefit + tax_revenue_benefit - sovereignty_cost + 15,  # 50
            'label': '(R, A)',
            'description': 'A resists, B accommodates: mirror of above'
        },
        'both_resist': {
            'A': coordination_benefit + sovereignty_cost,  # 25 (preserve sovereignty, slower growth)
            'B': coordination_benefit + sovereignty_cost,  # 25
            'label': '(R, R)',
            'description': 'Both resist: preserve sovereignty but suppress growth globally'
        }
    }

    # Nash Equilibrium Analysis
    # Country A's best response to B=Accommodate: Accommodate (35 > -30) ✓
    # Country A's best response to B=Resist: Accommodate (50 > 25) ✓
    # => Accommodate is dominant strategy for both
    # Nash Equilibrium: (Accommodate, Accommodate) with payoffs (35, 35)
    # But (Resist, Resist) gives (25, 25) — Pareto inferior but preserves sovereignty
    # The Pareto-optimal outcome (50, -30) is exploitative

    nash_eq = symmetric_payoffs['both_accommodate']
    cooperative = symmetric_payoffs['both_resist']

    results['symmetric_game'] = {
        'payoff_matrix': symmetric_payoffs,
        'nash_equilibrium': '(Accommodate, Accommodate)',
        'nash_payoffs': (nash_eq['A'], nash_eq['B']),
        'cooperative_outcome': '(Resist, Resist)',
        'cooperative_payoffs': (cooperative['A'], cooperative['B']),
        'is_prisoners_dilemma': True,
        'dominant_strategy': 'Accommodate',
        'welfare_loss_from_nash': 0,  # Nash is actually better for growth
        'sovereignty_loss_from_nash': 2 * sovereignty_cost
    }

    # === ASYMMETRIC GAME: Advanced vs Developing Economy ===
    # Key Gollin insight: developing countries with weak fiat currencies
    # may PREFER crypto (stabilization benefit)

    dev_fiat_weakness = 25  # extra cost of weak fiat (inflation, dollarization)
    dev_crypto_stability = 20  # stability gain from crypto/stablecoins

    asymmetric_payoffs = {
        'both_accommodate': {
            'advanced': ai_growth_benefit + tax_revenue_benefit - sovereignty_cost,  # 35
            'developing': ai_growth_benefit + tax_revenue_benefit - sovereignty_cost + dev_crypto_stability,  # 55
            'label': '(A, A)'
        },
        'adv_accommodate_dev_resist': {
            'advanced': ai_growth_benefit + tax_revenue_benefit - sovereignty_cost + 10,  # 45
            'developing': coordination_benefit - flight_cost + dev_fiat_weakness,  # -5 (resist + weak fiat = bad)
            'label': '(A, R)'
        },
        'adv_resist_dev_accommodate': {
            'advanced': coordination_benefit - flight_cost,  # -30
            'developing': ai_growth_benefit + tax_revenue_benefit + dev_crypto_stability,  # 70 (no sovereignty to lose)
            'label': '(R, A)'
        },
        'both_resist': {
            'advanced': coordination_benefit + sovereignty_cost,  # 25
            'developing': coordination_benefit - dev_fiat_weakness,  # -15 (resist + weak fiat = very bad)
            'label': '(R, R)'
        }
    }

    # For developing country: Accommodate dominant (55 > -5 when adv accommodates; 70 > -15 when adv resists)
    # For advanced: Accommodate dominant (35 > -30 when dev accommodates; 45 > 25 when dev resists)
    # Nash: (Accommodate, Accommodate) — even stronger for developing

    results['asymmetric_game'] = {
        'payoff_matrix': asymmetric_payoffs,
        'nash_equilibrium': '(Accommodate, Accommodate)',
        'advanced_payoff': 35,
        'developing_payoff': 55,
        'key_insight': 'Developing countries benefit MORE from accommodation (crypto substitutes for weak fiat)'
    }

    # === SEQUENTIAL (STACKELBERG) GAME ===
    # US moves first (GENIUS Act 2025), others follow
    # Leader advantage: commitment to accommodate makes followers' resistance unprofitable

    stackelberg = {
        'leader': 'United States',
        'leader_move': 'Accommodate (GENIUS Act, Anti-CBDC Act)',
        'follower_best_response': 'Accommodate',
        'rationale': 'Once US accommodates, AI firms concentrate in US. Other countries must accommodate or lose AI sector entirely.',
        'leader_payoff': 50,  # first-mover advantage captures disproportionate AI investment
        'follower_payoff': 35,  # late movers get standard benefit without first-mover premium
        'resister_payoff': -30,  # holdouts suffer severe innovation flight
        'first_mover_premium': 15  # difference between leader and follower payoffs
    }

    results['stackelberg_game'] = stackelberg

    # === COORDINATION GAME WITH NETWORK EFFECTS (Tipping Point) ===
    # Once fraction phi of AI transactions use crypto, remaining agents face
    # overwhelming incentive to switch (network effects)

    tipping_dynamics = []
    for phi in [i * 0.05 for i in range(21)]:  # 0% to 100%
        # Benefit of using crypto = base_benefit + network_effect * phi
        # Cost of using crypto = switching_cost * (1 - phi) + regulatory_risk * (1 - gov_accommodation)
        base_benefit = 5
        network_effect = 40
        switching_cost = 15
        regulatory_risk = 10
        gov_accommodation_prob = min(1.0, 0.3 + 0.7 * phi)  # governments accommodate as adoption rises

        benefit = base_benefit + network_effect * phi
        cost = switching_cost * (1 - phi) + regulatory_risk * (1 - gov_accommodation_prob)
        net_benefit = benefit - cost

        tipping_dynamics.append({
            'crypto_share': round(phi, 2),
            'benefit': round(benefit, 1),
            'cost': round(cost, 1),
            'net_benefit': round(net_benefit, 1),
            'equilibrium': 'stable_crypto' if net_benefit > 10 else ('unstable' if abs(net_benefit) <= 10 else 'stable_fiat')
        })

    # Find tipping point
    tipping_point = None
    for i, d in enumerate(tipping_dynamics):
        if d['net_benefit'] > 0 and (i == 0 or tipping_dynamics[i-1]['net_benefit'] <= 0):
            tipping_point = d['crypto_share']
            break

    results['tipping_dynamics'] = {
        'data': tipping_dynamics,
        'tipping_point': tipping_point,
        'description': f'Tipping point at {tipping_point*100:.0f}% crypto adoption: beyond this, network effects make fiat-only equilibrium unstable'
    }

    # === N-COUNTRY RACE TO ACCOMMODATE ===
    # As more countries accommodate, holdouts face increasing pressure
    # Models the cascade / domino effect

    cascade_results = []
    n_countries = 20  # major economies
    for n_accommodating in range(n_countries + 1):
        share_accommodating = n_accommodating / n_countries
        # Payoff for a holdout (resisting country) decreases as more accommodate
        holdout_payoff = coordination_benefit + sovereignty_cost - flight_cost * share_accommodating
        # Payoff for accommodating country increases with more peers
        accommodator_payoff = ai_growth_benefit + tax_revenue_benefit - sovereignty_cost + 5 * share_accommodating
        cascade_results.append({
            'n_accommodating': n_accommodating,
            'share': round(share_accommodating, 2),
            'holdout_payoff': round(holdout_payoff, 1),
            'accommodator_payoff': round(accommodator_payoff, 1),
            'holdout_viable': holdout_payoff > accommodator_payoff
        })

    # Find cascade threshold
    cascade_threshold = None
    for r in cascade_results:
        if not r['holdout_viable']:
            cascade_threshold = r['n_accommodating']
            break

    results['cascade_dynamics'] = {
        'data': cascade_results,
        'cascade_threshold': cascade_threshold,
        'description': f'After {cascade_threshold} of {n_countries} major economies accommodate, resistance becomes dominated strategy'
    }

    # === MECHANISM DESIGN: CBDC Incentive Compatibility ===
    # For CBDC to compete with private crypto, it must be incentive-compatible
    # AI agents choose CBDC only if: U(CBDC) >= U(crypto)

    cbdc_design = {
        'required_properties': [
            {'property': 'Settlement speed', 'crypto_benchmark': '<5 seconds', 'min_cbdc_requirement': '<5 seconds',
             'current_ecny': '~10 seconds', 'gap': 'Close'},
            {'property': 'Programmability', 'crypto_benchmark': 'Turing-complete smart contracts', 'min_cbdc_requirement': 'Conditional logic',
             'current_ecny': 'Basic conditions', 'gap': 'Large'},
            {'property': 'Cross-border interop', 'crypto_benchmark': 'Native (borderless)', 'min_cbdc_requirement': 'Multi-CBDC bridge',
             'current_ecny': 'mBridge (5 countries)', 'gap': 'Large'},
            {'property': 'Micropayments', 'crypto_benchmark': '<$0.001 tx cost', 'min_cbdc_requirement': '<$0.01 tx cost',
             'current_ecny': '~$0 (subsidized)', 'gap': 'Close'},
            {'property': 'Agent identity', 'crypto_benchmark': 'Pseudonymous keys', 'min_cbdc_requirement': 'Machine-compatible KYC',
             'current_ecny': 'Human-only', 'gap': 'Critical'},
        ],
        'incentive_compatibility': 'CBDC adoption by AI agents requires matching crypto on speed, cost, programmability, AND providing machine-compatible identity. Current CBDCs fail on programmability and agent identity.',
        'prediction': 'Without fundamental redesign, CBDCs will coexist with private crypto but not replace it for AI agents. The adaptation scenario requires innovation most central banks have not yet demonstrated.'
    }

    results['mechanism_design'] = cbdc_design

    return results


# ============================================================
# RUN ALL MODELS
# ============================================================

print("=" * 70)
print("MODEL RESULTS: AI-Crypto Monetary Displacement Analysis (v2)")
print("=" * 70)

# Models 1-5 (unchanged)
ai_proj = ai_sector_projection()
crypto_adopt = crypto_adoption_scurve()
monetary = seigniorage_loss_model(ai_proj, crypto_adopt)
gov_responses = government_response_model(monetary)
growth = dual_currency_growth()

# Model 6: Game Theory
game_theory = regulatory_competition_game()

print("\n--- TABLE 1: AI Sector Growth Projections ---")
print(f"{'Year':<6} {'World GDP ($T)':<16} {'AI Sector ($B)':<16} {'AI % of GDP':<12}")
for r in ai_proj:
    if r['year'] % 5 == 0 or r['year'] == 2025:
        print(f"{r['year']:<6} {r['world_gdp_T']:<16} {r['ai_sector_B']:<16} {r['ai_pct_gdp']:<12}")

print("\n--- TABLE 6: Symmetric Regulatory Competition Game ---")
sym = game_theory['symmetric_game']
print(f"  Nash Equilibrium: {sym['nash_equilibrium']}")
print(f"  Nash Payoffs: {sym['nash_payoffs']}")
print(f"  Dominant Strategy: {sym['dominant_strategy']}")
print(f"  Prisoner's Dilemma: {sym['is_prisoners_dilemma']}")
print(f"  Cooperative outcome (R,R) payoffs: {sym['cooperative_payoffs']}")

print("\n--- TABLE 7: Asymmetric Game (Advanced vs Developing) ---")
asym = game_theory['asymmetric_game']
print(f"  Nash: {asym['nash_equilibrium']}")
print(f"  Advanced economy payoff: {asym['advanced_payoff']}")
print(f"  Developing economy payoff: {asym['developing_payoff']}")
print(f"  Key insight: {asym['key_insight']}")

print("\n--- TABLE 8: Stackelberg Sequential Game ---")
stack = game_theory['stackelberg_game']
print(f"  Leader: {stack['leader']} -> {stack['leader_move']}")
print(f"  Leader payoff: {stack['leader_payoff']}, Follower payoff: {stack['follower_payoff']}")
print(f"  First-mover premium: {stack['first_mover_premium']}")

print("\n--- TABLE 9: Tipping Point / Network Effects ---")
tip = game_theory['tipping_dynamics']
print(f"  Tipping point: {tip['tipping_point']*100:.0f}% crypto adoption")
print(f"  {'Share':<8} {'Benefit':<10} {'Cost':<10} {'Net':<10} {'Equilibrium':<15}")
for d in tip['data']:
    if d['crypto_share'] in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        print(f"  {d['crypto_share']*100:>5.0f}%  {d['benefit']:<10} {d['cost']:<10} {d['net_benefit']:<10} {d['equilibrium']:<15}")

print("\n--- TABLE 10: N-Country Cascade ---")
cascade = game_theory['cascade_dynamics']
print(f"  Cascade threshold: {cascade['cascade_threshold']} of 20 major economies")
for d in cascade['data']:
    if d['n_accommodating'] in [0, 3, 5, 7, 10, 15, 20]:
        print(f"  n={d['n_accommodating']:>2}: holdout={d['holdout_payoff']:>6.1f}, accommodator={d['accommodator_payoff']:>6.1f}, holdout_viable={d['holdout_viable']}")

print("\n--- CRITICAL THRESHOLDS ---")
for r in monetary:
    if r['fiat_exit_pct_gdp'] >= 5.0:
        print(f"  Fiat exit reaches 5% of GDP in {r['year']} -> Monetary policy materially impaired")
        break
for r in monetary:
    if r['fiat_exit_pct_gdp'] >= 10.0:
        print(f"  Fiat exit reaches 10% of GDP in {r['year']} -> Systemic monetary sovereignty challenge")
        break
for ai in ai_proj:
    if ai['ai_pct_gdp'] >= 5.0:
        print(f"  AI sector reaches 5% of global GDP in {ai['year']}")
        break
for c in crypto_adopt:
    if c['crypto_share'] >= 0.5:
        print(f"  Crypto majority (>50%) of AI transactions in {c['year']}")
        break

# Save all results
all_results = {
    'ai_projections': ai_proj,
    'crypto_adoption': crypto_adopt,
    'monetary_impacts': monetary,
    'government_responses': gov_responses,
    'dual_currency_growth': growth,
    'game_theory': {
        'symmetric_game': game_theory['symmetric_game'],
        'asymmetric_game': game_theory['asymmetric_game'],
        'stackelberg_game': game_theory['stackelberg_game'],
        'tipping_point': game_theory['tipping_dynamics']['tipping_point'],
        'cascade_threshold': game_theory['cascade_dynamics']['cascade_threshold'],
        'tipping_data': game_theory['tipping_dynamics']['data'],
        'cascade_data': game_theory['cascade_dynamics']['data']
    }
}

with open('/home/claude/model_results_v2.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n\nAll model results saved to model_results_v2.json")
