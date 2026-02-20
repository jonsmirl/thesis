import numpy as np
import scipy.optimize as opt

# Configuration
N_FIRMS = 1000
TOTAL_RESOURCE = 10000.0
RHOS = [-5.0, -2.0, -0.5, 0.0, 0.5] # Different substitution parameters

def run_simulation(rho):
    print(f"\n--- Simulating Economy with Rho = {rho} ---")
    
    # 1. Random Productivities (A_i)
    # Log-normal distribution of firm productivities
    np.random.seed(42)
    A = np.random.lognormal(0, 1.0, N_FIRMS)
    
    # 2. Optimization Problem
    # Maximize Aggregate Output Y = ( sum (A_i * x_i)^rho )^(1/rho)
    # Subject to sum(x_i) = TOTAL_RESOURCE
    
    # Analytical Solution:
    # x_i ~ A_i^( rho / (1-rho) )
    # y_i = A_i * x_i ~ A_i^( 1 / (1-rho) )
    
    exponent = 1.0 / (1.0 - rho)
    print(f"Theoretical Firm Size Exponent (gamma): {exponent:.4f}")

    # Optimal firm sizes
    y_opt = A**exponent
    y_opt = y_opt * (TOTAL_RESOURCE / np.sum(y_opt**(1-rho))) # Normalization roughly

    # Actually, let's just compute the unnormalized distribution shape
    # The SHAPE depends only on A^exponent.
    
    # Let's measure the Gini coefficient of the resulting firm sizes y_opt.
    def gini(array):
        array = array.flatten()
        if np.amin(array) < 0:
            array -= np.amin(array)
        array += 0.0000001
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

    g = gini(y_opt)
    
    # Check for Heavy Tails (Kurtosis)
    from scipy.stats import kurtosis
    k = kurtosis(y_opt)
    
    print(f"Gini Coefficient of Firm Sizes: {g:.4f}")
    print(f"Excess Kurtosis: {k:.4f}")
    
    if k > 3.0:
        print("Result: Heavy Tailed (Leptokurtic)")
    else:
        print("Result: Thin Tailed")

    return g, k

print("--- CES Tail Exponent Simulation ---")
print("Prediction: Complements (rho < 0) equalize sizes? Substitutes (rho > 0) concentrate them?")

results = []
for r in RHOS:
    g, k = run_simulation(r)
    results.append((r, g, k))

print("\n--- Summary ---")
print(f"{'Rho':<10} | {'Sigma (1/1-r)':<15} | {'Gini':<10} | {'Kurtosis':<10}")
for r, g, k in results:
    sigma = 1/(1-r) if r != 1 else np.inf
    print(f"{r:<10.1f} | {sigma:<15.4f} | {g:<10.4f} | {k:<10.4f}")
