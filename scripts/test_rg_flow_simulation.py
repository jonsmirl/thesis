import numpy as np
import statsmodels.api as sm
import pandas as pd

# Configuration
DEPTH = 5
N_SAMPLES = 2000
NOISE_SCALE = 0.5 # High initial noise at micro level

np.random.seed(42)

def build_tree_output(inputs):
    # If base layer (depth from top), we add noise?
    # No, inputs are just numbers.
    # The structure is the tree.
    
    # We need to know the depth to know if we add noise.
    # Let's write a recursive function with depth tracking.
    return aggregate(inputs, current_depth=0)

def aggregate(inputs, current_depth):
    if len(inputs) == 1:
        return inputs[0]
    
    mid = len(inputs) // 2
    left = aggregate(inputs[:mid], current_depth + 1)
    right = aggregate(inputs[mid:], current_depth + 1)
    
    lL = np.log(left)
    lR = np.log(right)
    
    # Base Aggregator is Cobb-Douglas (a=0.5)
    ln_y = 0.5*lL + 0.5*lR
    
    # Add Translog Noise ONLY at the bottom aggregation steps (closest to inputs)
    # Total Depth is 5.
    # If current_depth is close to DEPTH-1, add noise.
    # If current_depth is 0 (top), we want to see if it's clean.
    
    # Let's say the production function is complex ONLY at the micro level.
    # i.e., "Firms are complex, but Sectors are simple".
    # We add noise if current_depth >= DEPTH - 2
    
    if current_depth >= DEPTH - 2:
        b11 = np.random.normal(0, NOISE_SCALE)
        b22 = np.random.normal(0, NOISE_SCALE)
        b12 = np.random.normal(0, NOISE_SCALE)
        ln_y += 0.5*b11*lL**2 + 0.5*b22*lR**2 + b12*lL*lR
    
    return np.exp(ln_y)

# 1. Generate Data
N_INPUTS = 2**DEPTH
X_data = np.random.lognormal(0, 0.5, (N_SAMPLES, N_INPUTS))
Y_data = []

valid_indices = []
for i in range(N_SAMPLES):
    try:
        y = build_tree_output(X_data[i,:])
        if not np.isnan(y) and y > 0:
            Y_data.append(y)
            valid_indices.append(i)
    except:
        pass

Y_data = np.array(Y_data)
X_data = X_data[valid_indices, :]

# 2. Define Macro Inputs (K and L)
mid = N_INPUTS // 2
K_data = np.sum(X_data[:, :mid], axis=1)
L_data = np.sum(X_data[:, mid:], axis=1)

# 3. Fit Translog to Macro Data
df = pd.DataFrame({
    'y': np.log(Y_data),
    'k': np.log(K_data),
    'l': np.log(L_data)
})

# Translog form
df['kk'] = 0.5 * df['k']**2
df['ll'] = 0.5 * df['l']**2
df['lk'] = df['k'] * df['l']

X = df[['k', 'l', 'kk', 'll', 'lk']]
X = sm.add_constant(X)
model = sm.OLS(df['y'], X).fit()

print(f"\n--- RG Flow Simulation (Depth {DEPTH}, Inputs {N_INPUTS}) ---")
print("Micro Layer: Translog Noise (Scale 0.5). Macro Layer: Clean Aggregation.")

params = model.params
alpha_norm = np.sqrt(params['k']**2 + params['l']**2)
beta_norm = np.sqrt(params['kk']**2 + params['ll']**2 + 2*params['lk']**2)
metric_macro = beta_norm / alpha_norm

print(f"\nMacro Metric (Beta/Alpha): {metric_macro:.4f}")

# Theoretical Micro Baseline (Single Layer Translog with Scale 0.5)
# Alpha ~ 0.7
# Beta ~ sqrt(3 * 0.5^2) ~ 0.86
# Ratio ~ 1.2
micro_base_ratio = np.sqrt(3 * NOISE_SCALE**2) / np.sqrt(0.5**2 + 0.5**2)
print(f"Micro Baseline Ratio: {micro_base_ratio:.4f}")

if metric_macro < micro_base_ratio * 0.5:
    print(f"Result: PASS. Significant decay ({metric_macro:.4f} vs {micro_base_ratio:.4f}).")
else:
    print(f"Result: FAIL. Decay not sufficient.")
