import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
from scipy.signal import find_peaks

DATA_DIR = "data_store"

def load_regime_energy():
    """Loads energy consumption by source to identify GPT regimes."""
    print("Loading Energy Regime Data (Coal, Oil, Gas, Renewables)...")
    df = pd.read_csv(os.path.join(DATA_DIR, "owid_energy_data.csv"))
    df_world = df[df['country'] == 'World'].copy()
    
    # Specific GPT-related energy sources
    sources = {
        'coal_consumption': 'Coal (1st/2nd Industrial)',
        'oil_consumption': 'Oil (ICE/Petrochem)',
        'gas_consumption': 'Gas (Modern Industrial)',
        'renewables_consumption': 'Renewables (Post-Modern)',
        'nuclear_consumption': 'Nuclear (Cold War/High-Tech)'
    }
    
    cols = ['year'] + list(sources.keys())
    df_regime = df_world[cols].dropna(subset=['year']).fillna(0)
    return df_regime, sources

def analyze_regime_shift(df, sources):
    """Identifies the 'dominance' of different technology regimes over time."""
    # Normalize to share of total energy
    total = df[list(sources.keys())].sum(axis=1)
    for col in sources.keys():
        df[f"{col}_share"] = df[col] / total
    return df

def detect_innovation_bursts(years, residuals, threshold=1.5):
    """Finds significant pulses in sectoral residuals to label tech eras."""
    peaks, _ = find_peaks(residuals, height=threshold)
    return years[peaks], residuals[peaks]

def run_enhanced_analysis():
    # 1. Load Data
    df_patents = pd.read_csv(os.path.join(DATA_DIR, "world_bank_patents.csv"))
    df_patents = df_patents[['date', 'value']].dropna()
    df_patents.columns = ['year', 'patents']
    df_patents = df_patents.groupby('year').sum().reset_index()
    
    df_regime, sources = load_regime_energy()
    
    # 2. Merge
    df = pd.merge(df_regime, df_patents, on='year', how='outer').sort_values('year')
    df = df[df['year'] >= 1960].interpolate().dropna()
    
    # 3. Perform Multi-GPT Wavelet Analysis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18))
    plt.subplots_adjust(hspace=0.4)
    
    # --- PANEL 1: Energy GPT Regimes (The Successive Waves) ---
    for col, label in sources.items():
        # Wavelet smoothing of the energy source to find its "Regime Curve"
        signal = df[col].values.copy() # Ensure writable copy
        coeffs = pywt.wavedec(signal, 'db4', level=2)
        trend = pywt.waverec([coeffs[0]] + [None]*(len(coeffs)-1), 'db4')[:len(signal)]
        ax1.plot(df['year'], trend / trend.max(), label=label, linewidth=2)
    
    ax1.set_title("Successive Technological Regimes (Normalized Energy GPTs)")
    ax1.set_ylabel("Relative Dominance")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- PANEL 2: Patent Innovation Pulse (Sectoral Shocks) ---
    pat_signal = (df['patents'].values - np.mean(df['patents'])) / np.std(df['patents'])
    # Extract GPT Trend vs Residuals
    coeffs_p = pywt.wavedec(pat_signal, 'db4', level=2)
    pat_gpt = pywt.waverec([coeffs_p[0]] + [None]*(len(coeffs_p)-1), 'db4')[:len(pat_signal)]
    pat_res = pat_signal - pat_gpt
    
    ax2.plot(df['year'], pat_res, color='purple', label='Innovation Residuals (Shocks)')
    ax2.fill_between(df['year'], 0, pat_res, color='purple', alpha=0.2)
    
    # Detect and Label Peaks
    peak_years, peak_vals = detect_innovation_bursts(df['year'].values, pat_res)
    labels = {
        1995: "Internet/ICT Boom",
        2010: "Mobile/Cloud Pivot",
        2018: "AI/DL Surge"
    }
    for py, pv in zip(peak_years, peak_vals):
        label = labels.get(int(py), f"Innovation Pulse ({int(py)})")
        ax2.annotate(label, xy=(py, pv), xytext=(py, pv+0.5),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    ax2.set_title("Sectoral Innovation Shocks (Denoised Patent Residuals)")
    ax2.set_ylabel("Amplitude (Sigma)")
    ax2.grid(alpha=0.3)

    # --- PANEL 3: Scalogram of ICT Era (1980-Present) ---
    scales = np.arange(1, 15)
    coef, freqs = pywt.cwt(pat_res, scales, 'morl')
    im = ax3.imshow(np.abs(coef), extent=[df['year'].min(), df['year'].max(), 1, 15],
                    interpolation='nearest', aspect='auto', cmap='viridis')
    ax3.set_title("Time-Frequency Map of Patent Bursts")
    ax3.set_ylabel("Scale (GPT Duration)")
    ax3.set_xlabel("Year")
    fig.colorbar(im, ax=ax3, label="Innovation Intensity")

    # Save Output
    fig.savefig("enhanced_gpt_regime_analysis.png", dpi=300)
    print("\nENHANCED ANALYSIS COMPLETE.")
    print("Output saved to 'enhanced_gpt_regime_analysis.png'")
    print("1. Successive Regimes: Shows how Coal, Oil, and Renewables hand off dominance.")
    print("2. Innovation Shocks: Identifies the exact years when ICT and AI broke the trend.")
    print("3. Time-Frequency Map: Visualizes the 'width' of these technological revolutions.")

if __name__ == "__main__":
    run_enhanced_analysis()
