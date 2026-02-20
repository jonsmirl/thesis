import os
import json
import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm

FRED_API_KEY = os.environ.get('FRED_API_KEY')
if not FRED_API_KEY:
    print("Error: FRED_API_KEY environment variable not set.")
    exit(1)

START_DATE = '1987-01-01'

SECTORS = {
    'Total': {
        'Y_proxy': 'IPMAN', 
        'K_prod': 'MPU9900072', 
        'L_prod': 'MPU9900063'
    },
    'Durable': {
        'Y_proxy': 'IPDMAN', 
        'K_prod': 'MPU9920072', 
        'L_prod': 'MPU9920063'
    },
    'Nondurable': {
        'Y_proxy': 'IPNMAN', 
        'K_prod': 'MPU9910072', 
        'L_prod': 'MPU9910063'
    }
}

def fetch_series_to_df(series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&observation_start={START_DATE}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        dates = []
        values = []
        
        for obs in data.get('observations', []):
            val_str = obs['value']
            if val_str != '.':
                dates.append(obs['date'])
                values.append(float(val_str))
                
        series = pd.Series(values, index=pd.to_datetime(dates), name=series_id)
        return series
    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
        return pd.Series(dtype=float)

results = {}

print(f"{'Sector':<15} | {'Alpha_L':<10} | {'Alpha_K':<10} | {'Norm(Beta)':<10} | {'Metric (Beta/Alpha)':<20}")
print("-" * 80)

for name, components in SECTORS.items():
    try:
        # 1. Fetch Data
        df_list = []
        for key, series_id in components.items():
            s = fetch_series_to_df(series_id)
            if s.empty:
                raise ValueError(f"Empty series for {series_id}")
            s.name = key # Rename to Y_proxy, K_prod, etc.
            df_list.append(s)
            
        df = pd.concat(df_list, axis=1).dropna()
        
        # 2. Derive Variables
        # K = Y / (Y/K) = Y_proxy / K_prod
        # L = Y / (Y/L) = Y_proxy / L_prod
        
        # Filter out zero/negative values before logging
        df = df[(df['Y_proxy'] > 0) & (df['K_prod'] > 0) & (df['L_prod'] > 0)]
        
        df['K'] = df['Y_proxy'] / df['K_prod']
        df['L'] = df['Y_proxy'] / df['L_prod']
        
        # Log-transform
        df['y'] = np.log(df['Y_proxy'])
        df['k'] = np.log(df['K'])
        df['l'] = np.log(df['L'])
        df['t'] = np.arange(len(df)) # Time trend
        
        # Translog interaction terms
        df['ll'] = 0.5 * df['l']**2
        df['kk'] = 0.5 * df['k']**2
        df['lk'] = df['l'] * df['k']
        
        # 3. Regression
        X = df[['l', 'k', 'll', 'kk', 'lk', 't']]
        X = sm.add_constant(X)
        y = df['y']
        
        model = sm.OLS(y, X).fit()
        params = model.params
        
        # 4. Metric Calculation
        norm_alpha = np.sqrt(params['l']**2 + params['k']**2)
        norm_beta = np.sqrt(params['ll']**2 + params['kk']**2 + 2 * params['lk']**2)
        
        metric = norm_beta / norm_alpha if norm_alpha > 0 else 0
        results[name] = metric
        
        print(f"{name:<15} | {params['l']:.4f}     | {params['k']:.4f}     | {norm_beta:.4f}     | {metric:.4f}")
        
    except Exception as e:
        print(f"Error processing {name}: {e}")

print("-" * 80)

if 'Durable' in results and 'Nondurable' in results and 'Total' in results:
    avg_sub = (results['Durable'] + results['Nondurable']) / 2
    total = results['Total']

    print(f"Average Disaggregated Metric: {avg_sub:.4f}")
    print(f"Aggregated Metric (Total):    {total:.4f}")

    if total < avg_sub:
        print("RESULT: PASS. Aggregation reduced the relative magnitude of interaction terms.")
    else:
        print("RESULT: FAIL. Aggregation did not reduce interaction terms.")
else:
    print("Incomplete results.")
