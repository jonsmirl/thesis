#!/usr/bin/env python3
"""Filter RPW data to thesis-relevant corridors and compute quarterly averages.
Run: python filter_rpw.py
Output: thesis_data/rpw_filtered.csv (small enough to upload)
"""
import pandas as pd

df = pd.read_csv("thesis_data/rpw_data.csv")
print(f"Raw: {len(df)} rows, {df['corridor'].nunique()} corridors, periods: {df['period'].nunique()}")

# Thesis countries (ISO3 codes used in RPW)
THESIS = {'IND','NGA','VNM','PHL','UKR','PAK','BRA','THA','RUS','CHN','TUR','ARG',
          'COL','KEN','IDN','MEX','ZAF','BGD','EGY','VEN','GHA','TZA','USA','GBR',
          'DEU','JPN','KOR','SGP','ARE','KAZ','MYS','PER','ETH','MMR','NPL','LKA',
          'CMR','SEN','UGA','BOL','SLV',
          # Common source codes that may differ
          'AGO','NAM','SAU','KWT','QAT','OMN'}

# Filter: either source or destination is a thesis country
mask = df['source_code'].isin(THESIS) | df['destination_code'].isin(THESIS)
filtered = df[mask].copy()
print(f"Filtered to thesis countries: {len(filtered)} rows, {filtered['corridor'].nunique()} corridors")

# Keep key columns, compute corridor-quarter averages
cols = ['period', 'source_code', 'source_name', 'destination_code', 'destination_name',
        'firm_type', 'cc1 total cost %', 'cc2 total cost %', 'cc1 fx margin', 'cc2 fx margin',
        'corridor']
filtered = filtered[cols].copy()

# Average cost by corridor-period (across firms)
avg = filtered.groupby(['period', 'source_code', 'source_name', 'destination_code',
                         'destination_name', 'corridor']).agg({
    'cc1 total cost %': 'mean',  # $200 transfer cost
    'cc2 total cost %': 'mean',  # $500 transfer cost
    'cc1 fx margin': 'mean',
    'cc2 fx margin': 'mean',
    'firm_type': 'count',
}).rename(columns={
    'cc1 total cost %': 'avg_cost_pct_200usd',
    'cc2 total cost %': 'avg_cost_pct_500usd',
    'cc1 fx margin': 'avg_fx_margin_200usd',
    'cc2 fx margin': 'avg_fx_margin_500usd',
    'firm_type': 'num_providers',
}).reset_index()

avg.to_csv("thesis_data/rpw_filtered.csv", index=False)
print(f"\nOutput: {len(avg)} corridor-quarter observations")
print(f"Corridors: {avg['corridor'].nunique()}")
print(f"Periods: {sorted(avg['period'].unique())[:5]} ... {sorted(avg['period'].unique())[-3:]}")
print(f"File size: {pd.io.common.file_exists('thesis_data/rpw_filtered.csv') and __import__('os').path.getsize('thesis_data/rpw_filtered.csv')/1024:.0f} KB")

# Quick stats
print(f"\nAvg cost $200 transfer: {avg['avg_cost_pct_200usd'].mean():.2f}%")
print(f"Avg cost $500 transfer: {avg['avg_cost_pct_500usd'].mean():.2f}%")
print(f"\nTop 10 most expensive corridors (latest period):")
latest = avg[avg['period'] == avg['period'].max()]
print(latest.nlargest(10, 'avg_cost_pct_200usd')[['corridor','avg_cost_pct_200usd','num_providers']].to_string(index=False))
