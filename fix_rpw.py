#!/usr/bin/env python3
"""
Fix RPW (Remittance Prices Worldwide) parsing.
Run from your thesis directory: python fix_rpw.py
"""
import pandas as pd
import os

RAW = "thesis_data/rpw_raw.xlsx"
OUT = "thesis_data/rpw_data.csv"

if not os.path.exists(RAW):
    print(f"ERROR: {RAW} not found. Run fetch_thesis_data.py first.")
    exit(1)

print(f"Inspecting {RAW} ({os.path.getsize(RAW)/1e6:.1f} MB)...")

# Step 1: List all sheets
xls = pd.ExcelFile(RAW)
print(f"Sheets: {xls.sheet_names}")

# Step 2: Try each sheet, find the one with actual data
best_sheet = None
best_rows = 0

for sheet in xls.sheet_names:
    try:
        # Read first 20 rows to inspect
        sample = pd.read_excel(xls, sheet_name=sheet, nrows=20, header=None)
        print(f"\n--- Sheet: '{sheet}' ---")
        print(f"  Shape (first 20 rows): {sample.shape}")
        # Show first few rows to find header
        for i in range(min(10, len(sample))):
            vals = [str(v)[:40] for v in sample.iloc[i] if pd.notna(v)]
            if vals:
                print(f"  Row {i}: {vals[:5]}{'...' if len(vals) > 5 else ''}")

        # Try to find the header row — look for common RPW column names
        header_keywords = ['source', 'destination', 'sending', 'receiving', 'corridor',
                           'firm', 'product', 'fee', 'cost', 'total', 'country', 'quarter', 'period']
        header_row = None
        for i in range(min(15, len(sample))):
            row_str = ' '.join(str(v).lower() for v in sample.iloc[i] if pd.notna(v))
            matches = sum(1 for kw in header_keywords if kw in row_str)
            if matches >= 3:
                header_row = i
                print(f"  >>> Likely header at row {i} ({matches} keyword matches)")
                break

        if header_row is not None:
            # Read full sheet with correct header
            df = pd.read_excel(xls, sheet_name=sheet, header=header_row)
            df = df.dropna(how='all')
            print(f"  Full read: {len(df)} rows × {len(df.columns)} cols")
            print(f"  Columns: {list(df.columns[:8])}...")
            if len(df) > best_rows:
                best_rows = len(df)
                best_sheet = (sheet, header_row, df)
        else:
            # Try header=0 as fallback
            df = pd.read_excel(xls, sheet_name=sheet, header=0)
            df = df.dropna(how='all')
            if len(df) > best_rows and len(df.columns) > 3:
                print(f"  Fallback read: {len(df)} rows × {len(df.columns)} cols")
                best_rows = len(df)
                best_sheet = (sheet, 0, df)
    except Exception as e:
        print(f"  Error reading '{sheet}': {e}")

if best_sheet:
    sheet_name, header_row, df = best_sheet
    print(f"\n{'='*60}")
    print(f"Best sheet: '{sheet_name}' (header row {header_row})")
    print(f"Shape: {len(df)} rows × {len(df.columns)} cols")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
    print(f"\nLast 3 rows:")
    print(df.tail(3).to_string())

    # Save
    df.to_csv(OUT, index=False)
    print(f"\n✓ Saved to {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
    print("  Re-run fetch_thesis_data.py to repackage into xlsx.")
else:
    print("\nCouldn't find data in any sheet. Open rpw_raw.xlsx in Excel manually")
    print("and note the sheet name + header row number, then edit this script.")
