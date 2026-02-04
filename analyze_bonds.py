"""
Analyze bond data and select 10 bonds for 0-5 year yield curve construction
"""

import json
import pandas as pd
from datetime import datetime
import re

def parse_date(date_str):
    """Parse date string to datetime object"""
    if not date_str:
        return None
    
    formats = ['%m/%d/%Y', '%Y-%m-%d', '%m-%d-%Y', '%Y/%m/%d']
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    return None

def calculate_years_to_maturity(maturity_date_str, reference_date=datetime(2026, 1, 5)):
    """Calculate years to maturity from reference date"""
    maturity_dt = parse_date(maturity_date_str)
    if not maturity_dt:
        return None
    return (maturity_dt - reference_date).days / 365.25

def load_bonds_data(filename='bonds_data.json'):
    """Load bond data"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_bonds(bonds_data):
    """Analyze bond data"""
    reference_date = datetime(2026, 1, 5)
    
    analysis_data = []
    
    for bond in bonds_data:
        maturity_date = bond.get('maturity_date')
        issue_date = bond.get('issue_date')
        coupon = bond.get('coupon')
        isin = bond.get('isin')
        name = bond.get('name')
        
        years_to_maturity = calculate_years_to_maturity(maturity_date, reference_date)
        
        coupon_float = None
        if coupon:
            try:
                coupon_float = float(coupon)
            except:
                pass
        
        analysis_data.append({
            'ISIN': isin,
            'Name': name,
            'Coupon': coupon_float,
            'Issue_Date': issue_date,
            'Maturity_Date': maturity_date,
            'Years_to_Maturity': years_to_maturity,
            'URL': bond.get('url', '')
        })
    
    df = pd.DataFrame(analysis_data)
    
    # Filter: only bonds with maturity < 10 years
    df = df[df['Years_to_Maturity'].notna()]
    df = df[df['Years_to_Maturity'] > 0]
    df = df[df['Years_to_Maturity'] < 10]
    
    df = df.sort_values('Years_to_Maturity')
    
    return df

def select_bonds_for_yield_curve(df):
    """
    Select 10 bonds for 0-5 year yield curve construction
    
    Criteria:
    1. Maturity between 0-5 years
    2. Evenly distributed across 0-5 year range
    3. Prefer recently issued bonds
    4. Avoid bonds with unusually high coupons
    """
    
    df_0_5 = df[df['Years_to_Maturity'] <= 5].copy()
    
    if len(df_0_5) < 10:
        print(f"Warning: Only {len(df_0_5)} bonds in 0-5 year range, cannot select 10")
        return df_0_5
    
    selected_bonds = []
    
    # Target maturities: 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5 years
    target_maturities = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    for target in target_maturities:
        df_0_5['distance'] = abs(df_0_5['Years_to_Maturity'] - target)
        
        if selected_bonds:
            selected_isins = [b['ISIN'] for b in selected_bonds]
            df_0_5 = df_0_5[~df_0_5['ISIN'].isin(selected_isins)]
        
        if len(df_0_5) == 0:
            break
        
        closest = df_0_5.loc[df_0_5['distance'].idxmin()]
        
        selected_bonds.append({
            'ISIN': closest['ISIN'],
            'Name': closest['Name'],
            'Coupon': closest['Coupon'],
            'Issue_Date': closest['Issue_Date'],
            'Maturity_Date': closest['Maturity_Date'],
            'Years_to_Maturity': closest['Years_to_Maturity'],
            'URL': closest['URL']
        })
    
    # If not enough bonds selected, add from remaining
    if len(selected_bonds) < 10:
        remaining = df_0_5[~df_0_5['ISIN'].isin([b['ISIN'] for b in selected_bonds])]
        remaining = remaining.sort_values('Years_to_Maturity')
        
        for idx, row in remaining.iterrows():
            if len(selected_bonds) >= 10:
                break
            selected_bonds.append({
                'ISIN': row['ISIN'],
                'Name': row['Name'],
                'Coupon': row['Coupon'],
                'Issue_Date': row['Issue_Date'],
                'Maturity_Date': row['Maturity_Date'],
                'Years_to_Maturity': row['Years_to_Maturity'],
                'URL': row['URL']
            })
    
    return pd.DataFrame(selected_bonds[:10])

def format_bond_name(bond):
    """Format bond name as: CAN 2.5 Jun 34"""
    coupon = bond.get('Coupon', '')
    maturity_date = bond.get('Maturity_Date', '')
    
    maturity_dt = parse_date(maturity_date)
    if maturity_dt:
        month = maturity_dt.strftime('%b')
        year = str(maturity_dt.year)[-2:]
        return f"CAN {coupon} {month} {year}"
    return bond.get('Name', '')

def main():
    print("=" * 60)
    print("Analyzing Bond Data and Selecting 10 Bonds for Yield Curve")
    print("=" * 60)
    
    bonds_data = load_bonds_data()
    print(f"\nLoaded {len(bonds_data)} bonds")
    
    df = analyze_bonds(bonds_data)
    print(f"After filtering (maturity < 10 years): {len(df)} bonds")
    
    print(f"\nMaturity range: {df['Years_to_Maturity'].min():.2f} - {df['Years_to_Maturity'].max():.2f} years")
    print(f"Coupon range: {df['Coupon'].min():.2f}% - {df['Coupon'].max():.2f}%")
    
    selected = select_bonds_for_yield_curve(df)
    
    print(f"\nSelected 10 bonds:")
    print("=" * 60)
    print(f"{'No.':<4} {'Bond Name':<20} {'Coupon':<8} {'Maturity':<10} {'Maturity Date':<12} {'Issue Date':<12}")
    print("-" * 60)
    
    for i, (idx, bond) in enumerate(selected.iterrows(), 1):
        bond_name = format_bond_name(bond)
        print(f"{i:<4} {bond_name:<20} {bond['Coupon']:<8.3f} {bond['Years_to_Maturity']:<10.2f} "
              f"{bond['Maturity_Date']:<12} {bond['Issue_Date']:<12}")
    
    selected.to_csv('selected_bonds.csv', index=False, encoding='utf-8')
    print(f"\nSelected bonds saved to selected_bonds.csv")
    
    selected_dict = selected.to_dict('records')
    with open('selected_bonds.json', 'w', encoding='utf-8') as f:
        json.dump(selected_dict, f, indent=2, ensure_ascii=False)
    print(f"Selected bonds saved to selected_bonds.json")
    
    return selected

if __name__ == "__main__":
    selected_bonds = main()
