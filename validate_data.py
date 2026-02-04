#!/usr/bin/env python3
"""
Data Validation Script for APM466 Assignment 1
This script helps verify data at each step of the analysis.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def validate_raw_data():
    """Step 1: Validate raw bond data collection"""
    print_section("STEP 1: Validating Raw Bond Data")
    
    try:
        with open('bonds_data.json', 'r', encoding='utf-8') as f:
            bonds_data = json.load(f)
        
        print(f"✓ Loaded {len(bonds_data)} bonds from bonds_data.json")
        
        # Check required fields
        required_fields = ['isin', 'coupon', 'issue_date', 'maturity_date', 'historical_prices']
        missing_fields = []
        
        for i, bond in enumerate(bonds_data):
            for field in required_fields:
                if field not in bond or bond[field] is None:
                    missing_fields.append(f"Bond {i+1} ({bond.get('isin', 'unknown')}): missing {field}")
        
        if missing_fields:
            print("\n⚠ WARNING: Missing fields found:")
            for msg in missing_fields[:10]:  # Show first 10
                print(f"  - {msg}")
        else:
            print("✓ All required fields present")
        
        # Check historical prices
        print("\n📊 Historical Price Data Summary:")
        price_stats = []
        for bond in bonds_data:
            prices = bond.get('historical_prices', {})
            if isinstance(prices, dict):
                price_count = len(prices)
                if price_count > 0:
                    price_values = [v for v in prices.values() if isinstance(v, (int, float))]
                    if price_values:
                        price_stats.append({
                            'isin': bond.get('isin', 'unknown'),
                            'count': price_count,
                            'min': min(price_values),
                            'max': max(price_values),
                            'mean': np.mean(price_values)
                        })
        
        if price_stats:
            df = pd.DataFrame(price_stats)
            print(f"  - Bonds with price data: {len(df)}")
            print(f"  - Average prices per bond: {df['count'].mean():.1f}")
            print(f"  - Price range: ${df['min'].min():.2f} - ${df['max'].max():.2f}")
            print("\n  Sample bonds with price data:")
            print(df.head(10).to_string(index=False))
        else:
            print("  ⚠ WARNING: No historical price data found!")
        
        # Check date range
        print("\n📅 Date Range Check:")
        all_dates = set()
        for bond in bonds_data:
            prices = bond.get('historical_prices', {})
            if isinstance(prices, dict):
                all_dates.update(prices.keys())
        
        if all_dates:
            sorted_dates = sorted(all_dates)
            print(f"  - First date: {sorted_dates[0]}")
            print(f"  - Last date: {sorted_dates[-1]}")
            print(f"  - Total unique dates: {len(sorted_dates)}")
            print(f"  - Expected dates (Jan 5-19, 2026, weekdays only): ~10 days")
        
        return bonds_data
        
    except FileNotFoundError:
        print("✗ ERROR: bonds_data.json not found!")
        print("  Run data_collection_selenium.py first to collect data.")
        return None
    except Exception as e:
        print(f"✗ ERROR loading bonds_data.json: {e}")
        return None

def validate_selected_bonds(bonds_data):
    """Step 2: Validate selected bonds"""
    print_section("STEP 2: Validating Selected Bonds")
    
    try:
        with open('selected_bonds.json', 'r', encoding='utf-8') as f:
            selected_bonds = json.load(f)
        
        print(f"✓ Loaded {len(selected_bonds)} selected bonds")
        
        # Display selected bonds
        print("\n📋 Selected Bonds:")
        print(f"{'No.':<4} {'ISIN':<15} {'Coupon':<8} {'Maturity':<12} {'Years':<8}")
        print("-" * 60)
        
        for i, bond in enumerate(selected_bonds, 1):
            isin = bond.get('ISIN', 'N/A')
            coupon = bond.get('Coupon', 'N/A')
            maturity = bond.get('Maturity_Date', 'N/A')
            years = bond.get('Years_to_Maturity', 'N/A')
            print(f"{i:<4} {isin:<15} {coupon:<8} {maturity:<12} {years:<8}")
        
        # Verify all selected bonds have data
        print("\n🔍 Verifying data availability for selected bonds:")
        missing_data = []
        for bond in selected_bonds:
            isin = bond.get('ISIN')
            # Find in bonds_data
            found = False
            for b in bonds_data:
                if b.get('isin') == isin:
                    found = True
                    prices = b.get('historical_prices', {})
                    if not prices or len(prices) == 0:
                        missing_data.append(f"{isin}: No price data")
                    break
            if not found:
                missing_data.append(f"{isin}: Not found in bonds_data.json")
        
        if missing_data:
            print("  ⚠ WARNING: Missing data for:")
            for msg in missing_data:
                print(f"    - {msg}")
        else:
            print("  ✓ All selected bonds have data")
        
        # Check maturity distribution
        print("\n📊 Maturity Distribution:")
        maturities = [bond.get('Years_to_Maturity', 0) for bond in selected_bonds]
        if maturities:
            maturities = sorted([m for m in maturities if m is not None])
            print(f"  - Shortest: {min(maturities):.2f} years")
            print(f"  - Longest: {max(maturities):.2f} years")
            print(f"  - Range: {max(maturities) - min(maturities):.2f} years")
            print(f"  - Distribution: {', '.join([f'{m:.2f}' for m in maturities])}")
        
        return selected_bonds
        
    except FileNotFoundError:
        print("✗ ERROR: selected_bonds.json not found!")
        print("  Run analyze_bonds.py first to select bonds.")
        return None
    except Exception as e:
        print(f"✗ ERROR loading selected_bonds.json: {e}")
        return None

def validate_ytm_data():
    """Step 3: Validate YTM calculations"""
    print_section("STEP 3: Validating YTM Data")
    
    try:
        ytm_df = pd.read_csv('ytm_data.csv')
        print(f"✓ Loaded YTM data: {len(ytm_df)} rows")
        
        print("\n📊 YTM Data Summary:")
        print(f"  - Unique dates: {ytm_df['Date'].nunique()}")
        print(f"  - Unique maturities: {ytm_df['Years_to_Maturity'].nunique()}")
        
        # Check for reasonable YTM values
        print("\n🔍 YTM Value Checks:")
        ytm_values = ytm_df['YTM'].values
        print(f"  - Min YTM: {ytm_values.min():.4f}%")
        print(f"  - Max YTM: {ytm_values.max():.4f}%")
        print(f"  - Mean YTM: {ytm_values.mean():.4f}%")
        print(f"  - Median YTM: {np.median(ytm_values):.4f}%")
        
        # Check for negative or extreme values
        negative = (ytm_values < 0).sum()
        extreme_high = (ytm_values > 20).sum()
        extreme_low = (ytm_values < -5).sum()
        
        if negative > 0:
            print(f"  ⚠ WARNING: {negative} negative YTM values found")
        if extreme_high > 0:
            print(f"  ⚠ WARNING: {extreme_high} YTM values > 20%")
        if extreme_low > 0:
            print(f"  ⚠ WARNING: {extreme_low} YTM values < -5%")
        
        if negative == 0 and extreme_high == 0 and extreme_low == 0:
            print("  ✓ All YTM values are reasonable")
        
        # Show sample data
        print("\n📋 Sample YTM Data (first 10 rows):")
        print(ytm_df.head(10).to_string(index=False))
        
        # Check by date
        print("\n📅 YTM by Date:")
        for date in sorted(ytm_df['Date'].unique())[:3]:  # Show first 3 dates
            date_data = ytm_df[ytm_df['Date'] == date]
            print(f"\n  {date}:")
            print(f"    Bonds: {len(date_data)}")
            print(f"    YTM range: {date_data['YTM'].min():.4f}% - {date_data['YTM'].max():.4f}%")
            print(f"    Maturities: {', '.join([f'{m:.2f}yr' for m in sorted(date_data['Years_to_Maturity'].unique())[:5]])}")
        
        return ytm_df
        
    except FileNotFoundError:
        print("✗ ERROR: ytm_data.csv not found!")
        print("  Run calculate_ytm.py first to calculate YTM.")
        return None
    except Exception as e:
        print(f"✗ ERROR loading ytm_data.csv: {e}")
        return None

def validate_spot_data():
    """Step 4: Validate spot rate data"""
    print_section("STEP 4: Validating Spot Rate Data")
    
    try:
        spot_df = pd.read_csv('spot_data.csv')
        print(f"✓ Loaded spot rate data: {len(spot_df)} rows")
        
        print("\n📊 Spot Rate Data Summary:")
        print(f"  - Unique dates: {spot_df['Date'].nunique()}")
        print(f"  - Unique maturities: {spot_df['Years_to_Maturity'].nunique()}")
        
        # Check for reasonable spot rate values
        print("\n🔍 Spot Rate Value Checks:")
        spot_values = spot_df['Spot_Rate'].values
        print(f"  - Min spot rate: {spot_values.min():.4f}%")
        print(f"  - Max spot rate: {spot_values.max():.4f}%")
        print(f"  - Mean spot rate: {spot_values.mean():.4f}%")
        
        # Check for negative or extreme values
        negative = (spot_values < 0).sum()
        extreme_high = (spot_values > 20).sum()
        
        if negative > 0:
            print(f"  ⚠ WARNING: {negative} negative spot rates found")
        if extreme_high > 0:
            print(f"  ⚠ WARNING: {extreme_high} spot rates > 20%")
        
        # Check if spot rates vary by maturity (should not be all the same)
        print("\n🔍 Spot Rate Variation Check:")
        for date in sorted(spot_df['Date'].unique())[:2]:  # Check first 2 dates
            date_data = spot_df[spot_df['Date'] == date]
            unique_rates = date_data['Spot_Rate'].nunique()
            if unique_rates == 1:
                print(f"  ⚠ WARNING: {date} - All spot rates are identical ({date_data['Spot_Rate'].iloc[0]:.4f}%)")
                print("    This suggests a problem with bootstrapping!")
            else:
                print(f"  ✓ {date} - {unique_rates} unique spot rates (good variation)")
                print(f"    Range: {date_data['Spot_Rate'].min():.4f}% - {date_data['Spot_Rate'].max():.4f}%")
        
        # Show sample data
        print("\n📋 Sample Spot Rate Data (first 10 rows):")
        print(spot_df.head(10).to_string(index=False))
        
        return spot_df
        
    except FileNotFoundError:
        print("✗ ERROR: spot_data.csv not found!")
        print("  Run calculate_curves.py first to calculate spot rates.")
        return None
    except Exception as e:
        print(f"✗ ERROR loading spot_data.csv: {e}")
        return None

def validate_forward_data():
    """Step 5: Validate forward rate data"""
    print_section("STEP 5: Validating Forward Rate Data")
    
    try:
        forward_df = pd.read_csv('forward_data.csv')
        print(f"✓ Loaded forward rate data: {len(forward_df)} rows")
        
        print("\n📊 Forward Rate Data Summary:")
        print(f"  - Unique dates: {forward_df['Date'].nunique()}")
        print(f"  - Unique forward terms: {forward_df['Forward_Term'].nunique()}")
        print(f"  - Forward terms: {', '.join(sorted(forward_df['Forward_Term'].unique()))}")
        
        # Check for reasonable forward rate values
        print("\n🔍 Forward Rate Value Checks:")
        forward_values = forward_df['Forward_Rate'].values
        print(f"  - Min forward rate: {forward_values.min():.4f}%")
        print(f"  - Max forward rate: {forward_values.max():.4f}%")
        print(f"  - Mean forward rate: {forward_values.mean():.4f}%")
        
        # Check for extreme negative values
        very_negative = (forward_values < -10).sum()
        if very_negative > 0:
            print(f"  ⚠ WARNING: {very_negative} forward rates < -10%")
        
        # Check if forward rates vary by term (should not be all the same)
        print("\n🔍 Forward Rate Variation Check:")
        for date in sorted(forward_df['Date'].unique())[:2]:  # Check first 2 dates
            date_data = forward_df[forward_df['Date'] == date]
            unique_rates = date_data['Forward_Rate'].nunique()
            if unique_rates == 1:
                print(f"  ⚠ WARNING: {date} - All forward rates are identical ({date_data['Forward_Rate'].iloc[0]:.4f}%)")
            else:
                print(f"  ✓ {date} - {unique_rates} unique forward rates")
                print(f"    Range: {date_data['Forward_Rate'].min():.4f}% - {date_data['Forward_Rate'].max():.4f}%")
        
        # Show sample data
        print("\n📋 Sample Forward Rate Data:")
        print(forward_df.head(10).to_string(index=False))
        
        return forward_df
        
    except FileNotFoundError:
        print("✗ ERROR: forward_data.csv not found!")
        print("  Run calculate_curves.py first to calculate forward rates.")
        return None
    except Exception as e:
        print(f"✗ ERROR loading forward_data.csv: {e}")
        return None

def cross_validate_relationships():
    """Step 6: Cross-validate relationships between data"""
    print_section("STEP 6: Cross-Validating Data Relationships")
    
    try:
        ytm_df = pd.read_csv('ytm_data.csv')
        spot_df = pd.read_csv('spot_data.csv')
        forward_df = pd.read_csv('forward_data.csv')
        
        # Check date consistency
        print("📅 Date Consistency Check:")
        ytm_dates = set(ytm_df['Date'].unique())
        spot_dates = set(spot_df['Date'].unique())
        forward_dates = set(forward_df['Date'].unique())
        
        all_dates = ytm_dates | spot_dates | forward_dates
        common_dates = ytm_dates & spot_dates & forward_dates
        
        print(f"  - YTM dates: {len(ytm_dates)}")
        print(f"  - Spot dates: {len(spot_dates)}")
        print(f"  - Forward dates: {len(forward_dates)}")
        print(f"  - Common dates: {len(common_dates)}")
        
        if len(common_dates) < len(all_dates):
            print(f"  ⚠ WARNING: Date mismatch! Some dates missing in one or more datasets")
        
        # Check YTM vs Spot relationship (spot should generally be close to YTM for similar maturities)
        print("\n🔗 YTM vs Spot Rate Relationship:")
        sample_date = sorted(common_dates)[0] if common_dates else None
        if sample_date:
            ytm_sample = ytm_df[ytm_df['Date'] == sample_date]
            spot_sample = spot_df[spot_df['Date'] == sample_date]
            
            # Try to match by similar maturity
            print(f"  Sample date: {sample_date}")
            print(f"  YTM maturities: {sorted(ytm_sample['Years_to_Maturity'].unique())[:5]}")
            print(f"  Spot maturities: {sorted(spot_sample['Years_to_Maturity'].unique())[:5]}")
            
            # Find closest matches
            for ytm_row in ytm_sample.head(3).itertuples():
                ytm_maturity = ytm_row.Years_to_Maturity
                ytm_value = ytm_row.YTM
                
                # Find closest spot rate
                spot_sample['diff'] = abs(spot_sample['Years_to_Maturity'] - ytm_maturity)
                closest_spot = spot_sample.loc[spot_sample['diff'].idxmin()]
                
                diff = abs(ytm_value - closest_spot['Spot_Rate'])
                print(f"    Maturity {ytm_maturity:.2f}yr: YTM={ytm_value:.4f}%, Spot={closest_spot['Spot_Rate']:.4f}%, Diff={diff:.4f}%")
        
        print("\n✓ Cross-validation complete")
        
    except Exception as e:
        print(f"✗ ERROR in cross-validation: {e}")

def main():
    """Run all validation steps"""
    print("\n" + "="*70)
    print("  APM466 Assignment 1 - Data Validation")
    print("="*70)
    
    # Step 1: Raw data
    bonds_data = validate_raw_data()
    if bonds_data is None:
        print("\n✗ Cannot proceed without raw bond data. Please collect data first.")
        return
    
    # Step 2: Selected bonds
    selected_bonds = validate_selected_bonds(bonds_data)
    if selected_bonds is None:
        print("\n⚠ Cannot validate selected bonds. Continuing with other checks...")
    
    # Step 3: YTM data
    ytm_df = validate_ytm_data()
    
    # Step 4: Spot rate data
    spot_df = validate_spot_data()
    
    # Step 5: Forward rate data
    forward_df = validate_forward_data()
    
    # Step 6: Cross-validation
    if ytm_df is not None and spot_df is not None and forward_df is not None:
        cross_validate_relationships()
    
    print("\n" + "="*70)
    print("  Validation Complete!")
    print("="*70)
    print("\n💡 Tips:")
    print("  - Check for ⚠ WARNING messages above")
    print("  - Verify that values are reasonable (YTM/Spot rates typically 0-10%)")
    print("  - Ensure data varies by maturity (not all identical values)")
    print("  - Check that dates are consistent across all datasets")
    print()

if __name__ == "__main__":
    main()
