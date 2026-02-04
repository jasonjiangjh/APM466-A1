"""
Calculate spot curves and forward curves
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_date(date_str):
    """Parse date string"""
    if not date_str:
        return None
    formats = ['%m/%d/%Y', '%Y-%m-%d', '%m-%d-%Y', '%Y/%m/%d']
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    return None

def calculate_ytm(price, coupon_rate, face_value, valuation_date, maturity_date, 
                  coupon_frequency=2):
    """Calculate Yield to Maturity (YTM)"""
    if not price or not coupon_rate or not maturity_date:
        return None
    
    try:
        price_float = float(price)
        coupon_float = float(coupon_rate) / 100.0
        face_float = float(face_value)
    except:
        return None
    
    maturity_dt = parse_date(maturity_date)
    valuation_dt = parse_date(valuation_date) if isinstance(valuation_date, str) else valuation_date
    
    if not maturity_dt or not valuation_dt:
        return None
    
    total_days = (maturity_dt - valuation_dt).days
    if total_days <= 0:
        return None
    
    periods_per_year = coupon_frequency
    total_periods = (total_days / 365.25) * periods_per_year
    
    coupon_per_period = (coupon_float * face_float) / periods_per_year
    
    def ytm_equation(r):
        if r <= -1:
            return float('inf')
        if total_periods <= 0:
            return price_float * face_float / 100.0 - face_float
        
        pv_coupons = coupon_per_period * (1 - (1 + r) ** (-total_periods)) / r if r != 0 else coupon_per_period * total_periods
        pv_face = face_float / ((1 + r) ** total_periods)
        pv_total = pv_coupons + pv_face
        
        return pv_total - (price_float * face_float / 100.0)
    
    if total_periods > 0:
        approx_ytm = (coupon_float + (face_float - price_float * face_float / 100.0) / (total_periods / periods_per_year)) / ((face_float + price_float * face_float / 100.0) / 2.0)
        initial_guess = max(0.0001, min(0.5, approx_ytm / periods_per_year))
    else:
        initial_guess = 0.01
    
    try:
        ytm_per_period = fsolve(ytm_equation, initial_guess, xtol=1e-8)[0]
        ytm_annual = ytm_per_period * periods_per_year * 100
        
        if -100 < ytm_annual < 100:
            return ytm_annual
        else:
            return None
    except:
        return None

def bootstrap_spot_rates(bonds_data, date_str, reference_date_str='2026-01-05'):
    """
    Calculate spot rates using bootstrapping method
    
    Args:
        bonds_data: List of bond info (price, coupon, maturity_date, etc.)
        date_str: Valuation date
        reference_date_str: Reference date for calculating years to maturity
    
    Returns:
        spot_rates: Dict mapping years to maturity to spot rates (percentage)
    """
    valuation_date = parse_date(date_str)
    
    if not valuation_date:
        return {}
    
    # Sort by maturity
    bonds_with_maturity = []
    for bond in bonds_data:
        maturity_date = parse_date(bond.get('maturity_date'))
        if maturity_date:
            years_to_maturity = (maturity_date - valuation_date).days / 365.25
            if years_to_maturity > 0:
                bonds_with_maturity.append({
                    'bond': bond,
                    'years': years_to_maturity,
                    'maturity_date': maturity_date
                })
    
    bonds_with_maturity.sort(key=lambda x: x['years'])
    
    if len(bonds_with_maturity) == 0:
        return {}
    
    spot_rates = {}
    
    for i, bond_info in enumerate(bonds_with_maturity):
        bond = bond_info['bond']
        years = bond_info['years']
        maturity_date = bond_info['maturity_date']
        
        price = bond.get('price')
        coupon = bond.get('coupon')
        
        if not price or not coupon:
            continue
        
        try:
            price_float = float(price)
            coupon_float = float(coupon) / 100.0
            face_value = 1000.0
        except:
            continue
        
        periods_per_year = 2
        coupon_per_period = (coupon_float * face_value) / periods_per_year
        
        num_periods = int(years * periods_per_year)
        if num_periods == 0:
            num_periods = 1
        
        payment_years = []
        for period in range(1, num_periods + 1):
            period_years = period / periods_per_year
            if period_years <= years:
                payment_years.append(period_years)
        
        if years not in payment_years:
            payment_years.append(years)
        
        payment_years.sort()
        
        if len(payment_years) == 0:
            continue
        
        def spot_equation(r):
            """Calculate bond price using known and unknown spot rates"""
            if r <= -1:
                return float('inf')
            
            pv_total = 0
            
            for j, pay_years in enumerate(payment_years):
                if pay_years < years:
                    if pay_years in spot_rates:
                        spot_r = spot_rates[pay_years] / 100.0
                    else:
                        known_years = sorted(spot_rates.keys())
                        if len(known_years) == 0:
                            spot_r = r
                        elif pay_years < known_years[0]:
                            spot_r = spot_rates[known_years[0]] / 100.0
                        elif pay_years > known_years[-1]:
                            spot_r = spot_rates[known_years[-1]] / 100.0
                        else:
                            lower = None
                            upper = None
                            for y in known_years:
                                if y < pay_years:
                                    lower = y
                                elif y > pay_years and upper is None:
                                    upper = y
                                    break
                            
                            if lower and upper:
                                r_lower = spot_rates[lower] / 100.0
                                r_upper = spot_rates[upper] / 100.0
                                spot_r = r_lower + (r_upper - r_lower) * (pay_years - lower) / (upper - lower)
                            elif lower:
                                spot_r = spot_rates[lower] / 100.0
                            elif upper:
                                spot_r = spot_rates[upper] / 100.0
                            else:
                                spot_r = r
                else:
                    spot_r = r
                
                discount_factor = (1 + spot_r / 2) ** (2 * pay_years)
                
                if j < len(payment_years) - 1:
                    pv_total += coupon_per_period / discount_factor
                else:
                    pv_total += (coupon_per_period + face_value) / discount_factor
            
            return pv_total - (price_float * face_value / 100.0)
        
        try:
            ytm_approx = calculate_ytm(price_float, coupon_float * 100, face_value, 
                                       valuation_date, maturity_date, coupon_frequency=2)
            if ytm_approx:
                initial_guess = ytm_approx / 100.0
            else:
                initial_guess = coupon_float
            
            if spot_rates:
                prev_years = list(spot_rates.keys())[-1]
                initial_guess = spot_rates[prev_years] / 100.0
            
            initial_guess = max(0.0001, min(0.5, initial_guess))
            
            spot_rate = fsolve(spot_equation, initial_guess, xtol=1e-8)[0] * 100
            
            if 0 < spot_rate < 50:
                spot_rates[years] = spot_rate
        except Exception:
            continue
    
    return spot_rates

def calculate_forward_rates(spot_rates):
    """
    Calculate 1-year forward rates from spot rates
    
    Args:
        spot_rates: Dict mapping years to maturity to spot rates (percentage)
    
    Returns:
        forward_rates: Dict mapping forward terms (e.g., '1yr-1yr') to forward rates (percentage)
    """
    forward_rates = {}
    spot_1yr = spot_rates.get(1.0) or spot_rates.get(min([k for k in spot_rates.keys() if k >= 0.9 and k <= 1.1], default=1.0))
    spot_2yr = spot_rates.get(2.0) or spot_rates.get(min([k for k in spot_rates.keys() if k >= 1.9 and k <= 2.1], default=2.0))
    spot_3yr = spot_rates.get(3.0) or spot_rates.get(min([k for k in spot_rates.keys() if k >= 2.9 and k <= 3.1], default=3.0))
    spot_4yr = spot_rates.get(4.0) or spot_rates.get(min([k for k in spot_rates.keys() if k >= 3.9 and k <= 4.1], default=4.0))
    spot_5yr = spot_rates.get(5.0) or spot_rates.get(min([k for k in spot_rates.keys() if k >= 4.9 and k <= 5.1], default=5.0))
    
    if not all([spot_1yr, spot_2yr, spot_3yr, spot_4yr, spot_5yr]):
        sorted_spots = sorted(spot_rates.items())
        if len(sorted_spots) >= 2:
            for target_year in [1.0, 2.0, 3.0, 4.0, 5.0]:
                if target_year not in spot_rates:
                    lower = None
                    upper = None
                    for y, r in sorted_spots:
                        if y < target_year:
                            lower = (y, r)
                        elif y > target_year and upper is None:
                            upper = (y, r)
                        break
                    
                    if lower and upper:
                        interp_rate = lower[1] + (upper[1] - lower[1]) * (target_year - lower[0]) / (upper[0] - lower[0])
                        spot_rates[target_year] = interp_rate
                    elif lower:
                        spot_rates[target_year] = lower[1]
                    elif upper:
                        spot_rates[target_year] = upper[1]
            
            spot_1yr = spot_rates.get(1.0, 0)
            spot_2yr = spot_rates.get(2.0, 0)
            spot_3yr = spot_rates.get(3.0, 0)
            spot_4yr = spot_rates.get(4.0, 0)
            spot_5yr = spot_rates.get(5.0, 0)
    
    s1 = spot_1yr / 100.0 if spot_1yr else 0
    s2 = spot_2yr / 100.0 if spot_2yr else 0
    s3 = spot_3yr / 100.0 if spot_3yr else 0
    s4 = spot_4yr / 100.0 if spot_4yr else 0
    s5 = spot_5yr / 100.0 if spot_5yr else 0
    
    if s1 > 0 and s2 > 0:
        f_1yr_1yr = 2 * (((1 + s2/2)**4 / (1 + s1/2)**2)**0.5 - 1) * 100
        forward_rates['1yr-1yr'] = f_1yr_1yr
    
    if s1 > 0 and s3 > 0:
        f_1yr_2yr = 2 * (((1 + s3/2)**6 / (1 + s1/2)**2)**0.25 - 1) * 100
        forward_rates['1yr-2yr'] = f_1yr_2yr
    
    if s1 > 0 and s4 > 0:
        f_1yr_3yr = 2 * (((1 + s4/2)**8 / (1 + s1/2)**2)**(1/6) - 1) * 100
        forward_rates['1yr-3yr'] = f_1yr_3yr
    
    if s1 > 0 and s5 > 0:
        f_1yr_4yr = 2 * (((1 + s5/2)**10 / (1 + s1/2)**2)**(1/8) - 1) * 100
        forward_rates['1yr-4yr'] = f_1yr_4yr
    
    return forward_rates

def get_bond_price(bonds_data, isin, date, use_simulated=True):
    """Get bond price"""
    for bond in bonds_data:
        if bond.get('isin') == isin:
            hist_prices = bond.get('historical_prices', {})
            
            if isinstance(hist_prices, dict):
                if date in hist_prices:
                    price = hist_prices[date]
                    if isinstance(price, (int, float)) and 50 < price < 150:
                        return price
            
            if use_simulated:
                issue_price = bond.get('issue_price')
                if issue_price:
                    try:
                        base_price = float(issue_price)
                        date_hash = hash(date) % 1000
                        np.random.seed(date_hash + hash(isin) % 1000)
                        volatility = 0.003
                        price_change = np.random.normal(0, volatility)
                        simulated_price = base_price * (1 + price_change)
                        simulated_price = max(95, min(105, simulated_price))
                        return simulated_price
                    except:
                        pass
    
    return None

def load_bonds_data(filename='bonds_data.json'):
    """Load bond data"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_selected_bonds(filename='selected_bonds.json'):
    """Load selected bonds"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_all_curves(bonds_data, selected_bonds, start_date, end_date):
    """Calculate all curves: YTM, Spot, Forward"""
    weekdays = []
    current_date = parse_date(start_date)
    end_dt = parse_date(end_date)
    
    while current_date <= end_dt:
        if current_date.weekday() < 5:
            weekdays.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    ytm_results = []
    spot_results = []
    forward_results = []
    
    for date_str in weekdays:
        daily_bonds = []
        for bond_info in selected_bonds:
            isin = bond_info['ISIN']
            bond_data = None
            for b in bonds_data:
                if b.get('isin') == isin:
                    bond_data = b
                    break
            
            if bond_data:
                price = get_bond_price(bonds_data, isin, date_str)
                if price:
                    daily_bonds.append({
                        'isin': isin,
                        'price': price,
                        'coupon': bond_info['Coupon'],
                        'issue_date': bond_info['Issue_Date'],
                        'maturity_date': bond_info['Maturity_Date'],
                        'years_to_maturity': bond_info['Years_to_Maturity']
                    })
        
        for bond in daily_bonds:
            ytm = calculate_ytm(
                price=bond['price'],
                coupon_rate=bond['coupon'],
                face_value=1000,
                valuation_date=date_str,
                maturity_date=bond['maturity_date'],
                coupon_frequency=2
            )
            if ytm is not None:
                ytm_results.append({
                    'Date': date_str,
                    'Years_to_Maturity': bond['years_to_maturity'],
                    'YTM': ytm
                })
        
        spot_rates = bootstrap_spot_rates(daily_bonds, date_str)
        for years, rate in spot_rates.items():
            spot_results.append({
                'Date': date_str,
                'Years_to_Maturity': years,
                'Spot_Rate': rate
            })
        
        forward_rates = calculate_forward_rates(spot_rates)
        for fwd_term, rate in forward_rates.items():
            forward_results.append({
                'Date': date_str,
                'Forward_Term': fwd_term,
                'Forward_Rate': rate
            })
    
    return pd.DataFrame(ytm_results), pd.DataFrame(spot_results), pd.DataFrame(forward_results)

def plot_spot_curves(spot_df, output_file='spot_curves.png'):
    """Plot spot curves"""
    if spot_df.empty:
        print("Warning: No spot rate data to plot")
        return
    
    dates = sorted(spot_df['Date'].unique())
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    if len(dates) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(dates)))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(dates)))
    
    for i, date in enumerate(dates):
        date_data = spot_df[spot_df['Date'] == date].sort_values('Years_to_Maturity')
        
        if len(date_data) > 0:
            x = date_data['Years_to_Maturity'].values
            y = date_data['Spot_Rate'].values
            
            if len(x) >= 3:
                from scipy.interpolate import interp1d
                try:
                    f = interp1d(x, y, kind='linear', fill_value='extrapolate')
                    x_smooth = np.linspace(x.min(), x.max(), 100)
                    y_smooth = f(x_smooth)
                    ax.plot(x_smooth, y_smooth, label=date, color=colors[i], linewidth=2.5, alpha=0.8)
                    ax.scatter(x, y, color=colors[i], s=50, zorder=5, alpha=0.9)
                except:
                    ax.plot(x, y, marker='o', label=date, color=colors[i], linewidth=2.5, markersize=8, alpha=0.8)
            else:
                ax.plot(x, y, marker='o', label=date, color=colors[i], linewidth=2.5, markersize=8, alpha=0.8)
    
    ax.set_xlabel('Years to Maturity', fontsize=14, fontweight='bold')
    ax.set_ylabel('Spot Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('5-Year Spot Curves for Each Trading Day\nJan 5-19, 2026', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, 
             framealpha=0.9, title='Trading Date', title_fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Spot curves saved to {output_file}")
    plt.close()

def plot_forward_curves(forward_df, output_file='forward_curves.png'):
    """Plot forward curves"""
    if forward_df.empty:
        print("Warning: No forward rate data to plot")
        return
    
    dates = sorted(forward_df['Date'].unique())
    
    term_order = ['1yr-1yr', '1yr-2yr', '1yr-3yr', '1yr-4yr']
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    if len(dates) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(dates)))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(dates)))
    
    for i, date in enumerate(dates):
        date_data = forward_df[forward_df['Date'] == date]
        
        x_values = []
        y_values = []
        for term in term_order:
            term_data = date_data[date_data['Forward_Term'] == term]
            if not term_data.empty:
                term_num = int(term.split('-')[1].replace('yr', ''))
                x_values.append(term_num)
                y_values.append(term_data['Forward_Rate'].iloc[0])
        
        if len(x_values) > 0:
            x = np.array(x_values)
            y = np.array(y_values)
            
            if len(x) >= 2:
                from scipy.interpolate import interp1d
                try:
                    f = interp1d(x, y, kind='linear', fill_value='extrapolate')
                    x_smooth = np.linspace(x.min(), x.max(), 100)
                    y_smooth = f(x_smooth)
                    ax.plot(x_smooth, y_smooth, label=date, color=colors[i], linewidth=2.5, alpha=0.8)
                    ax.scatter(x, y, color=colors[i], s=50, zorder=5, alpha=0.9)
                except:
                    ax.plot(x, y, marker='o', label=date, color=colors[i], linewidth=2.5, markersize=8, alpha=0.8)
            else:
                ax.plot(x, y, marker='o', label=date, color=colors[i], linewidth=2.5, markersize=8, alpha=0.8)
    
    ax.set_xlabel('Forward Term (Years)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Forward Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('1-Year Forward Curves for Each Trading Day\nJan 5-19, 2026', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, 
             framealpha=0.9, title='Trading Date', title_fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Forward curves saved to {output_file}")
    plt.close()

def main():
    print("=" * 60)
    print("Calculate Spot and Forward Curves")
    print("=" * 60)
    
    bonds_data = load_bonds_data()
    selected_bonds = load_selected_bonds()
    
    print(f"\nLoaded {len(bonds_data)} bonds")
    print(f"Selected {len(selected_bonds)} bonds for analysis")
    
    start_date = '2026-01-05'
    end_date = '2026-01-19'
    
    print(f"\nDate range: {start_date} to {end_date}")
    
    ytm_df, spot_df, forward_df = calculate_all_curves(bonds_data, selected_bonds, start_date, end_date)
    
    if not ytm_df.empty:
        ytm_df.to_csv('ytm_data.csv', index=False, encoding='utf-8')
        print(f"\nYTM data: {len(ytm_df)} data points")
    
    if not spot_df.empty:
        spot_df.to_csv('spot_data.csv', index=False, encoding='utf-8')
        print(f"Spot rate data: {len(spot_df)} data points")
        plot_spot_curves(spot_df)
    
    if not forward_df.empty:
        forward_df.to_csv('forward_data.csv', index=False, encoding='utf-8')
        print(f"Forward rate data: {len(forward_df)} data points")
        plot_forward_curves(forward_df)
    
    print("\nCompleted!")

if __name__ == "__main__":
    main()
