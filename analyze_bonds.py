"""
分析债券数据，选择10个债券用于构建0-5年收益率曲线
"""

import json
import pandas as pd
from datetime import datetime
import re

def parse_date(date_str):
    """解析日期字符串为datetime对象"""
    if not date_str:
        return None
    
    # 尝试多种日期格式
    formats = ['%m/%d/%Y', '%Y-%m-%d', '%m-%d-%Y', '%Y/%m/%d']
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    return None

def calculate_years_to_maturity(maturity_date_str, reference_date=datetime(2026, 1, 5)):
    """计算从参考日期到到期日的年数"""
    maturity_dt = parse_date(maturity_date_str)
    if not maturity_dt:
        return None
    return (maturity_dt - reference_date).days / 365.25

def load_bonds_data(filename='bonds_data.json'):
    """加载债券数据"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_bonds(bonds_data):
    """分析债券数据"""
    reference_date = datetime(2026, 1, 5)
    
    # 创建分析数据框
    analysis_data = []
    
    for bond in bonds_data:
        maturity_date = bond.get('maturity_date')
        issue_date = bond.get('issue_date')
        coupon = bond.get('coupon')
        isin = bond.get('isin')
        name = bond.get('name')
        
        # 计算到期年限
        years_to_maturity = calculate_years_to_maturity(maturity_date, reference_date)
        
        # 解析coupon为浮点数
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
    
    # 过滤：只保留到期日少于10年的债券
    df = df[df['Years_to_Maturity'].notna()]
    df = df[df['Years_to_Maturity'] > 0]
    df = df[df['Years_to_Maturity'] < 10]
    
    # 按到期年限排序
    df = df.sort_values('Years_to_Maturity')
    
    return df

def select_bonds_for_yield_curve(df):
    """
    选择10个债券用于构建0-5年收益率曲线
    
    选择标准：
    1. 到期年限在0-5年之间
    2. 尽量均匀分布在0-5年区间
    3. 优先选择最近发行的债券（issue date较新）
    4. 避免选择coupon异常高的债券（可能是特殊债券）
    5. 确保每个关键期限（1年、2年、3年、4年、5年）都有代表性债券
    """
    
    # 只保留0-5年的债券
    df_0_5 = df[df['Years_to_Maturity'] <= 5].copy()
    
    if len(df_0_5) < 10:
        print(f"警告: 只有 {len(df_0_5)} 个债券在0-5年范围内，无法选择10个")
        return df_0_5
    
    selected_bonds = []
    
    # 目标期限：0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5年
    target_maturities = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    for target in target_maturities:
        # 找到最接近目标期限的债券
        df_0_5['distance'] = abs(df_0_5['Years_to_Maturity'] - target)
        
        # 排除已经选择的债券
        if selected_bonds:
            selected_isins = [b['ISIN'] for b in selected_bonds]
            df_0_5 = df_0_5[~df_0_5['ISIN'].isin(selected_isins)]
        
        if len(df_0_5) == 0:
            break
        
        # 选择距离目标最近的债券
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
    
    # 如果还没选够10个，从剩余债券中选择
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
    """格式化债券名称，使用作业要求的格式：CAN 2.5 Jun 34"""
    coupon = bond.get('Coupon', '')
    maturity_date = bond.get('Maturity_Date', '')
    
    # 解析到期日期
    maturity_dt = parse_date(maturity_date)
    if maturity_dt:
        month = maturity_dt.strftime('%b')  # Jun, Feb等
        year = str(maturity_dt.year)[-2:]  # 最后两位数字
        return f"CAN {coupon} {month} {year}"
    return bond.get('Name', '')

def main():
    print("=" * 60)
    print("分析债券数据并选择10个债券用于构建收益率曲线")
    print("=" * 60)
    
    # 加载数据
    bonds_data = load_bonds_data()
    print(f"\n加载了 {len(bonds_data)} 个债券")
    
    # 分析数据
    df = analyze_bonds(bonds_data)
    print(f"过滤后（到期日<10年）: {len(df)} 个债券")
    
    # 显示统计信息
    print(f"\n到期年限范围: {df['Years_to_Maturity'].min():.2f} - {df['Years_to_Maturity'].max():.2f} 年")
    print(f"Coupon范围: {df['Coupon'].min():.2f}% - {df['Coupon'].max():.2f}%")
    
    # 选择10个债券
    selected = select_bonds_for_yield_curve(df)
    
    print(f"\n选择的10个债券:")
    print("=" * 60)
    print(f"{'序号':<4} {'债券名称':<20} {'Coupon':<8} {'到期年限':<10} {'到期日期':<12} {'发行日期':<12}")
    print("-" * 60)
    
    for i, (idx, bond) in enumerate(selected.iterrows(), 1):
        bond_name = format_bond_name(bond)
        print(f"{i:<4} {bond_name:<20} {bond['Coupon']:<8.3f} {bond['Years_to_Maturity']:<10.2f} "
              f"{bond['Maturity_Date']:<12} {bond['Issue_Date']:<12}")
    
    # 保存选择的债券
    selected.to_csv('selected_bonds.csv', index=False, encoding='utf-8')
    print(f"\n选择的债券已保存到 selected_bonds.csv")
    
    # 保存为JSON格式
    selected_dict = selected.to_dict('records')
    with open('selected_bonds.json', 'w', encoding='utf-8') as f:
        json.dump(selected_dict, f, indent=2, ensure_ascii=False)
    print(f"选择的债券已保存到 selected_bonds.json")
    
    return selected

if __name__ == "__main__":
    selected_bonds = main()
