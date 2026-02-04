"""
计算协方差矩阵和PCA分析
"""

import json
import pandas as pd
import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_ytm_data(filename='ytm_data.csv'):
    """加载YTM数据"""
    try:
        return pd.read_csv(filename)
    except:
        return pd.DataFrame()

def load_forward_data(filename='forward_data.csv'):
    """加载远期利率数据"""
    try:
        return pd.read_csv(filename)
    except:
        return pd.DataFrame()

def calculate_log_returns_matrix(df, rate_column, date_column, maturity_column):
    """
    计算每日对数收益率并组织成矩阵形式
    
    参数:
    - df: 包含利率数据的DataFrame
    - rate_column: 利率列名
    - date_column: 日期列名
    - maturity_column: 期限列名（用于YTM）或期限标识列（用于Forward）
    
    返回:
    - log_returns_matrix: numpy array，每行是一个期限的时间序列，每列是一天
    - maturities: 期限列表（对应矩阵的行）
    - dates: 日期列表（对应矩阵的列，从第二天开始，因为是对数收益率）
    """
    # 检查列是否存在
    if maturity_column not in df.columns:
        # 对于远期利率，使用Forward_Term
        if 'Forward_Term' in df.columns:
            maturity_column = 'Forward_Term'
        else:
            print(f"错误: 找不到期限列 {maturity_column} 或 Forward_Term")
            return np.array([]), [], []
    
    # 获取所有唯一的期限和日期
    maturities = sorted(df[maturity_column].unique())
    all_dates = sorted(df[date_column].unique())
    
    if len(all_dates) < 2:
        print("错误: 至少需要2天的数据来计算对数收益率")
        return np.array([]), [], []
    
    # 为每个期限创建时间序列
    time_series_dict = {}
    for maturity in maturities:
        maturity_data = df[df[maturity_column] == maturity].copy()
        maturity_data = maturity_data.sort_values(date_column)
        
        # 确保数据按日期排序且完整
        rates_dict = {}
        for _, row in maturity_data.iterrows():
            rates_dict[row[date_column]] = row[rate_column]
        
        # 按日期顺序提取利率
        rates = []
        for date in all_dates:
            if date in rates_dict:
                rates.append(rates_dict[date])
            else:
                rates.append(np.nan)
        
        # 计算对数收益率: log(r_{j+1} / r_j)
        log_returns = []
        for j in range(len(rates) - 1):
            if not np.isnan(rates[j]) and not np.isnan(rates[j+1]) and rates[j] > 0 and rates[j+1] > 0:
                log_ret = np.log(rates[j+1] / rates[j])
                log_returns.append(log_ret)
            else:
                log_returns.append(np.nan)
        
        time_series_dict[maturity] = log_returns
    
    # 转换为矩阵：每行是一个期限，每列是一天
    # 找到所有时间序列的共同长度
    lengths = [len(ts) for ts in time_series_dict.values()]
    if len(lengths) == 0:
        return np.array([]), [], []
    
    min_length = min(lengths)
    max_length = max(lengths)
    
    if min_length != max_length:
        print(f"警告: 时间序列长度不一致，使用最小长度 {min_length}")
    
    # 创建矩阵
    matrix_data = []
    valid_maturities = []
    for maturity in maturities:
        ts = time_series_dict[maturity][:min_length]
        # 检查是否有足够的有效数据
        valid_count = sum(1 for x in ts if not np.isnan(x))
        if valid_count >= 2:  # 至少需要2个有效值
            matrix_data.append(ts)
            valid_maturities.append(maturity)
    
    if len(matrix_data) == 0:
        return np.array([]), [], []
    
    log_returns_matrix = np.array(matrix_data)
    
    # 返回日期（从第二天开始，因为是对数收益率）
    return_dates = all_dates[1:min_length+1] if min_length <= len(all_dates) else all_dates[1:]
    
    return log_returns_matrix, valid_maturities, return_dates

def calculate_covariance_matrix(log_returns_matrix, maturities):
    """
    计算协方差矩阵
    
    参数:
    - log_returns_matrix: numpy array，每行是一个期限的时间序列，每列是一天
    - maturities: 期限列表（对应矩阵的行）
    
    返回:
    - cov_matrix: 协方差矩阵（numpy array）
    - maturities: 期限列表（对应矩阵的行和列）
    """
    if log_returns_matrix.size == 0 or len(maturities) == 0:
        return np.array([]), []
    
    # 处理NaN值：使用列均值填充，或者删除包含NaN的列
    # 更安全的方法：只使用没有NaN的列
    valid_cols = []
    for col_idx in range(log_returns_matrix.shape[1]):
        col = log_returns_matrix[:, col_idx]
        if not np.any(np.isnan(col)):
            valid_cols.append(col_idx)
    
    if len(valid_cols) == 0:
        print("错误: 没有完全有效的列")
        return np.array([]), []
    
    # 只使用有效列
    clean_matrix = log_returns_matrix[:, valid_cols]
    
    # 计算协方差矩阵（每行是一个变量，每列是一个观测值）
    # np.cov期望每行是一个变量，每列是一个观测值
    cov_matrix = np.cov(clean_matrix)
    
    return cov_matrix, maturities

def calculate_eigenvalues_eigenvectors(cov_matrix):
    """
    计算特征值和特征向量
    
    返回:
    - eigenvalues: 特征值（按降序排列）
    - eigenvectors: 特征向量（列对应特征值）
    """
    eigenvalues, eigenvectors = eig(cov_matrix)
    
    # 确保特征值是实数（去除小的虚部）
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    # 按特征值降序排列
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors

def plot_eigenvalues(eigenvalues, output_file='eigenvalues.png'):
    """绘制特征值"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n = len(eigenvalues)
    ax.bar(range(1, n+1), eigenvalues, alpha=0.7, color='steelblue')
    ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Eigenvalue', fontsize=12, fontweight='bold')
    ax.set_title('Eigenvalues of Covariance Matrix', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加解释方差百分比
    total_var = np.sum(eigenvalues)
    for i, val in enumerate(eigenvalues):
        pct = (val / total_var) * 100
        ax.text(i+1, val, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"特征值图已保存到 {output_file}")
    plt.close()

def plot_eigenvectors(eigenvectors, maturities, output_file='eigenvectors.png'):
    """绘制特征向量"""
    n_components = min(3, len(eigenvectors[0]))  # 显示前3个主成分
    
    fig, axes = plt.subplots(1, n_components, figsize=(6*n_components, 6))
    if n_components == 1:
        axes = [axes]
    
    for i in range(n_components):
        ax = axes[i]
        eigenvector = eigenvectors[:, i]
        
        # 如果maturities是数字，直接使用；否则转换为数字
        if isinstance(maturities[0], (int, float)):
            x = maturities
        else:
            # 对于远期利率，转换为数字
            x = [int(m.replace('yr-', '').replace('yr', '')) if isinstance(m, str) else m for m in maturities]
        
        ax.plot(x, eigenvector, marker='o', linewidth=2, markersize=8, color='steelblue')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Maturity', fontsize=11, fontweight='bold')
        ax.set_ylabel('Eigenvector Component', fontsize=11, fontweight='bold')
        ax.set_title(f'Principal Component {i+1}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"特征向量图已保存到 {output_file}")
    plt.close()

def main():
    print("=" * 60)
    print("计算协方差矩阵和PCA分析")
    print("=" * 60)
    
    # 加载YTM数据
    ytm_df = load_ytm_data()
    if ytm_df.empty:
        print("错误: 无法加载YTM数据")
        return
    
    print(f"\n加载了 {len(ytm_df)} 个YTM数据点")
    
    # 计算YTM的对数收益率矩阵
    ytm_log_returns_matrix, ytm_maturities, ytm_dates = calculate_log_returns_matrix(
        ytm_df, 'YTM', 'Date', 'Years_to_Maturity'
    )
    
    if ytm_log_returns_matrix.size == 0:
        print("错误: 无法计算YTM对数收益率")
        return
    
    print(f"YTM对数收益率矩阵形状: {ytm_log_returns_matrix.shape}")
    print(f"YTM期限数量: {len(ytm_maturities)}")
    print(f"交易日数量: {len(ytm_dates)}")
    
    # 计算YTM协方差矩阵
    ytm_cov, ytm_maturities = calculate_covariance_matrix(ytm_log_returns_matrix, ytm_maturities)
    print(f"\nYTM协方差矩阵形状: {ytm_cov.shape}")
    print(f"YTM期限: {ytm_maturities}")
    
    # 保存YTM协方差矩阵
    ytm_cov_df = pd.DataFrame(ytm_cov, index=ytm_maturities, columns=ytm_maturities)
    ytm_cov_df.to_csv('ytm_covariance_matrix.csv')
    print("YTM协方差矩阵已保存到 ytm_covariance_matrix.csv")
    
    # 计算YTM特征值和特征向量
    ytm_eigenvalues, ytm_eigenvectors = calculate_eigenvalues_eigenvectors(ytm_cov)
    print(f"\nYTM特征值: {ytm_eigenvalues}")
    print(f"第一个特征值: {ytm_eigenvalues[0]:.6f}")
    print(f"第一个特征向量: {ytm_eigenvectors[:, 0]}")
    
    # 保存YTM特征值和特征向量
    ytm_eigen_df = pd.DataFrame({
        'Eigenvalue': ytm_eigenvalues,
        **{f'PC{i+1}': ytm_eigenvectors[:, i] for i in range(len(ytm_eigenvalues))}
    })
    ytm_eigen_df.to_csv('ytm_eigenvalues_eigenvectors.csv', index=False)
    print("YTM特征值和特征向量已保存到 ytm_eigenvalues_eigenvectors.csv")
    
    # 绘制YTM特征值
    plot_eigenvalues(ytm_eigenvalues, 'ytm_eigenvalues.png')
    
    # 绘制YTM特征向量
    plot_eigenvectors(ytm_eigenvectors, ytm_maturities, 'ytm_eigenvectors.png')
    
    # 加载远期利率数据
    forward_df = load_forward_data()
    if not forward_df.empty:
        print(f"\n加载了 {len(forward_df)} 个远期利率数据点")
        
        # 计算远期利率的对数收益率矩阵
        forward_log_returns_matrix, forward_terms, forward_dates = calculate_log_returns_matrix(
            forward_df, 'Forward_Rate', 'Date', 'Forward_Term'
        )
        
        if forward_log_returns_matrix.size == 0:
            print("错误: 无法计算远期利率对数收益率")
            return
        
        print(f"远期利率对数收益率矩阵形状: {forward_log_returns_matrix.shape}")
        print(f"远期期限数量: {len(forward_terms)}")
        print(f"交易日数量: {len(forward_dates)}")
        
        # 计算远期利率协方差矩阵
        forward_cov, forward_terms = calculate_covariance_matrix(forward_log_returns_matrix, forward_terms)
        print(f"\n远期利率协方差矩阵形状: {forward_cov.shape}")
        print(f"远期期限: {forward_terms}")
        
        # 保存远期利率协方差矩阵
        forward_cov_df = pd.DataFrame(forward_cov, index=forward_terms, columns=forward_terms)
        forward_cov_df.to_csv('forward_covariance_matrix.csv')
        print("远期利率协方差矩阵已保存到 forward_covariance_matrix.csv")
        
        # 计算远期利率特征值和特征向量
        forward_eigenvalues, forward_eigenvectors = calculate_eigenvalues_eigenvectors(forward_cov)
        print(f"\n远期利率特征值: {forward_eigenvalues}")
        print(f"第一个特征值: {forward_eigenvalues[0]:.6f}")
        print(f"第一个特征向量: {forward_eigenvectors[:, 0]}")
        
        # 保存远期利率特征值和特征向量
        forward_eigen_df = pd.DataFrame({
            'Eigenvalue': forward_eigenvalues,
            **{f'PC{i+1}': forward_eigenvectors[:, i] for i in range(len(forward_eigenvalues))}
        })
        forward_eigen_df.to_csv('forward_eigenvalues_eigenvectors.csv', index=False)
        print("远期利率特征值和特征向量已保存到 forward_eigenvalues_eigenvectors.csv")
        
        # 绘制远期利率特征值
        plot_eigenvalues(forward_eigenvalues, 'forward_eigenvalues.png')
        
        # 绘制远期利率特征向量
        plot_eigenvectors(forward_eigenvectors, forward_terms, 'forward_eigenvectors.png')
    else:
        print("\n警告: 无法加载远期利率数据")
    
    print("\n完成！")

if __name__ == "__main__":
    main()
