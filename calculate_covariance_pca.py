"""
Calculate covariance matrices and perform PCA analysis
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
    """Load YTM data"""
    try:
        return pd.read_csv(filename)
    except:
        return pd.DataFrame()

def load_forward_data(filename='forward_data.csv'):
    """Load forward rate data"""
    try:
        return pd.read_csv(filename)
    except:
        return pd.DataFrame()

def calculate_log_returns_matrix(df, rate_column, date_column, maturity_column):
    """
    Calculate daily log returns and organize into matrix form
    
    Args:
        df: DataFrame with rate data
        rate_column: Rate column name
        date_column: Date column name
        maturity_column: Maturity column name (for YTM) or term identifier (for Forward)
    
    Returns:
        log_returns_matrix: numpy array, each row is a time series for a maturity, each column is a day
        maturities: List of maturities (corresponding to matrix rows)
        dates: List of dates (corresponding to matrix columns, starting from day 2)
    """
    if maturity_column not in df.columns:
        if 'Forward_Term' in df.columns:
            maturity_column = 'Forward_Term'
        else:
            print(f"Error: Cannot find maturity column {maturity_column} or Forward_Term")
            return np.array([]), [], []
    
    maturities = sorted(df[maturity_column].unique())
    all_dates = sorted(df[date_column].unique())
    
    if len(all_dates) < 2:
        print("Error: Need at least 2 days of data to calculate log returns")
        return np.array([]), [], []
    time_series_dict = {}
    for maturity in maturities:
        maturity_data = df[df[maturity_column] == maturity].copy()
        maturity_data = maturity_data.sort_values(date_column)
        
        rates_dict = {}
        for _, row in maturity_data.iterrows():
            rates_dict[row[date_column]] = row[rate_column]
        
        rates = []
        for date in all_dates:
            if date in rates_dict:
                rates.append(rates_dict[date])
            else:
                rates.append(np.nan)
        
        log_returns = []
        for j in range(len(rates) - 1):
            if not np.isnan(rates[j]) and not np.isnan(rates[j+1]) and rates[j] > 0 and rates[j+1] > 0:
                log_ret = np.log(rates[j+1] / rates[j])
                log_returns.append(log_ret)
            else:
                log_returns.append(np.nan)
        
        time_series_dict[maturity] = log_returns
    
    lengths = [len(ts) for ts in time_series_dict.values()]
    if len(lengths) == 0:
        return np.array([]), [], []
    
    min_length = min(lengths)
    max_length = max(lengths)
    
    if min_length != max_length:
        print(f"Warning: Time series lengths inconsistent, using minimum length {min_length}")
    
    matrix_data = []
    valid_maturities = []
    for maturity in maturities:
        ts = time_series_dict[maturity][:min_length]
        valid_count = sum(1 for x in ts if not np.isnan(x))
        if valid_count >= 2:
            matrix_data.append(ts)
            valid_maturities.append(maturity)
    
    if len(matrix_data) == 0:
        return np.array([]), [], []
    
    log_returns_matrix = np.array(matrix_data)
    
    return_dates = all_dates[1:min_length+1] if min_length <= len(all_dates) else all_dates[1:]
    
    return log_returns_matrix, valid_maturities, return_dates

def calculate_covariance_matrix(log_returns_matrix, maturities):
    """
    Calculate covariance matrix
    
    Args:
        log_returns_matrix: numpy array, each row is a time series for a maturity, each column is a day
        maturities: List of maturities (corresponding to matrix rows)
    
    Returns:
        cov_matrix: Covariance matrix (numpy array)
        maturities: List of maturities (corresponding to matrix rows and columns)
    """
    if log_returns_matrix.size == 0 or len(maturities) == 0:
        return np.array([]), []
    
    valid_cols = []
    for col_idx in range(log_returns_matrix.shape[1]):
        col = log_returns_matrix[:, col_idx]
        if not np.any(np.isnan(col)):
            valid_cols.append(col_idx)
    
    if len(valid_cols) == 0:
        print("Error: No completely valid columns")
        return np.array([]), []
    
    clean_matrix = log_returns_matrix[:, valid_cols]
    
    cov_matrix = np.cov(clean_matrix)
    
    return cov_matrix, maturities

def calculate_eigenvalues_eigenvectors(cov_matrix):
    """
    Calculate eigenvalues and eigenvectors
    
    Returns:
        eigenvalues: Eigenvalues (sorted in descending order)
        eigenvectors: Eigenvectors (columns correspond to eigenvalues)
    """
    eigenvalues, eigenvectors = eig(cov_matrix)
    
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors

def plot_eigenvalues(eigenvalues, output_file='eigenvalues.png'):
    """Plot eigenvalues"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n = len(eigenvalues)
    ax.bar(range(1, n+1), eigenvalues, alpha=0.7, color='steelblue')
    ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Eigenvalue', fontsize=12, fontweight='bold')
    ax.set_title('Eigenvalues of Covariance Matrix', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    total_var = np.sum(eigenvalues)
    for i, val in enumerate(eigenvalues):
        pct = (val / total_var) * 100
        ax.text(i+1, val, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Eigenvalues plot saved to {output_file}")
    plt.close()

def plot_eigenvectors(eigenvectors, maturities, output_file='eigenvectors.png'):
    """Plot eigenvectors"""
    n_components = min(3, len(eigenvectors[0]))
    
    fig, axes = plt.subplots(1, n_components, figsize=(6*n_components, 6))
    if n_components == 1:
        axes = [axes]
    
    for i in range(n_components):
        ax = axes[i]
        eigenvector = eigenvectors[:, i]
        
        if isinstance(maturities[0], (int, float)):
            x = maturities
        else:
            x = [int(m.replace('yr-', '').replace('yr', '')) if isinstance(m, str) else m for m in maturities]
        
        ax.plot(x, eigenvector, marker='o', linewidth=2, markersize=8, color='steelblue')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Maturity', fontsize=11, fontweight='bold')
        ax.set_ylabel('Eigenvector Component', fontsize=11, fontweight='bold')
        ax.set_title(f'Principal Component {i+1}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Eigenvectors plot saved to {output_file}")
    plt.close()

def main():
    print("=" * 60)
    print("Calculate Covariance Matrices and PCA Analysis")
    print("=" * 60)
    
    ytm_df = load_ytm_data()
    if ytm_df.empty:
        print("Error: Cannot load YTM data")
        return
    
    print(f"\nLoaded {len(ytm_df)} YTM data points")
    
    ytm_log_returns_matrix, ytm_maturities, ytm_dates = calculate_log_returns_matrix(
        ytm_df, 'YTM', 'Date', 'Years_to_Maturity'
    )
    
    if ytm_log_returns_matrix.size == 0:
        print("Error: Cannot calculate YTM log returns")
        return
    
    print(f"YTM log returns matrix shape: {ytm_log_returns_matrix.shape}")
    print(f"YTM maturities: {len(ytm_maturities)}")
    print(f"Trading days: {len(ytm_dates)}")
    
    ytm_cov, ytm_maturities = calculate_covariance_matrix(ytm_log_returns_matrix, ytm_maturities)
    print(f"\nYTM covariance matrix shape: {ytm_cov.shape}")
    print(f"YTM maturities: {ytm_maturities}")
    
    ytm_cov_df = pd.DataFrame(ytm_cov, index=ytm_maturities, columns=ytm_maturities)
    ytm_cov_df.to_csv('ytm_covariance_matrix.csv')
    print("YTM covariance matrix saved to ytm_covariance_matrix.csv")
    
    ytm_eigenvalues, ytm_eigenvectors = calculate_eigenvalues_eigenvectors(ytm_cov)
    print(f"\nYTM eigenvalues: {ytm_eigenvalues}")
    print(f"First eigenvalue: {ytm_eigenvalues[0]:.6f}")
    print(f"First eigenvector: {ytm_eigenvectors[:, 0]}")
    
    ytm_eigen_df = pd.DataFrame({
        'Eigenvalue': ytm_eigenvalues,
        **{f'PC{i+1}': ytm_eigenvectors[:, i] for i in range(len(ytm_eigenvalues))}
    })
    ytm_eigen_df.to_csv('ytm_eigenvalues_eigenvectors.csv', index=False)
    print("YTM eigenvalues and eigenvectors saved to ytm_eigenvalues_eigenvectors.csv")
    
    plot_eigenvalues(ytm_eigenvalues, 'ytm_eigenvalues.png')
    plot_eigenvectors(ytm_eigenvectors, ytm_maturities, 'ytm_eigenvectors.png')
    
    forward_df = load_forward_data()
    if not forward_df.empty:
        print(f"\nLoaded {len(forward_df)} forward rate data points")
        
        forward_log_returns_matrix, forward_terms, forward_dates = calculate_log_returns_matrix(
            forward_df, 'Forward_Rate', 'Date', 'Forward_Term'
        )
        
        if forward_log_returns_matrix.size == 0:
            print("Error: Cannot calculate forward rate log returns")
            return
        
        print(f"Forward rate log returns matrix shape: {forward_log_returns_matrix.shape}")
        print(f"Forward terms: {len(forward_terms)}")
        print(f"Trading days: {len(forward_dates)}")
        
        forward_cov, forward_terms = calculate_covariance_matrix(forward_log_returns_matrix, forward_terms)
        print(f"\nForward rate covariance matrix shape: {forward_cov.shape}")
        print(f"Forward terms: {forward_terms}")
        
        forward_cov_df = pd.DataFrame(forward_cov, index=forward_terms, columns=forward_terms)
        forward_cov_df.to_csv('forward_covariance_matrix.csv')
        print("Forward rate covariance matrix saved to forward_covariance_matrix.csv")
        
        forward_eigenvalues, forward_eigenvectors = calculate_eigenvalues_eigenvectors(forward_cov)
        print(f"\nForward rate eigenvalues: {forward_eigenvalues}")
        print(f"First eigenvalue: {forward_eigenvalues[0]:.6f}")
        print(f"First eigenvector: {forward_eigenvectors[:, 0]}")
        
        forward_eigen_df = pd.DataFrame({
            'Eigenvalue': forward_eigenvalues,
            **{f'PC{i+1}': forward_eigenvectors[:, i] for i in range(len(forward_eigenvalues))}
        })
        forward_eigen_df.to_csv('forward_eigenvalues_eigenvectors.csv', index=False)
        print("Forward rate eigenvalues and eigenvectors saved to forward_eigenvalues_eigenvectors.csv")
        
        plot_eigenvalues(forward_eigenvalues, 'forward_eigenvalues.png')
        plot_eigenvectors(forward_eigenvectors, forward_terms, 'forward_eigenvectors.png')
    else:
        print("\nWarning: Cannot load forward rate data")
    
    print("\nCompleted!")

if __name__ == "__main__":
    main()
