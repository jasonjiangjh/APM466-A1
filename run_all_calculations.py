"""
主脚本：运行所有计算（YTM、Spot、Forward、Covariance、PCA）
"""

import sys
import os

# 检查环境
def check_environment():
    try:
        import pandas
        import numpy
        import scipy
        import matplotlib
        return
    except ImportError as e:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        venv_python = os.path.join(script_dir, 'venv', 'bin', 'python3')
        
        if os.path.exists(venv_python):
            print("=" * 60)
            print("检测到未在虚拟环境中运行")
            print("正在尝试使用虚拟环境中的Python...")
            print("=" * 60)
            import subprocess
            result = subprocess.run([venv_python, __file__] + sys.argv[1:])
            sys.exit(result.returncode)
        else:
            print("=" * 60)
            print("错误: 未找到必要的Python模块")
            print("请激活虚拟环境: source venv/bin/activate")
            print("=" * 60)
            sys.exit(1)

check_environment()

import subprocess
import sys

def run_script(script_name, description):
    """运行Python脚本"""
    print(f"\n{'='*60}")
    print(f"运行: {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✓ {description} 完成")
            return True
        else:
            print(f"✗ {description} 失败 (退出码: {result.returncode})")
            return False
    except Exception as e:
        print(f"✗ 运行 {script_name} 时出错: {e}")
        return False

def main():
    print("=" * 60)
    print("运行所有计算脚本")
    print("=" * 60)
    
    scripts = [
        ('calculate_ytm.py', '计算YTM和收益率曲线'),
        ('calculate_curves.py', '计算即期曲线和远期曲线'),
        ('calculate_covariance_pca.py', '计算协方差矩阵和PCA分析')
    ]
    
    results = []
    for script, desc in scripts:
        success = run_script(script, desc)
        results.append((script, success))
    
    print(f"\n{'='*60}")
    print("运行结果总结")
    print(f"{'='*60}")
    for script, success in results:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{script}: {status}")
    
    all_success = all(success for _, success in results)
    if all_success:
        print("\n所有计算完成！")
    else:
        print("\n部分计算失败，请检查错误信息")

if __name__ == "__main__":
    main()
