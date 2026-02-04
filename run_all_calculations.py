"""
Main script: Run all calculations (YTM, Spot, Forward, Covariance, PCA)
"""

import sys
import os

def check_environment():
    """Check if required packages are available"""
    try:
        import pandas
        import numpy
        import scipy
        import matplotlib
        return
    except ImportError:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        venv_python = os.path.join(script_dir, 'venv', 'bin', 'python3')
        
        if os.path.exists(venv_python):
            print("=" * 60)
            print("Not running in virtual environment")
            print("Attempting to use virtual environment Python...")
            print("=" * 60)
            import subprocess
            result = subprocess.run([venv_python, __file__] + sys.argv[1:])
            sys.exit(result.returncode)
        else:
            print("=" * 60)
            print("ERROR: Required Python modules not found")
            print("Please activate virtual environment: source venv/bin/activate")
            print("=" * 60)
            sys.exit(1)

check_environment()

import subprocess

def run_script(script_name, description):
    """Run a Python script"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✓ {description} completed")
            return True
        else:
            print(f"✗ {description} failed (exit code: {result.returncode})")
            return False
    except Exception as e:
        print(f"✗ Error running {script_name}: {e}")
        return False

def main():
    print("=" * 60)
    print("Running All Calculation Scripts")
    print("=" * 60)
    
    scripts = [
        ('calculate_ytm.py', 'Calculate YTM and yield curves'),
        ('calculate_curves.py', 'Calculate spot and forward curves'),
        ('calculate_covariance_pca.py', 'Calculate covariance matrices and PCA')
    ]
    
    results = []
    for script, desc in scripts:
        success = run_script(script, desc)
        results.append((script, success))
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for script, success in results:
        status = "✓ Success" if success else "✗ Failed"
        print(f"{script}: {status}")
    
    all_success = all(success for _, success in results)
    if all_success:
        print("\nAll calculations completed!")
    else:
        print("\nSome calculations failed. Please check error messages.")

if __name__ == "__main__":
    main()
