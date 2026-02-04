# APM466 Assignment 1

This repository contains code, data, and analysis for Assignment 1 of APM466 (Mathematical Finance) at the University of Toronto.

## Overview

This assignment involves:
- Data collection of Canadian Government bonds from Business Insider Markets
- Yield to Maturity (YTM) calculation and yield curve construction
- Spot rate curve derivation using bootstrapping
- Forward rate curve calculation
- Principal Component Analysis (PCA) on yield and forward rate movements

## Repository Contents

### Python Scripts
- `data_collection_selenium.py` - Collects bond data and historical prices
- `analyze_bonds.py` - Analyzes bonds and selects 10 bonds for yield curve construction
- `calculate_ytm.py` - Calculates YTM and generates yield curves
- `calculate_curves.py` - Calculates spot and forward rate curves
- `calculate_covariance_pca.py` - Computes covariance matrices and performs PCA
- `run_all_calculations.py` - Runs all calculation scripts sequentially
- `validate_data.py` - Validates data at various stages

### Data Files
- `bonds_data.json` - Collected bond data (40 bonds)
- `selected_bonds.json` - Selected 10 bonds for analysis
- `ytm_data.csv`, `spot_data.csv`, `forward_data.csv` - Calculated rates
- `*_covariance_matrix.csv` - Covariance matrices
- `*_eigenvalues_eigenvectors.csv` - PCA results

### Output Files
- `assignment_2.1.1.pdf` - Final report (LaTeX source: `assignment_2.1.1.tex`)
- `*.png` - Plots (yield curves, spot curves, forward curves, eigenvalues, eigenvectors)

## Requirements

See `requirements.txt` for Python dependencies.

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Collect data: `python data_collection_selenium.py`
3. Run all calculations: `python run_all_calculations.py`

## Note

This repository is provided for grader reference only.
