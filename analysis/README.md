# SWE-bench PCA Analysis

Principal Component Analysis (PCA) framework for understanding the natural structure of software engineering problems in the SWE-bench Verified dataset.

## Quick Start

### 1. Install Dependencies

```bash
# Using Poetry (recommended)
poetry install --with analysis

# Or using pip
pip install -r "analysis/requirements.txt"
```

### 2. Extract Features

```bash
poetry run python analysis/extract_dataset_features.py
```

Processes all 500 SWE-bench Verified instances and creates `dataset_features.csv` with 25+ features per instance.

### 3. Run Analysis

Then run all cells of `analysis/pca_analysis.ipynb` (Cell → Run All) to perform PCA, generate visualizations, and interpret results.

## What This Does

1. **Extracts features** from SWE-bench instances (problem length, patch size, test counts, complexity metrics)
2. **Performs PCA** to reduce 25+ features to 2-3 principal components
3. **Visualizes** problem space with 2D/3D plots, biplots, heatmaps
4. **Interprets** components to understand what drives variance
5. **Compares** repositories (django, flask, matplotlib, etc.)
6. **Model Performance Overlaying** overlaying model performance with PCA components

## Files Structure

```
analysis/
├── config.py                      # Configuration (paths, PCA parameters)
├── feature_utils.py               # Feature extraction utilities
├── viz_utils.py                   # Visualization functions
├── extract_dataset_features.py    # Main extraction script
├── pca_analysis.ipynb            # Interactive analysis notebook
├── README.md                     # This file
```

## Features Extracted

### From SWE-bench Paper (Table 11)

- **Problem**: Length (characters, words, lines)
- **Context**: Files, lines, directories in context
- **Patch**: Files edited, functions edited, lines added/removed/edited, hunks
- **Tests**: Fail-to-pass, pass-to-pass counts
- **Repository**: Repository type/name

### Additional Metrics

- Patch complexity score (weighted combination)
- Patch entropy (change diversity)
- Add/remove ratio
- Average hunk size
- Problem size category (small/medium/large)
- Patch scope category (focused/moderate/widespread)

**Total**: 25+ features per instance

## Configuration

Edit `config.py` to customize:

```python
# Dataset
DATASET_PATH = DATASETS_DIR / "verified_dataset.json"

# PCA parameters
PCA_N_COMPONENTS = 3
PCA_VARIANCE_THRESHOLD = 0.95
SCALING_METHOD = "standard"

# Visualization
FIGURE_DPI = 150
INTERACTIVE_PLOTS = True  # Use plotly for 3D plots
```

## References

- SWE-bench paper: https://arxiv.org/abs/2310.06770 (Table 11, Appendix A.5)
- SWE-Search paper: https://arxiv.org/abs/2410.20285 (Appendix E)
- Scikit-learn PCA: https://scikit-learn.org/stable/modules/decomposition.html#pca
