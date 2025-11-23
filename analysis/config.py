"""Configuration for PCA analysis of SWE-bench dataset."""

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
DATASETS_DIR = BASE_DIR / "datasets"
PLOTS_DIR = ANALYSIS_DIR / "plots"

# dataset configuration
# Use the full SWE-bench verified evaluation data
DATASET_PATH = BASE_DIR / "moatless" / "benchmark" / "swebench_verified_all_evaluations.json"
OUTPUT_CSV = ANALYSIS_DIR / "dataset_features.csv"

# PCA params
PCA_N_COMPONENTS = 3  # number of principal components to compute
PCA_VARIANCE_THRESHOLD = 0.95  # variance threshold for auto component selection
SCALING_METHOD = "standard"  # options: "standard", "minmax", "robust"; for now we'll use "standard"

# feature selection
COMPUTE_REPO_FEATURES = True  # may require repo cloning
COMPUTE_COMPLEXITY_METRICS = True
COMPUTE_PATCH_ENTROPY = True

# visualization settings
FIGURE_DPI = 150
FIGURE_SIZE = (10, 8)
INTERACTIVE_PLOTS = True  # this is to use plotly for interactive plots but we can change it

# cache settings
USE_CACHE = True
CACHE_DIR = ANALYSIS_DIR / ".cache"

# ensure directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
if USE_CACHE:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

