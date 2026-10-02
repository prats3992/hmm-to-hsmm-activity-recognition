"""Project paths and shared experiment settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

# Number of principal components kept after standardisation (~90% of variance).
N_PCA_COMPONENTS = 65

RANDOM_STATE = 42

# Test subjects used for the per-subject decoding plots and comparisons.
FOCUS_SUBJECTS = (2, 9, 12)

# Per-class emission model: hidden sub-states per activity and mixtures per sub-state.
N_SUBSTATES = 3
N_MIXTURES = 2

# Longest segment (in windows) considered by the duration-explicit Viterbi decoder.
HSMM_MAX_DURATION = 50
