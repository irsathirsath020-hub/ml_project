"""Central configuration. Edit this file to change the target, features or split."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---- Paths ---------------------------------------------------------------
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DATA_PATH = DATA_RAW_DIR / "training.csv"          # never modified by the project
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "cleaned.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_FILENAME = "best_pipeline.joblib"
METADATA_FILENAME = "model_metadata.json"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# ---- Columns -------------------------------------------------------------
TARGET_COLUMN = "C6H6(GT)"        # benzene, micrograms per cubic metre
DATE_COLUMN = "Date"              # dd/mm/yyyy
TIME_COLUMN = "Time"              # HH.MM.SS
DATETIME_COLUMN = "datetime"      # built during cleaning

# Cheap-sensor inputs named by the problem statement: metal-oxide sensor
# responses, temperature and relative humidity.
FEATURE_COLUMNS = [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
]

# Deliberately NOT used as features (documented in README):
EXCLUDED_COLUMNS = {
    "CO(GT)": "Reference-analyser ground truth; the goal is estimation from cheap sensors.",
    "NMHC(GT)": "Reference analyser; ~90% missing in this file.",
    "NOx(GT)": "Reference analyser; not available from a low-cost sensor array.",
    "NO2(GT)": "Reference analyser; not available from a low-cost sensor array.",
    "AH": "Absolute humidity is derived from T and RH (redundant).",
}

FEATURE_DESCRIPTIONS = {
    "PT08.S1(CO)": "Metal-oxide sensor response (nominally CO), sensor units",
    "PT08.S2(NMHC)": "Metal-oxide sensor response (nominally NMHC), sensor units",
    "PT08.S3(NOx)": "Metal-oxide sensor response (nominally NOx), sensor units",
    "PT08.S4(NO2)": "Metal-oxide sensor response (nominally NO2), sensor units",
    "PT08.S5(O3)": "Metal-oxide sensor response (nominally O3), sensor units",
    "T": "Air temperature, °C",
    "RH": "Relative humidity, %",
}

# ---- Cleaning rules ------------------------------------------------------
MISSING_SENTINEL = -200                     # UCI marks missing readings with -200
# Physically impossible values become missing (None = unbounded side).
PHYSICAL_BOUNDS = {
    "C6H6(GT)": (0.0, None),
    "RH": (0.0, 100.0),
    "T": (-60.0, 70.0),
}
MIN_ROWS_AFTER_CLEANING = 200

# ---- Splitting / validation ---------------------------------------------
TEST_FRACTION = 0.20       # LAST 20% of the timeline is the final test set
CV_SPLITS = 5              # expanding-window TimeSeriesSplit inside the training part
RANDOM_STATE = 42
SELECTION_METRIC = "mae"   # "mae" or "rmse" (cross-validated mean decides the model)
BASELINE_NAME = "Baseline (Mean)"

# Sensitivity check: in this dataset the target is an almost deterministic function of
# PT08.S2(NMHC). train.py also cross-validates the selected model WITHOUT these columns.
ABLATION_FEATURES = ["PT08.S2(NMHC)"]

SUPPORTED_SUFFIXES = {".csv", ".txt"}
ENCODINGS = ("utf-8", "utf-8-sig", "latin-1")
