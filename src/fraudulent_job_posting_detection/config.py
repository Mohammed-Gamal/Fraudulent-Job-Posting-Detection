"""Central configuration: filesystem layout, dataset schema, and shared constants.

Every other module imports paths and column names from here. Nothing else in the
codebase should hardcode a path or a column name -- if the dataset or layout
changes, this is the only file that needs editing.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------
# config.py lives at <root>/src/fraudulent_job_posting_detection/config.py,
# so the project root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

MODELS_DIR = PROJECT_ROOT / "models"

# The CSV as downloaded from Kaggle (dataset: shivamb/real-or-fake-fake-jobposting-prediction).
RAW_CSV = RAW_DIR / "fake_job_postings.csv"

# --------------------------------------------------------------------------
# Dataset schema (EMSCAD -- University of the Aegean)
# --------------------------------------------------------------------------
TARGET = "fraudulent"
ID_COLUMN = "job_id"

# Free-text fields. These carry the signal we mine.
TEXT_FIELDS: list[str] = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

# Short categorical fields. Low cardinality, usable as one-hot features.
CATEGORICAL_FIELDS: list[str] = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]

# Binary indicator flags already encoded as 0/1 in the source data.
BINARY_FLAGS: list[str] = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
]

EXPECTED_COLUMNS: list[str] = (
    [ID_COLUMN]
    + ["title", "location", "department", "salary_range", "company_profile",
       "description", "requirements", "benefits"]
    + BINARY_FLAGS
    + ["employment_type", "required_experience", "required_education",
       "industry", "function"]
    + [TARGET]
)

# Published characteristics of EMSCAD, used as integrity assertions on load.
EXPECTED_ROWS = 17_880
EXPECTED_N_COLUMNS = 18

# --------------------------------------------------------------------------
# Experiment settings
# --------------------------------------------------------------------------
RANDOM_STATE = 42
CV_FOLDS = 5

# The CFPB-style redaction marker. EMSCAD anonymises some entities with
# placeholder tokens; these must be stripped or they become leaked features.
REDACTION_PATTERNS: list[str] = [r"#URL_[0-9a-f]+#", r"#EMAIL_[0-9a-f]+#", r"#PHONE_[0-9a-f]+#"]


def ensure_directories() -> None:
    """Create every output directory. Safe to call repeatedly."""
    for directory in (
        RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
        FIGURES_DIR, TABLES_DIR, MODELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
