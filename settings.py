"""Define global paths for easy file find."""
from pathlib import Path
from sys import exit as sysexit

# Define basic paths for global use
ROOT = Path(__file__).resolve().parent
try:
    # Documentation
    ASPELL_DIR_PATH = (ROOT / "docs" / "aspell").resolve(strict=True)
    FILTER_DIR_PATH = (ROOT / "docs" / "filters").resolve(strict=True)
    REPORT_DIR_PATH = (ROOT / "docs" / "report").resolve(strict=True)
    FIGURES_DIR_PATH = (REPORT_DIR_PATH / "figures").resolve(strict=True)
    # Code
    REQUIREMENTS_DIR_PATH = (ROOT / "requirements").resolve(strict=True)
    SOURCE_DIR_PATH = (ROOT / "src").resolve(strict=True)
    DATASETS_DIR_PATH = (SOURCE_DIR_PATH / "datasets").resolve(strict=True)
    PROBEN1_DIR_PATH = (DATASETS_DIR_PATH / "proben1").resolve(strict=True)
except FileNotFoundError as error:
    sysexit(f"{error.strerror}: {error.filename}")

# Possibly not existent folders
LOGS_DIR_PATH = (SOURCE_DIR_PATH / "logs")
LOGS_DIR_PATH.mkdir(exist_ok=True)

# Find report name from the only TeX file in the report directory
REPORT_NAME = list(REPORT_DIR_PATH.glob("*.tex"))[0].stem