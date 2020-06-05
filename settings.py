from pathlib import Path
from sys import exit as sysexit

# Define basic paths for global use
ROOT = Path(__file__).resolve().parent
try:
    ASPELL_DIR_PATH = ROOT / "docs" / "aspell"
    REPORT_DIR_PATH = ROOT / "docs" / "report"
    FILTER_DIR_PATH = ROOT / "docs" / "filters"
except FileNotFoundError as error:
    sysexit(f"{error.strerror}: {error.filename}")

# Find report name from the only TeX file in the report directory
REPORT_NAME = list(REPORT_DIR_PATH.glob("*.tex"))[0].stem
