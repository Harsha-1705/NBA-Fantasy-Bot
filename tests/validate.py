from pathlib import Path
import pytest
from src.validators import validate_processed_csv

# find the newest processed file (or hard-code one)
latest = sorted(Path("data/processed").glob("fantasy_points_*.csv"))[-1]

def test_processed_csv_validates():
    """Fail if schema rules are broken."""
    validate_processed_csv(latest)   # will raise if invalid
