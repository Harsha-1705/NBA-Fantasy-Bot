from pathlib import Path
import pandas as pd
import pandera as pa
from pandera import Column, Check

#: ----------------- schema definition --------------------------------------
player_schema = pa.DataFrameSchema(
    {
        "PLAYER_ID": pa.Column(int, checks=Check.gt(0)),
        "GAME_DATE": pa.Column(pa.DateTime),
        "MIN": Column(float, checks=Check.ge(4)),              # at least 4 min
        "fantasy_points": Column(float, checks=Check.not_null()),
    },
    checks=[
        Check(lambda df: df["GAME_DATE"].is_monotonic_increasing,  # dates sorted
              error="GAME_DATE must be monotonic ascending"),
    ],
)

#: ----------------- public API ---------------------------------------------
def validate_processed_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a processed CSV and raise an error if it violates the schema.
    Returns the validated DataFrame so callers can reuse it.
    """
    df = pd.read_csv(path, parse_dates=["GAME_DATE"])
    player_schema.validate(df, lazy=False)   # lazy=False so it fails fast
    return df
