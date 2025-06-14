from pathlib import Path
from typing import Iterable, List

import pandas as pd

def add_rolling_features(
    df: pd.DataFrame,
    col: str,
    windows: Iterable[int] = (3, 5, 10),
) -> pd.DataFrame:
    """Add player-level rolling means for the given column."""
    for w in windows:
        df[f"{col}_rolling{w}"] = (
            df.groupby("PLAYER_ID")[col]
            .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
        )
    return df

def add_home_away_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Create IS_HOME (1 = home, 0 = away) from MATCHUP column."""
    df["IS_HOME"] = df["MATCHUP"].str.contains(r"vs\.").astype(int)
    return df

def add_days_rest(df: pd.DataFrame) -> pd.DataFrame:
    """Add DAYS_REST based on previous GAME_DATE per player."""
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    last = df.groupby("PLAYER_ID")["GAME_DATE"].shift(1)
    df["DAYS_REST"] = (df["GAME_DATE"] - last).dt.days.fillna(0).astype(int)
    df["LAST_GAME_DATE"] = last.dt.strftime("%Y-%m-%d").fillna("")  # string format for CSV
    return df

def make_features(latest_rows: pd.DataFrame, full_history: pd.DataFrame) -> pd.DataFrame:
    """Build a feature matrix and combine with original data."""
    df = latest_rows.copy()

    rolls = full_history.copy()
    rolls = add_rolling_features(rolls, "fantasy_points")
    if "MIN" in rolls.columns:
        rolls = add_rolling_features(rolls, "MIN")
    rolls = add_home_away_flag(rolls)
    rolls = add_days_rest(rolls)

    feature_cols: List[str] = [c for c in rolls.columns if "rolling" in c] + [
        "IS_HOME",
        "DAYS_REST",
        "LAST_GAME_DATE",
    ]

    combined = pd.concat([latest_rows.reset_index(drop=True), rolls[feature_cols].reset_index(drop=True)], axis=1)
    return combined

def _cli():
    import click

    @click.command()
    @click.option(
        "--input",
        type=str,
        default="data/processed/fantasy_points_2023_24.csv",
        show_default=True,
    )
    @click.option(
        "--output",
        type=str,
        default="data/processed/features_combined.csv",
        show_default=True,
    )
    def main(input: str, output: str):
        print(f"Loading {input} …")
        hist = pd.read_csv(input)
        feats = make_features(hist, hist)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(output, index=False)
        print(f"Saved {output}  (shape={feats.shape})")

    main()

if __name__ == "__main__":
    _cli()

