# featurizer.py
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def add_rolling_features(
    df: pd.DataFrame,
    col: str,
    windows: Iterable[int] = (3, 5, 10),
) -> pd.DataFrame:
    """Add player-level rolling means for the given column using only past games."""
    # shift so current game is excluded
    shifted = df.groupby("PLAYER_ID")[col].shift(1)
    for w in windows:
        df[f"{col}_rolling{w}"] = (
            shifted
            .rolling(window=w, min_periods=1)
            .mean()
            .fillna(df[col].mean())
        )
    return df


def add_home_away_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Create IS_HOME (1 = home, 0 = away) from MATCHUP column."""
    df["IS_HOME"] = df["MATCHUP"].str.contains(r"vs\.?").astype(int)
    return df


def add_days_rest(df: pd.DataFrame) -> pd.DataFrame:
    """Add DAYS_REST based on previous GAME_DATE per player."""
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    last_game = df.groupby("PLAYER_ID")["GAME_DATE"].shift(1)
    df["DAYS_REST"] = (df["GAME_DATE"] - last_game).dt.days.fillna(0).astype(int)
    # keep last_game for possible feature engineering
    df["LAST_GAME_DATE"] = last_game
    return df


def make_features(full_history: pd.DataFrame) -> pd.DataFrame:
    """Generate feature matrix from full game history data."""
    df = full_history.copy()
    # rolling features
    df = add_rolling_features(df, "fantasy_points")
    if "MIN" in df.columns:
        df = add_rolling_features(df, "MIN")
    # context flags
    df = add_home_away_flag(df)
    df = add_days_rest(df)

    # select features
    feature_cols: List[str] = [c for c in df.columns if "rolling" in c] + [
        "IS_HOME", "DAYS_REST"
    ]

    # join original identifiers and features
    return df.loc[:, ["PLAYER_ID", "GAME_DATE", "fantasy_points"] + feature_cols]


def _cli():
    import click

    @click.command()
    @click.option(
        "--input",
        type=str,
        default="data/processed/fantasy_points_2023-24.csv",
        show_default=True,
    )
    @click.option(
        "--output",
        type=str,
        default="data/processed/features_new.csv",
        show_default=True,
    )
    def main(input: str, output: str):
        print(f"Loading {input} …")
        hist = pd.read_csv(input)
        feats = make_features(hist)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(output, index=False)
        print(f"Saved {output}  (shape={feats.shape})")

    main()


if __name__ == "__main__":
    _cli()