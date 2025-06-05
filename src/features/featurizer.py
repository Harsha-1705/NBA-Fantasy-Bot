# ──────────────────────────────────────────────────────────────────────────────
#  src/featurizer.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Feature-engineering helpers.

How to use
----------
>>> from src.featurizer import make_features
>>> X = make_features(latest_games_df, full_history_df)
"""

from pathlib import Path
from typing import Iterable, List

import pandas as pd


# --------------------------------------------------------------------------- #
# Rolling helpers                                                             #
# --------------------------------------------------------------------------- #
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
    return df


# --------------------------------------------------------------------------- #
# Public entry-point                                                          #
# --------------------------------------------------------------------------- #
def make_features(latest_rows: pd.DataFrame, full_history: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature matrix for model inference/training.

    Parameters
    ----------
    latest_rows : pd.DataFrame
        One row per (PLAYER_ID, GAME_DATE) you want predictions for
        — typically the **latest** record of each player.
    full_history : pd.DataFrame
        Complete history **up to the same date**; used for rolling windows.

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned with `latest_rows` index.
    """
    df = latest_rows.copy()

    # join rolling windows computed from full history
    rolls = full_history.copy()
    rolls = add_rolling_features(rolls, "fantasy_points")
    if "MIN" in rolls.columns:
        rolls = add_rolling_features(rolls, "MIN")
    rolls = add_home_away_flag(rolls)
    rolls = add_days_rest(rolls)

    # Keep only latest rows after features added
    feature_cols: List[str] = [c for c in rolls.columns if "rolling" in c] + [
        "IS_HOME",
        "DAYS_REST",
    ]
    df = rolls.loc[df.index, feature_cols].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Stand-alone CLI so you can still call "python src/featurizer.py"
# --------------------------------------------------------------------------- #
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
        default="data/processed/features_2023_24.csv",
        show_default=True,
    )
    def main(input: str, output: str):
        print(f"Loading {input} …")
        hist = pd.read_csv(input)
        # for demo: treat whole file as "latest" (one row per game)
        feats = make_features(hist, hist)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(output, index=False)
        print(f"Saved {output}  (shape={feats.shape})")

    main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    _cli()
