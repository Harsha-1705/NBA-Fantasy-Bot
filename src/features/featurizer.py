import pandas as pd
import numpy as np
from pathlib import Path

def add_rolling_features(df: pd.DataFrame, col: str, windows: list[int] = [3, 5, 10]) -> pd.DataFrame:
    grp = df.groupby("Player_ID")[col]
    for w in windows:
        df[f"{col}_roll{w}_mean"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f"{col}_roll{w}_std"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0))
    return df

def add_ewm_features(df: pd.DataFrame, col: str, halflives: list[int] = [3, 5]) -> pd.DataFrame:
    grp = df.groupby("Player_ID")[col]
    for hl in halflives:
        df[f"{col}_ewm_hl{hl}"] = grp.transform(lambda x: x.shift(1).ewm(halflife=hl, adjust=False).mean())
    return df

def add_game_flags(df: pd.DataFrame) -> pd.DataFrame:
    # Strip quotes and convert to datetime
    df["GAME_DATE"] = (
        df["GAME_DATE"]
        .astype(str)
        .str.replace('"', '', regex=False)
        .str.strip()
    )
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y", errors="coerce")

    if df["GAME_DATE"].isna().sum() > 0:
        print(f"⚠️ Warning: {df['GAME_DATE'].isna().sum()} GAME_DATE values could not be parsed and were set to NaT")

    df["IS_HOME"] = df["MATCHUP"].str.contains(r"vs\.").astype(int)
    df["OPPONENT_PARSED"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s*([A-Z]{3})")

    df["GAME_DOW"] = df["GAME_DATE"].dt.dayofweek
    df["GAME_MONTH"] = df["GAME_DATE"].dt.month

    df = df.sort_values(["Player_ID", "GAME_DATE"])
    df["LAST_GAME_DATE"] = df.groupby("Player_ID")["GAME_DATE"].shift(1)
    df["LAST_DOW"] = df["LAST_GAME_DATE"].dt.dayofweek.fillna(-1).astype(int)
    df["LAST_MONTH"] = df["LAST_GAME_DATE"].dt.month.fillna(0).astype(int)
    df["DAYS_REST"] = (df["GAME_DATE"] - df["LAST_GAME_DATE"]).dt.days.fillna(0).astype(int)

    return df

def add_cumulative_count(df: pd.DataFrame) -> pd.DataFrame:
    df["GAME_NUMBER"] = df.groupby("Player_ID").cumcount() + 1
    return df

def make_features(input_csv: str, output_csv: str) -> None:
    df = pd.read_csv(input_csv)

    df = add_game_flags(df)
    df = add_rolling_features(df, "fantasy_points", windows=[3, 5, 10])
    if "MIN" in df.columns:
        df = add_rolling_features(df, "MIN", windows=[3, 5, 10])

    df = add_ewm_features(df, "fantasy_points", halflives=[3, 5])
    if "MIN" in df.columns:
        df = add_ewm_features(df, "MIN", halflives=[3, 5])

    df = add_cumulative_count(df)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Enhanced features saved to {output_csv} (shape={df.shape})")

if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--input", "-i", default="data/processed/fantasy_points_together.csv", help="Input CSV path")
    @click.option("--output", "-o", default="data/processed/features_enhanced.csv", help="Output CSV path")
    def cli(input: str, output: str):
        make_features(input, output)

    cli()

