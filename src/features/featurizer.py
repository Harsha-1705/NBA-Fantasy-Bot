import pandas as pd
import numpy as np
import os

def add_rolling_features(df, col, windows=[3, 5, 10]):
    """
    Adds rolling mean features for specified windows.
    """
    for w in windows:
        df[f'{col}_rolling_{w}'] = (
            df.groupby('PLAYER_ID')[col]
            .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
        )
    return df

def add_home_away_flag(df):
    """
    Adds IS_HOME column: 1 if home game, 0 if away.
    """
    df['IS_HOME'] = df['MATCHUP'].str.contains(r'vs\.').astype(int)
    return df

def add_days_rest(df):
    """
    Adds DAYS_REST column: days between games for each player.
    """
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
    df['LAST_GAME_DATE'] = df.groupby('PLAYER_ID')['GAME_DATE'].shift(1)
    df['DAYS_REST'] = (df['GAME_DATE'] - df['LAST_GAME_DATE']).dt.days.fillna(0).astype(int)
    return df

def main():
    # Load processed fantasy points data
    input_file = 'data/processed/fantasy_points_2023_24.csv'
    output_file = 'data/processed/features_2023_24.csv'

    print(f"Loading {input_file} ...")
    df = pd.read_csv(input_file)

    # Add rolling fantasy point averages
    df = add_rolling_features(df, 'fantasy_points', windows=[3, 5, 10])
    # Add rolling minutes averages
    if 'MIN' in df.columns:
        df = add_rolling_features(df, 'MIN', windows=[3, 5, 10])
    # Add home/away flag
    df = add_home_away_flag(df)
    # Add days of rest
    df = add_days_rest(df)

    # Drop rows with any missing values in new features (optional)
    feature_cols = [col for col in df.columns if 'rolling' in col or col in ['IS_HOME', 'DAYS_REST']]
    df = df.dropna(subset=feature_cols)

    # Save out
    df.to_csv(output_file, index=False)
    print(f"Feature-engineered dataset saved to {output_file}")
    print(f"Columns now include: {feature_cols}")

if __name__ == "__main__":
    main()

