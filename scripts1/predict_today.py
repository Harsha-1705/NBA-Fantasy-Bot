#!/usr/bin/env python3
"""
Daily NBA Fantasy Points Prediction Script
ALWAYS uses features_combined.csv in data/processed/.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
from joblib import load

warnings.filterwarnings('ignore')

PREDICTION_DATE = datetime(2024, 3, 10).date()  # Change this to today's date as needed
PREDICTION_DATE_STR = PREDICTION_DATE.strftime("%Y%m%d")

def load_model():
    """Load the trained XGBoost model"""
    model_path = 'model/nba_fantasy_xgb_model_with_player.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = load(model_path)
    print(f"✓ Model loaded from {model_path}")
    return model

def load_recent_data():
    """Load and clean the features data"""
    features_path = 'data/processed/features_combined.csv'
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"No features file found at {features_path}")
    df = pd.read_csv(features_path)

    df.columns = df.columns.str.upper()
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"✓ Loaded {len(df)} records from {features_path} (columns: {len(df.columns)})")
    return df

def get_latest_player_data(df):
    """Get the most recent game record for each player"""
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['LAST_GAME_DATE'] = pd.to_datetime(df['LAST_GAME_DATE'], errors='coerce')

    cutoff_date = pd.Timestamp(PREDICTION_DATE) - pd.Timedelta(days=30)
    recent_df = df[df['GAME_DATE'] >= cutoff_date]

    if recent_df.empty:
        raise ValueError(f"No games found between {cutoff_date.date()} and {PREDICTION_DATE}")

    latest_data = recent_df.loc[recent_df.groupby('PLAYER_ID')['GAME_DATE'].idxmax()]
    print(f"✓ Found {len(latest_data)} active players")
    return latest_data

def engineer_features(df):
    """Prepare features for prediction"""
    pred_df = df.copy()
    pred_df['DAYS_REST'] = (pd.Timestamp(PREDICTION_DATE) - pred_df['LAST_GAME_DATE']).dt.days
    pred_df['LAST_GAME_DAYOFWEEK'] = pred_df['LAST_GAME_DATE'].dt.dayofweek
    pred_df['LAST_GAME_MONTH'] = pred_df['LAST_GAME_DATE'].dt.month

    player_mean = df.groupby('PLAYER_ID')['FANTASY_POINTS'].mean()
    pred_df['PLAYER_MEAN_FP'] = pred_df['PLAYER_ID'].map(player_mean).fillna(df['FANTASY_POINTS'].mean())

    if 'IS_HOME' not in pred_df.columns:
        np.random.seed(42)
        pred_df['IS_HOME'] = np.random.choice([0, 1], size=len(pred_df))

    feature_cols = [
        'PLAYER_MEAN_FP',
        'FANTASY_POINTS_ROLL3_MEAN', 'FANTASY_POINTS_ROLL5_MEAN', 'FANTASY_POINTS_ROLL10_MEAN',
        'MIN_ROLL3_MEAN', 'MIN_ROLL5_MEAN', 'MIN_ROLL10_MEAN',
        'IS_HOME', 'DAYS_REST',
        'LAST_GAME_DAYOFWEEK', 'LAST_GAME_MONTH',
        'FANTASY_POINTS_EWM_HL3', 'FANTASY_POINTS_EWM_HL5',
        'MIN_EWM_HL3', 'MIN_EWM_HL5'
    ]

    available_cols = [col for col in feature_cols if col in pred_df.columns]
    missing_cols = set(feature_cols) - set(available_cols)
    if missing_cols:
        print(f"⚠️ Warning: Missing columns in input data: {missing_cols}")

    X_pred = pred_df[available_cols].copy()
    X_pred = X_pred.fillna(X_pred.mean())

    print(f"✓ Feature engineering complete. Using {len(available_cols)} features.")
    return X_pred, pred_df

def get_player_name_column(df):
    for col in df.columns:
        if 'NAME' in col:
            return col
    raise KeyError(f"No player name column found. Existing columns: {df.columns.tolist()}")

def make_predictions(model, X_pred, player_data):
    predictions = model.predict(X_pred)
    name_col = get_player_name_column(player_data)
    results = pd.DataFrame({
        'PLAYER_ID': player_data['PLAYER_ID'].values,
        'PLAYER_NAME': player_data[name_col].values,
        'PRED_FP': predictions
    }).sort_values('PRED_FP', ascending=False)

    print(f"✓ Generated predictions for {len(results)} players")
    return results

def save_predictions(predictions):
    pred_dir = 'predictions'
    os.makedirs(pred_dir, exist_ok=True)
    filename = f"{pred_dir}/{PREDICTION_DATE_STR}.csv"
    predictions.to_csv(filename, index=False)
    print(f"✓ Predictions saved to {filename}")
    print("Top 5 predicted performers:")
    print(predictions.head(5)[['PLAYER_NAME', 'PRED_FP']].to_string(index=False))
    return filename

def main():
    try:
        print("🏀 Starting NBA Fantasy Predictions...")
        print(f"Prediction Date: {PREDICTION_DATE}")
        print("-" * 50)

        model = load_model()
        df = load_recent_data()
        latest_data = get_latest_player_data(df)
        X_pred, player_data = engineer_features(latest_data)
        predictions = make_predictions(model, X_pred, player_data)
        filename = save_predictions(predictions)

        print("-" * 50)
        print("✅ Prediction pipeline completed successfully!")
        print(f"📁 Results saved to: {filename}")

    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        raise

if __name__ == "__main__":
    main()

