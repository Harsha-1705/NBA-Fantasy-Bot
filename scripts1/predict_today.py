#!/usr/bin/env python3
"""
Daily NBA Fantasy Points Prediction Script
Uses data/processed/features_enhanced.csv for prediction.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
from joblib import load

warnings.filterwarnings('ignore')

PREDICTION_DATE = datetime(2024, 2,12).date()
PREDICTION_DATE_STR = PREDICTION_DATE.strftime("%Y%m%d")

def load_model():
    model_path = 'model/nba_fantasy_xgb_model_with_player_weighted.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = load(model_path)
    print(f"✓ Model loaded from {model_path}")
    return model

def load_recent_data():
    path = 'data/processed/features_enhanced.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)

    # Uppercase columns and drop duplicates (fixes Grouper error on PLAYER_ID)
    df.columns = df.columns.str.upper()
    df = df.loc[:, ~df.columns.duplicated()]

    print(f"✓ Loaded {len(df)} rows from {path}")
    return df

def get_latest_player_data(df):
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['LAST_GAME_DATE'] = pd.to_datetime(df['LAST_GAME_DATE'], errors='coerce')

    cutoff = pd.Timestamp(PREDICTION_DATE) - pd.Timedelta(days=30)
    recent = df[df['GAME_DATE'] >= cutoff]
    if recent.empty:
        raise ValueError(f"No games found between {cutoff.date()} and {PREDICTION_DATE}")

    # Group by single PLAYER_ID column
    idx = recent.groupby('PLAYER_ID')['GAME_DATE'].idxmax()
    latest = recent.loc[idx].reset_index(drop=True)
    print(f"✓ Found {len(latest)} active players")
    return latest

def engineer_features(df):
    pred_df = df.copy()
    pred_df['DAYS_REST'] = (pd.Timestamp(PREDICTION_DATE) - pred_df['LAST_GAME_DATE']).dt.days
    pred_df['LAST_GAME_DAYOFWEEK'] = pred_df['LAST_GAME_DATE'].dt.dayofweek
    pred_df['LAST_GAME_MONTH'] = pred_df['LAST_GAME_DATE'].dt.month

    player_mean = df.groupby('PLAYER_ID')['FANTASY_POINTS'].mean()
    pred_df['PLAYER_MEAN_FP'] = pred_df['PLAYER_ID'].map(player_mean).fillna(df['FANTASY_POINTS'].mean())

    # Define expected feature columns
    feature_cols = [
        'PLAYER_MEAN_FP',
        'FANTASY_POINTS_ROLL3_MEAN', 'FANTASY_POINTS_ROLL5_MEAN', 'FANTASY_POINTS_ROLL10_MEAN',
        'MIN_ROLL3_MEAN', 'MIN_ROLL5_MEAN', 'MIN_ROLL10_MEAN',
        'IS_HOME', 'DAYS_REST',
        'LAST_GAME_DAYOFWEEK', 'LAST_GAME_MONTH',
        'FANTASY_POINTS_EWM_HL3', 'FANTASY_POINTS_EWM_HL5',
        'MIN_EWM_HL3', 'MIN_EWM_HL5'
    ]

    # Fill missing expected features with 0
    missing = [c for c in feature_cols if c not in pred_df.columns]
    if missing:
        print(f"⚠️ Adding missing columns: {missing}")
        for c in missing:
            pred_df[c] = 0.0

    X_pred = pred_df[feature_cols].fillna(0.0)
    print(f"✓ Engineered features; using columns: {feature_cols}")
    return X_pred, pred_df

def get_player_name_column(df):
    for col in df.columns:
        if 'NAME' in col:
            return col
    raise KeyError("No PLAYER_NAME column found!")

def make_predictions(model, X_pred, player_data):
    preds = model.predict(X_pred)
    name_col = get_player_name_column(player_data)
    results = pd.DataFrame({
        'PLAYER_ID': player_data['PLAYER_ID'],
        'PLAYER_NAME': player_data[name_col],
        'PRED_FP': preds
    }).sort_values('PRED_FP', ascending=False).reset_index(drop=True)
    print(f"✓ Generated predictions for {len(results)} players")
    return results

def save_predictions(preds):
    os.makedirs('predictions', exist_ok=True)
    out_path = f'predictions/{PREDICTION_DATE_STR}.csv'
    preds.to_csv(out_path, index=False)
    print(f"✓ Saved predictions to {out_path}")
    print("Top 5 performers:")
    print(preds.head(5)[['PLAYER_NAME', 'PRED_FP']].to_string(index=False))
    return out_path

def main():
    print("🏀 Starting NBA Fantasy Predictions...")
    print(f"📅 Prediction Date: {PREDICTION_DATE}")
    print("-" * 50)
    try:
        model = load_model()
        df = load_recent_data()
        latest = get_latest_player_data(df)
        X_pred, player_data = engineer_features(latest)
        predictions = make_predictions(model, X_pred, player_data)
        save_predictions(predictions)
        print("✅ Prediction pipeline completed successfully!")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        raise

if __name__ == "__main__":
    main()

