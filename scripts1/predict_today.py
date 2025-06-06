#!/usr/bin/env python3
"""
Daily NBA Fantasy Points Prediction Script
ALWAYS uses features_2023_24.csv in data/processed/.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
from joblib import load

warnings.filterwarnings('ignore')

PREDICTION_DATE = datetime(2024, 3, 10).date()  # Use a date within your data range
PREDICTION_DATE_STR = '20240310'

def load_model():
    """Load the trained Random Forest model"""
    model_path = 'model/nba_fantasy_model.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = load(model_path)
    print(f"✓ Model loaded from {model_path}")
    return model

def load_recent_data():
    """Load and sanitize feature data (ALWAYS uses features_2023_24.csv)"""
    features_path = 'data/processed/features_2023_24.csv'
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"No features file found at {features_path}")
    df = pd.read_csv(features_path)

    # Standardize column names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).strip().upper() for col in df.columns.values]
    else:
        df.columns = df.columns.str.upper()

    # Remove duplicate columns (keep first occurrence)
    before_cols = len(df.columns)
    df = df.loc[:, ~df.columns.duplicated()]
    after_cols = len(df.columns)
    if before_cols != after_cols:
        print(f"✓ Removed {before_cols - after_cols} duplicate columns.")

    print(f"✓ Loaded {len(df)} records from {features_path} (columns: {len(df.columns)})")
    print("ACTUAL COLUMNS AFTER PROCESSING:", df.columns.tolist())
    return df

def get_latest_player_data(df):
    """Get most recent player data"""
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['LAST_GAME_DATE'] = pd.to_datetime(df['LAST_GAME_DATE'])
    
    cutoff_date = pd.Timestamp(PREDICTION_DATE) - pd.Timedelta(days=30)
    recent_df = df[df['GAME_DATE'] >= cutoff_date]
    
    if recent_df.empty:
        raise ValueError(f"No games found between {cutoff_date.date()} and {PREDICTION_DATE}")
    
    latest_data = recent_df.loc[recent_df.groupby('PLAYER_ID')['GAME_DATE'].idxmax()]
    print(f"✓ Found {len(latest_data)} active players")
    return latest_data

def engineer_features(df):
    """Create prediction features"""
    pred_df = df.copy()
    pred_df['PRED_GAME_DATE'] = pd.Timestamp(PREDICTION_DATE)
    pred_df['DAYS_REST'] = (pd.Timestamp(PREDICTION_DATE) - pred_df['LAST_GAME_DATE']).dt.days
    
    np.random.seed(42)
    pred_df['IS_HOME'] = np.random.choice([0, 1], size=len(pred_df))
    pred_df['LAST_GAME_DAYOFWEEK'] = pred_df['LAST_GAME_DATE'].dt.dayofweek
    pred_df['LAST_GAME_MONTH'] = pred_df['LAST_GAME_DATE'].dt.month

    feature_cols = [
        'PLAYER_ID', 'FANTASY_POINTS_ROLLING_3', 'FANTASY_POINTS_ROLLING_5',
        'FANTASY_POINTS_ROLLING_10', 'MIN_ROLLING_3', 'MIN_ROLLING_5',
        'MIN_ROLLING_10', 'IS_HOME', 'DAYS_REST', 'LAST_GAME_DAYOFWEEK',
        'LAST_GAME_MONTH'
    ]
    
    available_cols = [col for col in feature_cols if col in pred_df.columns]
    X_pred = pred_df[available_cols].copy()
    X_pred = X_pred.fillna(X_pred.mean())
    
    print(f"✓ Feature engineering complete. Features: {X_pred.columns.tolist()}")
    return X_pred, pred_df

def get_player_name_column(df):
    """Get the correct player name column, flexible matching"""
    for col in df.columns:
        if 'NAME' in col.upper():
            return col
    raise KeyError(f"No player name column found. Existing columns: {df.columns.tolist()}")

def make_predictions(model, X_pred, player_data):
    """Generate predictions (TEMPORARY FIX - KEEPING PLAYER_ID)"""
    predictions = model.predict(X_pred)
    # Find the correct player name column
    name_col = get_player_name_column(player_data)
    results = pd.DataFrame({
        'PLAYER_ID': X_pred['PLAYER_ID'],
        'PLAYER_NAME': player_data[name_col].values,
        'PRED_FP': predictions
    }).sort_values('PRED_FP', ascending=False)
    
    print(f"✓ Generated predictions for {len(results)} players")
    return results

def save_predictions(predictions):
    """Save results to CSV"""
    pred_dir = 'predictions'
    os.makedirs(pred_dir, exist_ok=True)
    filename = f"{pred_dir}/{PREDICTION_DATE_STR}.csv"
    predictions.to_csv(filename, index=False)
    print(f"✓ Predictions saved to {filename}")
    print(f"Top 5 predicted performers:")
    print(predictions.head()[['PLAYER_NAME', 'PRED_FP']].to_string(index=False))
    return filename

def main():
    """Main pipeline"""
    try:
        print("🏀 Starting NBA Fantasy Predictions...")
        print(f"Prediction Date: {PREDICTION_DATE} (hardcoded)")
        print("-" * 50)
        
        model = load_model()
        df = load_recent_data()
        latest_data = get_latest_player_data(df)
        X_pred, player_data = engineer_features(latest_data)
        predictions = make_predictions(model, X_pred, player_data)
        filename = save_predictions(predictions)
        
        print("-" * 50)
        print(f"✅ Prediction pipeline completed successfully!")
        print(f"📁 Results saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        raise

if __name__ == "__main__":
    main()

