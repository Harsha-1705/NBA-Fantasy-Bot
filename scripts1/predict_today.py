#!/usr/bin/env python3
"""
Daily NBA Fantasy Points Prediction Script
Loads yesterday's data, applies feature engineering, and generates predictions
for March 10, 2025 (hardcoded date).
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Hardcoded prediction date
PREDICTION_DATE = datetime(2025, 3, 10).date()
PREDICTION_DATE_STR = '20250310'

def load_model():
    """Load the trained Random Forest model"""
    model_path = 'model/nba_fantasy_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✓ Model loaded from {model_path}")
    return model

def load_recent_data():
    """Load recent game data for feature engineering"""
    # Load the latest processed features file
    features_path = 'data/processed/features_2024.csv'
    if not os.path.exists(features_path):
        features_path = 'data/processed/features_2023_24.csv'
    
    if not os.path.exists(features_path):
        raise FileNotFoundError("No features file found in data/processed/")
    
    df = pd.read_csv(features_path)
    print(f"✓ Loaded {len(df)} records from {features_path}")
    return df

def get_latest_player_data(df):
    """Get the most recent data for each active player"""
    # Convert dates
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['LAST_GAME_DATE'] = pd.to_datetime(df['LAST_GAME_DATE'])
    
    # Get the most recent game for each player (last 30 days to ensure active players)
    cutoff_date = PREDICTION_DATE - timedelta(days=30)
    recent_df = df[df['GAME_DATE'] >= cutoff_date]
    
    # Get latest record per player
    latest_data = recent_df.loc[recent_df.groupby('PLAYER_ID')['GAME_DATE'].idxmax()]
    
    print(f"✓ Found {len(latest_data)} active players")
    return latest_data

def engineer_features(df):
    """Apply the same feature engineering as training"""
    
    # Create a copy for predictions
    pred_df = df.copy()
    
    # Use hardcoded prediction date
    pred_df['PRED_GAME_DATE'] = PREDICTION_DATE
    
    # Calculate days rest from last game
    pred_df['DAYS_REST'] = (pd.to_datetime(PREDICTION_DATE) - pred_df['LAST_GAME_DATE']).dt.days
    
    # For prediction, assume alternating home/away (you might want to get actual schedule)
    np.random.seed(42)  # For reproducible results
    pred_df['IS_HOME'] = np.random.choice([0, 1], size=len(pred_df))
    
    # Extract datetime features from LAST_GAME_DATE
    pred_df['LAST_GAME_DAYOFWEEK'] = pred_df['LAST_GAME_DATE'].dt.dayofweek
    pred_df['LAST_GAME_MONTH'] = pred_df['LAST_GAME_DATE'].dt.month
    
    # Feature columns (same as training)
    feature_cols = [
        'PLAYER_ID', 'fantasy_points_rolling_3', 'fantasy_points_rolling_5', 
        'fantasy_points_rolling_10', 'MIN_rolling_3', 'MIN_rolling_5', 
        'MIN_rolling_10', 'IS_HOME', 'DAYS_REST', 'LAST_GAME_DAYOFWEEK', 
        'LAST_GAME_MONTH'
    ]
    
    # Handle missing columns (adjust column names to match your actual data)
    available_cols = []
    for col in feature_cols:
        if col in pred_df.columns:
            available_cols.append(col)
        elif col.upper() in pred_df.columns:
            pred_df[col] = pred_df[col.upper()]
            available_cols.append(col)
    
    X_pred = pred_df[available_cols]
    
    # Fill any missing values
    X_pred = X_pred.fillna(X_pred.mean())
    
    print(f"✓ Feature engineering complete. Shape: {X_pred.shape}")
    return X_pred, pred_df

def make_predictions(model, X_pred, player_data):
    """Generate predictions using the trained model"""
    
    # Remove PLAYER_ID from features for prediction
    X_features = X_pred.drop(columns=['PLAYER_ID'])
    
    # Make predictions
    predictions = model.predict(X_features)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'PLAYER_ID': X_pred['PLAYER_ID'],
        'PLAYER_NAME': player_data['PLAYER_NAME'],
        'PRED_FP': predictions
    })
    
    # Sort by predicted fantasy points (descending)
    results = results.sort_values('PRED_FP', ascending=False)
    
    print(f"✓ Generated predictions for {len(results)} players")
    return results

def save_predictions(predictions):
    """Save predictions to CSV file"""
    # Create predictions directory if it doesn't exist
    pred_dir = 'predictions'
    os.makedirs(pred_dir, exist_ok=True)
    
    # Use hardcoded date string for filename
    filename = f"{pred_dir}/{PREDICTION_DATE_STR}.csv"
    
    # Save predictions
    predictions.to_csv(filename, index=False)
    
    print(f"✓ Predictions saved to {filename}")
    print(f"Top 5 predicted performers:")
    print(predictions.head()[['PLAYER_NAME', 'PRED_FP']].to_string(index=False))
    
    return filename

def main():
    """Main prediction pipeline"""
    try:
        print("🏀 Starting NBA Fantasy Predictions...")
        print(f"Prediction Date: {PREDICTION_DATE} (hardcoded)")
        print("-" * 50)
        
        # Step 1: Load trained model
        model = load_model()
        
        # Step 2: Load recent data
        df = load_recent_data()
        
        # Step 3: Get latest player data
        latest_data = get_latest_player_data(df)
        
        # Step 4: Engineer features
        X_pred, player_data = engineer_features(latest_data)
        
        # Step 5: Make predictions
        predictions = make_predictions(model, X_pred, player_data)
        
        # Step 6: Save results
        filename = save_predictions(predictions)
        
        print("-" * 50)
        print(f"✅ Prediction pipeline completed successfully!")
        print(f"📁 Results saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Error in prediction pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()

