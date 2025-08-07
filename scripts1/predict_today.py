# predict_for_date.py
import pandas as pd
from joblib import load
from datetime import datetime

# === Step 1: Load trained model ===
model = load("model/nba_fantasy_histgb_model_with_weights.joblib")

# === Step 2: Load enhanced features dataset ===
df = pd.read_csv("data/processed/features_enhanced.csv")
df.columns = df.columns.str.upper()

# === Step 3: Filter for 2nd Feb 2025 games ===
target_date = "2025-02-04"
df_target = df[df["GAME_DATE"] == target_date]

# === Step 4: Ensure required features exist ===
feature_cols = [
    'PLAYER_MEAN_FP',
    'FANTASY_POINTS_ROLL3_MEAN', 'FANTASY_POINTS_ROLL3_STD',
    'FANTASY_POINTS_ROLL5_MEAN', 'FANTASY_POINTS_ROLL5_STD',
    'FANTASY_POINTS_ROLL10_MEAN', 'FANTASY_POINTS_ROLL10_STD',
    'FANTASY_POINTS_EWM_HL3', 'FANTASY_POINTS_EWM_HL5',
    'MIN', 'MIN_ROLL3_MEAN', 'MIN_ROLL3_STD',
    'MIN_ROLL5_MEAN', 'MIN_ROLL5_STD',
    'MIN_ROLL10_MEAN', 'MIN_ROLL10_STD',
    'MIN_EWM_HL3', 'MIN_EWM_HL5',
    'OPP_DEF_RATING',       # opponent defense
    'PTS_VS_DEFADJ'         # adjusted points vs defense
]


# Fill missing feature columns with 0
for col in feature_cols:
    if col not in df_target.columns:
        df_target[col] = 0.0

# Drop rows with missing values in feature columns
df_target = df_target.dropna(subset=feature_cols)

# === Step 5: Predict ===
df_target["PREDICTED_FANTASY_POINTS"] = model.predict(df_target[feature_cols])

# === Step 6: Save results ===
output_file = f"predictions/predictions_{target_date.replace('-', '')}.csv"
df_target[["PLAYER_NAME", "TEAM_ABBREVIATION", "OPPONENT", "GAME_DATE", "PREDICTED_FANTASY_POINTS"]].to_csv(output_file, index=False)

# === Step 7: Display Top 10 ===
print("Top 10 Predictions for", target_date)
print(df_target[["PLAYER_NAME", "TEAM_ABBREVIATION", "OPPONENT", "PREDICTED_FANTASY_POINTS"]].sort_values(by="PREDICTED_FANTASY_POINTS", ascending=False).head(10))
