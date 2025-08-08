#!/usr/bin/env python3
"""
predict_today.py  –  7-day matchup-aware predictions
Window :  2 Feb 2025 → 8 Feb 2025  (inclusive)
Outputs:  predictions/20250202.csv ... 20250208.csv
"""

import os, warnings
from datetime import timedelta
import pandas as pd
from joblib import load

warnings.filterwarnings("ignore")

# ───────── SETTINGS ────────────────────────────────────────
START_DATE   = pd.Timestamp("2025-02-02")
NUM_DAYS     = 7
END_DATE     = START_DATE + timedelta(days=NUM_DAYS - 1)

MODEL_PATH     = "model/nba_fantasy_histgb_model_with_weights.joblib"
HIST_DATA_PATH = "data/processed/features_enhanced.csv"
FIXTURE_PATH   = "src/data/prediction_fixtures.csv"     # team rows
PRED_DIR       = "predictions"
# ───────────────────────────────────────────────────────────

FEATURE_COLS = [
    "PLAYER_MEAN_FP",
    "FANTASY_POINTS_ROLL3_MEAN", "FANTASY_POINTS_ROLL5_MEAN", "FANTASY_POINTS_ROLL10_MEAN",
    "MIN_ROLL3_MEAN", "MIN_ROLL5_MEAN", "MIN_ROLL10_MEAN",
    "IS_HOME", "DAYS_REST",
    "LAST_GAME_DAYOFWEEK", "LAST_GAME_MONTH",
    "FANTASY_POINTS_EWM_HL3", "FANTASY_POINTS_EWM_HL5",
    "MIN_EWM_HL3", "MIN_EWM_HL5",
    "OPP_DEF_RATING", "PTS_VS_DEFADJ",
]

# ───────── LOADERS ─────────────────────────────────────────
def load_model():
    return load(MODEL_PATH)

def load_hist():
    df = pd.read_csv(HIST_DATA_PATH)
    df.columns = df.columns.str.upper()
    df = df.loc[:, ~df.columns.duplicated()]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    # ensure TEAM_ID is Int64 (nullable int)
    df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce").astype("Int64")
    return df.dropna(subset=["TEAM_ID"])

def load_fixture():
    fx = pd.read_csv(FIXTURE_PATH)
    fx.columns = fx.columns.str.upper()
    fx["GAME_DATE"] = pd.to_datetime(fx["GAME_DATE"])
    fx["OPP_TEAM_ID"] = pd.to_numeric(fx["OPP_TEAM_ID"], errors="coerce").astype("Int64")
    fx["IS_HOME"]     = fx["IS_HOME"].astype("int64")
    fx = fx[(fx["GAME_DATE"] >= START_DATE) & (fx["GAME_DATE"] <= END_DATE)]
    return fx.reset_index(drop=True)

# ───────── FIXTURE ➜ PLAYERS ───────────────────────────────
def explode_fixture(team_fx: pd.DataFrame, hist_df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for _, row in team_fx.iterrows():
        team_id = int(row.get("TEAM_ID", row["OPP_TEAM_ID"]))     # if TEAM_ID present use it
        opp_id  = int(row["OPP_TEAM_ID"])
        home    = int(row["IS_HOME"])
        g_ts    = row["GAME_DATE"]

        roster_hist = hist_df[(hist_df["TEAM_ID"] == team_id) &
                              (hist_df["GAME_DATE"] <= g_ts)]
        if roster_hist.empty:
            print(f"  ⚠️  team {team_id} has 0 players in hist up to {g_ts.date()}")
            continue

        latest_idx = roster_hist.groupby("PLAYER_ID")["GAME_DATE"].idxmax()
        snap = roster_hist.loc[latest_idx, ["PLAYER_ID", "PLAYER_NAME"]]

        for _, p in snap.iterrows():
            out_rows.append(
                dict(PLAYER_ID=int(p["PLAYER_ID"]),
                     PLAYER_NAME=p["PLAYER_NAME"],
                     TEAM_ID=team_id,
                     OPP_TEAM_ID=opp_id,
                     IS_HOME=home,
                     GAME_DATE=g_ts.date())
            )
    return pd.DataFrame(out_rows)

# ───────── FEATURES ────────────────────────────────────────
def build_X(merged: pd.DataFrame, pred_ts: pd.Timestamp) -> pd.DataFrame:
    df = merged.copy()
    df["LAST_GAME_DATE"] = pd.to_datetime(df["LAST_GAME_DATE"], errors="coerce")
    df["DAYS_REST"] = (pred_ts - df["LAST_GAME_DATE"]).dt.days
    df["LAST_GAME_DAYOFWEEK"] = df["LAST_GAME_DATE"].dt.dayofweek
    df["LAST_GAME_MONTH"] = df["LAST_GAME_DATE"].dt.month
    df["PLAYER_MEAN_FP"] = df.groupby("PLAYER_ID")["FANTASY_POINTS"].transform("mean")
    for col in FEATURE_COLS:
        if col not in df:
            df[col] = 0.0
    return df[FEATURE_COLS].fillna(0.0)

# ───────── MAIN ────────────────────────────────────────────
def main():
    print(f"Predicting {START_DATE.date()} → {END_DATE.date()}")
    model   = load_model()
    hist_df = load_hist()
    fix_df  = load_fixture()
    ply_fx  = explode_fixture(fix_df, hist_df)
    print(f"✓ Player rows total: {len(ply_fx)}")

    os.makedirs(PRED_DIR, exist_ok=True)

    for day in pd.date_range(START_DATE, END_DATE):
        subset = ply_fx[ply_fx["GAME_DATE"] == day.date()]
        print(f"\n{day.date()}  players:{len(subset)}")
        if subset.empty:
            print("  • No data, skipping")
            continue

        hist_upto = hist_df[hist_df["GAME_DATE"] <= day]
        latest_idx = hist_upto.groupby("PLAYER_ID")["GAME_DATE"].idxmax()
        latest = hist_upto.loc[latest_idx]

        merged = subset.merge(latest, on="PLAYER_ID", how="left", suffixes=("", "_STAT"))
        X = build_X(merged, day)
        preds = model.predict(X)

        out = merged[["PLAYER_ID", "PLAYER_NAME"]].copy()
        out["PRED_FP"] = preds
        out.sort_values("PRED_FP", ascending=False, inplace=True, ignore_index=True)

        file = os.path.join(PRED_DIR, f"{day:%Y%m%d}.csv")
        out.to_csv(file, index=False)
        print(f"  • Saved → {file}")

    print("\n✅  seven-day batch finished.")

if __name__ == "__main__":
    main()
