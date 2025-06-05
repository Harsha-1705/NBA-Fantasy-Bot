#!/usr/bin/env python3
"""
predict_today.py
----------------
Project fantasy points for a given slate (default: *tomorrow*).

How it works
------------
1. Read processed history (fantasy_points_<season>.csv)
2. Trim rows strictly earlier than `--date`
3. Build feature set for players whose teams play on `--date`
4. Run trained model -> fantasy-point projection
5. Save predictions/YYYYMMDD.csv

Run examples
------------
python scripts/predict_today.py                    # predicts tomorrow
python scripts/predict_today.py --date 2024-10-29  # predict on a fixed date
"""

import datetime as dt
from pathlib import Path
import sys

import click
import joblib
import pandas as pd
from nba_api.stats.endpoints import ScoreboardV2

import os

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts1/
project_root = os.path.abspath(os.path.join(current_dir, '..'))  # Go one level up to root
sys.path.append(project_root)



# === import your own featurizer ===
from src.features.featurizer import make_features  # <- update if your path differs


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def next_day() -> dt.date:
    return dt.date.today() + dt.timedelta(days=1)

def schedule_teams(game_date: dt.date) -> set[int]:
    """Return set of TEAM_IDs that play on `game_date`."""
    mmddyyyy = game_date.strftime("%m/%d/%Y")
    print(f"Requesting games for: {mmddyyyy}")  # DEBUG LINE

    sb = ScoreboardV2(game_date=mmddyyyy)

    try:
        df_list = sb.get_data_frames()
    except Exception as e:
        print(f"Error retrieving scoreboard data: {e}")
        sys.exit(1)

    if not df_list or df_list[0].empty:
        click.echo(f"No NBA games found for {game_date}, exiting.")
        sys.exit(0)

    df = df_list[0]
    return set(df["HOME_TEAM_ID"]).union(df["VISITOR_TEAM_ID"])


def load_history(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["GAME_DATE"])


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
@click.command()
@click.option(
    "--date",
    "input_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Game date to predict (YYYY-MM-DD). Defaults to tomorrow.",
)
@click.option(
    "--season",
    default="2024",  # adjust if you switch seasons
    show_default=True,
    help="Season label used in processed filename & model filename.",
)
def main(input_date, season):
    game_date: dt.date = (input_date.date() if input_date else next_day())

    # ---------- paths ---------- #
    proc_file = Path(f"data/processed/fantasy_points_{season}.csv")
    model_file = Path(f"models/baseline.pkl")
    out_dir = Path("predictions")
    out_dir.mkdir(exist_ok=True)

    # ---------- load data ---------- #
    hist = load_history(proc_file)
    hist = hist[hist["GAME_DATE"].dt.date < game_date]  # no leakage
    teams_today = schedule_teams(game_date)
    latest_rows = (
        hist[hist["TEAM_ID"].isin(teams_today)]
        .sort_values("GAME_DATE")
        .groupby("PLAYER_ID")
        .tail(1)
        .reset_index(drop=True)
    )
    if latest_rows.empty:
        click.echo("No players found for that slate—abort.")
        sys.exit(1)

    # ---------- features & predict ---------- #
    X_tonight = make_features(latest_rows, hist)
    model = joblib.load(model_file)
    latest_rows["PRED_FP"] = model.predict(X_tonight)

    # ---------- save ---------- #
    ymd = game_date.strftime("%Y%m%d")
    outfile = out_dir / f"{ymd}.csv"
    latest_rows[["PLAYER_ID", "PLAYER_NAME", "PRED_FP"]].to_csv(
        outfile, index=False
    )
    click.echo(f"✅  wrote {len(latest_rows)} rows → {outfile}")


if __name__ == "__main__":
    main()
