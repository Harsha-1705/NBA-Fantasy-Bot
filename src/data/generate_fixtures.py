#!/usr/bin/env python3
"""
Build a player-level fixture for games on 3 Feb 2025
===================================================

Output file:
    src/data/player_fixture_20250203.csv

Columns:
    PLAYER_ID, PLAYER_NAME,
    TEAM_ID, OPP_TEAM_ID,
    IS_HOME (1 = home, 0 = away),
    GAME_DATE  (always 2025-02-03)

Assumptions:
• historical table: data/processed/features_enhanced.csv
• NBA schedule available via nba_api (season 2024-25)
• We treat the latest team for each player up to 2 Feb 2025
  as their current roster.
"""

import os
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

# ------------- paths ---------------------------------------------------------
HIST_PATH = "data/processed/features_enhanced.csv"
OUT_PATH  = "src/data/player_fixture_20250203.csv"
GAME_DAY  = pd.Timestamp("2025-02-03")     # prediction date
CUTOFF    = GAME_DAY - pd.Timedelta(days=1)  # use data up to 2 Feb 2025

# ------------- 1.  load historical data --------------------------------------
hist = pd.read_csv(HIST_PATH)
hist.columns = hist.columns.str.upper()
hist = hist.loc[:, ~hist.columns.duplicated()]
hist["GAME_DATE"] = pd.to_datetime(hist["GAME_DATE"])

# keep only rows up to cutoff
hist = hist[hist["GAME_DATE"] <= CUTOFF]
hist["TEAM_ID"] = pd.to_numeric(hist["TEAM_ID"], errors="coerce").astype("Int64")

# latest row per player (roster snapshot)
idx_latest = hist.groupby("PLAYER_ID")["GAME_DATE"].idxmax()
roster_snap = hist.loc[idx_latest, ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID"]]

# ------------- 2.  fetch Feb-3 schedule --------------------------------------
print("⏳  downloading 2024-25 schedule …")
sched = leaguegamefinder.LeagueGameFinder(
            season_nullable="2024-25"
        ).get_data_frames()[0]

sched["GAME_DATE"] = pd.to_datetime(sched["GAME_DATE"])
day_games = sched[sched["GAME_DATE"] == GAME_DAY]

# remove duplicates (each team appears many times in leaguegamefinder)
day_games = day_games.drop_duplicates(subset=["MATCHUP"])

abbr_to_id = {t["abbreviation"]: t["id"] for t in teams.get_teams()}
fixture_rows = []

for _, g in day_games.iterrows():
    m = g["MATCHUP"].strip()      # e.g. "GSW @ LAL", "LAL vs. GSW"
    if " @ " in m:
        away_abbr, home_abbr = m.split(" @ ")
    elif " vs. " in m:
        home_abbr, away_abbr = m.split(" vs. ")
    elif " vs " in m:
        home_abbr, away_abbr = m.split(" vs ")
    else:
        print(f"⚠️  skipping unparsable matchup: {m!r}")
        continue

    # skip non-NBA teams (All-Star, etc.)
    if home_abbr not in abbr_to_id or away_abbr not in abbr_to_id:
        print(f"⚠️  unknown team(s) in matchup {m!r} — skipping")
        continue

    home_id, away_id = abbr_to_id[home_abbr], abbr_to_id[away_abbr]

    fixture_rows.append(dict(TEAM_ID=home_id, OPP_TEAM_ID=away_id,
                             IS_HOME=1))
    fixture_rows.append(dict(TEAM_ID=away_id, OPP_TEAM_ID=home_id,
                             IS_HOME=0))

team_fixture = pd.DataFrame(fixture_rows).drop_duplicates()

# ------------- 3.  explode to players ----------------------------------------
player_rows = []

for _, row in team_fixture.iterrows():
    team_id   = row["TEAM_ID"]
    opp_id    = row["OPP_TEAM_ID"]
    is_home   = row["IS_HOME"]

    roster_players = roster_snap[roster_snap["TEAM_ID"] == team_id]

    for _, ply in roster_players.iterrows():
        player_rows.append(dict(
            PLAYER_ID   = int(ply["PLAYER_ID"]),
            PLAYER_NAME = ply["PLAYER_NAME"],
            TEAM_ID     = team_id,
            OPP_TEAM_ID = opp_id,
            IS_HOME     = is_home,
            GAME_DATE   = GAME_DAY.date()
        ))

players_df = pd.DataFrame(player_rows)
print(f"✓ generated {len(players_df)} player rows for 3 Feb 2025")

# ------------- 4.  save -------------------------------------------------------
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
players_df.to_csv(OUT_PATH, index=False)
print(f"✅ saved → {OUT_PATH}")
