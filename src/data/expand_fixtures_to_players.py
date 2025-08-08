#!/usr/bin/env python3
"""
expand_fixtures_to_players.py
-------------------------------------------------
Convert team-level fixture rows into player-level
rows so the prediction pipeline can be matchup-aware.

Input : prediction_fixtures.csv   (team rows)
Output: player_prediction_fixtures.csv (player rows)

Columns in output:
    PLAYER_ID
    PLAYER_NAME
    TEAM_ID         – player’s own team
    OPP_TEAM_ID     – opponent team
    IS_HOME         – 1 if player’s team is at home
    GAME_DATE       – date of the game (YYYY-MM-DD)
"""

import pandas as pd
from nba_api.stats.endpoints import commonteamroster
from tqdm import tqdm                         # pip install tqdm
import os

# ---------- CONFIG -----------------------------------------------------------
FIXTURE_PATH      = "prediction_fixtures.csv"   # team-level fixture
OUT_PATH          = "player_prediction_fixtures.csv"
SEASON            = "2024-25"   # first choice
FALLBACK_SEASON   = "2023-24"   # if 24-25 roster not yet on stats.nba.com
# -----------------------------------------------------------------------------

# 1️⃣  Load the team-level fixture
if not os.path.exists(FIXTURE_PATH):
    raise FileNotFoundError(f"Cannot find {FIXTURE_PATH}")

fixtures = pd.read_csv(FIXTURE_PATH)
print(f"Team-level rows loaded: {len(fixtures)}")

# 2️⃣  Roster-fetch helper with fallback
roster_cache = {}

def fetch_roster(team_id: int, season_tag: str) -> pd.DataFrame:
    """
    Return a DataFrame with PLAYER_ID, PLAYER columns
    for the given team_id and season_tag.
    """
    df = commonteamroster.CommonTeamRoster(
            team_id=team_id,
            season=season_tag
         ).get_data_frames()[0]
    # Only need player id & name
    return df[['PLAYER_ID', 'PLAYER']]

def get_roster(team_id: int) -> pd.DataFrame:
    """
    Try to load the roster for the primary season;
    fall back to the prior season if NBA stats payload
    is missing the 'Coaches' set (KeyError).
    """
    if team_id in roster_cache:
        return roster_cache[team_id]

    try:
        df_players = fetch_roster(team_id, SEASON)
    except KeyError:
        print(f" ↪️  {team_id}: {SEASON} roster unavailable – "
              f"falling back to {FALLBACK_SEASON}")
        df_players = fetch_roster(team_id, FALLBACK_SEASON)

    roster_cache[team_id] = df_players
    return df_players

# 3️⃣  Expand each team row into player rows
player_rows = []

for _, row in tqdm(fixtures.iterrows(),
                   total=len(fixtures),
                   desc="Expanding to players"):
    team_id     = row['OPP_TEAM_ID']   # this team’s own ID
    opp_team_id = row['OPP_TEAM_ID']   # from fixture perspective
    is_home     = row['IS_HOME']
    game_date   = row['GAME_DATE']

    roster_df = get_roster(team_id)

    for _, p in roster_df.iterrows():
        player_rows.append({
            "PLAYER_ID"  : int(p['PLAYER_ID']),
            "PLAYER_NAME": p['PLAYER'],
            "TEAM_ID"    : team_id,
            "OPP_TEAM_ID": opp_team_id,
            "IS_HOME"    : is_home,
            "GAME_DATE"  : game_date
        })

players_fixture = pd.DataFrame(player_rows)
print(f"Player-level rows generated: {len(players_fixture)}")

# 4️⃣  Save
players_fixture.to_csv(OUT_PATH, index=False)
print(f"✅ Player fixture saved as {os.path.abspath(OUT_PATH)}")
