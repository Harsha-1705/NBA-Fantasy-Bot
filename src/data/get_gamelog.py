#!/usr/bin/env python3
"""
Fetch NBA player game logs with opponent defensive rating.
Parses opponent from the MATCHUP string and merges team DEF_RATING.
"""

from pathlib import Path
import time
import click
import pandas as pd
from nba_api.stats.endpoints import PlayerGameLog, LeagueDashTeamStats
from nba_api.stats.static import players as players_static

# Map full team names → 3-letter abbreviations
TEAM_NAME_TO_ABBR = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}


def fetch_player_season(player_id: int, season: str) -> pd.DataFrame | None:
    """Return one-row-per-game DataFrame for the player/season, or None if empty."""
    try:
        gl = PlayerGameLog(player_id=player_id, season=season)
        df = gl.get_data_frames()[0]
        if df.empty:
            return None
        df["PLAYER_ID"] = player_id
        df["SEASON"] = season
        return df
    except Exception as e:
        print(f"[WARN] PID {player_id} → {e}")
        return None


def extract_opponent(matchup: str) -> str:
    """
    MATCHUP is always '<YourTeam> vs <Opp>' or '<YourTeam> @ <Opp>';
    the opponent is the 3rd token.
    """
    parts = matchup.split()
    return parts[2] if len(parts) >= 3 else ""


def get_team_def_ratings(season: str) -> pd.DataFrame:
    """Fetch each team’s defensive rating for the season, keyed by 3-letter abbr."""
    try:
        stats = LeagueDashTeamStats(season=season, measure_type_detailed_defense="Advanced")
        df = stats.get_data_frames()[0]
        df["TEAM_ABBREVIATION"] = df["TEAM_NAME"].map(TEAM_NAME_TO_ABBR)
        return df[["TEAM_ABBREVIATION", "DEF_RATING"]]
    except Exception as e:
        print(f"[WARN] Could not load DEF_RATING for {season}: {e}")
        return pd.DataFrame(columns=["TEAM_ABBREVIATION", "DEF_RATING"])


@click.command()
@click.option("--seasons", multiple=True, required=True,
              help='NBA seasons in "YYYY-YY" format, e.g. 2022-23 2023-24')
@click.option("--top", type=int, default=None,
              help="Limit to N active players (for quick tests)")
@click.option("--sleep", "sleep_s", type=float, default=0.6,
              help="Seconds to sleep between API calls")
def main(seasons: list[str], top: int | None, sleep_s: float):
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    active = [p for p in players_static.get_players() if p["is_active"]]
    if top:
        active = active[:top]

    all_rows: list[pd.DataFrame] = []

    for season in seasons:
        print(f"\n📅 Season {season} → {len(active)} players…")
        def_ratings = get_team_def_ratings(season)

        for idx, p in enumerate(active, 1):
            print(f"({idx}/{len(active)}) {p['full_name']:<20}", end="")
            df = fetch_player_season(p["id"], season)
            if df is None:
                print(" – no games")
                continue

            df["PLAYER_NAME"] = p["full_name"]
            df["OPPONENT"] = df["MATCHUP"].apply(extract_opponent)

            # merge on opponent abbreviation → defensive rating
            df = df.merge(def_ratings,
                          left_on="OPPONENT",
                          right_on="TEAM_ABBREVIATION",
                          how="left")
            df.rename(columns={"DEF_RATING": "OPP_DEF_RATING"}, inplace=True)

            # add normalized points
            if "PTS" in df.columns:
                df["PTS_vs_DefAdj"] = df["PTS"] / df["OPP_DEF_RATING"]

            all_rows.append(df)
            print(f" – {len(df)} rows")

            time.sleep(sleep_s)

    if not all_rows:
        print("No data fetched.")
        raise SystemExit(1)

    out = pd.concat(all_rows, ignore_index=True)
    fn = out_dir / f"player_gamelog_{'_'.join(seasons)}.csv"
    out.to_csv(fn, index=False)
    print(f"\n✅ Saved {len(out):,} rows → {fn}")


if __name__ == "__main__":
    main()

