#!/usr/bin/env python3
"""
Download full-season game logs for every active player across multiple seasons
and save to:

    data/raw/player_gamelog_all_seasons.csv

Usage examples
--------------
# All players, multiple seasons
python scripts/get_gamelog_season.py --seasons 2022-23 2023-24

# First 20 players, quick test
python scripts/get_gamelog_season.py --seasons 2023-24 --top 20
"""

from pathlib import Path
import time
import click
import pandas as pd
from nba_api.stats.endpoints import PlayerGameLog
from nba_api.stats.static import players as players_static


def fetch_player_season(player_id: int, season: str) -> pd.DataFrame | None:
    """Return a DataFrame with one row per game for `player_id` in `season`."""
    try:
        gl = PlayerGameLog(player_id=player_id, season=season)
        df = gl.get_data_frames()[0]
        if df.empty:
            return None
        df["PLAYER_ID"] = player_id
        df["SEASON"] = season  # <- add season column
        return df
    except Exception as exc:
        print(f"[WARN] could not fetch PID {player_id}: {exc}")
        return None


@click.command()
@click.option(
    "--seasons",
    multiple=True,
    required=True,
    help='One or more NBA seasons in "YYYY-YY" format (e.g., 2022-23 2023-24)',
)
@click.option(
    "--top",
    type=int,
    default=None,
    help="Limit to N active players (useful for quick tests)",
)
@click.option(
    "--sleep",
    "sleep_s",
    type=float,
    default=0.6,
    show_default=True,
    help="Seconds to sleep between API calls (avoid 429s)",
)
def main(seasons: list[str], top: int | None, sleep_s: float) -> None:
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    active_players = [p for p in players_static.get_players() if p["is_active"]]
    if top:
        active_players = active_players[:top]

    all_frames: list[pd.DataFrame] = []

    for season in seasons:
        print(f"\n📅 Season {season} → querying {len(active_players)} players…")
        for idx, p in enumerate(active_players, start=1):
            pid, name = p["id"], p["full_name"]
            print(f"({idx}/{len(active_players)}) {name:<25}", end="")

            df = fetch_player_season(pid, season)
            if df is not None:
                df["PLAYER_NAME"] = name
                all_frames.append(df)
                print(f" – {len(df)} games ✓")
            else:
                print(" – skipped")

            time.sleep(sleep_s)

    if not all_frames:
        print("No game logs retrieved — exiting.")
        raise SystemExit(1)

    combined = pd.concat(all_frames, ignore_index=True)
    outfile = out_dir / "player_gamelog_2021_2022.csv"
    combined.to_csv(outfile, index=False)

    try:
        pretty_path = outfile.relative_to(Path.cwd())
    except ValueError:
        pretty_path = outfile

    print(
        f"\n✅  Saved {len(combined):,} rows → {pretty_path}\n"
        f"    Columns: {list(combined.columns)}"
    )


if __name__ == "__main__":
    main()
