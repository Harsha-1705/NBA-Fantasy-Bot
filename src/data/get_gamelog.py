#!/usr/bin/env python3
"""
Download full-season game logs for every active player and save to:

    data/raw/player_gamelog_<season>.csv

Usage examples
--------------
# Full active roster (slow)
python scripts/get_gamelog_season.py --season 2023-24

# Just first 50 active players (quick test)
python scripts/get_gamelog_season.py --season 2023-24 --top 50
"""

from pathlib import Path
import time
import click
import pandas as pd
from nba_api.stats.endpoints import PlayerGameLog
from nba_api.stats.static import players as players_static


# --------------------------------------------------------------------------- #
# Helper: fetch one player's season                                           #
# --------------------------------------------------------------------------- #
def fetch_player_season(player_id: int, season: str) -> pd.DataFrame | None:
    """
    Return a DataFrame with one row per game for player_id in season.
    Returns None on any error (network, empty, etc.).
    """
    try:
        gl = PlayerGameLog(player_id=player_id, season=season)
        df = gl.get_data_frames()[0]
        if df.empty:
            return None
        df["PLAYER_ID"] = player_id
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] could not fetch PID {player_id}: {exc}")
        return None


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
@click.command()
@click.option(
    "--season",
    default="2023-24",
    show_default=True,
    help='NBA season in "YYYY-YY" format (stats.nba.com style)',
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
def main(season: str, top: int | None, sleep_s: float) -> None:
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- build player list ---------------- #
    active_players = [p for p in players_static.get_players() if p["is_active"]]
    if top:
        active_players = active_players[:top]

    print(f"Season {season} → querying {len(active_players)} players…")

    frames: list[pd.DataFrame] = []

    for idx, p in enumerate(active_players, start=1):
        pid, name = p["id"], p["full_name"]
        print(f"({idx}/{len(active_players)}) {name:<25}", end="")

        df = fetch_player_season(pid, season)
        if df is not None:
            df["PLAYER_NAME"] = name
            frames.append(df)
            print(f" – {len(df)} games ✓")
        else:
            print(" – skipped")

        time.sleep(sleep_s)

    if not frames:
        print("No game logs retrieved — exiting.")
        raise SystemExit(1)

    combined = pd.concat(frames, ignore_index=True)
    outfile = out_dir / f"player_gamelog_{season.replace('-', '_')}.csv"
    combined.to_csv(outfile, index=False)

    # -------- pretty-print path without crashing -------- #
    try:
        pretty_path = outfile.relative_to(Path.cwd())
    except ValueError:
        pretty_path = outfile

    print(
        f"\n✅  Saved {len(combined):,} rows → {pretty_path}\n"
        f"    Columns: {list(combined.columns)}"
    )


if _name_ == "_main_":
    main()