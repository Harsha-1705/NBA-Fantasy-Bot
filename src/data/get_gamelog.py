import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import os
import time

def get_player_gamelog(player_id, season='2023-24'):
    """
    Pull game logs for a specific player and season
    """
    try:
        # Get game logs for the season
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gamelog.get_data_frames()[0]  # Fix: get first dataframe
        return df
    except Exception as e:
        print(f"Error fetching data for player ID {player_id}: {e}")
        return None

def main():
    # Create data/raw directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Get all active players (limit to top players for testing)
    player_dict = players.get_players()
    
    # For full season, use all players. For testing, use subset:
    # player_ids = [p['id'] for p in player_dict if p['is_active']][:50]  # First 50 active
    player_ids = [2544, 201939, 1629029]  # LeBron, Curry, Luka for testing
    
    season = '2023-24'
    all_gamelogs = []
    
    for idx, pid in enumerate(player_ids):
        print(f"Fetching data for player ID {pid} ({idx+1}/{len(player_ids)})...")
        df = get_player_gamelog(pid, season)
        
        if df is not None and not df.empty:
            df['PLAYER_ID'] = pid
            all_gamelogs.append(df)
            print(f"  -> Got {len(df)} games")
        
        time.sleep(0.6)  # Rate limiting
    
    if all_gamelogs:
        combined_df = pd.concat(all_gamelogs, ignore_index=True)
        output_file = 'data/raw/player_gamelog_2024.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
        print(f"Shape: {combined_df.shape}")
        print(f"Columns: {list(combined_df.columns)}")
    else:
        print("No data fetched")

if __name__ == "__main__":
    main()

