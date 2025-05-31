import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import os
import time

def get_player_gamelog(player_name, season='2023-24'):
    """
    Pull game logs for a specific player and season
    """
    try:
        # Get all players
        player_dict = players.get_players()
        
        # Find the specific player (try exact match first, then partial)
        player = [p for p in player_dict if p['full_name'].lower() == player_name.lower()]
        
        if not player:
            # Try partial match
            player = [p for p in player_dict if player_name.lower() in p['full_name'].lower()]
        
        if not player:
            print(f"Player {player_name} not found")
            return None
        
        player_id = player[0]['id']
        print(f"Found {player[0]['full_name']} (ID: {player_id})")
        
        # Add delay to avoid rate limiting
        time.sleep(1)
        
        # Get game logs for the season
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gamelog.get_data_frames()[0]
        
        print(f"Fetched {len(df)} games for {player[0]['full_name']}")
        return df
        
    except Exception as e:
        print(f"Error fetching data for {player_name}: {e}")
        return None

def main():
    # Create data/raw directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Use correct player names
    players_to_fetch = ['LeBron James', 'Stephen Curry', 'Luka Dončić']  # Note the correct spelling
    season = '2023-24'
    
    all_gamelogs = []
    
    for player_name in players_to_fetch:
        print(f"\nFetching data for {player_name}...")
        df = get_player_gamelog(player_name, season)
        
        if df is not None and not df.empty:
            df['PLAYER_NAME'] = player_name
            all_gamelogs.append(df)
            print(f"Added {len(df)} games")
        else:
            print(f"No data for {player_name}")
    
    # Combine all player data
    if all_gamelogs:
        combined_df = pd.concat(all_gamelogs, ignore_index=True)
        
        # Save to CSV
        output_file = f'data/raw/player_gamelog_{season.replace("-", "_")}.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
        print(f"Shape: {combined_df.shape}")
        print(f"Columns: {list(combined_df.columns)}")
        
        # Show sample data
        print("\nSample data:")
        print(combined_df[['PLAYER_NAME', 'GAME_DATE', 'PTS', 'REB', 'AST']].head())
    else:
        print("No data fetched")

if __name__ == "__main__":
    main()

