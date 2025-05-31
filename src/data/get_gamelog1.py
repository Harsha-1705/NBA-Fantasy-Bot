import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import os

def get_player_gamelog(player_name, season='2023-24'):
    """
    Pull game logs for a specific player and season
    """
    # Get all players
    player_dict = players.get_players()
    
    # Find the specific player
    player = [p for p in player_dict if p['full_name'] == player_name]
    
    if not player:
        print(f"Player {player_name} not found")
        return None
    
    player_id = player[0]['id']
    
    # Get game logs for the season
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = gamelog.get_data_frames()[0]
    
    return df

def main():
    # Create data/raw directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Test with a few star players for one day/season
    players_to_fetch = ['LeBron James', 'Stephen Curry', 'Luka Doncic']
    season = '2023-24'
    
    all_gamelogs = []
    
    for player_name in players_to_fetch:
        print(f"Fetching data for {player_name}...")
        df = get_player_gamelog(player_name, season)
        
        if df is not None:
            df['PLAYER_NAME'] = player_name
            all_gamelogs.append(df)
    
    # Combine all player data
    if all_gamelogs:
        combined_df = pd.concat(all_gamelogs, ignore_index=True)
        
        # Save to CSV
        output_file = f'data/raw/player_gamelog_{season.replace("-", "_")}.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        print(f"Shape: {combined_df.shape}")
    else:
        print("No data fetched")

if __name__ == "__main__":
    main()

