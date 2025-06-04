import pandas as pd
import os

def calculate_fantasy_points(df):
    """
    Calculate fantasy points based on scoring rules from GitHub issue
    
    Scoring Rules:
    - PTS: 1 point
    - REB: 1.25 points  
    - AST: 1.5 points
    - STL: 2 points
    - BLK: 3 points
    - TO: -1 point
    """
    fantasy_points = (
        df['PTS'] * 1.0 +
        df['REB'] * 1.25 +
        df['AST'] * 1.5 +
        df['STL'] * 2.0 +
        df['BLK'] * 3.0 +
        df['TOV'] * -1.0  # TOV is turnovers in NBA API
    )
    
    return fantasy_points

def main():
    # Ensure processed directory exists
    os.makedirs('../data/processed', exist_ok=True)
    
    # Read raw CSV
    print("Reading raw player game log data...")
    try:
        df = pd.read_csv('data/raw/player_gamelog_2023_24.csv')
        print(f"Loaded {len(df)} rows of data")
    except FileNotFoundError:
        print("Error: ../data/raw/player_gamelog_2023_24.csv not found")
        print("Make sure you've run the get_gamelog.py script first")
        return
    
    # Calculate fantasy points
    print("Calculating fantasy points...")
    df['fantasy_points'] = calculate_fantasy_points(df)
    
    # Show some stats
    print(f"Fantasy points calculated for {len(df)} games")
    print(f"Average fantasy points: {df['fantasy_points'].mean():.2f}")
    print(f"Max fantasy points: {df['fantasy_points'].max():.2f}")
    print(f"Min fantasy points: {df['fantasy_points'].min():.2f}")
    
    # Save processed data
    output_file = '../data/processed/fantasy_points_2023_24.csv'
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    main()
