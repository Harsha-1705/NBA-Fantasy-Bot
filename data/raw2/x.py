import pandas as pd
import glob

# Option 1: Manually specify file names
files = [
    'player_gamelog_2020-21.csv',
    'player_gamelog_2021-22.csv',
    'player_gamelog_2022-23.csv',
    'player_gamelog_2023-24.csv',
    'player_gamelog_2024-25.csv'
]

# Read and concatenate all files
df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

# Save to a new CSV
df.to_csv('combined_output.csv', index=False)

print("All files combined successfully into 'combined_output.csv'")
