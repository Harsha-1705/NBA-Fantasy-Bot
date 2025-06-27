import pandas as pd
import os

# List of CSV file paths (update as needed)
files = [
    'data/raw2/player_gamelog_2020-21.csv',
    'data/raw2/player_gamelog_2021-22.csv',
    'data/raw2/player_gamelog_2022-23.csv',
    'data/raw2/player_gamelog_2023-24.csv',
    'data/raw2/player_gamelog_2024-25.csv'
]

# Combine the files
df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

# Get folder path from first file
output_folder = os.path.dirname(files[0])

# Create output path in same folder
output_path = os.path.join(output_folder, 'player_gamelog_combined.csv')

# Save combined CSV
df.to_csv(output_path, index=False)

print(f"Combined CSV saved at: {output_path}")
