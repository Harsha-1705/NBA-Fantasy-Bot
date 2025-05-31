# Dataset Registry

## Raw Data Files

### player_gamelog_2024.csv
- **Description**: NBA player game logs for 2023-24 season
- **Source**: NBA API via nba_api library
- **Date Created**: 2025-01-01
- **File Size**: ~X MB
- **Rows**: ~X records
- **Columns**: GAME_ID, PLAYER_ID, GAME_DATE, PTS, REB, AST, etc.
- **SHA-256**: 8e11888aa295ae626d828d4b00b8f7fc9d64ff264e1ea16204286a43f4054fb7

### Usage

import pandas as pd
df = pd.read_csv('data/raw/player_gamelog_2024.csv')


## Data Quality Notes
- Missing values handled: [To be documented]
- Known issues: [To be documented]
- Validation status: [To be updated]

## Version History
- v1.0 (2025-01-01): Initial dataset with 2023-24 season data

## Access Instructions
1. Clone the repository
2. Ensure Git LFS is installed: `git lfs install`
3. Pull LFS files: `git lfs pull`
4. Access data from `data/raw/` directory

