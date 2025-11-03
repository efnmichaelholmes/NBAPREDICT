import pandas as pd
from tqdm import tqdm
import glob
import os

# Folder where your CSV files are stored
DATA_FOLDER = "."  # current folder
OUTPUT_FILE = "merged_player_stats.csv"

# List of all player stats files
csv_files = sorted(glob.glob(os.path.join(DATA_FOLDER, "player_stats_*.csv")))

all_rows = []

def expand_player_season(row):
    """If player has multiple teams in a season, split into separate rows"""
    if row['Team'] == '2TM':
        # Find the rows in the same season and player with actual teams
        # For now, just return the row as-is (will adjust if needed)
        return [row]
    else:
        return [row]

print(f"Processing {len(csv_files)} files...")

for file in csv_files:
    print(f"\nLoading {file}...")
    df = pd.read_csv(file)
    
    # Skip header duplicates if present
    df = df.loc[~df['Rk'].astype(str).str.contains('Rk')]
    
    # Expand multi-team rows
    print("Expanding player seasons...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        all_rows.extend(expand_player_season(row))
    
# Create final DataFrame
merged_df = pd.DataFrame(all_rows)

# Reset index
merged_df.reset_index(drop=True, inplace=True)

# Save merged CSV
merged_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nMerged CSV saved as {OUTPUT_FILE}. Total rows: {len(merged_df)}")
