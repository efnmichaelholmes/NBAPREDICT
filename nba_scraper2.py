import os
import time
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Output directory
OUTPUT_DIR = "nba_data_2025"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup Selenium Chrome
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
driver = webdriver.Chrome(options=chrome_options)

# List of NBA teams abbreviations
TEAMS = ["ATL","BOS","BRK","CHI","CHO","CLE","DAL","DEN","DET","GSW",
         "HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK",
         "OKC","ORL","PHI","PHO","POR","SAC","SAS","TOR","UTA","WAS"]

def fetch_schedule(team):
    url = f"https://www.basketball-reference.com/teams/{team}/2025_games.html"
    driver.get(url)
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    # Handle tables inside comments
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    table_html = None
    for c in comments:
        if 'id="games"' in c:
            table_html = c
            break
    if not table_html:
        table = soup.find("table", {"id":"games"})
        if table:
            table_html = str(table)
    if not table_html:
        print(f"No schedule table for {team}")
        return pd.DataFrame()
    
    df = pd.read_html(StringIO(str(table_html)))[0]
    df = df.dropna(subset=["Date"])
    df["Team"] = team

    # Detect Boxscore column dynamically
    box_col = None
    for col in df.columns:
        if "Box" in str(col):
            box_col = col
            break
    if box_col:
        df = df[df[box_col].notna()]
        df.rename(columns={box_col: "Box Score"}, inplace=True)
    else:
        df["Box Score"] = None

    return df

def fetch_player_boxscores(box_url):
    if not box_url:
        return pd.DataFrame()
    
    url = f"https://www.basketball-reference.com{box_url}"
    driver.get(url)
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Look for player box score table
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    table_html = None
    for c in comments:
        if 'id="box-' in c and 'basic"' in c:
            table_html = c
            break
    if not table_html:
        print(f"No box scores found for {url}")
        return pd.DataFrame()

    df = pd.read_html(StringIO(str(table_html)))[0]

    # Only keep relevant columns
    keep_cols = ["Player","PTS","REB","AST","STL","BLK","3P"]
    df = df[[c for c in keep_cols if c in df.columns]]
    df["Game URL"] = url
    return df

def main():
    all_boxscores = []
    for team in TEAMS:
        print(f"Fetching games for {team}...")
        schedule = fetch_schedule(team)
        if schedule.empty:
            continue
        for idx, row in schedule.iterrows():
            box_url = row["Box Score"]
            box_df = fetch_player_boxscores(box_url)
            if not box_df.empty:
                box_df["Team"] = team
                box_df["Date"] = row["Date"]
                all_boxscores.append(box_df)

    if all_boxscores:
        combined = pd.concat(all_boxscores, ignore_index=True)
        combined.to_csv(os.path.join(OUTPUT_DIR, "nba_2025_boxscores.csv"), index=False)
        print(f"Saved all 2025 box scores! Total rows: {len(combined)}")
    else:
        print("No box scores collected.")

if __name__ == "__main__":
    main()
    driver.quit()
