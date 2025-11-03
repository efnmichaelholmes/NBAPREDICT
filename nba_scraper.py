import os
import time
from io import StringIO
import pandas as pd
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# OUTPUT directory
OUTPUT_DIR = "nba_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup Selenium Chrome
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
driver = webdriver.Chrome(options=chrome_options)

BASE_URL = "https://www.basketball-reference.com"

def get_all_player_links(season):
    """Fetch all player URLs for a given season"""
    url = f"{BASE_URL}/leagues/NBA_{season}_per_game.html"
    driver.get(url)
    time.sleep(2)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    table_html = None
    for c in comments:
        if 'id="per_game_stats"' in c:
            table_html = c
            break

    if not table_html:
        table = soup.find("table", {"id": "per_game_stats"})
        if table:
            table_html = str(table)

    if not table_html:
        print(f"Could not fetch player page for {season}: No tables found")
        return []

    df = pd.read_html(StringIO(table_html))[0]
    player_links = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if href and "/players/" in href and href.endswith(".html"):
            if "gamelog" not in href:
                player_links.append(BASE_URL + href.replace(".html", f"/gamelog/{season}"))
    return list(set(player_links))  # unique links

def scrape_player_game_log(player_url, season):
    """Scrape individual player's box scores for a season"""
    try:
        driver.get(player_url)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        table = soup.find('table', id='pgl_basic')

        if table is None:
            # Check inside comments
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for c in comments:
                if 'id="pgl_basic"' in c:
                    table = BeautifulSoup(c, 'html.parser').find('table', id='pgl_basic')
                    break

        if table is None:
            return pd.DataFrame()

        rows = table.find('tbody').find_all('tr')
        data = []

        player_name = soup.find('h1').text.strip() if soup.find('h1') else "Unknown"

        for row in rows:
            if row.get('class') and 'thead' in row.get('class'):
                continue

            game_date = row.find('td', {'data-stat':'date_game'})
            opponent = row.find('td', {'data-stat':'opp_id'})
            location = row.find('td', {'data-stat':'game_location'})
            team = row.find('td', {'data-stat':'team_id'})

            if not game_date or not opponent or not team:
                continue  # skip empty rows

            # Matchup format
            matchup = f"{team.text} {'@' if location and location.text=='@' else 'vs.'} {opponent.text}"

            game = {
                "PLAYER_NAME": player_name,
                "GAME_DATE": game_date.text.strip(),
                "MATCHUP": matchup,
                "PTS": row.find('td', {'data-stat':'pts'}).text.strip() if row.find('td', {'data-stat':'pts'}) else '0',
                "REB": row.find('td', {'data-stat':'trb'}).text.strip() if row.find('td', {'data-stat':'trb'}) else '0',
                "AST": row.find('td', {'data-stat':'ast'}).text.strip() if row.find('td', {'data-stat':'ast'}) else '0',
                "STL": row.find('td', {'data-stat':'stl'}).text.strip() if row.find('td', {'data-stat':'stl'}) else '0',
                "BLK": row.find('td', {'data-stat':'blk'}).text.strip() if row.find('td', {'data-stat':'blk'}) else '0',
                "FG3M": row.find('td', {'data-stat':'fg3m'}).text.strip() if row.find('td', {'data-stat':'fg3m'}) else '0',
                "TEAM": team.text.strip(),
                "Season": season
            }
            data.append(game)

        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error scraping {player_url}: {e}")
        return pd.DataFrame()

def main():
    all_data = []

    for season in range(2020, 2026):
        print(f"Starting season {season}")
        player_links = get_all_player_links(season)
        print(f"Found {len(player_links)} players for {season}")

        for idx, player_url in enumerate(player_links, start=1):
            print(f"[{idx}/{len(player_links)}] Scraping {player_url}")
            df = scrape_player_game_log(player_url, season)
            if not df.empty:
                all_data.append(df)

        # Save season data
        if all_data:
            season_df = pd.concat(all_data, ignore_index=True)
            season_df.to_csv(os.path.join(OUTPUT_DIR, f"player_boxscores_{season}.csv"), index=False)
            print(f"Saved season {season} data with {len(season_df)} rows")
            all_data = []  # reset for next season

    # Optional: combine all seasons into one CSV
    combined = []
    for season in range(2020, 2026):
        path = os.path.join(OUTPUT_DIR, f"player_boxscores_{season}.csv")
        if os.path.exists(path):
            combined.append(pd.read_csv(path))
    if combined:
        pd.concat(combined, ignore_index=True).to_csv(os.path.join(OUTPUT_DIR, "all_player_boxscores_2020_2025.csv"), index=False)
        print("All seasons combined CSV saved!")

if __name__ == "__main__":
    main()
    driver.quit()
