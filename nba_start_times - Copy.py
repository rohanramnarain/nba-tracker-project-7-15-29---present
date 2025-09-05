import csv
import re
import requests
from datetime import datetime
from PyPDF2 import PdfReader
from io import BytesIO

# === Team name to abbreviation map (with aliases) ===
team_abbr = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS"
}

# === PDF URL Generator ===
def build_pdf_url(game_date, away, home):
    try:
        date_obj = datetime.strptime(game_date.strip(), "%a, %b %d, %Y")
    except ValueError:
        date_obj = datetime.strptime(game_date.strip(), "%a %b %d %Y")
    date_str = date_obj.strftime("%Y%m%d")
    return f"https://statsdmz.nba.com/pdfs/{date_str}/{date_str}_{away}{home}_book.pdf"

# === Enhanced Extract Start of 1st Quarter from PDF ===
def extract_first_quarter_start(pdf_url, debug=False):
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()

        reader = PdfReader(BytesIO(response.content))
        all_text = ""
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            all_text += text + "\n"
            
            if debug and page_num < 2:  # Show first 2 pages for debugging
                print(f"\n--- PAGE {page_num + 1} ---")
                print(text[:1000] + "..." if len(text) > 1000 else text)
            
            # Try multiple patterns for start time - ordered from most specific to least
            patterns = [
                r"Start of 1st Quarter\s*[:\-]?\s*(\d{1,2}:\d{2}\s*[AP]M)",
                r"Start of 1st quarter\s*[:\-]?\s*(\d{1,2}:\d{2}\s*[AP]M)",
                r"1st Quarter Start\s*[:\-]?\s*(\d{1,2}:\d{2}\s*[AP]M)",
                r"Game Start\s*[:\-]?\s*(\d{1,2}:\d{2}\s*[AP]M)",
                r"Start Time\s*[:\-]?\s*(\d{1,2}:\d{2}\s*[AP]M)",
                r"Tip[- ]?off\s*[:\-]?\s*(\d{1,2}:\d{2}\s*[AP]M)",
                r"Opening Tip\s*[:\-]?\s*(\d{1,2}:\d{2}\s*[AP]M)",
                r"Game Time\s*[:\-]?\s*(\d{1,2}:\d{2}\s*[AP]M)",
                # Look for time patterns near game-related words
                r"(?:Game|Start|Tip|Quarter|Begin).*?(\d{1,2}:\d{2}\s*[AP]M)",
                # Last resort - any time that looks like a game time
                r"(\d{1,2}:\d{2}\s*[AP]M)",
            ]
            
            for i, pattern in enumerate(patterns):
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # For the last pattern (any time), filter out obviously wrong times
                    if i == len(patterns) - 1:
                        # Filter times that are likely game times (between 6 PM and 11 PM typically)
                        valid_times = []
                        for time_str in matches:
                            try:
                                time_obj = datetime.strptime(time_str.strip(), "%I:%M %p")
                                hour = time_obj.hour
                                # NBA games typically start between 6 PM and 11 PM
                                if 18 <= hour <= 23 or 12 <= hour <= 15:  # Also allow afternoon games
                                    valid_times.append(time_str.strip())
                            except:
                                continue
                        if valid_times:
                            result = valid_times[0]  # Take first valid time
                            if debug:
                                print(f"✅ FOUND with pattern {i+1} '{pattern}': {result}")
                            return result
                    else:
                        result = matches[0].strip()
                        if debug:
                            print(f"✅ FOUND with pattern {i+1} '{pattern}': {result}")
                        return result
        
        if debug:
            print(f"\n⚠️ No time found. Searching for common keywords in full text...")
            keywords = ["start", "quarter", "time", "game", "tip", "PM", "AM", "opening", "begin"]
            found_any = False
            for keyword in keywords:
                if keyword.lower() in all_text.lower():
                    if not found_any:
                        print(f"Keywords found in PDF:")
                        found_any = True
                    print(f"- '{keyword}'")
                    
            # Show any time-like patterns found
            all_times = re.findall(r"\d{1,2}:\d{2}\s*[AP]M", all_text, re.IGNORECASE)
            if all_times:
                print(f"All time patterns found: {all_times[:10]}")  # Show first 10
            
    except Exception as e:
        if debug:
            print(f"⚠️ Error reading PDF: {pdf_url} - {str(e)}")
        else:
            print(f"⚠️ Error reading PDF: {pdf_url}")
    
    return None

# === Main Script ===
input_csv = "ba_dates_of_games.csv"
output_csv = "Actual_nba_start_times.csv"
unknown_teams = set()

# Test function for debugging a specific game
def test_single_game():
    test_date = "Sun Oct 27 2024"
    away_abbr = "LAC"  # LA Clippers
    home_abbr = "GSW"  # Golden State Warriors
    
    pdf_url = build_pdf_url(test_date, away_abbr, home_abbr)
    print(f"Testing PDF: {pdf_url}")
    start_time = extract_first_quarter_start(pdf_url, debug=True)
    
    if start_time:
        print(f"✅ Found start time: {start_time}")
    else:
        print("❌ No start time found")
    return start_time

# Uncomment the next line to test a single game first
# test_single_game()

with open(input_csv, newline='', encoding='utf-8') as infile, open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = ['Date', 'Away Team', 'Home Team', 'PDF Link', 'Start of 1st Quarter']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    count = 0
    kept = 0
    for row in reader:
        count += 1
        game_date = row['Date']
        away_team = row['Away Team'].strip()
        home_team = row['Home Team'].strip()

        away_abbr = team_abbr.get(away_team)
        home_abbr = team_abbr.get(home_team)

        if not away_abbr or not home_abbr:
            unknown_teams.update([t for t in [away_team, home_team] if t not in team_abbr])
            continue

        pdf_url = build_pdf_url(game_date, away_abbr, home_abbr)
        
        # Enable debug for first 3 games to see what's happening
        debug_mode = count <= 3
        start_time = extract_first_quarter_start(pdf_url, debug=debug_mode)

        # Always write to CSV, even if no start time found
        writer.writerow({
            'Date': game_date,
            'Away Team': away_team,
            'Home Team': home_team,
            'PDF Link': pdf_url,
            'Start of 1st Quarter': start_time if start_time else 'Not Found'
        })
        
        if start_time:
            kept += 1
            print(f"✅ Found: {away_team} @ {home_team} ({game_date}) - {start_time}")
        else:
            print(f"⏭️ Not found: {away_team} @ {home_team} ({game_date})")

    print(f"\n✅ Finished processing {count} games.")
    print(f"📝 Successfully found start times for {kept} games.")
    print(f"❌ Could not find start times for {count - kept} games.")

    if unknown_teams:
        print("⚠️ Unknown team names found:", ", ".join(sorted(unknown_teams)))