"""Public NBA schedule and game-log ingestion; public, unofficial sources.

No sportsbook market lines, injury status, or real-time availability are invented.
The game-log source is a public, community-maintained GitHub dataset (FanDuel
fantasy scoring derived from NBA box scores). It is NOT an official league feed.
"""
from __future__ import annotations
from datetime import date, datetime
from io import BytesIO
import re

import pandas as pd
import requests

GITHUB_LIST = "https://api.github.com/repos/xtheredshirtx/fanduel-player-data/contents/NBA/csv"
RAW_ROOT = "https://raw.githubusercontent.com/xtheredshirtx/fanduel-player-data/main/NBA/csv"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
REQUIRED = {"Date", "Player", "ID", "Team", "Opp", "Minutes", "Points", "Rebounds", "Assists", "3PM", "Steals", "Blocks", "Turnovers"}
MARKET_COLS = {"Points": "pts", "Rebounds": "reb", "Assists": "ast",
               "Threes": "threes", "Steals": "stl", "Blocks": "blk", "Turnovers": "tov"}
TEAM_ALIASES = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP",
                "PHO": "PHX", "UTAH": "UTA", "WSH": "WAS", "BRK": "BKN"}

def norm_team(value):
    label = str(value).strip().upper()
    return TEAM_ALIASES.get(label, label)

def season_key(when: date):
    start = when.year if when.month >= 10 else when.year - 1
    return f"{start}-{str(start + 1)[-2:]}"

def _get_json(url, params=None, timeout=15):
    resp = requests.get(url, params=params, headers={"User-Agent": "TeamAI2Pro/1.0 (sports research)"}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def scoreboard(when: date):
    data = _get_json(ESPN_SCOREBOARD, params={"dates": when.strftime("%Y%m%d")})
    rows = []
    for event in data.get("events", []):
        for comp in event.get("competitions", []):
            entries = comp.get("competitors", [])
            teams = {x.get("homeAway"): norm_team(x.get("team", {}).get("abbreviation", ""))
                     for x in entries}
            home, away = teams.get("home", ""), teams.get("away", "")
            if not home or not away:
                continue
            status = event.get("status", {}).get("type", {})
            rows.append({
                "event_id": str(event.get("id", "")),
                "home": home, "away": away,
                "matchup": f"{away} @ {home}",
                "state": status.get("state", "unknown"),
                "status": status.get("shortDetail", status.get("description", "Unknown")),
                "start_utc": str(event.get("date", "")),
            })
    return pd.DataFrame(rows, columns=["event_id", "home", "away", "matchup", "state", "status", "start_utc"])

def discover_game_log_files(when: date):
    """Find the newest public snapshot for the current and previous 2 seasons."""
    payload = _get_json(GITHUB_LIST)
    matches = {}
    pattern = re.compile(r"^NBA_players_game_log_(\d{4}-\d{2})_as_of_(\d{4}-\d{2}-\d{2})\.csv$")
    begin = int(season_key(when)[:4])
    allowed = {f"{y}-{str(y+1)[-2:]}" for y in [begin-2, begin-1, begin]}
    for item in payload:
        name = str(item.get("name", ""))
        m = pattern.match(name)
        if not m or m.group(1) not in allowed:
            continue
        # Prefer the most recent source snapshot; retain its stated as-of timestamp.
        if m.group(1) not in matches or m.group(2) > matches[m.group(1)]["as_of"]:
            matches[m.group(1)] = {"name": name, "season": m.group(1), "as_of": m.group(2)}
    if not matches:
        raise ValueError("No matching player game-log CSVs found in the public source.")
    return sorted(matches.values(), key=lambda item: item["season"])

def minutes_number(s):
    if isinstance(s, str) and ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60.0
        except (ValueError, IndexError):
            return None
    return pd.to_numeric(s, errors="coerce")

def normalize_logs(raw: pd.DataFrame):
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError("Missing player history columns: " + ", ".join(sorted(missing)))
    rename = {"Date":"date", "Player":"player", "ID":"player_id",
              "Team":"team", "Opp":"opponent", "Minutes":"minutes",
              "Points":"pts", "Rebounds":"reb", "Assists":"ast",
              "3PM":"threes", "Steals":"stl", "Blocks":"blk", "Turnovers":"tov"}
    x = raw.rename(columns=rename)[list(rename.values())].copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["player_id"] = x["player_id"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    for col in MARKET_COLS.values():
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x["minutes"] = x["minutes"].map(minutes_number)
    x["team"] = x["team"].map(norm_team)
    x["opponent"] = x["opponent"].map(norm_team)
    x["player"] = x["player"].astype(str).str.strip()
    x = x.dropna(subset=["date", "minutes", "pts", "reb", "ast", "player_id"])
    x = x.loc[(x["minutes"] >= 5) & (x["minutes"] <= 53) & x["player"].ne("")]
    x[list(MARKET_COLS.values())] = x[list(MARKET_COLS.values())].fillna(0).clip(lower=0)
    x = x.sort_values(["date", "player_id"]).drop_duplicates(["date", "player_id"], keep="last")
    return x.reset_index(drop=True)

def load_public_logs(when: date):
    """Return records STRICTLY before target date, and provenance metadata."""
    files = discover_game_log_files(when)
    frames = []
    for obj in files:
        response = requests.get(f"{RAW_ROOT}/{obj['name']}", timeout=30, headers={"User-Agent": "TeamAI2Pro/1.0"})
        response.raise_for_status()
        frame = pd.read_csv(BytesIO(response.content), low_memory=False)
        frames.append(frame)
    data = normalize_logs(pd.concat(frames, ignore_index=True))
    data = data.loc[data["date"] < pd.Timestamp(when)].copy()
    if data.empty:
        raise ValueError("No pre-game historical NBA observations were returned.")
    metadata = {
        "provider": "Community NBA player-game-log CSV (xtheredshirtx/fanduel-player-data)",
        "provider_url": "https://github.com/xtheredshirtx/fanduel-player-data/tree/main/NBA/csv",
        "snapshot": max(obj["as_of"] for obj in files),
        "latest_game_date": str(data["date"].max().date()),
        "seasons": [obj["season"] for obj in files],
        "rows": len(data),
    }
    return data, metadata
