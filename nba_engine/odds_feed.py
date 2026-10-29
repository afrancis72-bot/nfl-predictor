"""Optional licensed real NBA player-prop prices via The Odds API V4.

This module never treats a bookmaker line as a projection. The credentials
remain client-side in Streamlit's password input (or deployment secrets).
Each explicit refresh may consume API quota; no periodic polling is performed.
"""
from __future__ import annotations
from datetime import datetime
from difflib import SequenceMatcher
import re
import unicodedata

import pandas as pd
import requests

from nba_engine.data_sources import norm_team

BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

BOOKMAKERS = {
    "Hard Rock Bet Florida": "hardrockbet_fl",
    "DraftKings": "draftkings",
    "FanDuel": "fanduel",
    "BetMGM": "betmgm",
}

MARKET_TO_MODEL = {
    "player_points": "Points",
    "player_rebounds": "Rebounds",
    "player_assists": "Assists",
    "player_threes": "Threes",
    "player_blocks": "Blocks",
    "player_steals": "Steals",
    "player_turnovers": "Turnovers",
    "player_points_rebounds_assists": "PRA",
    "player_points_rebounds": "PR",
    "player_points_assists": "PA",
    "player_rebounds_assists": "RA",
    "player_blocks_steals": "Stocks",
}
MODEL_TO_MARKET = {v:k for k,v in MARKET_TO_MODEL.items()}

NBA_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets",
    "IND": "Indiana Pacers", "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}
NAME_TO_TEAM = {v.lower():k for k,v in NBA_NAMES.items()}

def _norm_name(name):
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", s)
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())

def player_id_for_name(name, eligible: pd.DataFrame):
    """Only automatic unambiguous high-quality player matches are admitted."""
    target = _norm_name(name)
    if not target:
        return None
    catalog = [(str(r.player_id), _norm_name(r.player)) for r in eligible.itertuples(index=False)]
    exact = sorted(set(pid for pid, normalized in catalog if normalized == target))
    if len(exact) == 1:
        return exact[0]
    if exact:
        return None
    scores = sorted([(SequenceMatcher(None,target,text).ratio(),pid)
                     for pid,text in catalog], reverse=True)
    if not scores or scores[0][0] < .91:
        return None
    if len(scores) > 1 and scores[0][0] - scores[1][0] < .09:
        return None
    return scores[0][1]

def parse_event_props(event: dict, projections: pd.DataFrame, bookmaker_key: str):
    """Pure parser: return strict market-only rows and unresolved player labels."""
    expected_teams = {
        NAME_TO_TEAM.get(str(event.get("home_team","")).lower()),
        NAME_TO_TEAM.get(str(event.get("away_team","")).lower()),
    }
    if None in expected_teams or len(expected_teams) != 2:
        return [], []
    eligible = projections[projections["team"].map(norm_team).isin(expected_teams)]
    rows, unmatched = [], set()
    book = next((b for b in event.get("bookmakers",[]) if b.get("key")==bookmaker_key), None)
    if not book:
        return rows, []
    for market in book.get("markets",[]):
        label = MARKET_TO_MODEL.get(market.get("key"))
        if not label:
            continue
        for item in market.get("outcomes",[]):
            side = str(item.get("name","")).title()
            if side not in ("Over","Under"):
                continue
            name = str(item.get("description","")).strip()
            pid = player_id_for_name(name,eligible)
            if not pid:
                unmatched.add(name)
                continue
            price = item.get("price")
            line = item.get("point")
            try:
                price, line = int(price), float(line)
                if price == 0 or -100 < price < 100 or line < 0:
                    continue
            except (TypeError, ValueError):
                continue
            player = eligible.loc[eligible["player_id"].astype(str)==pid].iloc[0]
            rows.append({
                "player_id":pid, "player":str(player["player"]),
                "team":str(player["team"]), "market":label,
                "line":line, "side":side, "odds":price,
                "sportsbook":str(book.get("title") or bookmaker_key),
                "event_id":str(event.get("id","")),
                "book_update_utc":str(market.get("last_update") or book.get("last_update") or ""),
            })
    return rows, sorted(unmatched)

def _api_get(url, api_key, **params):
    """Don't include secrets in the UI or in exception messages."""
    try:
        response = requests.get(
            url, params={"apiKey":api_key, **params},
            timeout=18, headers={"Accept":"application/json"},
        )
        credits = {h:response.headers.get(h)
                   for h in ("x-requests-remaining","x-requests-used","x-requests-last")
                   if response.headers.get(h) is not None}
        if response.status_code == 401:
            raise ValueError("The API key was rejected. Check it in your sportsbook data provider account.")
        if response.status_code in (402,403):
            raise ValueError("The provider rejected the requested player-prop access or plan quota.")
        if response.status_code == 429:
            raise ValueError("The provider rate limit or free quota was reached.")
        if not response.ok:
            raise ValueError(f"Market provider request failed (HTTP {response.status_code}).")
        return response.json(), credits
    except requests.RequestException as exc:
        # Exception URLs may include the key; never stringify them.
        raise ValueError("The odds provider did not respond. Check connectivity and try again.") from None

def load_current_props(projections: pd.DataFrame, teams_map: dict, api_key: str,
                       bookmaker_key: str = "hardrockbet_fl", markets=None,
                       max_events: int = 10):
    """Fetch only matching events, explicitly requested once by the user.

    Makes one 'events' request and ONE priced event request per matched matchup.
    Missing bookmaker markets are missing data, not betting edges.
    """
    if not api_key or len(api_key.strip()) < 6:
        raise ValueError("A valid The Odds API key is required for automatic sportsbook odds.")
    if bookmaker_key not in BOOKMAKERS.values():
        raise ValueError("Unsupported bookmaker.")
    markets = list(markets or ["Points","Rebounds","Assists","Threes","PRA"])
    unknown = set(markets) - set(MODEL_TO_MARKET)
    if unknown:
        raise ValueError("Unsupported markets: " + ", ".join(sorted(unknown)))
    if not markets:
        raise ValueError("Select at least one market.")
    raw_events, credits = _api_get(f"{BASE}/sports/{SPORT}/events", api_key)
    allowed_pairs = {frozenset((str(t),str(opp))) for t,opp in teams_map.items()}
    matching = []
    for ev in raw_events:
        home = NAME_TO_TEAM.get(str(ev.get("home_team","")).lower())
        away = NAME_TO_TEAM.get(str(ev.get("away_team","")).lower())
        if home and away and frozenset((home,away)) in allowed_pairs:
            matching.append(ev)
    if len(matching) > max_events:
        raise ValueError(f"{len(matching)} events found; select fewer games to control API quota.")
    rows, unresolved = [], set()
    markets_arg=",".join(MODEL_TO_MARKET[m] for m in markets)
    for event in matching:
        data, current_credits = _api_get(
            f"{BASE}/sports/{SPORT}/events/{event['id']}/odds", api_key,
            bookmakers=bookmaker_key,
            markets=markets_arg,
            oddsFormat="american",
        )
        credits.update(current_credits)
        new_rows, unmatched = parse_event_props(data, projections, bookmaker_key)
        rows.extend(new_rows)
        unresolved.update(unmatched)
    frame = pd.DataFrame(rows, columns=[
        "player_id","player","team","market","line","side","odds",
        "sportsbook","event_id","book_update_utc"
    ])
    frame = frame.drop_duplicates(
        ["player_id","market","side","line","sportsbook"], keep="last"
    ).reset_index(drop=True)
    return frame, {
        "sportsbook":bookmaker_key,
        "event_count":len(matching),
        "matched_market_rows":len(frame),
        "unmatched_player_names":sorted(unresolved)[:30],
        "retrieved_utc":datetime.now().astimezone().isoformat(),
        "quota":credits,
    }
