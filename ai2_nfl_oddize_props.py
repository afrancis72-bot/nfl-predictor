"""Oddize documented NFL player props API adapter (free monthly credits).

Requires an API key supplied via Streamlit secrets or environment; never log it.
Market lines are validation inputs only, not simulator projections.
"""
import json
import os
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from ai2_nfl_prop_comparison import SUPPORTED

API_ROOT = "https://oddize.com/api/v1"
MARKETS = {
    "player_pass_yds": "passing_yards",
    "player_pass_tds": "passing_tds",
    "player_pass_attempts": "pass_attempts",
    "player_pass_completions": "completions",
    "player_pass_interceptions": "interceptions",
    "player_rush_yds": "rushing_yards",
    "player_rush_attempts": "carries",
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
}
def _request(path, key, timeout=12, params=None):
    url = API_ROOT + path + ("?" + urlencode(params) if params else "")
    req = Request(url, headers={"X-API-Key": key, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response), dict(response.headers)

def fetch_oddize_props(key, prop_types, *, timeout=12):
    """One 2-credit call per selected market; return only documented main lines.

    Callers should restrict market count and cache results to conserve credits.
    """
    if not key or not str(key).strip():
        raise ValueError("Oddize API key is not configured")
    if not prop_types or len(prop_types) > 4:
        raise ValueError("Choose 1 to 4 prop markets per refresh")
    rows = []
    remaining = None
    for prop_type in prop_types:
        if prop_type not in MARKETS:
            raise ValueError("Unsupported Oddize prop market: " + str(prop_type))
        payload, headers = _request("/props/nfl", key, timeout,
                                    {"prop_type": prop_type})
        remaining = headers.get("X-Credits-Remaining", remaining)
        for event in payload.get("events", []):
            for player in event.get("players", []):
                name = player.get("player_name")
                main_line = player.get("main_line")
                if not name or not isinstance(main_line, (int, float)):
                    continue
                rows.append({"player": name, "stat": MARKETS[prop_type],
                             "line": float(main_line),
                             "sportsbook": "Oddize main line (book unspecified)",
                             "event_id": event.get("event_id", "")})
    return rows, {"source": "Oddize NFL player props API",
                  "checked_utc": datetime.now(timezone.utc).isoformat(),
                  "prop_rows": len(rows), "credits_remaining": remaining,
                  "markets_checked": len(prop_types)}
