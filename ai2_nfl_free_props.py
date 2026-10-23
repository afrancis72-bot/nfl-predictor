"""Optional free/public NFL prop data via ESPN's public event odds summary.

This source is undocumented and can omit player props entirely. Fail closed.
Never synthesize sportsbook lines from model projections.
"""
import json
from datetime import datetime, timezone
from urllib.request import Request, urlopen

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event_id}"

def _json(url, timeout=8):
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout) as response:
        return json.load(response)

def fetch_public_espn_props(timeout=8, max_events=16):
    """Return (rows, metadata). No rows is a normal, honest outcome."""
    scoreboard = _json(SCOREBOARD, timeout)
    events = scoreboard.get("events", [])[:max_events]
    rows = []
    checked = datetime.now(timezone.utc).isoformat()
    # ESPN's public endpoints generally offer game odds, not player props.
    # Only accept explicit player/market/line fields if provided.
    from ai2_nfl_prop_comparison import norm_stat, SUPPORTED
    for event in events:
        eid = event.get("id")
        if not eid:
            continue
        try:
            summary = _json(SUMMARY.format(event_id=eid), timeout)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        for item in summary.get("props", []):
            if not isinstance(item, dict):
                continue
            player = item.get("playerName")
            stat = norm_stat(item.get("market", ""))
            line = item.get("line")
            book = item.get("provider", "ESPN public")
            if not player or stat not in SUPPORTED or not isinstance(line, (int, float)):
                continue
            rows.append({"player": player, "stat": stat, "line": line,
                         "sportsbook": str(book), "event_id": str(eid)})
    return rows, {"source": "ESPN public NFL event summaries",
                  "checked_utc": checked, "events_checked": len(events),
                  "prop_rows": len(rows), "warning":
                  "Public ESPN event summaries may not expose player props; zero rows is not a successful prop feed."}
