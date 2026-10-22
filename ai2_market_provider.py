"""Read-only ParlayAPI prop retrieval, strictly after simulation freeze.

Explicit athlete ID mapping required; no fuzzy name joins, no simulator imports.
Caller provides API key from Streamlit secrets/environment, never source code.
"""
from datetime import datetime, timezone
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from ai2_market_challenge import FrozenSimulation, SPORT_KEYS

BASE_URL = "https://parlay-api.com"
MARKETS = {
    "NFL": ("player_pass_yds", "player_pass_tds", "player_pass_completions",
            "player_rush_yds", "player_rec_yds", "player_receptions"),
    "NBA": ("player_points", "player_rebounds", "player_assists", "player_threes"),
}

def retrieve_market_props(frozen, *, sport, api_key, identity_map,
                          fetcher=None, max_rows=5000):
    """Fetch a bounded flat prop board and normalize verified ID matches.

    identity_map maps (event_id, player_name) -> internal player_id.
    Unmapped athletes are skipped, never guessed. Network call occurs only
    after validating FrozenSimulation and required parameters.
    """
    if not isinstance(frozen, FrozenSimulation):
        raise TypeError("Freeze the simulation before retrieving market odds")
    if sport not in SPORT_KEYS:
        raise ValueError("Unsupported sport")
    if not api_key or not str(api_key).strip():
        raise ValueError("API key required")
    if not isinstance(identity_map, dict):
        raise TypeError("Explicit identity map required")
    if not isinstance(max_rows, int) or not 1 <= max_rows <= 10000:
        raise ValueError("Invalid row limit")
    query = urlencode({"markets": ",".join(MARKETS[sport]),
                       "limit": max_rows, "maxAgeSec": 900})
    url = f"{BASE_URL}/v1/sports/{SPORT_KEYS[sport]}/props?{query}"
    headers = {"X-API-Key": str(api_key), "Accept": "application/json"}
    if fetcher is None:
        def fetcher(url, headers):
            with urlopen(Request(url, headers=headers), timeout=12) as response:
                return json.loads(response.read(5_000_001).decode("utf-8"))
    data = fetcher(url, headers)
    if not isinstance(data, list):
        raise ValueError("Unexpected props response; cannot establish completeness")
    if len(data) >= max_rows:
        raise ValueError("Prop board may be truncated; narrow query or paginate")
    fetched_at = datetime.now(timezone.utc).isoformat()
    allowed = {key for _, key, _ in frozen.observations}
    normalized = []
    for item in data:
        if not isinstance(item, dict):
            continue
        player_id = identity_map.get((str(item.get("event_id", "")),
                                      str(item.get("player_name", ""))))
        if not player_id or item.get("market_key") not in allowed:
            continue
        if str(item.get("period", "")).upper() != "FULL":
            continue
        normalized.append({
            "player_id": str(player_id), "market_key": item["market_key"],
            "period": "FULL", "line": item.get("line"),
            "over_price": item.get("over_price"),
            "under_price": item.get("under_price"),
            "bookmaker": item.get("bookmaker"),
            "event_id": item.get("event_id"),
            "last_update": item.get("last_update"),
            "fetched_at": fetched_at,
        })
    return normalized
