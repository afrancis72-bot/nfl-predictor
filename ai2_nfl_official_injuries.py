"""Best-effort official NFL injury statuses; never infer OUT from missed practice.

Only explicit game designations (Out, Doubtful, Questionable) are applied.
If the page format changes or retrieval fails, callers must surface that fact.
"""
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

SOURCE_URL = "https://www.nfl.com/injuries/"
GAME_STATUSES = {"OUT", "DOUBTFUL", "QUESTIONABLE"}

def fetch_official_injury_statuses(timeout=12):
    """Return (normalized player-name -> game status, metadata).

    A player name with conflicting designations is discarded, not guessed.
    """
    req = Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0 TEAM-AI2-Pro/1.0"})
    with urlopen(req, timeout=timeout) as response:
        html = response.read()
    soup = BeautifulSoup(html, "html.parser")
    statuses = {}
    conflicts = set()
    table_count = 0
    for table in soup.select("table"):
        headers = [h.get_text(" ", strip=True).lower() for h in table.select("thead th")]
        if not headers:
            first = table.select_one("tr")
            if first:
                headers = [h.get_text(" ", strip=True).lower() for h in first.select("th")]
        name_col = next((i for i, h in enumerate(headers) if h in {"player", "name"}), None)
        status_col = next((i for i, h in enumerate(headers) if "game status" in h), None)
        if name_col is None or status_col is None:
            continue
        table_count += 1
        for tr in table.select("tbody tr"):
            cells = tr.find_all(["td", "th"], recursive=False)
            if len(cells) <= max(name_col, status_col):
                continue
            name = cells[name_col].get_text(" ", strip=True)
            status = cells[status_col].get_text(" ", strip=True).upper()
            if not name or status not in GAME_STATUSES:
                continue
            key = "".join(ch for ch in name.casefold() if ch.isalnum())
            if key in statuses and statuses[key] != status:
                conflicts.add(key)
            statuses[key] = status
    for key in conflicts:
        statuses.pop(key, None)
    if table_count == 0:
        raise ValueError("Official NFL injury page has no recognizable player/game-status tables")
    return statuses, {"source": SOURCE_URL, "checked_utc": datetime.now(timezone.utc).isoformat(),
                      "tables": table_count, "designated_players": len(statuses),
                      "conflicts": len(conflicts)}

def match_injury_statuses(salary_rows, statuses):
    """Match only unambiguous normalized names to DK player IDs."""
    from collections import Counter
    counts = Counter("".join(ch for ch in str(r.get("Name", "")).casefold() if ch.isalnum())
                     for r in salary_rows)
    result = {}
    for row in salary_rows:
        key = "".join(ch for ch in str(row.get("Name", "")).casefold() if ch.isalnum())
        if counts[key] == 1 and key in statuses:
            result[str(row["ID"])] = statuses[key]
    return result
