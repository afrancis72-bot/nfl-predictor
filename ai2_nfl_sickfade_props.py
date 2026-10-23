"""Experimental read-only extraction of publicly displayed NFL prop lines.

Source page structure may change; a zero-row result is NOT a successful feed.
No login, bypass, hidden API, or synthetic odds are used.
"""
import re
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from ai2_nfl_prop_comparison import norm_stat, SUPPORTED

URL = "https://sickfade.com/odds-screen/nfl/props"
STAT_NAMES = {
    "pass yards": "passing_yards", "passing yards": "passing_yards",
    "pass tds": "passing_tds", "pass attempts": "pass_attempts",
    "pass completions": "completions", "int": "interceptions",
    "rush yards": "rushing_yards", "rush attempts": "carries",
    "receiving yards": "receiving_yards", "receptions": "receptions",
}
OVER = re.compile(r"\\bo\\s*(\\d+(?:\\.\\d+)?)\\s*[+\\-]\\d+", re.I)

def parse_sickfade_html(html, *, prefer="Hard Rock"):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    tables_seen = 0
    for table in soup.select("table"):
        headers = [th.get_text(" ", strip=True) for th in table.select("thead th")]
        if not headers:
            first = table.find("tr")
            headers = [th.get_text(" ", strip=True) for th in first.find_all("th")] if first else []
        lower = [h.casefold() for h in headers]
        player_idx = next((i for i, h in enumerate(lower) if h == "player"), None)
        stat_idx = next((i for i, h in enumerate(lower) if h == "stat"), None)
        if player_idx is None or stat_idx is None:
            continue
        books = [(i, h) for i, h in enumerate(headers) if
                 any(b in h.casefold() for b in ("hard rock", "draftkings", "fanduel", "betmgm", "caesars"))]
        if not books:
            continue
        tables_seen += 1
        for tr in table.select("tr"):
            cells = tr.find_all(["td", "th"], recursive=False)
            if len(cells) < len(headers):
                continue
            player = cells[player_idx].get_text(" ", strip=True)
            raw_stat = cells[stat_idx].get_text(" ", strip=True).casefold()
            stat = STAT_NAMES.get(raw_stat, norm_stat(raw_stat))
            if not player or stat not in SUPPORTED:
                continue
            choices = sorted(books, key=lambda x: (prefer.casefold() not in x[1].casefold(), x[0]))
            for i, book in choices:
                match = OVER.search(cells[i].get_text(" ", strip=True))
                if match:
                    rows.append({"player": player, "stat": stat,
                                 "line": float(match.group(1)), "sportsbook": book})
                    break
    return rows, tables_seen

def fetch_sickfade_props(timeout=12):
    req = Request(URL, headers={"User-Agent": "Mozilla/5.0 (compatible; TEAM-AI2-Pro/1.0)"})
    with urlopen(req, timeout=timeout) as response:
        html = response.read()
    rows, tables = parse_sickfade_html(html)
    if not rows:
        raise ValueError("Public page returned no parseable player prop table; site markup may differ.")
    return rows, {"source": URL, "checked_utc": datetime.now(timezone.utc).isoformat(),
                  "events_checked": tables, "prop_rows": len(rows)}
