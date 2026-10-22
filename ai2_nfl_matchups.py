"""Infer NFL opponent from DraftKings Game Info (e.g. BUF@MIA 1:00PM ET).

Unknown or inconsistent matchups stay unverified.
"""
import re

MATCHUP = re.compile(r"^\s*([A-Za-z]{2,3})\s*@\s*([A-Za-z]{2,3})(?=\s|$)")

ALIASES = {"JAC": "JAX", "WSH": "WAS", "ARZ": "ARI", "LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV"}

def canonical_team(value):
    team = str(value or "").upper().strip()
    return ALIASES.get(team, team)

def opponent_from_game_info(team, game_info):
    match = MATCHUP.match(str(game_info or ""))
    if not match:
        return None
    away, home = (canonical_team(part) for part in match.groups())
    team = canonical_team(team)
    if away == home:
        return None
    if team == away:
        return home
    if team == home:
        return away
    return None
