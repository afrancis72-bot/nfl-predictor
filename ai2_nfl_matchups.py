"""Infer NFL opponent from DraftKings Game Info (e.g. BUF@MIA 1:00PM ET).

Unknown or inconsistent matchups stay unverified.
"""
import re

MATCHUP = re.compile(r"^\s*([A-Za-z]{2,3})\s*@\s*([A-Za-z]{2,3})(?=\s|$)")

def opponent_from_game_info(team, game_info):
    match = MATCHUP.match(str(game_info or ""))
    if not match:
        return None
    away, home = (part.upper() for part in match.groups())
    team = str(team or "").upper().strip()
    if away == home:
        return None
    if team == away:
        return home
    if team == home:
        return away
    return None
