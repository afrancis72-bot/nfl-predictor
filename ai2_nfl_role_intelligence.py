"""Observed-workload reentry watch for NFL running-back committees.

A player who previously handled substantial carries but has missed multiple
team games may reclaim volume when he returns. This is NOT a live medical
clearance, a trusted depth chart, or a calibrated playing probability.
Every watch lists observed source, missed weeks and current status, and can
only reduce the portfolio's confidence in a teammate, never fabricate points.
"""
import math
import pandas as pd
from ai2_nfl_matchups import canonical_team
from ai2_nfl_auto_baselines import _norm_name

ABSENT = {"IR", "INACTIVE", "PUP", "NFI", "SUSPENDED"}
UNCERTAIN = {"Q", "QUESTIONABLE", "", "ACTIVE", "HEALTHY", "O", "OUT"}


def returning_rb_competition(baselines, raw_rates, *, season, week):
    """Return (per-player risk, auditable teammate candidates) from pregame data.

    Presence in the DK slate does not confirm availability. Only candidates
    with >=3 observed games, >=10 historic carries and >=2 weeks of missed
    actual games create a watch for other RBs on their team.
    """
    if not isinstance(baselines, pd.DataFrame) or not isinstance(raw_rates, pd.DataFrame):
        raise ValueError("Returning player review needs source tables")
    required_b = {"Name","Team","Position","current_status","ID"}
    required_r = {"player","team","position","games_played",
                  "carries_pg","latest_observed_season","latest_observed_week"}
    if not required_b.issubset(baselines.columns) or not required_r.issubset(raw_rates.columns):
        raise ValueError("Returning RB review missing player or historical source fields")
    latest_rates = {}
    for row in raw_rates.to_dict("records"):
        if str(row.get("position","")).upper() != "RB":
            continue
        key = (_norm_name(row.get("player","")),canonical_team(row.get("team","")))
        # Conflicting historical identity must not be silently merged.
        if key in latest_rates:
            latest_rates[key] = None
        else:
            latest_rates[key] = row
    watched = []
    roster = [x for x in baselines.to_dict("records")
              if str(x.get("Position","")).upper() == "RB"]
    for item in roster:
        team = canonical_team(item.get("Team",""))
        rate = latest_rates.get((_norm_name(item.get("Name","")),team))
        if not rate:
            continue
        try:
            games = int(rate["games_played"])
            carries = float(rate["carries_pg"])
            last_season = int(rate["latest_observed_season"])
            last_week = int(rate["latest_observed_week"])
        except (TypeError,ValueError,KeyError):
            continue
        if not (math.isfinite(carries) and games >= 3 and carries >= 10
                and last_season == int(season)
                and 2 <= int(week)-last_week <= 8):
            continue
        status = str(item.get("current_status") or "").strip().upper()
        if status in ABSENT or status not in UNCERTAIN:
            continue
        # An OUT designation from a generic undated NFL.com injury page
        # might be for the *previous* game. It is not treated as clearance
        # to play, but cannot alone eliminate return uncertainty for the
        # next slate. Keep a conservative teammate exposure watch until
        # an explicitly target-week designation is independently checked.
        status_note = ("OUT designation; report's target week not independently verified"
                       if status in {"OUT","O"} else
                       "Current status does not verify game-day availability")
        watched.append({
            "status_note":status_note,
            "returning_player":str(item["Name"]),
            "returning_player_id":str(item["ID"]),
            "team":team,"current_status":status or "NOT_DESIGNATED",
            "last_observed_week":last_week,"weeks_since_last_game":int(week)-last_week,
            "observed_carries_pg":round(carries,2),"observed_games":games,
            "risk_level":"REENTRY_UNCONFIRMED",
            "source":"nflverse current-season pregame rushing usage + DK/official injury designation",
            "interpretation":"Potential returning high-volume teammate; do not assume prior backup workload persists or that he is active",
        })
    risk = {}
    for watch in watched:
        for item in roster:
            if (canonical_team(item.get("Team","")) == watch["team"]
                    and str(item.get("ID")) != watch["returning_player_id"]):
                ident = str(item["ID"])
                risk.setdefault(ident,[]).append(watch)
    return risk,watched
