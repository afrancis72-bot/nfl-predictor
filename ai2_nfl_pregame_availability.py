"""Research-only, fail-closed NFL game-day availability evidence.

An injury designation is NOT a game-day ACTIVE list. A player's historical
stats, DK salary listing, or absence from an inactive list are NOT proof the
player is active. This validator only accepts an explicit positive game-day
active record from an authoritative source, or explicit OUT evidence.

Real snapshots MUST be acquired/persisted before kickoff. A retrospective
nflverse weekly roster or a postgame box score cannot be used as a pregame
availability source. No network I/O or unsupported scraping is performed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlparse

from ai2_nfl_auto_baselines import _norm_name
from ai2_nfl_matchups import canonical_team


ACTIVE="VERIFIED_ACTIVE"
OUT="VERIFIED_OUT"
UNKNOWN="UNCERTAIN"
VALID_STATUSES={ACTIVE,OUT,UNKNOWN}
# Explicit positive ACTIVE required; the official inactive list's omission is
# never sufficient evidence that a player was on the active game-day roster.
ALLOWED_SOURCE_STATUS={
    "NFL_GAME_DAY_ACTIVE_ROSTER":{ACTIVE},
    "NFL_OFFICIAL_INACTIVES":{OUT},
    "NFL_OFFICIAL_GAME_STATUS_REPORT":{OUT,UNKNOWN},
}
APPROVED_HOSTS={"www.nfl.com","nfl.com","amp.nfl.com","operations.nfl.com"}
DATE_KEYS=("kickoff_utc","captured_at_utc")
PROOF_KEYS=("player","team","status","source_type","source_url","published_at_utc")


def _utc(value,field):
    if not isinstance(value,str) or not value:
        raise ValueError(f"{field} must be a timestamp with UTC timezone")
    try:
        dt=datetime.fromisoformat(value.replace("Z","+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid {field}") from exc
    if dt.tzinfo is None or dt.utcoffset().total_seconds()!=0:
        raise ValueError(f"{field} must be explicitly UTC")
    return dt.astimezone(timezone.utc)


def _url(value):
    if not isinstance(value,str) or not value:
        raise ValueError("Evidence must link to the official NFL source URL")
    parsed=urlparse(value)
    if parsed.scheme!="https" or parsed.hostname not in APPROVED_HOSTS or not parsed.path.strip("/"):
        raise ValueError("Availability evidence URL must be official NFL HTTPS page")
    if parsed.username or parsed.password or parsed.port or parsed.fragment:
        raise ValueError("Invalid official source URL metadata")
    return value


def validate_availability_snapshot(snapshot,*,season,week,game,as_of_utc=None):
    """Return indexed verified pregame evidence; strict identity/time checks.

    The input may be partial. A missing entry is UNCERTAIN (never automatically
    ACTIVE or OUT). All rows must refer to the same scheduled game, time before
    kickoff, and exact source, with zero target-game output information.

    The source's original publication timestamp is required, and it must be
    earlier than the capture and kickoff. Postgame or backfilled files fail.
    """
    if not isinstance(snapshot,dict):
        raise ValueError("Availability snapshot must be an explicit record")
    needed=set(DATE_KEYS)|{"season","week","game","records"}
    if not needed.issubset(snapshot):
        raise ValueError("Availability snapshot missing game/time/records provenance")
    if int(snapshot["season"])!=int(season) or int(snapshot["week"])!=int(week):
        raise ValueError("Availability snapshot season/week mismatches target slate")
    match=str(game).split()[0]
    snap_match=str(snapshot["game"]).split()[0]
    def split_game(v):
        sides=v.split("@")
        if len(sides)!=2 or not all(sides):
            raise ValueError("Invalid explicit game-team identity")
        return tuple(canonical_team(side) for side in sides)
    teams=split_game(match)
    if split_game(snap_match)!=teams:
        raise ValueError("Availability snapshot game identity mismatches target slate")
    kickoff=_utc(snapshot["kickoff_utc"],"kickoff_utc")
    captured=_utc(snapshot["captured_at_utc"],"captured_at_utc")
    if captured>=kickoff:
        raise ValueError("Post-kickoff snapshot cannot support pregame model")
    if as_of_utc is not None and captured>_utc(as_of_utc,"as_of_utc"):
        raise ValueError("Future availability snapshot relative to model as-of time")
    records=snapshot["records"]
    if not isinstance(records,list):
        raise ValueError("Snapshot records must be an explicit list")
    indexed={}
    for entry in records:
        if not isinstance(entry,dict) or not set(PROOF_KEYS).issubset(entry):
            raise ValueError("Incomplete player status source/provenance")
        team=canonical_team(entry["team"])
        player=_norm_name(entry["player"])
        if team not in teams or not player:
            raise ValueError("Availability evidence player/team identity incorrect")
        key=(team,player)
        if key in indexed:
            # Contradictions must fail rather than choosing the latest string.
            raise ValueError("Duplicate/conflicting player status in one snapshot")
        status=str(entry["status"]).upper()
        source=str(entry["source_type"])
        if status not in VALID_STATUSES or source not in ALLOWED_SOURCE_STATUS:
            raise ValueError("Unrecognized NFL status or source type")
        if status not in ALLOWED_SOURCE_STATUS[source]:
            raise ValueError("Source cannot prove this game-day status")
        link=_url(entry["source_url"])
        published=_utc(entry["published_at_utc"],"published_at_utc")
        if published>captured or published>=kickoff:
            raise ValueError("Availability publication after pregame capture/kickoff")
        indexed[key]={
            "team":team,"player":str(entry["player"]), "status":status,
            "source_type":source,"source_url":link,
            "published_at_utc":published.isoformat(),
            "captured_at_utc":captured.isoformat(),
            "kickoff_utc":kickoff.isoformat(),
        }
    return indexed


def classify_game_players(pool,rates,snapshot,*,season,week,game,as_of_utc=None):
    """Document ACTIVE/OUT/UNCERTAIN classifications for all historical teammates.

    No-weekly-roster-after-game heuristic, no 'questionable means OUT', no
    assumed activation from lack of inactive announcement.
    """
    indexed=validate_availability_snapshot(
        snapshot,season=season,week=week,game=game,as_of_utc=as_of_utc)
    state={"verified_active":[],"verified_out":[],"uncertain":[]}
    known={(canonical_team(row["TeamAbbrev"]),_norm_name(row["Name"]))
           for row in pool.to_dict("records") if bool(row["optimizer_eligible"])}
    observed=set()
    for row in rates.to_dict("records"):
        team=canonical_team(row["team"])
        if str(row.get("position","")).upper() not in {"RB","WR","TE"}:
            continue
        identity=(team,_norm_name(row["player"]))
        if team not in {canonical_team(t) for t in str(game).split("@")}:
            continue
        if identity in observed:
            raise ValueError("Duplicate historical teammate identity")
        observed.add(identity)
        proof=indexed.get(identity)
        if identity in known and proof and proof["status"]==OUT:
            raise ValueError("Confirmed-OUT player remains DraftKings optimizer eligible")
        if identity in known:
            continue
        if proof and proof["status"]==ACTIVE:
            state["verified_active"].append(identity)
        elif proof and proof["status"]==OUT:
            state["verified_out"].append(identity)
        else:
            state["uncertain"].append(identity)
    state["records_with_explicit_evidence"]=len(indexed)
    state["as_of_utc"]=snapshot["captured_at_utc"]
    state["no_evidence_is_not_active"]=True
    return state
