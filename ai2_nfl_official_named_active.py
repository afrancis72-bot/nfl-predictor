"""Strict capture of named NFL.com game-day 'is active' news claims.

This does NOT turn an injury report's EXPECTED TO PLAY, QUESTIONABLE, lack
of OUT designation, or practice participation into verified active status.

A source URL, metadata, article title, source bytes, team news section and
exact sentence must all agree. A historical article may be used to smoke-test
parsing, but it cannot retrospectively become pregame evidence.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

from ai2_nfl_matchups import canonical_team
from ai2_nfl_official_inactives_capture import (
    _article_date,_allowed_url,_TEAM_ALIASES,fetch_official_html,MAX_BYTES,
)
from ai2_nfl_pregame_availability import (
    ACTIVE,validate_availability_snapshot,_utc,
)

_POS={"QB","RB","HB","FB","WR","TE"}
_ACTIVE_RE=re.compile(
    r"\b(?:is\s+(?:officially\s+)?active|"
    r"(?:was|is)\s+(?:officially\s+)?listed\s+as\s+active|"
    r"has\s+been\s+(?:officially\s+)?declared\s+active)\b",
    re.IGNORECASE)
_NEGATIVE_RE=re.compile(
    r"\b(?:not\s+(?:officially\s+)?active|"
    r"not\s+listed\s+as\s+active|"
    r"might\s+be\s+active|"
    r"could\s+be\s+active)\b",
    re.IGNORECASE)
_END={"related content","more news","latest news"}


def _team_from_link(tag):
    if tag.name!="a" or tag.find_parent("li"):
        return None
    path=str(tag.get("href") or "").lower()
    if "/teams/" not in path:
        return None
    words=" ".join(tag.get_text(" ",strip=True).upper().split())
    for fullname,code in _TEAM_ALIASES.items():
        if words==fullname or words.startswith(fullname+" "):
            return code
    return None


def _headline(soup,week):
    title=soup.find("h1")
    headline=title.get_text(" ",strip=True) if title else ""
    low=headline.lower()
    if (f"week {int(week)}" not in low or
        not ("injur" in low and ("sunday" in low or "pregame" in low))):
        raise ValueError("Not a week-specific official NFL Sunday injury roundup")
    return title,headline


def parse_named_active_news(raw,url,*,season,week,game):
    """Extract ONLY explicitly active named players under exact NFL team cards.

    The source must have unambiguous actual present-tense confirmations.
    Expected-to-play text alone is deliberately excluded.
    """
    _allowed_url(url)
    if not isinstance(raw,(bytes,bytearray)) or not 200<len(raw)<=MAX_BYTES:
        raise ValueError("Invalid article HTML source bytes")
    parts=str(game).split("@")
    if len(parts)!=2 or len(set(parts))!=2:
        raise ValueError("Explicit two-team matchup required")
    teams={canonical_team(t) for t in parts}
    soup=BeautifulSoup(bytes(raw),"html.parser")
    h1,headline=_headline(soup,week)
    published=_article_date(soup)
    if _utc(published,"published_at_utc").year!=int(season):
        raise ValueError("Wrong NFL season publication year")
    root=soup.find("main")
    if root is None or h1 not in root.descendants:
        raise ValueError("Official NFL news main-content boundary missing")
    after=False
    current=None
    found={t:[] for t in teams}
    for node in root.find_all(["h1","h2","h3","a","li"]):
        if node is h1:
            after=True
            continue
        if not after:
            continue
        if node.name in ("h2","h3") and node.get_text(" ",strip=True).lower() in _END:
            break
        if node.name=="a":
            t=_team_from_link(node)
            if t:
                current=t
            continue
        if node.name!="li" or current not in teams:
            continue
        player_link=None
        for a in node.find_all("a",href=True):
            if "/players/" in a["href"]:
                player_link=a
                break
        if player_link is None:
            continue
        player=" ".join(player_link.get_text(" ",strip=True).split())
        text=" ".join(node.get_text(" ",strip=True).replace("\ufeff","").split())
        match=re.match(r"^(QB|RB|HB|FB|WR|TE)\s+",text,re.IGNORECASE)
        if not match:
            continue
        position=match.group(1).upper()
        remaining=text[match.end():]
        if not remaining.casefold().startswith(player.casefold()+" "):
            continue
        statement=remaining[len(player):].strip()
        if not _ACTIVE_RE.search(statement) or _NEGATIVE_RE.search(statement):
            continue
        # The actual team card + week are identity anchors, not 'active'
        # elsewhere in unrelated editorial story copy.
        found[current].append({
            "team":current,"player":player,"position":position,
            "article_sentence":text[:450],
            "source_evidence_type":"NFL_NEWS_PLAYER_BULLET_EXPLICIT_IS_ACTIVE",
        })
    emitted=[r for rows in found.values() for r in rows]
    if not emitted:
        raise ValueError("Official article contains no unambiguous named game-day ACTIVE claims for matchup")
    seen=set()
    for row in emitted:
        key=(row["team"],row["player"].casefold())
        if key in seen:
            raise ValueError("Conflicting repeated active-status evidence for player")
        seen.add(key)
    return {
        "headline":headline,"season":int(season),"week":int(week),"game":str(game),
        "source_url":url,"published_at_utc":published,
        "source_html_sha256":hashlib.sha256(bytes(raw)).hexdigest(),
        "active_by_team":{k:v for k,v in found.items() if v},
        "explicit_only":True,
        "inferred_from_no_inactive_listing":False,
    }


def make_named_active_snapshot(parsed,*,kickoff_utc,captured_at_utc):
    kickoff=_utc(kickoff_utc,"kickoff_utc")
    capture=_utc(captured_at_utc,"captured_at_utc")
    if capture>=kickoff:
        raise ValueError("Past-kickoff ACTIVE-source capture cannot inform pregame")
    if _utc(parsed["published_at_utc"],"published_at_utc")>capture:
        raise ValueError("Official named-active article published after capture")
    rows=[]
    for team,players in parsed["active_by_team"].items():
        for row in players:
            rows.append({
                "team":team,"player":row["player"],"status":ACTIVE,
                "source_type":"NFL_OFFICIAL_EXPLICIT_ACTIVE_NEWS",
                "source_url":parsed["source_url"],
                "published_at_utc":parsed["published_at_utc"],
                "source_html_sha256":parsed["source_html_sha256"],
                "source_evidence_type":row["source_evidence_type"],
                "article_sentence":row["article_sentence"],
            })
    result={
        "season":parsed["season"],"week":parsed["week"],"game":parsed["game"],
        "kickoff_utc":kickoff.isoformat(),"captured_at_utc":capture.isoformat(),
        "records":rows,
    }
    validate_availability_snapshot(
        result,season=parsed["season"],week=parsed["week"],
        game=parsed["game"],as_of_utc=capture.isoformat())
    return result


def verify_named_active_raw(raw,parsed,snapshot):
    rerun=parse_named_active_news(raw,parsed["source_url"],
        season=parsed["season"],week=parsed["week"],game=parsed["game"])
    if rerun!=parsed:
        raise ValueError("Named ACTIVE source parser output differs from archived HTML")
    expected=make_named_active_snapshot(
        rerun,kickoff_utc=snapshot["kickoff_utc"],
        captured_at_utc=snapshot["captured_at_utc"])
    if expected!=snapshot:
        raise ValueError("Archived explicit ACTIVE list/source metadata was altered")
    return True
