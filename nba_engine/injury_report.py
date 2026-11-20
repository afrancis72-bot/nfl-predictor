"""Source-dated league NBA injury reports for the unified NBA prop model.

Retrieve a *real official* NBA PDF, inspect its player rows, and remove only
explicitly OUT/Doubtful players. The absence of a name is UNKNOWN, not ACTIVE.
Never backfill a past target game with a future injury report.
"""
from __future__ import annotations
from datetime import date,datetime,timedelta
from io import BytesIO
from zoneinfo import ZoneInfo
import re
import unicodedata

import pandas as pd
import requests
from pypdf import PdfReader

PDF_ROOT="https://ak-static.cms.nba.com/referee/injury"
EASTERN=ZoneInfo("America/New_York")
TEAM_NAMES={
    "Atlanta Hawks":"ATL","Boston Celtics":"BOS","Brooklyn Nets":"BKN",
    "Charlotte Hornets":"CHA","Chicago Bulls":"CHI",
    "Cleveland Cavaliers":"CLE","Dallas Mavericks":"DAL",
    "Denver Nuggets":"DEN","Detroit Pistons":"DET",
    "Golden State Warriors":"GSW","Houston Rockets":"HOU",
    "Indiana Pacers":"IND","LA Clippers":"LAC",
    "Los Angeles Clippers":"LAC","Los Angeles Lakers":"LAL",
    "Memphis Grizzlies":"MEM","Miami Heat":"MIA",
    "Milwaukee Bucks":"MIL","Minnesota Timberwolves":"MIN",
    "New Orleans Pelicans":"NOP","New York Knicks":"NYK",
    "Oklahoma City Thunder":"OKC","Orlando Magic":"ORL",
    "Philadelphia 76ers":"PHI","Phoenix Suns":"PHX",
    "Portland Trail Blazers":"POR","Sacramento Kings":"SAC",
    "San Antonio Spurs":"SAS","Toronto Raptors":"TOR",
    "Utah Jazz":"UTA","Washington Wizards":"WAS",
}
STATUSES={"Out","Doubtful","Questionable","Probable","Available"}
EXCLUDE={"Out","Doubtful"}


def normalize_person(name):
    s=unicodedata.normalize("NFKD",str(name))
    s="".join(c for c in s if not unicodedata.combining(c))
    s=s.replace("’","'").casefold()
    return re.sub(r"[^a-z0-9]","",s)


def _name_from_pdf(label):
    # League injury reports display family name first.
    s=" ".join(str(label).split())
    if "," not in s:
        raise ValueError("Official NBA player-name field missing comma")
    family,first=s.split(",",1)
    return " ".join((first.strip(),family.strip()))


def parse_official_injury_pdf(blob, *, target_games, document_date):
    """Read genuine league table columns; never guess a player's team.

    pypdf's layout extraction returns visual lines bottom-first for this
    published NBA report layout. Reversing those lines makes teammate/team
    context continuous even across PDF page breaks.
    """
    if not isinstance(blob,(bytes,bytearray)) or not blob.startswith(b"%PDF"):
        raise ValueError("NBA injury source did not return a PDF")
    reader=PdfReader(BytesIO(blob))
    if not 1<=len(reader.pages)<=40:
        raise ValueError("Unexpected NBA official report page count")
    games={str(x).upper() for x in target_games}
    all_records=[]
    current_game=None
    current_team=None
    for page in reader.pages:
        content=page.extract_text(extraction_mode="layout")
        if not content:
            raise ValueError("Official NBA injury report text could not be read")
        for line in reversed(content.splitlines()):
            if "Report:" in line and "Injury" in line:
                continue
            pieces=[x.strip() for x in re.split(r" {3,}",line.strip()) if x.strip()]
            if not pieces:
                continue
            # PDF layout can separate the two words of a team name with
            # several spaces. Identify the TEAM on the full visual row,
            # not an arbitrary whitespace-fragmented cell.
            normalized_line=" ".join(line.split())
            for full_name,abbrev in sorted(
                    TEAM_NAMES.items(),key=lambda item:-len(item[0])):
                if re.search(
                        r"(?<![A-Za-z])"+re.escape(full_name)+r"(?![A-Za-z])",
                        normalized_line,re.I):
                    current_team=abbrev
                    break
            for seg in pieces:
                m=re.search(r"\b([A-Z]{2,3}@[A-Z]{2,3})\b",seg)
                if m:
                    current_game=m.group(1)
            for i,item in enumerate(pieces):
                if item not in STATUSES or i<1:
                    continue
                label=pieces[i-1]
                # Source PDF's variable kerning can put three or more spaces
                # BETWEEN last/first names, so column-based splitting may
                # yield ["Mobley,", "Evan", "Out"]. Rejoin only the precise
                # adjacent comma-terminated family-name token.
                if ("," not in label and i>=2 and
                        pieces[i-2].rstrip().endswith(",")):
                    label=pieces[i-2]+" "+label
                # Players such as Udeh Jr., Ernest may be laid out as
                # ["Udeh", "Jr.,Ernest", "Out"]. Preserve full surname.
                if (i>=2 and re.match(
                        r"^(?:Jr\.?|Sr\.?|II|III|IV|V),",label,re.I)
                        and "," not in pieces[i-2]):
                    label=pieces[i-2]+" "+label
                if "," not in label:
                    continue
                if current_game is None or current_team is None:
                    raise ValueError("Official injury row lacks verified team/game context")
                if current_game not in games:
                    continue
                player=_name_from_pdf(label)
                all_records.append({
                    "game":current_game,"team":current_team,
                    "player":player,"player_key":normalize_person(player),
                    "status":item,
                })
    records=pd.DataFrame(all_records)
    if records.empty:
        raise ValueError("No matching player injury rows found in official NBA PDF")
    if records.duplicated(["game","team","player_key"]).any():
        combined=records.groupby(["game","team","player_key"])["status"].nunique()
        if (combined>1).any():
            raise ValueError("Conflicting player injury status in official report")
        records=records.drop_duplicates(["game","team","player_key"])
    actual_games=set(records["game"].astype(str))
    if not actual_games.issubset(games):
        raise ValueError("Cross-game injury rows must not enter this slate")
    return records.reset_index(drop=True)


def fetch_current_official_injuries(when:date, *, target_games,
                                    now=None, timeout=8,
                                    get=None):
    """Poll exact dated 15-minute league reports, newest FIRST.

    No opaque API, no invented status, and no assumption that a player absent
    from a published list has been positively certified ACTIVE. Reports for
    future/past games are only used when published on the current date.
    """
    moment=now or datetime.now(EASTERN)
    if moment.tzinfo is None:
        raise ValueError("NBA status check requires timezone-aware clock")
    moment=moment.astimezone(EASTERN)
    if when<moment.date() or when>moment.date()+timedelta(days=1):
        raise ValueError(
            "Official NBA injury verification only supports today's and "
            "tomorrow's games; historical replay needs archived as-of data")
    if not target_games:
        raise ValueError("No NBA matchup to verify for official injuries")
    get=get or requests.get
    quarter=(moment.minute//15)*15
    current=moment.replace(minute=quarter,second=0,microsecond=0)
    errors=[]
    for back in range(8):
        instant=current-timedelta(minutes=15*back)
        suffix=instant.strftime("%Y-%m-%d_%I_%M%p")
        url=f"{PDF_ROOT}/Injury-Report_{suffix}.pdf"
        try:
            r=get(url,timeout=timeout,headers={
                "User-Agent":"TeamAI2Pro/1.0 (public NBA official injury source)"})
            if r.status_code!=200:
                errors.append(f"{suffix}: HTTP {r.status_code}")
                continue
            document=r.content
            statuses=parse_official_injury_pdf(
                document,target_games=target_games,
                document_date=instant.date())
            if not (statuses["game"].isin(target_games)).all():
                raise ValueError("Other NBA game injury records leaked into slate")
            return statuses,{
                "source_url":url,
                "as_of_eastern":instant.isoformat(),
                "status":"OFFICIAL_NBA_REPORT_PARSED",
                "games_reported":sorted(set(statuses["game"])),
                "rows":len(statuses),
                "excluded_statuses":sorted(EXCLUDE),
                "uncertain_statuses":["Questionable","Probable"],
                "healthy_unlisted_are_not_certified":True,
            }
        except (requests.RequestException,ValueError,OSError,KeyError) as exc:
            errors.append(f"{suffix}: {type(exc).__name__}: {str(exc)[:130]}")
    raise ValueError(
        "NBA official injury report unavailable or not reliably parsed. "
        "No injury-unverified 10,000-game projection was generated. "
        "Recent attempts: "+"; ".join(errors[:4]))


def injury_exclusions(statuses, team_map):
    """(excluded player IDs by name+team, flagged uncertain) as name keys."""
    if not isinstance(statuses,pd.DataFrame):
        raise ValueError("Official NBA status rows must be a DataFrame")
    required={"team","player_key","status","game"}
    if not required.issubset(statuses):
        raise ValueError("Official NBA status rows incomplete")
    teams=set(team_map)
    excluded={
        (r.team,r.player_key) for r in statuses.itertuples(index=False)
        if r.team in teams and r.status in EXCLUDE}
    unsure={
        (r.team,r.player_key):r.status for r in statuses.itertuples(index=False)
        if r.team in teams and r.status in {"Questionable","Probable"}}
    return excluded,unsure


def match_excluded_ids(logs,excluded,team_map,when):
    """Bind official family/given names to unique same-team historical IDs."""
    current=logs.loc[logs["date"]<pd.Timestamp(when)].sort_values("date")
    current=current.groupby("player_id",sort=False).tail(1)
    current=current.loc[current["team"].isin(team_map)]
    ids=set()
    unmatched=set(excluded)
    for row in current.itertuples(index=False):
        key=(str(row.team),normalize_person(row.player))
        if key in excluded:
            ids.add(str(row.player_id))
            unmatched.discard(key)
    # Some league OUT players have never appeared in the current season.
    # Unmatched exclusions can safely be logged; they must never create
    # a new player or be applied to a same-surname teammate.
    return sorted(ids),sorted(unmatched)
