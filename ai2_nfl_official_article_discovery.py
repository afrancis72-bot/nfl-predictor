"""Bounded, fail-closed discovery of *official* NFL.com game-day news links.

This module searches ONLY publicly available official NFL.com index/news pages,
not a general web-search engine and not untrusted third-party 'actives' reports.
Discovery NEVER establishes any player's game-day status. Every URL found
must subsequently pass the existing official HTML/article/UTC/source checks.

Only one page per official hub is fetched; redirects and unsafe hosts fail.
News index visibility is not guaranteed. If articles aren't linked, return
UNAVAILABLE, never invent a status or URL.
"""
from __future__ import annotations

from collections import OrderedDict
import re
from urllib.parse import urljoin,urlparse

from bs4 import BeautifulSoup

from ai2_nfl_official_inactives_capture import fetch_official_html, _allowed_url

OFFICIAL_DISCOVERY_HUBS=(
    "https://www.nfl.com/news",
    "https://www.nfl.com/inactives/",
    "https://www.nfl.com/news/",
)
MAX_LINKS_PER_PAGE=6000
_SLUG=re.compile(r"^/news/([a-z0-9-]+)$")
_WEEK=re.compile(r"(?:^|-)week-(\d{1,2})(?:-|$)")


def _kind(path,*,week):
    slug=path.rsplit("/",1)[-1]
    if not _SLUG.fullmatch(path):
        return None
    if not (m:=_WEEK.search(slug)) or int(m.group(1))!=int(week):
        return None
    if "inactives" in slug and not any(s in slug for s in (
        "preview","prediction","how-to","explained")):
        return "OFFICIAL_INACTIVES"
    if (("injuries" in slug or "injury" in slug)
        and any(w in slug for w in ("monitor","pregame","sunday"))):
        return "EXPLICIT_INJURY_UPDATES"
    return None


def extract_candidate_links(raw,hub_url,*,week):
    """Return ordered absolute official NFL.com article links + discovery kinds.

    Only actual <a href> links are accepted; free text, JS script contents and
    article claims from other domains are NOT approved as source discovery.
    """
    hub=urlparse(hub_url)
    if hub.scheme!="https" or hub.hostname not in ("www.nfl.com","nfl.com"):
        raise ValueError("Article discovery requires exact official NFL hub")
    soup=BeautifulSoup(raw,"html.parser")
    links=soup.find_all("a",href=True)
    if len(links)>MAX_LINKS_PER_PAGE:
        raise ValueError("Unbounded official article discovery HTML")
    found=OrderedDict()
    for link in links:
        address=urljoin(hub_url,link["href"]).split("#",1)[0].split("?",1)[0]
        parsed=urlparse(address)
        if parsed.hostname not in ("www.nfl.com","nfl.com"):
            continue
        kind=_kind(parsed.path,week=week)
        if not kind:
            continue
        try:
            _allowed_url(address)
        except ValueError:
            continue
        found[address]=kind
    return [{"url":url,"kind":kind} for url,kind in found.items()]


def discover_official_links(*,week,hubs=OFFICIAL_DISCOVERY_HUBS,
                            fetcher=fetch_official_html):
    """Search official hub links with bounded scope. Empty is meaningful.

    Failed source hubs are reported, not silently replaced by search results.
    This function must not be used to infer ACTIVE from a missing report.
    """
    if not isinstance(week,int) or not 1<=week<=22:
        raise ValueError("NFL season week must be 1–22")
    report={"week":week,"discovery_only":True,"attempted_hubs":[],
            "errors":[],"articles":[]}
    seen=set()
    for url in hubs:
        if not isinstance(url,str):
            raise ValueError("Invalid hub URL")
        host=urlparse(url).hostname
        if host not in ("nfl.com","www.nfl.com"):
            raise ValueError("Untrusted NFL article discovery hub")
        report["attempted_hubs"].append(url)
        try:
            raw=fetcher(url)
            links=extract_candidate_links(raw,url,week=week)
        except (ValueError,OSError) as e:
            report["errors"].append({"hub":url,"error":str(e)[:180]})
            continue
        for item in links:
            if item["url"] not in seen:
                seen.add(item["url"])
                report["articles"].append({**item,"discovered_via":url})
    report["candidate_count"]=len(report["articles"])
    report["content_and_status_verified"]=False
    return report
