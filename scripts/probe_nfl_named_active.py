"""Smoke only: parse an existing NFL.com injury article, NO pregame snapshot."""
from ai2_nfl_official_inactives_capture import fetch_official_html
from ai2_nfl_official_named_active import parse_named_active_news
url="https://www.nfl.com/news/week-10-injuries-sunday-pregame-2026-nfl-season"
raw=fetch_official_html(url)
result=parse_named_active_news(raw,url,season=2026,week=10,game="NE@DET")
print({"mode":"POSTGAME_PARSER_SMOKE_NO_STATUS_SNAPSHOT",
       "game":result["game"],"positive_explicit_active_claims":
           {k:[p["player"] for p in v] for k,v in result["active_by_team"].items()},
       "captured_before_kickoff":False,
       "source_html_sha256":result["source_html_sha256"],
       "production_approved":False})
