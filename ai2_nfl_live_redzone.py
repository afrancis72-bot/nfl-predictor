"""Bounded live draft-only nflverse PBP acquisition; no invented factors."""
import io
from functools import lru_cache
import pandas as pd
import requests
from ai2_nfl_redzone_usage import FIELDS,pregame_redzone_profiles
from ai2_nfl_pbp_drives import PBP_RELEASE

@lru_cache(maxsize=2)
def _read_season(season):
    url=PBP_RELEASE.format(season=int(season))
    with requests.get(url,timeout=(8,18),stream=True) as response:
        response.raise_for_status()
        buf=io.BytesIO()
        for chunk in response.iter_content(1024*1024):
            buf.write(chunk)
            if buf.tell()>100*1024*1024:
                raise ValueError("PBP source exceeds 100 MB limit")
        buf.seek(0)
        return pd.read_parquet(buf,columns=sorted(FIELDS))

def load_draft_redzone(history,season,week):
    data=_read_season(int(season))
    profile=pregame_redzone_profiles(data,history,season,week)
    if profile["last_observed_season"]!=int(season) or profile["last_observed_week"]<int(week)-2:
        raise ValueError("Red-zone source stale for target slate")
    if not profile["players"]:
        raise ValueError("No source-matched red-zone player factors")
    return profile
