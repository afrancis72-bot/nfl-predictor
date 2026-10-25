"""V10.6.3 multi-source international course/weather ingestion adapters.

This module fetches and normalizes external inputs only. Predictive logic is unchanged.
Golf Courses API: international course search/detail/scorecards (free API key required).
Open-Meteo: weather by course coordinates.
"""
from __future__ import annotations

from datetime import date, timedelta
from difflib import SequenceMatcher
from typing import Any
import re
import requests
import pandas as pd

GCA = "https://www.golfcoursesapi.com/api/v1"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
UA = "PGA-Predictor-V10.6.3/1.0 (personal golf research app)"
NOMINATIM = "https://nominatim.openstreetmap.org/search"

class IngestionError(RuntimeError):
    pass

def _norm(s: Any) -> str:
    s = str(s or "").casefold()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())

def _headers(api_key: str | None = None) -> dict:
    h = {"User-Agent": UA, "Accept": "application/json"}
    if api_key:
        # GCA keys are bearer credentials. X-API-Key is included as a compatibility
        # header so a provider-side auth convention change produces a useful HTTP
        # response instead of a silent UI failure.
        h["Authorization"] = f"Bearer {api_key.strip()}"
        h["X-API-Key"] = api_key.strip()
    return h

def _get_json(url: str, *, params: dict | None = None, api_key: str | None = None, timeout: int = 20) -> Any:
    try:
        r = requests.get(url, params=params, timeout=timeout, headers=_headers(api_key))
    except requests.RequestException as exc:
        raise IngestionError(f"Network error contacting course/weather source: {exc}") from exc
    if not r.ok:
        body = (r.text or "").strip().replace("\n", " ")[:240]
        hint = ""
        if r.status_code in (401, 403):
            hint = " Check the Golf Courses API key in the sidebar."
        elif r.status_code == 429:
            hint = " The API rate limit was reached; try again later."
        raise IngestionError(f"Source returned HTTP {r.status_code}.{hint} {body}".strip())
    try:
        return r.json()
    except ValueError as exc:
        raise IngestionError("Source returned a non-JSON response.") from exc

def _records(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("data", "results", "courses", "items"):
            val = payload.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
            if isinstance(val, dict):
                return [val]
        return [payload] if payload else []
    return []

def search_courses(query: str, api_key: str, country_hint: str = "", limit: int = 15) -> list[dict]:
    if not query.strip():
        raise IngestionError("Enter a course name before searching.")
    if not api_key.strip():
        raise IngestionError("Golf Courses API key is required for international course lookup.")
    payload = _get_json(f"{GCA}/courses", params={"q": query.strip(), "per_page": min(limit, 25)}, api_key=api_key)
    rows = _records(payload)
    qn, cn = _norm(query), _norm(country_hint)
    out = []
    for row in rows:
        cid = row.get("id") or row.get("course_id")
        name = row.get("name") or row.get("course_name") or row.get("club")
        if cid is None or not name:
            continue
        city = row.get("city") or ""
        state = row.get("state") or row.get("province") or ""
        country = row.get("country") or ""
        club = row.get("club") or ""
        hay = _norm(" ".join(map(str, (name, club, city, state, country))))
        score = SequenceMatcher(None, qn, _norm(name)).ratio()
        if qn and qn in hay: score += 0.35
        if cn and cn in hay: score += 0.30
        out.append({
            "id": str(cid), "name": str(name), "club": str(club),
            "city": str(city), "state": str(state), "country": str(country),
            "latitude": row.get("latitude"), "longitude": row.get("longitude"),
            "match_score": round(score, 3), "raw": row,
        })
    out.sort(key=lambda x: x["match_score"], reverse=True)
    return out[:limit]


def geocode_course(query: str, country_hint: str = "", limit: int = 5) -> list[dict]:
    """One-off, user-triggered OSM Nominatim fallback. Cached by Streamlit caller.

    This intentionally makes ONE search request per user action and does not
    autocomplete, bulk-query, or scrape Nominatim details.
    """
    if not query.strip():
        raise IngestionError("Enter a course name before searching.")
    q = ", ".join(x for x in (query.strip(), country_hint.strip()) if x)
    payload = _get_json(NOMINATIM, params={
        "q": q, "format": "jsonv2", "addressdetails": 1,
        "limit": max(1, min(int(limit), 10)),
    })
    if not isinstance(payload, list):
        return []
    qn, cn = _norm(query), _norm(country_hint)
    out=[]
    for row in payload:
        if not isinstance(row, dict):
            continue
        display = str(row.get("display_name") or "")
        addr = row.get("address") if isinstance(row.get("address"), dict) else {}
        name = str(row.get("name") or addr.get("golf_course") or display.split(",")[0] or query)
        hay = _norm(display)
        score = SequenceMatcher(None, qn, _norm(name)).ratio()
        if qn and qn in hay: score += 0.35
        if cn and cn in hay: score += 0.30
        try:
            lat, lon = float(row.get("lat")), float(row.get("lon"))
        except (TypeError, ValueError):
            continue
        out.append({
            "id": f"osm:{row.get('osm_type','')}:{row.get('osm_id','')}",
            "name": name, "club": name,
            "city": str(addr.get("city") or addr.get("town") or addr.get("village") or addr.get("municipality") or ""),
            "state": str(addr.get("state") or addr.get("region") or ""),
            "country": str(addr.get("country") or country_hint or ""),
            "latitude": lat, "longitude": lon,
            "match_score": round(score, 3), "display_name": display,
            "source": "OpenStreetMap / Nominatim", "raw": row,
        })
    out.sort(key=lambda x: x["match_score"], reverse=True)
    return out

def location_detail(match: dict) -> dict:
    """Normalize a geocoder match to the detail shape expected by weather."""
    return {
        "name": match.get("name"),
        "latitude": match.get("latitude"),
        "longitude": match.get("longitude"),
        "city": match.get("city"), "state": match.get("state"),
        "country": match.get("country"), "source": match.get("source"),
    }

def fetch_course_detail(course_id: str, api_key: str) -> dict:
    payload = _get_json(f"{GCA}/courses/{course_id}", api_key=api_key)
    rows = _records(payload)
    return rows[0] if rows else {}

def _extract_holes(detail: dict) -> list[dict]:
    # Direct holes shape.
    if isinstance(detail.get("holes"), list):
        return [x for x in detail["holes"] if isinstance(x, dict)]
    # Scorecard/tees shapes. Prefer the longest tee because PGA events generally
    # play near championship yardages; par is course-level and stable across tees.
    tees = detail.get("tees") or detail.get("teeboxes") or detail.get("tee_boxes") or detail.get("scorecards")
    if isinstance(tees, dict): tees = list(tees.values())
    if isinstance(tees, list):
        candidates=[]
        for tee in tees:
            if not isinstance(tee, dict): continue
            holes = tee.get("holes") or tee.get("scorecard")
            if isinstance(holes, list):
                total=0
                for h in holes:
                    if isinstance(h, dict):
                        try: total += float(h.get("yardage") or h.get("yards") or h.get("length") or 0)
                        except (TypeError, ValueError): pass
                candidates.append((total, holes))
        if candidates:
            return max(candidates, key=lambda x:x[0])[1]
    return []

def fetch_course_holes(course_id: str, tournament: str, api_key: str) -> tuple[pd.DataFrame, dict]:
    detail = fetch_course_detail(course_id, api_key)
    rows = _extract_holes(detail)
    normalized=[]
    for i,row in enumerate(rows,start=1):
        normalized.append({
            "tournament": tournament,
            "hole": row.get("hole") or row.get("number") or row.get("hole_number") or i,
            "par": row.get("par"),
            "yardage": row.get("yardage") or row.get("yards") or row.get("length"),
            # Unknown course-DNA fields stay neutral; no fabricated course edge.
            "fairway_width": 30.0, "rough_severity": 0.5, "water": 0.5,
            "bunker_density": 0.5, "green_size": 6000.0, "wind_exposure": 0.5,
            "elevation_change": 25.0,
        })
    df=pd.DataFrame(normalized)
    if not df.empty:
        df["par"]=pd.to_numeric(df["par"],errors="coerce")
        df["yardage"]=pd.to_numeric(df["yardage"],errors="coerce")
    return df, detail

def _coords(detail: dict) -> tuple[float,float] | None:
    candidates=[detail]
    for key in ("location","coordinates"):
        if isinstance(detail.get(key),dict): candidates.append(detail[key])
    for obj in candidates:
        lat=obj.get("latitude",obj.get("lat")); lon=obj.get("longitude",obj.get("lon",obj.get("lng")))
        try:
            if lat is not None and lon is not None: return float(lat),float(lon)
        except (TypeError,ValueError): pass
    return None

def fetch_weather(detail: dict, tournament: str, start_date: date | None=None, days: int=7) -> pd.DataFrame:
    coords=_coords(detail)
    if not coords: raise IngestionError("Selected course has no latitude/longitude; upload weather.csv instead.")
    lat,lon=coords; start=start_date or date.today(); end=start+timedelta(days=max(1,min(days,16))-1)
    payload=_get_json(OPEN_METEO,params={
        "latitude":lat,"longitude":lon,
        "hourly":"temperature_2m,precipitation_probability,wind_speed_10m,wind_gusts_10m",
        "temperature_unit":"fahrenheit","wind_speed_unit":"mph","precipitation_unit":"inch",
        "timezone":"auto","start_date":start.isoformat(),"end_date":end.isoformat(),
    })
    hourly=payload.get("hourly",{}) if isinstance(payload,dict) else {}; times=hourly.get("time",[])
    if not times: raise IngestionError("Open-Meteo returned no hourly forecast for the selected course/date window.")
    return pd.DataFrame({"tournament":tournament,"time":times,
        "temperature_f":hourly.get("temperature_2m",[]),"rain_prob":hourly.get("precipitation_probability",[]),
        "wind_mph":hourly.get("wind_speed_10m",[]),"wind_gust_mph":hourly.get("wind_gusts_10m",[])})
