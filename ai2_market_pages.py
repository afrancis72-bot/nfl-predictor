"""Strict ParlayAPI pagination with completeness-header verification.

This helper does not normalize athlete identities or access the simulator.
Only call after a simulation has been frozen.
"""
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

def fetch_complete_props(sport_key, markets, api_key, *, transport=None,
                         page_size=1000, max_pages=10):
    if not sport_key or not markets or not api_key:
        raise ValueError("Sport, markets and API key are required")
    if not 1 <= page_size <= 10000 or not 1 <= max_pages <= 10:
        raise ValueError("Invalid pagination limits")
    if transport is None:
        def transport(url, headers):
            with urlopen(Request(url, headers=headers), timeout=12) as response:
                body = response.read(5_000_001)
                if len(body) > 5_000_000:
                    raise ValueError("Response too large")
                return json.loads(body), dict(response.headers.items())
    offset = 0
    rows = []
    for _ in range(max_pages):
        url = ("https://parlay-api.com/v1/sports/" + sport_key + "/props?"
               + urlencode({"markets": ",".join(markets), "limit": page_size,
                            "offset": offset, "maxAgeSec": 900}))
        page, raw_headers = transport(url, {"X-API-Key": api_key,
                                            "Accept": "application/json"})
        headers = {str(k).lower(): str(v).lower() for k, v in raw_headers.items()}
        if not isinstance(page, list):
            raise ValueError("Unexpected provider response shape")
        if len(page) > page_size:
            raise ValueError("Provider returned more rows than requested")
        if "x-result-has-more" not in headers:
            raise ValueError("Missing provider completeness header")
        rows.extend(page)
        has_more = headers["x-result-has-more"] == "true"
        if headers["x-result-has-more"] not in ("true", "false"):
            raise ValueError("Invalid completeness header")
        if not has_more:
            return rows
        if not page:
            raise ValueError("Provider reports more results but returned an empty page")
        try:
            next_offset = int(headers["x-next-offset"])
        except (KeyError, ValueError) as exc:
            raise ValueError("Missing pagination offset") from exc
        if next_offset <= offset or next_offset > 10000:
            raise ValueError("Invalid or exhausted pagination offset")
        offset = next_offset
    raise ValueError("Incomplete prop board: page budget exceeded")
