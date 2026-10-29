"""Real Week 8 DraftKings acceptance: live nflverse + NFL official designations.

This workflow uses the actual uploaded DK salary slate. Unlike the offline
synthetic CI smoke tests, this must produce 20 validated *real-player* lineups.
It checks software function, not predictive accuracy, or contest profitability.
"""
from pathlib import Path
from collections import Counter
import csv
import io
import json
import hashlib
import traceback

from ai2_nfl_oneclick import build_oneclick_portfolio, _slate_week
from ai2_nfl_lineup import check_classic_lineup
from ai2_nfl_review import (review_tables, named_csv, named_excel,
                            snapshot_approved, load_approved, premium_risk_checks)

SLATE = Path("tests/fixtures/2026_week8_user_dk_slate.csv")
OUTPUT = Path("artifacts/nfl_live_week8")

def main():
    contents = SLATE.read_bytes()
    saved = (SLATE.name, contents, "2026-10-28T22:00:00+00:00")
    rows = list(csv.DictReader(io.StringIO(contents.decode("utf-8-sig"))))
    print("Uploaded slate rows:",len(rows),
          "games:",len(set(r["Game Info"] for r in rows)),
          "year/week:",_slate_week(saved),
          "sha256:",hashlib.sha256(contents).hexdigest())
    try:
        result = build_oneclick_portfolio(saved, seed=42)
    except Exception:
        print("LIVE ACCEPTANCE FAILED - complete error follows")
        traceback.print_exc()
        raise
    assert len(result["ranked"]) == 20, "Must contain 20 complete lineups"
    assert len(result["summary"]) == 20
    official_ids = {str(row["ID"]) for row in rows}
    appearances = Counter()
    for lineup in result["ranked"]:
        players = lineup["candidate"]["players"]
        assert len(players) == 9
        assert check_classic_lineup(players, salary_floor=49600) == [], "Illegal roster"
        for p in players:
            assert str(p["ID"]) in official_ids, "Non-DraftKings player was selected"
            appearances[str(p["ID"])] += 1
    assert max(appearances.values()) <= 7, "Overexposure"
    roster_csv = list(csv.reader(io.StringIO(result["csv"])))
    assert len(roster_csv) == 21, "Invalid DraftKings roster CSV"
    assert sum(item["contest_tier"] == "$3" for item in result["ranked"]) == 5
    premium = [entry for entry in result["ranked"] if entry["contest_tier"] == "$3"]
    premium_qbs = Counter()
    premium_pass_teams = Counter()
    for entry in premium:
        roster = entry["candidate"]["players"]
        premium_qbs.update(str(p["ID"]) for p in roster if p["Position"] == "QB")
        premium_pass_teams.update({
            str(p["TeamAbbrev"]).upper() for p in roster
            if p["Position"] in {"QB", "WR", "TE"}})
    assert max(premium_qbs.values()) <= 2, "Premium QB concentration exceeded 2/5"
    assert max(premium_pass_teams.values()) <= 3, "Premium passing-team concentration exceeded 3/5"
    gray_premium = sum(
        any(str(p["Name"]).strip().casefold() == "noah gray"
            for p in entry["candidate"]["players"])
        for entry in premium)
    gray_regular = sum(
        any(str(p["Name"]).strip().casefold() == "noah gray"
            for p in entry["candidate"]["players"])
        for entry in result["ranked"] if entry["contest_tier"] != "$3")
    assert gray_premium == 0, "Noah Gray must not enter Week 8 premium contests"
    assert result["audit"].get("premium_only_exclusions") == ["noah gray"]
    print("Noah Gray premium exposures:", gray_premium,
          "regular exposures:", gray_regular)
    print("Premium quarterback exposures:", dict(premium_qbs))
    print("Premium passing-team exposures:", dict(premium_pass_teams))

    report = result["pregame_report"]
    assert report["official_injury_url"].startswith("https://www.nfl.com/"), "Injury source missing"
    assert report["checked_utc"], "Injury retrieval timestamp missing"
    assert report["injuries"], "Injury dashboard has no player rows"
    assert "No verified late-breaking news feed" in report["news_feed"], "News coverage overstated"
    assert not premium_risk_checks(result["ranked"],
                                  result["audit"].get("premium_only_exclusions", ()))
    overview, detail, exposure = review_tables(result["ranked"])
    assert len(overview) == 20 and len(detail) == 180 and not exposure.empty
    assert "Trevor Lawrence" in named_csv(result["ranked"]).decode("utf-8-sig")
    approval = snapshot_approved(result["ranked"], hashlib.sha256(contents).hexdigest(), 42)
    assert len(load_approved(approval)["lineups"]) == 20
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "real_week8_named_review.csv").write_bytes(named_csv(result["ranked"]))
    (OUTPUT / "real_week8_named_review.xlsx").write_bytes(named_excel(result["ranked"]))
    (OUTPUT / "sample_approved_20_snapshot_REVIEW_ONLY.json").write_bytes(approval)
    (OUTPUT / "injury_news_sources.json").write_text(json.dumps(report, indent=2))

    (OUTPUT / "real_week8_lineups_REVIEW_ONLY.csv").write_text(result["csv"])
    (OUTPUT / "real_week8_summary.json").write_text(json.dumps(result["summary"],indent=2))
    (OUTPUT / "real_week8_audit.json").write_text(json.dumps(result["audit"],indent=2))
    print("SUCCESS: 20/20 lineups, player IDs matched, 5 premium, salary and exposure checks passed")
    print("Audit:",result["audit"])
    print("Average simulated points:",
          round(sum(item["Mean"] for item in result["summary"]) / 20,2))

if __name__ == "__main__":
    main()
