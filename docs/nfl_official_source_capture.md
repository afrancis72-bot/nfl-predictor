# Official NFL inactive article capture — research-only

**Branch:** research/nfl-official-source-capture-20261117

This collector reads the **actual NFL.com inactive article** over HTTPS,
finds the exact two teams' inactive lists, verifies the article's published
timestamp and NFL week, cross-checks the **actual kickoff** using independently
published nflverse schedules, and retains the raw original HTML, SHA-256 digest,
derived OUT-only status rows, timestamps and matching manifest.

No positive game-day ACTIVE evidence is generated from an inactive list.
Unlisted players remain UNKNOWN. No historical box score is used to infer
availability. The research NFL player universe remains disabled in the live
Team AI² Pro Preview one-click app.

## What has been verified

The safe article parser was tested against an actual (already completed)
NFL.com Week 10 2026 Chiefs–Falcons article. It extracted seven inactive
bullets per team; it recorded `PARSER_SMOKE_ONLY_NOT_PREGAME` and wrote **no**
pregame availability snapshot. Unit tests also cover malformed article HTML,
nonadjacent teams, wrong season/week, fraudulent URLs, missing publication
metadata, post-kickoff collection, altered archive bytes/statuses, and
kickoff-time falsification.

The real source is:
https://www.nfl.com/news/nfl-week-10-inactives-players-ruled-out-sunday-2026-season

That historic article is **not** retrospective pregame evidence. NFL.com's
current site layout uses a `<main>` element rather than an `<article>` element
around its inactive lists. If that structure changes again, the parser should
fail rather than infer statuses from arbitrary page prose.

## Manual research capture for a *future* game

From GitHub Actions on this research branch, select workflow
**NFL official inactive article source capture research**,
then choose **Run workflow**, mode `capture`, exact posted NFL.com /news/
inactive article URL, season, week, game (`AWAY@HOME`), and scheduled kickoff in
UTC. Do not run a capture before the official lists have been published.

Equivalent CLI:

~~~bash
PYTHONPATH=. python scripts/capture_nfl_official_inactives.py \
  --url "https://www.nfl.com/news/<EXACT_FUTURE_OFFICIAL_INACTIVES_ARTICLE>" \
  --season 2026 --week 12 --game "TEAM1@TEAM2" \
  --kickoff-utc "YYYY-MM-DDTHH:MM:SSZ" \
  --out-base nfl_official_captures/
~~~

The code independently reads
https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv
and verifies that the exact matchup/season/week kickoff matches the caller's
UTC timestamp (nflverse `gameday` and `gametime` use US Eastern time).
If the match is missing, delayed, flexed without an update, or inconsistent,
the capture fails closed. NFL.com itself supplies the official inactive
content; nflverse only supplies an independent schedule check.

Artifacts are uploaded to GitHub Actions with 90-day retention and contain:

- `official_source.html` — original page bytes;
- `inactive_player_snapshot.json` — pregame OUT-only player rows;
- `verified_parser.json` — exact teams, names, publication and SHA-256 digest;
- `manifest.json` — captured URL/time/kickoff, hash, verification claims;
- `schedule_crosscheck.json` — published schedule URL and matched kickoff.

The archive directory name incorporates matchup, capture time, and source
digest; preexisting archive directories cannot be overwritten. It can be
reloaded/revalidated with:

~~~python
from ai2_nfl_official_inactives_capture import load_verified_capture
snapshot, manifest = load_verified_capture(
    "nfl_official_captures/<CAPTURE_DIR>",
    as_of_utc="YYYY-MM-DDTHH:MM:SSZ")
~~~

The verified archive can be passed into the existing **research-only** evaluator
without retyping source claims:

~~~python
from ai2_nfl_official_inactives_capture import load_verified_capture_index
snapshots = load_verified_capture_index(
    ["nfl_official_captures/<CAPTURE_DIR>"],
    as_of_utc="YYYY-MM-DDTHH:MM:SSZ")
# evaluate(..., full_player_universe=True, availability_gate=True,
#          availability_snapshots=snapshots)
~~~

This strict loader refuses duplicates, altered source content, or archives
captured after the intended model-as-of time.

**Limits:** GitHub Actions artifact retention isn't permanent storage or a
cryptographic signature of NFL authorship. The archive proves the bytes that
the capture job saw and flags subsequent tampering. It does not guarantee
NFL.com won't revise an article after collection. The caller provides the
article URL; this research workflow has **no unattended article discovery
or scheduled live capture yet**. A real verified game-day ACTIVE roster feed
is still required to safely add non-DK players as positively active. Until
that exists, all unlisted teammates remain unverified and preserved in OTHER.

**Release requirements:** official-source autodiscovery, true pre-kickoff
snapshots, positive active-roster evidence (not absent-from-inactives inference),
correct injury/out redistribution, and independent untouched-slate accuracy
validation with player MAE, WR1/WR2 targets, and P90 coverage. No production
change was made.
