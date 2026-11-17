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


## Automated research monitoring now installed

A **read-only scheduled launcher** has been installed on the repository's
default `main` branch at
`.github/workflows/nfl-official-pregame-source-monitor.yml`. This is necessary
because GitHub Actions schedules execute only from the default branch.
It checks out `research/nfl-official-source-capture-20261117` and does not
check out or modify the unified app's active production branch.

The workflow runs approximately every 30 minutes during NFL regular-season
months (September–January), within the UTC windows 00:17–05:47 and
12:17–23:47. GitHub may delay or skip individual scheduled runs. This should
not be treated as a hard real-time injury alert.

The research collector:

1. Retrieves the current independently published nflverse game schedule.
2. Considers only games with kickoff in the next 160 minutes (future only).
3. Searches actual official NFL.com news and inactives hub links for week-
   specific inactive lists and Sunday pregame injury-monitor articles.
4. Fetches and reparses the official article. Unmatched team/game or wrong
   publication year/week/format does **not** become a player status.
5. Records explicit **OUT** from official paired-team inactives and explicitly
   named present-tense **ACTIVE** claims from NFL's injury articles.
6. Keeps omitted players **UNKNOWN**. “Expected to play” or absence from an
   inactive list never counts as confirmed active.
7. Archives raw official HTML, published/capture times, SHA-256, source-specific
   player statuses, and independently checked kickoff in a 90-day Actions
   artifact, **only if actual valid prekickoff source data was captured**.
   Errors or no articles appear in run logs without manufacturing a file.
8. Performs **zero** app production updates, optimizer actions, lineups,
   changes to approved lineups, or betting actions.

Research-only direct invocation:

~~~bash
PYTHONPATH=. python scripts/auto_capture_nfl_status.py \
  --out-base nfl_official_captures \
  --report nfl_official_capture_run.json
~~~

Source discovery is **best effort**, not guaranteed. A live Tuesday (Nov. 17)
test found no link to the prior Sunday's Week 11 inactives on the currently
visible official hubs; therefore it polls repeatedly before future games.
No third-party article title or URL is allowed to count as original evidence.
Historical NFL.com explicit-ACTIVE article parsing was demonstrated on the
already-completed 2026 Week 10 pregame injury monitor story, identifying
specific present-tense ACTIVE statements. That smoke test was NOT backdated.

Archived OUT and named ACTIVE reports for the same game can be merged using
`combined_verified_game_snapshot` from
`ai2_nfl_verified_pregame_sources.py`. Every original source is reloaded and
reparsed first; duplicate/contradictory player claims fail closed. The merged
snapshot can then be supplied to the availability-gated **research**
historical evaluator.

**Still not certified for production:** The official news roundup does not
supply a complete game-day active roster. Only named explicit-positive
confirmations qualify. Players unmentioned anywhere remain UNKNOWN; a
reconstructed historical salary pool is not a roster. The collector has
not yet been observed successfully capturing a *new upcoming game* before
kickoff under unattended scheduling. We must assess that first, then freeze
actual prospective player forecasts and compare fantasy MAE, WR1/WR2 targets
and P90 calibration against outcomes with a predeclared evaluation protocol.
