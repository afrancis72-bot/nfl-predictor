# TEAM AI² Pro — unified development app

## Run the NFL development version

Install requirements and launch `streamlit run unified_app.py`.
In Streamlit Community Cloud, select repository `afrancis72-bot/nfl-predictor`,
branch `feature/unified-ai2-platform`, and main file `unified_app.py`.
This configuration is **not** deployed automatically by a GitHub commit.
The repository's original `main/app.py` remains a separate entry point.

## NASCAR one-button workflow — native unified sports tab

The NASCAR sport selector is now connected to a native one-page simulator and
six-driver DraftKings portfolio builder, using the user's existing NASCAR
Predictor V2.14 racing/track/DFS engine, not a placeholder or another app.

1. In the **same unified app** select **NASCAR**.
2. Choose the **race track** (and check scheduled laps).
3. Upload the **post-qualifying DraftKings salary CSV** with actual driver Name,
   ID, Salary, and a complete unique Starting grid. This is the only required file.
4. Choose race simulations (default 25,000), portfolio count (default 10),
   maximum driver exposure (default 35%), overlap, minimum salary and a
   reproducible seed. Optional Fibonacci seed presets and a random-seed mode
   are available. An optional driver veto removes a driver only from fantasy
   lineup selection, not the simulated NASCAR race.
5. Click **Run race simulation + build DraftKings lineups** once.
6. Inspect final lineups, simulation-based driver outcomes, scripts, and driver
   exposures. Download a readable lineup review, six-slot D/ID CSV, projections
   and exposure CSV.

Actual practice *speed/ranks* improve single-lap and 5/10/15/20-lap pace
signals. An optional upload supports those fields, but **it is not required**.
Neither lap speeds nor starting positions are fabricated. The core attempts
published NASCAR Reference recent-form and comparable-track research through
bounded parallel requests. Source unavailability is disclosed; the app can
produce a **lower-confidence market-based research output** from salary/FPPG,
track assumptions, and the verified grid but cannot claim verified speed.
Final position/official roster changes must be reviewed before entry.

The engine corrects a prior race-matrix/driver identity mismatch by reordering
simulated race matrix columns with sorted driver projections. It rejects
incomplete qualifying positions, duplicate/missing IDs, incorrect simulation
column alignment, illegal salary, overlap, excessive exposure, and incomplete
portfolios. The salary floor applies while optimizing rather than after.
The resulting six D/ID columns are **not guaranteed to match every DK contest
entry upload template**; if the user supplies the DKEntries file, match its
exact schema before submitting. No actual wagering, uploads to DraftKings, or
live-deployment claim is made.

The separate `afrancis72-bot/NASCAR-predictor` standalone repository remains
unchanged. All unified functionality lives under
`afrancis72-bot/nfl-predictor@feature/unified-ai2-platform`.

## PGA — native V11 golf simulation, three-balls, Showdown

Select **PGA** from the same unified sport selector. The PGA selector
now opens a native workspace with four tabs:

- **Tournament DFS:** enter the real event, venue, week, course par and
  cut/no-cut format; upload the current DraftKings salary file. Click
  **Run PGA simulation + build DFS lineups**. The existing V11 engine
  simulates hole-by-hole scores for 72 holes, produces DK Classic scores,
  and constructs 6-golfer, $50K-cap portfolios. Defaults: 3,000
  tournaments and 10 lineups. Settings include exposure, minimum unique
  golfers, minimum salary, balanced vs GPP vs cut-equity objectives,
  fixed/Fibonacci/random seed, golfer locks/vetoes.
- **Round Group Matchups (2-Ball / 3-Ball) & 6-Leg Research:** retrieves
  official round groupings and verified prior round scores from the PGA TOUR
  adapter, or user-provided CSV fallbacks. Recognizes weekend two-person
  tee groups alongside three-person groups and can construct a mixed six-leg
  matchup shortlist. For **2-ball 3-way head-to-head markets**, score ties
  lose a golfer pick, and a separate Tie outcome is reported; for **2-ball
  2-way markets**, ties push. Users select the settlement convention shown
  on their sportsbook. Three-ball ties use the original push convention.
  Six-leg candidate generation prioritizes named golfer picks, not
  automatically the Tie outcome, and refuses to force a six-leg ticket
  when fewer than six fully matched groups exist. Round 2+ requires
  verified completed-round scoring. Modeled probabilities are **not**
  actual sportsbook prices, calibrated win rates, or validated profit edges.
- **Round Showdown:** imports round-specific DK salaries and uses
  completed-round evidence to construct 6-golfer showdown lineups.
- **Course DNA / Data Quality:** reports actual input coverage, measured
  SG vs salary-market-only baseline, hole template provenance, weather,
  and player-specific skill assumptions.

**Optional RickRunGood Course Fit / Player Fit CSV:** Under PGA → Tournament DFS
→ Optional golfer research, upload the **separate Course Fit** export (not the
Stats Bank SG export). Accepted fields include `Player`, `FIT`, and optionally
the observed course-fit categories `OTT`, `APP`, `ARG`, `PUTT`, and
`SCORING`. Player names are matched to the current DraftKings slate; at
least 70% of the field must have usable fit data. No file is required.

The app uses its **own** category weights: APP 35%, OTT 25%, SCORING 18%,
PUTT 14%, ARG 8%. Components are standardized within the matched field,
and the resulting per-round adjustment is shrunk and capped at **0.22
strokes**. If only the vendor's overall `FIT` score exists, the
composite-only adjustment is half-strength. RRG rankings and salaries
are kept out of the course-fit adjustment. The Course DNA / Data Quality
tab shows matched-field coverage, fitted component values, signal,
the exact per-golfer adjustment and before/after projections, plus a
downloadable audit. This is **not** a validated golf prediction edge;
the default weights and magnitude are provisional, and the original
Stats Bank SG inputs continue to anchor skill when available. Without
Stats Bank or OTIS, DK salary is still the golfer-quality baseline even
if an RRG FIT export provides a course-specific tilt.

**Measured golfer data is optional, but it matters.** A single DK salary
upload enables a **lower-confidence market-prior experiment** that derives
expected skill solely from the DK salary distribution; it is not
independently predictive and does not invent historical strokes gained.
Optional observed **RickRunGood/independent SG Stats Bank exports** take
priority in the V10 golfer model. Optional OTIS Advanced Course-Fit
exports inform skill, recent form and course-fit only when independently
observed SG has not been provided. OTIS ranking/model are audit-only
and never direct inputs. Each supplied research file requires at least
70% valid matching to the current DK field. The app never substitutes
the standalone PGA repo's stale results/field data as current.

Actual 18-hole course pars can be imported; otherwise generic pars
matching course par are clearly labeled provisional. A user-selected,
verified venue from public OpenStreetMap geocoding can optionally feed
Open-Meteo hourly weather into the V10 golfer model when measured input
data are present. No individualized tee-wave exposure is fabricated.
This is an experimental golf-first scoring model, not a calibrated
hole-level/strokes-gained simulation or demonstrated profitable
DraftKings strategy. Full covariance-aware portfolio search, verified
ownership and real DFS contest ROI evaluation remain research tasks.

Files include `ai2_pga_workspace.py`, `ai2_pga_oneclick.py`, the
ported V11 golf/round modules, and
`.github/workflows/unified-pga-acceptance.yml`.
The independent `afrancis72-bot/pga-predictor` repository remains unchanged.
As with NASCAR, the six-G/real-ID roster CSV must be compared to the
actual DKEntries contest template before uploading. No direct
DraftKings betting, entries, or live-deployment claim is made.

## NFL Classic one-button workflow

1. Open NFL from the home screen and upload the DraftKings salaries CSV.
2. Optionally set the simulation seed (default 42).
3. Click **BUILD MY 20 LINEUPS**.

The experimental model fetches nflverse observed player history and official NFL
game-status designations, builds a game-level 10,000-scenario simulation, and
generates 20 checked NFL Classic roster combinations. No projection CSV needed.

The app displays **human-readable names, team, position and scores** and has
separate downloadable named Excel, named CSV, and DraftKings roster-slot ID CSV.
The roster-ID CSV is **not a verified contest-entry upload template**: match
it against your DraftKings contest entry requirements before submitting.

## News and injury handling

The same build retrieves the official NFL injury table, including practice
participation, injury note, official game designation, source URL and UTC
retrieval time where available. Current-week questionable and otherwise
unverified players are conservatively excluded. Confirmed unavailable
players have their observed workloads redistributed in the simulation
once. Practice participation **is not** a final game designation,
and active status alone does not confirm workload.

There is **no independently verified, comprehensive breaking-player-news
or depth-chart feed yet**. The app explicitly warns about this gap.
It is necessary to recheck late-breaking news and final Sunday inactives.

## Opposing defensive injuries and matchup research

The NFL build now joins official team-scoped defensive injury reports (CB/S,
front seven, pass rush) to the current DK matchup, alongside observed prior-game
opponent allowance by offensive position. It shows evidence, data timestamps,
sample sizes and a neutral game-script hypothesis when no trusted defender
coverage/snap role information is available. Unknown or ambiguous teams are not
assigned to a defense.

The **Opposing defensive injuries and matchup intelligence** panel allows the
user to inspect these findings and download a team-by-position CSV. It does not
invent coverage assignments, a WR's cornerback matchup, or individual target
increases based on a missing defender.

A chronological 2026 Weeks 4–7 exploratory test of a bounded opponent
position-allowance adjustment used 1,267 historical player-game observations:
unadjusted partial-DK MAE **4.6037** versus adjusted **4.6072** (worse by 0.08%).
**No automatic fantasy-point opponent multipliers are enabled.** This early
holdout only evaluates one simplified adjustment, not the full proposed
injury/game-script model and not profitable contest ROI.

A future verified scoring model should combine defensive snap roles, position-
specific pressure and coverage context, offensive line injuries, QB protection,
receiver route alignment, game pace and shared team-level script scenarios.
Historical data must be timestamped before kickoff to avoid look-ahead bias;
any new adjustments must beat a no-injury opponent baseline in separate
held-out weeks before affecting the actual optimizer.

## Human review and approvals

The five $3 lineups and fifteen $0.50 lineups can be reviewed; any proposed
contest-tier swap must comply with QB/team diversification and
current-slate premium-only exclusions. Swaps change **contest assignments only**
and cannot alter rosters. A one-click "Approve and save" action stores a
separate JSON lineup snapshot, without overwriting it when new simulations
are run. JSON snapshots are downloadable for backup. Previously approved
DK entries can also be imported and saved separately. Historical
vetoes are **not** automatically carried forward to new slates.

## Data storage

SQLite upload storage lives under `AI2_DATA_DIR` or
`~/TEAM_AI2_Pro_Data/slates.sqlite3`. Replacement archives prior
versions. Streamlit Community Cloud storage may be ephemeral; download
portable backups, or use a durable mounted volume, before relying
on saved files through redeployments. Nothing uploaded in ChatGPT is
automatically copied into Streamlit.

## Release limits

The NFL engine passed controlled and real 2026 Week 8 data tests and
performs legality and exposure checks. It is **experimental**, not
backtested for profitability or independently calibrated for current
depth charts, player news, ownership, defense, or fantasy-scoring distributions.
Browser verification of the deployed Streamlit URL remains a separate acceptance
step. NBA now routes directly into the one-click NBA Predictor on the same feature branch (not the separate team-ai2-pro repository). It loads the current ESPN schedule and community-maintained NBA game logs, projects minutes/player stats, simulates correlated stat lines and compares actual sportsbook prop lines. No DK salary upload is required. Official injury status and sportsbook prices are not connected; the NBA engine remains experimental and uncalibrated. PGA/NASCAR retain their previous preview behavior.
