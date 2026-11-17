# Team AI² Pro — NFL availability-aware Classic mode (same app)

## Where it lives

The current four-sport unified app is **`unified_app.py`** on
`feature/unified-ai2-platform`, not the legacy main-branch `app.py`. The new
feature is **inside the existing NFL tab**, not another Streamlit deployment.
It uses the same DraftKings upload, simulation seeds, player projections,
candidate builder, portfolio optimizer, approval workflow, and exports.

A deployment pointing at the unified app's branch does not automatically pick
up an unmerged research branch. The feature becomes available only after the
tested integration is merged into the active unified branch and that deployment
has synchronized/restarted. The git SHA shown at the top of the app remains
the authoritative running-code identifier.

## How to use

1. Open **NFL** from the Team AI² Pro home page.
2. Upload an actual current-game **DraftKings Classic salary file**, choose
   your lineups and seed as usual.
3. Choose **Existing game engine (default)** to retain the current program.
   This deliberately leaves the expanded player universe OFF.
4. To experiment, select **Availability-aware game engine
   (experimental)**, then press the same **BUILD MY LINEUPS** button.
5. The app checks independently published upcoming NFL schedules and NFL.com
   official-source article links, then reads and revalidates locally saved
   pre-kickoff source captures. It keeps confirmed NFL.com inactive names
   separate from explicit named-active article claims; **questionable,
   expected-to-play, and unlisted are not active proof**.
6. In the finished review, inspect **NFL engine and intelligence used for THIS
   build** for known active and OUT claims, per-game article coverage, actual
   non-DK-selectable football-only players added to simulation, source URLs,
   and rejected/unavailable source notes.
7. Download the **exact NFL engine + intelligence manifest (JSON)** with every
   build. It records the deployed git commit, engine version, slate SHA-256,
   seeds, engine mode, older official injury-feed timestamp, new evidence
   URLs/coverage, and that predictive improvement is **not** established.
   The manifest is separately preserved in the app's existing file storage.

For a 40-lineup build, the same official-source check is shared among all
three simulation seeds. Approved portfolios are not overwritten.

## Honesty rules / availability limits

- Unselected offensive teammates become simulated internal football-only
  players ONLY with documented prior-game opportunity *and* an explicit
  independently revalidated NFL.com source claim that they are ACTIVE.
  They are never DraftKings roster-selectable.
- Explicitly OUT players are blocked from the current eligible player pool.
  Historical missing teammate opportunity is not magically gifted to WR1.
- No verified official pregame source = **zero added player-universe
  intelligence**. The UI warns when no status evidence was obtained; the
  optional engine falls back to the existing scoring setup. The old official
  NFL injury report and prior-game opportunity inputs still run.
- The GitHub scheduled **research collector** stores workflow artifacts
  separately. Streamlit does NOT automatically see those artifact ZIPs.
  This app performs its own publisher check on Build and reuses only archives
  actually saved within its own durable `AI2_DATA_DIR`. Do not claim the
  GitHub research capture is already available inside the app.
- GitHub/website article discovery is best-effort: when the current NFL.com
  news hub does not link the required pregame article, the UI reports that
  absence. A weekend prospective real-world end-to-end capture is needed.
- This is **research opt-in**, NOT a validated accuracy improvement, published
  positive 48-man official active-list integration, or profitable DFS proof.
  We previously rejected naive historical-full-universe simulations because
  WR1/WR2 errors and P90 coverage worsened. The mode must be forward-audited
  on untouched slates before it can replace the default.
- No automatic betting, DraftKings submissions, changes to approved lineups,
  or side effects in PGA/NBA/NASCAR.

## Showdown follow-up — same NFL section

Single-game Showdown is **not** activated by this Classic integration.
Once the Classic status feed and forecast validation have proven reliable,
the Showdown option will use the same availability/source collector and
stat-by-stat simulation, but with a dedicated single-game game-world,
Captain 1.5x salary/points math, FLEX role validation, six distinct players,
correlations/ownership tailored for Showdown, and the user's existing
**QB vs opposing DST allowed in Showdown** exception. It will remain inside
the single Team AI² Pro NFL section, never a standalone sport app.
