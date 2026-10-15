# NFL Showdown game-script development (experimental)

This branch is isolated from production `main`.

## Baseline verified
- Current `app.py` calls `simulate_showdown_players(players, teams, n_sims, seed)`.
- This function samples correlated fantasy outcomes using pace, team offense, passing/rushing factors and player noise. It does **not** simulate snaps, possessions or box scores.
- The root of `main` does not currently contain `showdown_game_script_engine.py`.

## Acceptance criteria before any production change
1. Preserve existing Showdown and Classic modes, including confirmed-starter QB gate.
2. Add a separately selectable experimental game simulator; keep baseline as default.
3. Generate reproducible 10,000-game outputs: final scores, team plays, player passing/rushing/receiving stats, and DraftKings scores from the same game.
4. Verify accounting invariants (receptions <= targets, TD <= opportunities, team totals, roster constraints).
5. Benchmark 10k run performance and compare output against baseline with identical salary pool and seed.
6. Audit all three ranking options using a fixed candidate bank; compare selected lineup IDs and selection scores.
7. Test on Seattle–Denver DK salaries and confirm no inactive/backup QB is automatically overvalued.
8. Merge only after passing regression tests and explicit review.

No production code has been changed by this planning commit.
