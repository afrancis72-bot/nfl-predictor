# NFL DFS V3 — Stat-Driven Game Simulation

## What changed
Classic DFS now defaults to a football-stat simulation instead of drawing fantasy points directly.

Each scenario generates team dropbacks/rushes, explicit sacks, pass attempts, targets, receptions, carries, yards, touchdowns, interceptions, fumble recoveries, defensive touchdowns, safeties and points allowed. DraftKings points are calculated afterward from those simulated football outcomes.

The opposing DST is tied to the same game scenario. A QB can therefore throw for substantial yardage while the defense still scores through sacks/turnovers, and a high-yardage offense does not automatically force a poor DST score.

## DraftKings scoring implemented
QB: 0.04/pass yard, 4/pass TD, -1/INT, 3-point 300-yard bonus; rushing 0.1/yard, 6/TD, 3-point 100-yard bonus.
RB/WR/TE: full PPR, 0.1/rush or receiving yard, 6/TD, 3-point 100-yard rushing/receiving bonuses.
DST: 1/sack, 2/INT, 2/fumble recovery, 6/defensive TD, 2/safety, plus standard DK points-allowed tiers.

## Data
The engine downloads pregame nflverse weekly player statistics and uses only games before the target week. props_data.py now also retains passing TD, interception, rushing TD, receiving TD and sacks-suffered rates when available.

## Important current limitation
This is a game-level statistical simulator, not yet a literal play-by-play possession engine. Until a dedicated team defensive sack-rate feed is added, sack probability combines the QB's observed sacks-suffered rate with a conservative opponent-DST strength adjustment. That assumption is intentionally bounded and should be backtested before treating the model as calibrated.

The legacy fantasy-point simulator remains selectable on the Simulation page for A/B comparison. The Scenario Portfolio optimizer now uses the stat-driven engine by default.
