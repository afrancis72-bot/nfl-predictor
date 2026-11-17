# NFL pregame availability evidence gate — research phase

**Status:** research-only; not connected to one-click Team AI² Pro; no production
release or verified player-status predictions. This does not alter the original
four-sport app or approved lineups.

## The problem this fixes

DraftKings salaries and last week's statistical participation are not the
football game-day roster. In the previous retrospective, 279 of 868 additional
historical player-game rows had no matching week-of-game boxscore row.
Automatically giving every one of those historical players a share of targets
made WR1/WR2 target prediction and P90 coverage worse.

A pregame **snapshot** is the only route for safely modeling more players:

- **Explicit game-day active claim:** potentially simulate as an internal
  football-only player, *never* make them DK-selectable. The source content
  must be independently checked before a real deployment.
- **Confirmed OUT report:** do not model this player as active. The initial
  safe policy keeps historically attributable omitted-player opportunity
  in OTHER; it does **not** automatically gift all of it to WR1/WR2.
- **Questionable / doubtful / limited / unlisted / missing source:** uncertain,
  not automatically active and not automatically OUT. Keep protected OTHER
  opportunity until a dated, stronger game-day source arrives.
- **Selected DraftKings player explicitly OUT:** fail closed rather than
  building a lineup containing him.

## Input contract (not proof that NFL.com contents have been verified)

The JSON below is *illustrative only*: fake teams, fake players, fake
publication timestamps, and a placeholder source URL. It must never be used to
assert real NFL status.

```json
{
  "season": 2026,
  "week": 7,
  "game": "AAA@BBB",
  "kickoff_utc": "2026-10-18T17:00:00Z",
  "captured_at_utc": "2026-10-18T16:15:00Z",
  "records": [
    {
      "team": "AAA",
      "player": "Example WR",
      "status": "VERIFIED_ACTIVE",
      "source_type": "NFL_GAME_DAY_ACTIVE_ROSTER",
      "source_url": "https://www.nfl.com/news/synthetic-example",
      "published_at_utc": "2026-10-18T15:42:00Z"
    },
    {
      "team": "AAA",
      "player": "Another WR",
      "status": "VERIFIED_OUT",
      "source_type": "NFL_OFFICIAL_INACTIVES",
      "source_url": "https://www.nfl.com/news/synthetic-example",
      "published_at_utc": "2026-10-18T15:42:00Z"
    }
  ]
}
```

Explicit source types currently accepted:
`NFL_GAME_DAY_ACTIVE_ROSTER` only for positive ACTIVE evidence;
`NFL_OFFICIAL_INACTIVES` only for OUT;
`NFL_OFFICIAL_GAME_STATUS_REPORT` only for OUT or UNCERTAIN.
An inactive-list article which **does not name a player** is NOT positive
ACTIVE proof. The system does not infer their status by omission.

Every source URL must use NFL.com HTTPS, must be for the correct game/team,
and the original published and captured timestamps must be UTC and pre-kickoff.
Source claims must be nonduplicated. The validator rejects missing/late,
contradictory or inconsistent evidence.

**Critical limitation:** the current validator verifies the *format and
consistency* of supplied URLs, timestamps and player records. It does **not**
fetch the page and verify its contents or authentic publication time; a user
or algorithm can still submit fabricated metadata. The field
`source_content_independently_fetched=false` is explicit. The live interface
must not treat these records as verified actual game-day actives until an
independent official-source acquisition and saved-content check is added.

## Research call

```python
from ai2_nfl_full_player_universe import build_football_only_players
from dfs_stats_engine import simulate_stat_driven_dfs

extra = build_football_only_players(
    dk_pool, pregame_rates, season=2026, week=11,
    require_verified_active=True,
    availability_snapshot=source_snapshot,
    as_of_utc="2026-11-22T17:00:00Z", # example; must be a verified pregame time
)
players, draws, stat_draws = simulate_stat_driven_dfs(
    dk_pool, pregame_rates,
    football_only_players=extra,
    football_universe_target_season=2026,
    football_universe_target_week=11,
    football_availability_snapshot=source_snapshot,
    football_availability_as_of_utc="2026-11-22T17:00:00Z",
)
```

A missing snapshot with the evidence gate enabled produces **zero** additional
football-only players. The original salary-pool scoring path remains
unchanged; this is intentional. In the research backtest function, pass
`full_player_universe=True`, `availability_gate=True` and a dictionary
`availability_snapshots[(season, week, "AWAY@HOME")]=snapshot`. Never
use postgame weekly rosters, target-week box scores or retroactively fabricated
capture dates as proof of pregame status.

## Research acceptance gates

1. Preserve original eligible DraftKings roster, IDs, ordering, salary and
   projection schema; no extra football-only row can enter the optimizer.
2. Do not infer any player's active status merely from their historic snap
   count or absence from an inactive list.
3. Explicit UTC kickoff, article publication, capture and modeling as-of
   timestamps all precede the actual start.
4. Conserve team pass attempts, targets, completions, carries and touchdowns.
5. With no independently confirmed source, output should remain the original
   model bit for bit, with documented unresolved availability.
6. Validate a true dated source capture system and run future untouched weeks;
   historical studies already inspected are **not** independent validation.
7. Require a genuinely improved player fantasy MAE, WR1/WR2 targets and
   85–95% P90 coverage at sufficient sample size before any production release.

A full 10,000-scenario **synthetic** regression proves only data-flow,
accounting and interface safety. It cannot prove football prediction accuracy
or that any named 2026 player actually played.
