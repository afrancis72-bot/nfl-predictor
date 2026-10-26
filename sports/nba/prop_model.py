"""NBA prop research baseline for Team AI² Pro.

No sportsbook lines enter these pregame projections. The baseline is a simple
minutes-adjusted mean, NOT an injury-aware possession simulation. Every
assumption and missing feature is explicit.
"""
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PlayerRates:
    player: str
    team: str
    minutes_per_game: float
    points_per_36: float
    rebounds_per_36: float
    assists_per_36: float
    threes_per_36: float
    sample_games: int


def project_props(rates: PlayerRates, expected_minutes: float) -> dict:
    if not (0 <= expected_minutes <= 48):
        raise ValueError("Expected minutes must be between 0 and 48.")
    if rates.minutes_per_game <= 0 or rates.sample_games < 1:
        raise ValueError("Historical player rates require a nonempty sample.")
    values = [
        rates.points_per_36, rates.rebounds_per_36,
        rates.assists_per_36, rates.threes_per_36,
    ]
    if any(v < 0 for v in values):
        raise ValueError("Per-36 rates must be nonnegative.")
    factor = expected_minutes / 36.0
    pts, reb, ast, threes = (round(v * factor, 2) for v in values)
    return {
        "player": rates.player,
        "team": rates.team,
        "projected_minutes": expected_minutes,
        "points": pts,
        "rebounds": reb,
        "assists": ast,
        "threes": threes,
        "points_rebounds_assists": round(pts + reb + ast, 2),
        "source_games": rates.sample_games,
        "method": "Historical per-36 rate × projected minutes",
        "limitations": "Baseline only: no lineup injuries, opponent adjustment, correlated possessions, or sportsbook odds.",
    }
