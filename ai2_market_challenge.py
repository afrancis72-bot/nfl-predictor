"""Post-simulation market comparison. No odds access during simulation.

The caller must persist a frozen simulation before invoking this module.
Never writes to or modifies the simulation object.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
from statistics import median

NFL_MARKETS = frozenset({"player_pass_yds", "player_pass_tds", "player_pass_completions",
                         "player_rush_yds", "player_rec_yds", "player_receptions"})
NBA_MARKETS = frozenset({"player_points", "player_rebounds", "player_assists", "player_threes"})
SPORT_KEYS = {"NFL": "americanfootball_nfl", "NBA": "basketball_nba"}

@dataclass(frozen=True)
class FrozenSimulation:
    run_id: str
    frozen_at: str
    digest: str
    observations: tuple

def _utc(value):
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Timezone-aware timestamp required")
    return stamp.astimezone(timezone.utc)

def freeze_simulation(run_id, observations, frozen_at):
    """Freeze player/market simulation samples, with a reproducible digest."""
    if not str(run_id).strip():
        raise ValueError("Run ID required")
    _utc(frozen_at)
    canonical = []
    seen = set()
    for obs in observations:
        key = (str(obs["player_id"]), str(obs["market_key"]))
        if not all(key) or key in seen:
            raise ValueError("Duplicate or missing player/market")
        seen.add(key)
        samples = tuple(float(v) for v in obs["samples"])
        if not samples or any(not isfinite(v) or v < 0 for v in samples):
            raise ValueError("Simulation samples must be finite and nonnegative")
        canonical.append({"player_id": key[0], "market_key": key[1],
                          "samples": samples})
    if not canonical:
        raise ValueError("No simulation observations")
    canonical.sort(key=lambda o: (o["player_id"], o["market_key"]))
    payload = json.dumps({"run_id": run_id, "frozen_at": frozen_at,
                          "observations": canonical}, sort_keys=True, separators=(",", ":"))
    return FrozenSimulation(str(run_id), str(frozen_at),
                            sha256(payload.encode()).hexdigest(), tuple(
                                (o["player_id"], o["market_key"], o["samples"]) for o in canonical))

def _american_implied(price):
    price = float(price)
    if not isfinite(price) or price == 0 or abs(price) < 100:
        raise ValueError("Invalid American odds")
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)

def compare_frozen_to_props(frozen, props, *, min_gap=0.15):
    """Return review flags only; no model updates or automatic bets.

    Requires explicit player_id matching, full-game period, and odds timestamps.
    Provider names alone are not accepted as proof of player identity.
    """
    if not isinstance(frozen, FrozenSimulation):
        raise TypeError("A frozen simulation is required")
    if not 0 <= min_gap <= 1:
        raise ValueError("min_gap must be between 0 and 1")
    samples_by_key = {(pid, market): samples for pid, market, samples in frozen.observations}
    findings = []
    for prop in props:
        if str(prop.get("period", "")).upper() != "FULL":
            continue
        key = (str(prop.get("player_id", "")), str(prop.get("market_key", "")))
        samples = samples_by_key.get(key)
        if samples is None:
            continue
        try:
            observed_at = _utc(prop["last_update"])
            fetched_at = _utc(prop["fetched_at"])
            line = float(prop["line"])
            over = _american_implied(prop["over_price"])
            under = _american_implied(prop["under_price"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if (fetched_at < _utc(frozen.frozen_at) or observed_at > fetched_at
                or (fetched_at - observed_at).total_seconds() > 900
                or not isfinite(line) or line < 0):
            continue
        fair_over = over / (over + under)
        model_over = sum(v > line for v in samples) / len(samples)
        model_push = sum(v == line for v in samples) / len(samples)
        gap = model_over - fair_over
        if abs(gap) < min_gap:
            continue
        findings.append({"run_id": frozen.run_id, "simulation_digest": frozen.digest,
                         "player_id": key[0], "market_key": key[1],
                         "bookmaker": str(prop.get("bookmaker", "")),
                         "line": line, "simulation_median": median(samples),
                         "model_over_probability": model_over,
                         "model_push_probability": model_push,
                         "market_no_vig_over_probability": fair_over,
                         "probability_gap": gap, "status": "REVIEW_ONLY",
                         "market_observed_at": observed_at.isoformat()})
    return sorted(findings, key=lambda f: abs(f["probability_gap"]), reverse=True)
