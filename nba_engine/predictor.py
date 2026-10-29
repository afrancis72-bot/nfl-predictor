"""NBA player-stat projection + joint bootstrap simulation.

Produces point estimates from historical games and *correlated* hypothetical
stat-line draws. The model is research only: rates, matchup factors, and
dependencies have not yet been walk-forward calibrated to sportsbook markets.
"""
from __future__ import annotations
from datetime import date
import numpy as np
import pandas as pd

from nba_engine.data_sources import MARKET_COLS, season_key, norm_team

MARKETS = {
    "Points": ("pts",),
    "Rebounds": ("reb",),
    "Assists": ("ast",),
    "Threes": ("threes",),
    "Steals": ("stl",),
    "Blocks": ("blk",),
    "Turnovers": ("tov",),
    "PRA": ("pts", "reb", "ast"),
    "PR": ("pts", "reb"),
    "PA": ("pts", "ast"),
    "RA": ("reb", "ast"),
    "Stocks": ("stl", "blk"),
}
STATS = tuple(MARKET_COLS.values())

def _recency_weights(size, rate=0.08):
    age = np.arange(size - 1, -1, -1)
    w = np.exp(-rate * age)
    return w / w.sum()

def opponent_context(logs: pd.DataFrame):
    """Small, capped team scoring-environment adjustment; NOT a defensive matchup model."""
    games = (logs.groupby(["date", "team", "opponent"], observed=True, as_index=False)
             .agg(team_points=("pts", "sum")))
    if games.empty:
        return {}
    recent = games.sort_values("date").groupby("team", group_keys=False).tail(24)
    league = float(recent["team_points"].mean())
    if league < 50:
        return {}
    defensive = {}
    for opponent, group in recent.groupby("opponent"):
        if len(group) < 10:
            continue
        raw = float(group["team_points"].mean()) / league
        defensive[opponent] = float(np.clip(1.0 + 0.40 * (raw - 1.0), 0.95, 1.05))
    return defensive

def build_projections(logs: pd.DataFrame, teams, when: date, min_minutes: float=12.0,
                      excluded=None, minute_overrides=None):
    """Only players observed with their current team in the target season are eligible."""
    if isinstance(teams, dict):
        opponents = {norm_team(k): norm_team(v) for k,v in teams.items()}
        teams = set(opponents)
    else:
        teams = {norm_team(t) for t in teams}
        opponents = {t: next((u for u in teams if u != t), "") for t in teams}
    excluded = {str(x) for x in (excluded or [])}
    minute_overrides = {str(k): float(v) for k, v in (minute_overrides or {}).items()}
    history = logs.loc[logs["date"] < pd.Timestamp(when)].copy()
    if history.empty:
        raise ValueError("No history exists strictly before the selected game date.")
    season_start = season_key(when).split("-")[0]
    current_start = pd.Timestamp(f"{season_start}-10-01")
    active = history.loc[(history["date"] >= current_start) & history["team"].isin(teams)]
    if active.empty:
        raise ValueError("No current-season player-game observations for these teams; cannot infer a roster.")
    # Latest observed team per ID; prevent silently projecting a traded player for the old team.
    current_player = history.sort_values("date").groupby("player_id", sort=False).tail(1)
    current_player = current_player[current_player["team"].isin(teams)]
    active_ids = set(active["player_id"])
    current_player = current_player[current_player["player_id"].isin(active_ids)]
    context = opponent_context(history)
    rows=[]
    for p in current_player.itertuples(index=False):
        if str(p.player_id) in excluded:
            continue
        h = history.loc[history["player_id"] == p.player_id].sort_values("date").tail(30).copy()
        if h.empty:
            continue
        if (pd.Timestamp(when) - h["date"].iloc[-1]).days > 30:
            continue
        recent_mins = h["minutes"].tail(5)
        longer_mins = h["minutes"].tail(15)
        forecast_minutes = .65 * recent_mins.mean() + .35 * longer_mins.mean()
        if str(p.player_id) in minute_overrides:
            forecast_minutes = minute_overrides[str(p.player_id)]
        if not 0 <= forecast_minutes <= 48:
            raise ValueError(f"Minutes override for {p.player} must be between 0 and 48.")
        if forecast_minutes < min_minutes:
            continue
        h_rates = h[list(STATS)].to_numpy(dtype=float) / h["minutes"].to_numpy(dtype=float)[:,None]
        w = _recency_weights(len(h))
        means = np.average(h_rates, axis=0, weights=w)
        # High minutes players get normal historical per-minute variance; low sample size is flagged.
        env = context.get(opponents.get(p.team, ""), 1.0)
        outputs = means * forecast_minutes
        outputs[0] *= env  # Only score environment proxy, not multi-stat opponent adjustments.
        current_games = int((h["date"] >= current_start).sum())
        record = {
            "player_id": str(p.player_id),
            "player": str(p.player), "team": p.team,
            "opponent": opponents.get(p.team, ""),
            "minutes": round(float(forecast_minutes), 2),
            "minutes_sd": round(float(max(2.5, h["minutes"].tail(15).std(ddof=0))), 2),
            "sample_games": len(h), "current_season_games": current_games,
            "last_played": str(h["date"].iloc[-1].date()),
            "sample_flag": ("Limited" if current_games < 3 or len(h) < 10 else "Standard"),
            "environment_factor": round(env, 3),
        }
        for key, val in zip(STATS, outputs):
            record[key] = round(max(0, float(val)), 3)
        for name, cols in MARKETS.items():
            record[name] = round(sum(record[col] for col in cols), 2)
        rows.append(record)
    if not rows:
        raise ValueError("No recently active players met the playing-time threshold.")
    return (pd.DataFrame(rows).sort_values(["minutes", "Points"], ascending=False)
            .reset_index(drop=True))

def simulate_stat_lines(projections: pd.DataFrame, logs: pd.DataFrame,
                        trials: int=10000, seed: int=42):
    """Multivariate empirical player-game bootstrap with shared game/team factors.

    A single historic game is sampled per player per iteration: this preserves
    observed within-player stat relationships. Shared game and team multipliers
    approximate tempo correlations; magnitude is an explicit uncalibrated assumption.
    """
    if not 1000 <= trials <= 50000:
        raise ValueError("Simulation count must be 1,000–50,000.")
    if projections.empty:
        raise ValueError("Run projections before simulations.")
    rng = np.random.default_rng(int(seed))
    games = {"-".join(sorted([str(p.team), str(p.opponent)])) for p in projections.itertuples()}
    game_factors = {g: np.clip(1 + rng.normal(0, 0.045, trials), 0.85, 1.18) for g in games}
    team_factors = {t: np.clip(1 + rng.normal(0, 0.027, trials), 0.90, 1.10)
                    for t in projections["team"].unique()}
    result = {}
    for p in projections.itertuples(index=False):
        hist = logs.loc[logs["player_id"] == p.player_id].sort_values("date").tail(30)
        values = hist[list(STATS)].to_numpy(dtype=float)
        mins = hist["minutes"].to_numpy(dtype=float)
        if len(hist) == 0 or np.any(mins <= 0):
            continue
        weights = _recency_weights(len(hist))
        chosen = rng.choice(len(hist), size=trials, p=weights)
        historic_rates = values[chosen] / mins[chosen, None]
        sampled_minutes = np.clip(rng.normal(p.minutes, max(1., p.minutes_sd), trials), 0, 48)
        game_key = "-".join(sorted([p.team, p.opponent]))
        factor = (sampled_minutes * game_factors[game_key] * team_factors[p.team])
        mean_rate = np.average(values / mins[:,None], axis=0, weights=weights)
        expected = np.array([getattr(p, col) for col in STATS])
        # Center bootstrap means to the point projections to avoid drift.
        scale = np.divide(expected, np.maximum(1e-6, mean_rate * p.minutes),
                          out=np.ones(len(STATS)), where=mean_rate > 1e-6)
        raw = historic_rates * factor[:,None] * scale[None,:]
        draws = np.rint(np.maximum(raw, 0)).astype(np.int16)
        single = {key: draws[:, j] for j, key in enumerate(STATS)}
        for name, cols in MARKETS.items():
            if len(cols) == 1:
                result[(p.player_id, name)] = single[cols[0]]
            else:
                result[(p.player_id, name)] = sum((single[c].astype(np.int32) for c in cols))
    return result

def prepare_bet_lines(lines: pd.DataFrame, projections: pd.DataFrame):
    """Only user-supplied sportsbook lines are scored; no fictional odds."""
    x = lines.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    needed = {"player_id", "market", "line", "side"}
    if not needed.issubset(x.columns):
        raise ValueError("Lines must include player_id, market, line and side.")
    x["player_id"] = x["player_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    x["line"] = pd.to_numeric(x["line"], errors="coerce")
    if "odds" not in x.columns:
        x["odds"] = np.nan
    x["odds"] = pd.to_numeric(x["odds"], errors="coerce")
    x["side"] = x["side"].astype(str).str.title().str.strip()
    x["market"] = x["market"].astype(str).str.strip()
    x = x[x["line"].notna()].copy()
    if x.empty:
        return x
    if (~x["side"].isin(["Over","Under"])).any():
        raise ValueError("Each side must be Over or Under.")
    if (~x["market"].isin(MARKETS)).any():
        raise ValueError("Unsupported market. Supported: " + ", ".join(MARKETS))
    if (x["line"] < 0).any():
        raise ValueError("All sportsbook lines must be >= 0.")
    valid_ids = set(projections["player_id"].astype(str))
    unknown = set(x["player_id"]) - valid_ids
    if unknown:
        raise ValueError(f"Unknown/unavailable players in betting lines: {', '.join(sorted(unknown)[:5])}")
    x = x.merge(projections[["player_id", "player", "team", "opponent"]],
                on="player_id", how="left", validate="many_to_one", suffixes=("", "_model"))
    x["label"] = x.apply(lambda row: f"{row.player} · {row.market} {row.side} {row.line:g}", axis=1)
    x["label"] = [f"{i+1}. {s}" for i,s in enumerate(x["label"])]
    return x.reset_index(drop=True)

def evaluate_bets(lines: pd.DataFrame, samples):
    """Return per-leg estimates and tri-state results: WIN +1, PUSH 0, LOSS -1."""
    status = np.empty((len(next(iter(samples.values()))), len(lines)), dtype=np.int8)
    report = lines.copy()
    pwin=[]; ppush=[]; est_mean=[]
    for j, row in enumerate(lines.itertuples(index=False)):
        sample = np.asarray(samples[(row.player_id, row.market)], dtype=float)
        wins = sample > row.line if row.side == "Over" else sample < row.line
        pushes = np.isclose(sample, row.line)
        status[:,j] = np.where(wins, 1, np.where(pushes, 0, -1))
        pwin.append(float(np.mean(wins)))
        ppush.append(float(np.mean(pushes)))
        est_mean.append(float(np.mean(sample)))
    report["sim_mean"] = est_mean
    report["win_probability"] = pwin
    report["push_probability"] = ppush
    report["conditional_win_rate"] = report["win_probability"] / (1-report["push_probability"]).clip(lower=.0001)
    def american_implied(x):
        if not np.isfinite(x) or x == 0 or (-100 < x < 100):
            return np.nan
        return 100/(100+x) if x > 0 else abs(x)/(abs(x)+100)
    report["book_implied"] = report["odds"].map(american_implied)
    report["model_edge"] = report["conditional_win_rate"] - report["book_implied"]
    return report, status

def parlay_summary(status: np.ndarray, indices):
    indices = list(indices)
    if len(indices) < 2 or len(indices) > 8 or len(set(indices)) != len(indices):
        raise ValueError("Select 2–8 distinct betting legs.")
    chosen = status[:,indices]
    win = np.all(chosen == 1, axis=1)
    loss = np.any(chosen == -1, axis=1)
    push = ~(win|loss)
    return {"all_win": float(np.mean(win)),
            "any_loss": float(np.mean(loss)),
            "any_push_no_loss": float(np.mean(push)),
            "has_push": bool(np.any(chosen == 0))}
