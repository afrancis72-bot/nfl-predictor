"""Independent sportsbook prop comparison for NFL stat-driven simulation.

Market lines are diagnostics, never inputs to simulation or optimizer.
"""
import csv
import io
import numpy as np

SUPPORTED = {
    "passing_yards", "passing_tds", "pass_attempts", "completions",
    "interceptions", "rushing_yards", "carries", "receiving_yards",
    "receptions", "targets", "rushing_tds", "receiving_tds",
}
ALIASES = {
    "pass yards": "passing_yards", "passing yards": "passing_yards",
    "pass touchdowns": "passing_tds", "passing touchdowns": "passing_tds",
    "rush yards": "rushing_yards", "rushing yards": "rushing_yards",
    "receiving yards": "receiving_yards", "receptions": "receptions",
    "passing attempts": "pass_attempts", "pass completions": "completions",
    "rushing attempts": "carries", "carries": "carries",
    "targets": "targets", "interceptions": "interceptions",
    "rushing touchdowns": "rushing_tds", "receiving touchdowns": "receiving_tds",
}
def norm_name(name):
    return "".join(ch for ch in str(name).casefold() if ch.isalnum())

def norm_stat(value):
    raw = str(value).strip().casefold().replace("_", " ")
    return ALIASES.get(raw, raw.replace(" ", "_"))

def compare_props(stat_draws, prop_rows, *, portfolio=None):
    """Return comparison records; fail closed on ambiguous player names."""
    from collections import Counter
    names = {norm_name(k): k for k in stat_draws if not str(k).startswith("__")}
    collisions = Counter(norm_name(k) for k in stat_draws if not str(k).startswith("__"))
    exposure = Counter()
    if portfolio:
        for entry in portfolio:
            lineup = entry.get("candidate", entry)
            exposure.update(norm_name(p["Name"]) for p in lineup["players"])
    results = []
    for row in prop_rows:
        player = str(row.get("player", row.get("Player", ""))).strip()
        stat = norm_stat(row.get("stat", row.get("Stat", "")))
        source = str(row.get("sportsbook", row.get("Sportsbook", "Unknown"))).strip()
        try:
            line = float(row.get("line", row.get("Line", "")))
        except (TypeError, ValueError):
            continue
        key = norm_name(player)
        if not np.isfinite(line) or stat not in SUPPORTED or collisions[key] != 1:
            continue
        actual_name = names[key]
        stats = stat_draws[actual_name]
        if stat not in stats:
            continue
        samples = np.asarray(stats[stat], dtype=float)
        if samples.size == 0 or not np.isfinite(samples).all():
            continue
        results.append({
            "Player": actual_name, "Stat": stat, "Sportsbook": source,
            "Prop line": line, "Sim mean": round(float(np.mean(samples)), 2),
            "Sim median": round(float(np.median(samples)), 2),
            "Sim P25": round(float(np.percentile(samples, 25)), 2),
            "Sim P75": round(float(np.percentile(samples, 75)), 2),
            "Over probability": round(float(np.mean(samples > line)), 4),
            "Under probability": round(float(np.mean(samples < line)), 4),
            "Push probability": round(float(np.mean(samples == line)), 4),
            "Mean minus line": round(float(np.mean(samples) - line), 2),
            "Portfolio lineups": exposure[key],
        })
    return sorted(results, key=lambda r: (-r["Portfolio lineups"],
                                          -abs(r["Mean minus line"]), r["Player"]))

def read_prop_csv(payload):
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(payload)))
    return rows
