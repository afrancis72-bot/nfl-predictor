"""Public-feed NBA end-to-end diagnostic.

Run from GitHub Actions or locally (requires internet). Verifies the actual
data providers and computes an actual 1-game NBA prediction without fake input.
"""
from datetime import date
from nba_engine.data_sources import scoreboard, load_public_logs
from nba_engine.predictor import build_projections, simulate_stat_lines

def main():
    game_date = date(2026, 10, 30)
    games = scoreboard(game_date)
    if games.empty:
        raise RuntimeError("ESPN scoreboard returned zero NBA games for 2026-10-30")
    print("Schedule provider OK:", len(games), "matchups")
    print(games[["matchup","state","status"]].to_string(index=False))
    eligible = games[games["state"].isin(["pre","in"])]
    if eligible.empty:
        eligible = games
    game = eligible.iloc[0]
    logs, meta = load_public_logs(game_date)
    print("Historical provider OK:", meta)
    proj = build_projections(logs, {game["home"]: game["away"], game["away"]: game["home"]},
                             game_date, min_minutes=12)
    sims = simulate_stat_lines(proj, logs, trials=1000)
    print("Projections made:", len(proj), "draws:", len(sims))
    print(proj[["player","team","opponent","minutes","Points","Rebounds","Assists"]].head(5).to_string(index=False))
    if len(proj) < 8 or len(sims) < 20:
        raise RuntimeError("NBA live pipeline produced too few projections for an expected full game.")
    matchup_map = {}
    for row in eligible.itertuples(index=False):
        matchup_map[row.home] = row.away
        matchup_map[row.away] = row.home
    full = build_projections(logs, matchup_map, game_date, min_minutes=12)
    full_sims = simulate_stat_lines(full, logs, trials=10000)
    print("Full-slate projections:", len(full), "stat-market sample arrays:", len(full_sims))
    if len(full) < 100 or len(full_sims) != len(full) * 12:
        raise RuntimeError("Full-slate NBA simulation is missing expected players or markets.")
    print("End-to-end NBA public pipeline completed successfully.")

if __name__ == "__main__":
    main()
