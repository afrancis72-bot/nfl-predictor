"""Team AI² Pro NBA — automatic slate research + actual sportsbook prop analysis."""
from __future__ import annotations
from datetime import datetime, date
from io import BytesIO
from pathlib import Path
import hashlib
from zoneinfo import ZoneInfo
import os
import secrets
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from nba_engine.data_sources import scoreboard, load_public_logs, norm_team
from nba_engine.predictor import (
    MARKETS, build_projections, simulate_stat_lines,
    prepare_bet_lines, evaluate_bets, parlay_summary,
    summarize_simulated_projections,
)
from nba_engine.props_sim import expected_net_roi
from nba_engine.odds_feed import BOOKMAKERS, load_current_props, normalize_csv_lines


st.subheader("NBA Predictor · Player Props & Parlays")
st.caption("Choose the games → Run engine → Compare real player-prop lines → Review parlays")

@st.cache_data(ttl=900, show_spinner=False)
def cached_schedule(day):
    return scoreboard(date.fromisoformat(day))

@st.cache_data(ttl=14400, show_spinner=False)
def cached_history(day):
    return load_public_logs(date.fromisoformat(day))

now = datetime.now(ZoneInfo("America/New_York")).date()
col_date, col_game, col_sims = st.columns([1,2.5,1.2])
with col_date:
    target = st.date_input("Game date (Eastern)", value=now)
with col_sims:
    trials = st.selectbox("Simulations", [5000,10000,25000], index=1)

seed_left, seed_right = st.columns([1.15, 1.85])
with seed_left:
    seed_mode = st.radio(
        "Simulation seed mode",
        ["Fixed", "Fibonacci preset", "Fresh random"],
        key="nba_seed_mode",
    )
with seed_right:
    if seed_mode == "Fixed":
        selected_seed = int(st.number_input(
            "Seed · repeatable", min_value=0, max_value=2147483647,
            value=42, step=1, key="nba_fixed_seed",
        ))
    elif seed_mode == "Fibonacci preset":
        selected_seed = int(st.selectbox(
            "Fibonacci seed · repeatable",
            [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946],
            index=7, key="nba_fibonacci_seed",
        ))
    else:
        selected_seed = None
        st.caption("A new random seed is chosen whenever you press Run. The number is displayed for later replay.")
st.caption("Seeds change the sampled simulation scenarios, not our independent player projections. Fibonacci presets are research choices, not predictive advantages.")

with st.expander("Model transparency and source limitations"):
    st.write(
        "Schedule: ESPN's undocumented public NBA scoreboard endpoint; "
        "player-game logs: community-maintained NBA box-score-derived CSVs hosted on GitHub. "
        "The latest publicly available snapshot may lag the real NBA results. "
        "No sportsbook lines, odds, or injury statuses are invented. "
        "Projections use observed player minutes, historical per-minute rates, "
        "a capped opponent scoring proxy, and empirical joint-stat resampling. "
        "The simulation correlation factors and reported probabilities are research estimates "
        "**not yet calibrated** through a full historical betting-market backtest."
    )
    st.markdown("[Player stats provenance](https://github.com/xtheredshirtx/fanduel-player-data/tree/main/NBA/csv) · [ESPN](https://www.espn.com/nba/schedule)")

schedule_issue = None
try:
    schedule = cached_schedule(target.isoformat())
except Exception as exc:
    schedule_issue = str(exc)
    schedule = pd.DataFrame()

selected = None
with col_game:
    if not schedule.empty:
        valid = schedule.loc[schedule["state"].isin(["pre","in"])]
        if valid.empty:
            st.info("No upcoming/in-progress NBA games on this date. You can select a historical date or a manual matchup.")
        labels = ["All available games"] + valid["matchup"].tolist() if not valid.empty else []
        selection = st.selectbox("Today's slate", labels + ["Manual matchup"])
        if selection == "Manual matchup":
            selected = None
        elif selection == "All available games":
            selected = valid
        else:
            selected = valid.loc[valid["matchup"] == selection]
    else:
        st.warning("The live schedule is temporarily unavailable." if schedule_issue else "No game schedule was found.")
        selection = "Manual matchup"
if selected is None:
    NBA_TEAMS = sorted(("ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET","GSW",
                        "HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP",
                        "NYK","OKC","ORL","PHI","PHX","POR","SAC","SAS","TOR","UTA","WAS"))
    left,right = st.columns(2)
    with left: manual_away = st.selectbox("Away team (manual; schedule not verified)", NBA_TEAMS, index=NBA_TEAMS.index("LAL"))
    with right: manual_home = st.selectbox("Home team (manual; schedule not verified)", NBA_TEAMS, index=NBA_TEAMS.index("GSW"))
    if manual_away == manual_home:
        st.error("Away and home teams must differ.")
    teams_map = {manual_away: manual_home, manual_home: manual_away}
    schedule_key = f"{manual_away}-{manual_home}"
else:
    teams_map = {}
    for g in selected.itertuples(index=False):
        teams_map[g.home] = g.away
        teams_map[g.away] = g.home
    schedule_key = "-".join(sorted(teams_map))
    st.caption("Selected: " + ", ".join(selected["matchup"]))

st.info("Injury availability is NOT automatically confirmed. Check official reports before evaluating bets; exclude OUT players and adjust teammates' minutes when their roles change.")

market_key = f"{target}-{schedule_key}"  # Current bookmaker lines are independent of RNG seed.
run_key = f"{market_key}-{trials}-{seed_mode}-{selected_seed}"
if st.button("▶ Run NBA Prediction Engine", type="primary", use_container_width=True):
    if len(teams_map) != len(set(teams_map.values())) and len(teams_map) != 2:
        st.error("Please select a valid slate with one opponent per team.")
    elif len(teams_map) == 1:
        st.error("Choose two different teams.")
    else:
        try:
            run_seed = int(selected_seed) if selected_seed is not None else secrets.randbelow(2147483647)
            with st.spinner("Loading recent NBA box scores, projecting minutes and simulating correlated stat lines…"):
                history, provenance = cached_history(target.isoformat())
                projections = build_projections(history, teams_map, target, min_minutes=12)
                samples = simulate_stat_lines(projections, history, trials=trials, seed=run_seed)
            st.session_state["nba_auto_result"] = {
                "run_key": run_key, "history":history, "provenance":provenance,
                "original":projections, "projections":projections,
                "samples":samples, "teams_map":teams_map, "seed":run_seed,
            }
            st.session_state.pop("nba_auto_bets", None)
            st.success(f"Engine complete: {len(projections)} player projections and {trials:,} simulations. Seed: {run_seed}.")
        except Exception as exc:
            st.error(f"NBA data or simulation failed: {exc}")
            st.info("The public data feeds may be unavailable. No synthetic statistics will be substituted.")

run = st.session_state.get("nba_auto_result")
if run is not None and run["run_key"] == run_key:
    projections = run["projections"]
    history = run["history"]
    provenance = run["provenance"]
    samples = run["samples"]
    st.caption(f"Active simulation seed: {run['seed']} · Mode: {seed_mode} · {trials:,} scenarios. Reuse the same seed and inputs to reproduce the draws.")
    st.caption(f"Player data as of {provenance['snapshot']} | Latest observed game {provenance['latest_game_date']} | "
               f"{provenance['rows']:,} historical player-game records | {'/'.join(provenance['seasons'])}")
    age_days = (target - date.fromisoformat(provenance["latest_game_date"])).days
    if age_days > 4:
        st.warning(f"Historical data is {age_days} days old relative to the target game. Confirm roster moves and current playing time.")

    projection_tab, props_tab, adjustments_tab = st.tabs(["📊 Player projections", "🎯 Props & parlays", "⚙️ Injuries / minutes"])
    with projection_tab:
        st.subheader("Simulation-based player projections")
        st.caption(
            "Our historical/matchup baseline is calculated BEFORE simulation and should not change with the seed. "
            "The simulated averages, medians, P10 and P90 come from the actual sampled games and DO change across seeds."
        )
        visible_markets = ("Points", "Rebounds", "Assists", "Threes", "PRA")
        simulated_summary = summarize_simulated_projections(
            projections, samples, markets=visible_markets
        )
        means = simulated_summary.pivot(
            index="player_id", columns="market", values="simulation_mean"
        ).reset_index()
        means = means.rename(columns={
            "Points":"Sim PTS", "Rebounds":"Sim REB",
            "Assists":"Sim AST", "Threes":"Sim 3PM", "PRA":"Sim PRA",
        })
        display = projections[[
            "player_id", "player", "team", "opponent",
            "minutes", "Points", "sample_flag",
        ]].copy().merge(means, on="player_id", how="left", validate="one_to_one")
        display = display[[
            "player", "team", "opponent", "minutes", "Points",
            "Sim PTS", "Sim REB", "Sim AST", "Sim 3PM", "Sim PRA",
            "sample_flag",
        ]].rename(columns={
            "player":"Player", "team":"Team", "opponent":"Opponent",
            "minutes":"Expected minutes", "Points":"Baseline PTS",
            "sample_flag":"Sample quality",
        })
        for col in ("Baseline PTS", "Sim PTS", "Sim REB", "Sim AST",
                    "Sim 3PM", "Sim PRA"):
            display[col] = display[col].round(3)
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.caption(
            f"Simulation seed: {run['seed']}. The simulated mean columns change slightly with the seed; "
            "the independent baseline is fixed. Small differences between seeds reflect Monte Carlo sampling noise."
        )
        detail_market = st.selectbox(
            "Explore simulated distribution for one stat",
            list(visible_markets), key=f"nba_distribution_market_{run_key}",
        )
        detail = simulated_summary.loc[
            simulated_summary["market"] == detail_market,
            ["player","team","baseline_projection","simulation_mean",
             "simulation_median","simulation_p10","simulation_p90",
             "monte_carlo_se","sample_flag"],
        ].copy()
        detail = detail.rename(columns={
            "player":"Player", "team":"Team",
            "baseline_projection":"Historical baseline",
            "simulation_mean":"Simulation mean",
            "simulation_median":"Simulation median",
            "simulation_p10":"P10",
            "simulation_p90":"P90",
            "monte_carlo_se":"Monte Carlo std error",
            "sample_flag":"Sample quality",
        })
        for col in ("Historical baseline", "Simulation mean",
                    "Simulation median", "P10", "P90", "Monte Carlo std error"):
            detail[col] = detail[col].round(3)
        st.dataframe(detail, hide_index=True, use_container_width=True)
        st.caption(
            "P10/P90 are 10th/90th percentile simulated game outcomes, not confidence intervals "
            "for our forecast. Monte Carlo std error measures uncertainty from finite simulation draws; "
            "it does not capture player-model errors, injuries or unmodeled matchups."
        )

        projections_export = projections.copy()
        for market in visible_markets:
            per_market = simulated_summary.loc[
                simulated_summary["market"] == market
            ].set_index("player_id")
            label = {
                "Points":"points", "Rebounds":"rebounds",
                "Assists":"assists", "Threes":"threes", "PRA":"pra",
            }[market]
            indices = projections_export["player_id"].astype(str)
            for attr, suffix in (
                ("simulation_mean","mean"),("simulation_median","median"),
                ("simulation_p10","p10"),("simulation_p90","p90"),
                ("monte_carlo_se","mc_se"),
            ):
                projections_export[f"sim_{label}_{suffix}"] = (
                    indices.map(per_market[attr]).round(4)
                )
        projections_export["simulation_seed"] = int(run["seed"])
        projections_export["simulation_trials"] = int(trials)
        st.download_button(
            "Export player projections — baseline + simulated outcomes (CSV)",
            projections_export.to_csv(index=False).encode("utf-8-sig"),
            "team_ai2_nba_projections.csv", mime="text/csv",
        )
        detail_export = simulated_summary.copy()
        detail_export["simulation_seed"] = int(run["seed"])
        st.download_button(
            "Export full simulated stat distributions (CSV)",
            detail_export.to_csv(index=False).encode("utf-8-sig"),
            "team_ai2_nba_simulated_distributions.csv", mime="text/csv",
        )
        st.caption(
            "Limited history means fewer than three current-season games or fewer than ten "
            "prior observations. Seed sensitivity does not establish a betting advantage."
        )

    with props_tab:
        st.subheader("Enter real sportsbook prop lines")
        st.caption("The app never invents a line or betting price. Type offered lines into the grid, or upload a CSV.")
        names = {f"{p.player} ({p.team})": str(p.player_id) for p in projections.itertuples(index=False)}
        defaults = list(names)[:12]
        chosen = st.multiselect("Players to compare", list(names), default=defaults)
        market_choices = st.multiselect("Markets", list(MARKETS),
                                        default=["Points","Rebounds","Assists","Threes","PRA"])
        # Our projections are calculated independently. Real sportsbook markets
        # are ingested only when explicitly requested with an API key or upload.
        with st.expander("Automatic sportsbook lines — Hard Rock Bet FL and others", expanded=True):
            st.caption("Compare our independent forecasts against LIVE quoted markets. The Odds API supports Hard Rock Bet (Florida), subject to currently listed events, markets, and API quota. Quotes may differ from your Hard Rock Bet app.")
            book_title = st.selectbox("Sportsbook to retrieve", list(BOOKMAKERS),
                                     key="nba_odds_book")
            try:
                saved_api_key = str(st.secrets.get("THE_ODDS_API_KEY",""))
            except Exception:
                saved_api_key = ""
            provided_api_key = st.text_input(
                "The Odds API key (password input; not saved by this app)",
                value=os.environ.get("THE_ODDS_API_KEY", saved_api_key),
                type="password", key="nba_odds_access_key"
            )
            st.caption("The fetch button is intentional: each request may consume your provider's API credits. No automatic or repeated polling occurs. You can start with a free key, but verify your plan's prop-market availability.")
            if st.button("Import current actual sportsbook prop lines",
                         type="secondary", disabled=not provided_api_key.strip(),
                         key=f"nba_fetch_props_{market_key}"):
                try:
                    with st.spinner("Retrieving quoted NBA player props for the selected games…"):
                        incoming, provenance = load_current_props(
                            projections, run["teams_map"], provided_api_key.strip(),
                            bookmaker_key=BOOKMAKERS[book_title], markets=market_choices,
                            target_day=target,
                        )
                    st.session_state["nba_odds_snapshot"] = {
                        "market_key":market_key, "book":book_title,
                        "data":incoming, "details":provenance,
                    }
                    if incoming.empty:
                        st.warning("This bookmaker returned no matching player props for the selected games. No prices or lines have been substituted.")
                    else:
                        st.success(f"Imported {len(incoming)} actual quoted Over/Under rows from {book_title}.")
                except Exception as exc:
                    st.error(f"Unable to retrieve actual sportsbook odds: {exc}")
            latest_odds = st.session_state.get("nba_odds_snapshot")
            if latest_odds and latest_odds.get("market_key") == market_key:
                details = latest_odds["details"]
                st.caption(
                    f"Last fetch: {details.get('retrieved_utc','unknown')} · "
                    f"{details.get('event_count',0)} matching events · "
                    f"{len(latest_odds['data'])} quoted lines"
                )
                if details.get("quota"):
                    st.caption("API quota: " + ", ".join(f"{k}: {v}" for k,v in details["quota"].items()))
                if details.get("unmatched_player_names"):
                    st.warning("Unmatched player names (not silently assigned): " +
                               ", ".join(details["unmatched_player_names"][:10]))
                if not latest_odds["data"].empty:
                    st.download_button(
                        "Export imported sportsbook odds (CSV)",
                        latest_odds["data"].to_csv(index=False).encode("utf-8-sig"),
                        "team_ai2_nba_actual_sportsbook_odds.csv", mime="text/csv"
                    )
        st.caption("**No API key?** You can still upload a sportsbook CSV or type real lines directly below.")
        upload = st.file_uploader(
            "Optional: upload existing sportsbook CSV (accepts player names or IDs)",
            type=["csv"], key=f"nba_market_{schedule_key}_{target}"
        )
        bet_grid = None
        if upload is not None:
            try:
                uploaded_lines = pd.read_csv(BytesIO(upload.getvalue()))
                bet_grid, imported = normalize_csv_lines(uploaded_lines, projections)
                st.caption(
                    f"Imported file: {imported['matched']} matched prop lines, "
                    f"{imported['missing_lines']} rows without sportsbook lines."
                )
                if imported["unmatched"]:
                    st.warning("Unmatched player rows: " + ", ".join(imported["unmatched"][:10]))
            except Exception as exc:
                st.error(f"Could not import prop CSV: {exc}")
        elif (st.session_state.get("nba_odds_snapshot") or {}).get("market_key") == market_key:
            bet_grid = st.session_state["nba_odds_snapshot"]["data"].copy()
            if not bet_grid.empty:
                st.caption("Using actual lines imported from " +
                           st.session_state["nba_odds_snapshot"]["book"] + ".")
        if bet_grid is None:
            grid = [{"player_id": names[n], "player": n, "market": market,
                     "side": "Over", "line": None, "odds": None}
                    for n in chosen for market in market_choices]
            if len(grid) > 350:
                st.warning("Choose fewer players or markets to keep the worksheet manageable.")
                grid = grid[:350]
            bet_grid = st.data_editor(
                pd.DataFrame(grid, columns=["player_id","player","market","side","line","odds"]),
                hide_index=True, use_container_width=True,
                disabled=["player_id","player","market"],
                column_config={
                    "line": st.column_config.NumberColumn("Actual sportsbook line", min_value=0, step=0.5),
                    "odds": st.column_config.NumberColumn("Actual American odds", step=5),
                    "side": st.column_config.SelectboxColumn("Bet side", options=["Over","Under"]),
                },
                key=f"nba_lines_{market_key}",
            )
        # Give the user a player-populated worksheet instead of an empty header-only
        # CSV. These are MODEL projections, not sportsbook lines or quoted odds.
        roster_lookup = projections.set_index("player_id")
        worksheet_rows = []
        for display_name in chosen:
            player_id = names[display_name]
            player_row = roster_lookup.loc[player_id]
            for market in market_choices:
                worksheet_rows.append({
                    "player_id": player_id,
                    "player": str(player_row["player"]),
                    "team": str(player_row["team"]),
                    "market": market,
                    "model_projection": float(player_row[market]),
                    "line": None,
                    "side": "Over",
                    "odds": None,
                })
        worksheet = pd.DataFrame(
            worksheet_rows,
            columns=["player_id", "player", "team", "market",
                     "model_projection", "line", "side", "odds"]
        )
        st.download_button(
            "Download prefilled player-prop worksheet",
            worksheet.to_csv(index=False).encode("utf-8-sig"),
            "team_ai2_nba_player_props_prefilled.csv",
            mime="text/csv",
            disabled=worksheet.empty,
            help="Player IDs and model projections are included. Enter your real sportsbook line and odds before uploading."
        )
        st.caption("Prefilled worksheet: player names, IDs, and our model projections are included. **Actual sportsbook lines and odds remain blank.** You can also enter those directly in the table above—no download required.")
        with st.expander("Advanced: blank CSV format"):
            st.download_button("Download header-only CSV",
                               b"player_id,market,line,side,odds\n",
                               "nba_real_lines_template.csv", mime="text/csv")
        try:
            bets = prepare_bet_lines(bet_grid, projections)
            if bets.empty:
                st.info("Fill in at least one actual sportsbook line to see probabilities.")
            else:
                report,status = evaluate_bets(bets,samples)
                # This is the user's actual decision question: compare OUR independent
                # expected stat line to the external bookmaker line, then price.
                proj_lookup = projections.set_index("player_id")
                report["our_projection"] = [
                    float(proj_lookup.loc[str(r.player_id),str(r.market)])
                    for r in report.itertuples(index=False)
                ]
                report["projection_minus_book_line"] = report["our_projection"] - report["line"]
                if "sportsbook" not in report.columns:
                    report["sportsbook"] = "Manually entered line"
                st.subheader("Independent model vs. actual sportsbook lines")
                shown = report[[
                    "label", "sportsbook", "our_projection", "line",
                    "projection_minus_book_line", "sim_mean",
                    "win_probability", "push_probability",
                    "odds", "book_implied", "model_edge",
                ]].copy()
                for c in ["win_probability","push_probability","book_implied","model_edge"]:
                    shown[c] = (100 * shown[c]).round(1)
                for c in ["our_projection","line","projection_minus_book_line","sim_mean"]:
                    shown[c] = shown[c].round(2)
                shown = shown.rename(columns={
                    "label":"Player · market · side",
                    "sportsbook":"Sportsbook",
                    "our_projection":"Our projected stat",
                    "line":"Actual sportsbook line",
                    "projection_minus_book_line":"Projection − line",
                    "sim_mean":"Simulation average",
                    "win_probability":"Model win %",
                    "push_probability":"Push %",
                    "odds":"Offered American odds",
                    "book_implied":"Book break-even %",
                    "model_edge":"Uncalibrated model edge %",
                })
                st.dataframe(shown, hide_index=True, use_container_width=True)
                st.caption("Positive projection-minus-line supports the OVER direction; negative supports UNDER. Sportsbook implied prices include bookmaker margin. A projection difference or model edge is not proof of a profitable bet.")
                with st.expander("🔬 Compare two simulation seeds on the SAME sportsbook lines", expanded=True):
                    st.write("Player forecasts and book prices stay fixed. This checks only whether simulated hit rates change when the random seed changes.")
                    alternate_seed_default = 34 if int(run["seed"]) != 34 else 42
                    compare_seed = int(st.number_input(
                        "Comparison seed", min_value=0, max_value=2147483647,
                        value=alternate_seed_default, step=1,
                        key=f"nba_compare_seed_{run_key}",
                    ))
                    market_fingerprint = hashlib.sha256(
                        bets[["player_id", "market", "line", "side", "odds"]]
                        .to_csv(index=False).encode("utf-8")
                    ).hexdigest()
                    comparison_key = (run_key, int(run["seed"]), compare_seed, market_fingerprint)
                    if st.button(
                        "Compare simulated hit rates",
                        disabled=(compare_seed == int(run["seed"])),
                        key=f"nba_compare_go_{run_key}",
                    ):
                        with st.spinner("Running independent simulation using comparison seed…"):
                            other_samples = simulate_stat_lines(
                                projections, history, trials=int(trials), seed=compare_seed
                            )
                            other_report, _ = evaluate_bets(bets, other_samples)
                        sensitivity = report[["label", "player_id", "market", "side", "line",
                                              "our_projection", "win_probability"]].copy()
                        sensitivity["comparison_probability"] = other_report["win_probability"].to_numpy()
                        sensitivity["delta_percentage_points"] = 100 * (
                            sensitivity["comparison_probability"] - sensitivity["win_probability"]
                        )
                        sensitivity["absolute_delta_pp"] = sensitivity["delta_percentage_points"].abs()
                        sensitivity["baseline_seed"] = int(run["seed"])
                        sensitivity["comparison_seed"] = compare_seed
                        st.session_state["nba_seed_sensitivity"] = {
                            "key": comparison_key, "results": sensitivity,
                        }
                    comparison = st.session_state.get("nba_seed_sensitivity", {})
                    if comparison.get("key") == comparison_key:
                        sensitivity = comparison["results"]
                        changed = int((sensitivity["absolute_delta_pp"] > 1e-8).sum())
                        avg_delta = float(sensitivity["absolute_delta_pp"].mean())
                        max_delta = float(sensitivity["absolute_delta_pp"].max())
                        d1, d2, d3 = st.columns(3)
                        d1.metric("Legs with changed probabilities", f"{changed}/{len(sensitivity)}")
                        d2.metric("Avg absolute change", f"{avg_delta:.2f} pp")
                        d3.metric("Largest change", f"{max_delta:.2f} pp")
                        preview = sensitivity.sort_values(
                            "absolute_delta_pp", ascending=False
                        ).head(15).copy()
                        preview["win_probability"] *= 100
                        preview["comparison_probability"] *= 100
                        preview = preview[[
                            "label", "our_projection", "line",
                            "win_probability", "comparison_probability",
                            "delta_percentage_points",
                        ]].round(2).rename(columns={
                            "label":"Player / bet",
                            "our_projection":"Our forecast (unchanged)",
                            "line":"Book line (unchanged)",
                            "win_probability":f"Seed {int(run['seed'])} hit %",
                            "comparison_probability":f"Seed {compare_seed} hit %",
                            "delta_percentage_points":"Change (percentage points)",
                        })
                        st.dataframe(preview, hide_index=True, use_container_width=True)
                        st.download_button(
                            "Export seed sensitivity comparison",
                            sensitivity.to_csv(index=False).encode("utf-8-sig"),
                            "team_ai2_nba_seed_comparison.csv", mime="text/csv",
                        )
                        st.caption("Different seeds sample different outcomes, but with enough trials hit-rate estimates should be close. Larger seed-to-seed swings point to Monte Carlo uncertainty, not a special predictive seed.")
                    elif compare_seed == int(run["seed"]):
                        st.info("Pick a DIFFERENT comparison seed to test simulation sensitivity.")
                    else:
                        st.caption("Click Compare simulated hit rates to see numerical differences between the two seeds.")
                report_export = report.copy()
                report_export["simulation_seed"] = int(run["seed"])
                report_export["simulation_trials"] = int(trials)
                st.download_button("Export prop comparison CSV", report_export.to_csv(index=False).encode(),
                                   "team_ai2_nba_props_evaluation.csv", mime="text/csv")
                st.subheader("Correlated parlay research")
                legs = st.multiselect("Select 2–8 actual bet legs", list(report["label"]), max_selections=8)
                combined_odds = st.number_input("Actual combined American odds offered by your sportsbook (0 = unknown)",
                                                value=0, step=10)
                if len(legs) >= 2:
                    indices = [int(report.index[report["label"] == lbl][0]) for lbl in legs]
                    selected_rows = report.iloc[indices]
                    if selected_rows.duplicated(["player_id","market"]).any():
                        st.error("Do not include two sides of the same player/market in the same parlay.")
                    else:
                        ticket = parlay_summary(status, indices)
                        independent = float(selected_rows["win_probability"].prod())
                        a,b,c = st.columns(3)
                        a.metric("All legs win", f"{100*ticket['all_win']:.2f}%")
                        b.metric("Naive independent estimate", f"{100*independent:.2f}%")
                        c.metric("Push without loss", f"{100*ticket['any_push_no_loss']:.2f}%")
                        if ticket["has_push"]:
                            st.warning("At least one leg can push. A push reduces the parlay payout; joint ROI cannot be evaluated correctly from only one combined price.")
                        elif combined_odds:
                            try:
                                roi = expected_net_roi(ticket["all_win"], int(combined_odds))
                                st.metric("Uncalibrated modeled net ROI at offered price", f"{100*roi:+.1f}%")
                                st.caption("These are simulation estimates only. Do not treat a positive result as a validated betting edge.")
                            except ValueError as exc:
                                st.error(str(exc))
                        else:
                            st.info("Enter the actual combined price to estimate ROI; same-game parlay pricing differs from multiplied individual odds.")
                elif legs:
                    st.info("Choose at least two legs to evaluate a parlay.")
        except Exception as exc:
            st.error(f"Could not evaluate these betting lines: {exc}")

    with adjustments_tab:
        st.subheader("Manual availability and playing time")
        st.write("Exclude confirmed OUT players. Adjust minutes for players whose roles change. No unsupported automatic injury redistribution is performed.")
        original = run["original"]
        edit = original[["player_id","player","team","minutes"]].copy()
        edit["Exclude"] = False
        edited = st.data_editor(edit, disabled=["player_id","player","team"],
                                hide_index=True, use_container_width=True,
                                column_config={"minutes": st.column_config.NumberColumn("Expected minutes", min_value=0., max_value=48., step=1.),
                                               "Exclude": st.column_config.CheckboxColumn("Confirmed OUT / exclude")},
                                key=f"nba_injury_{run_key}")
        if st.button("Apply adjustments and re-simulate", use_container_width=True):
            try:
                omit = edited.loc[edited["Exclude"],"player_id"].astype(str).tolist()
                overrides = {str(r.player_id):float(r.minutes) for r in edited.itertuples(index=False)}
                with st.spinner("Rebuilding player distributions with approved playing time assumptions…"):
                    new_proj = build_projections(history,run["teams_map"],target,min_minutes=12,
                                                 excluded=omit,minute_overrides=overrides)
                    new_samples = simulate_stat_lines(new_proj,history,trials=trials,seed=int(run["seed"]))
                run["projections"] = new_proj
                run["samples"] = new_samples
                st.success("Updated projections and simulations. Review the other tabs before considering bets.")
            except Exception as exc:
                st.error(f"Adjustment could not be applied: {exc}")
elif run is not None:
    st.caption("The game, date or simulation settings changed. Run the engine again to use the new selection.")
