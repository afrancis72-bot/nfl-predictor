"""Team AI² Pro NBA — automatic slate research + actual sportsbook prop analysis."""
from __future__ import annotations
from datetime import datetime, date
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo
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
)
from nba_engine.props_sim import expected_net_roi


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

run_key = f"{target}-{schedule_key}-{trials}"
if st.button("▶ Run NBA Prediction Engine", type="primary", use_container_width=True):
    if len(teams_map) != len(set(teams_map.values())) and len(teams_map) != 2:
        st.error("Please select a valid slate with one opponent per team.")
    elif len(teams_map) == 1:
        st.error("Choose two different teams.")
    else:
        try:
            with st.spinner("Loading recent NBA box scores, projecting minutes and simulating correlated stat lines…"):
                history, provenance = cached_history(target.isoformat())
                projections = build_projections(history, teams_map, target, min_minutes=12)
                samples = simulate_stat_lines(projections, history, trials=trials, seed=20261029)
            st.session_state["nba_auto_result"] = {
                "run_key": run_key, "history":history, "provenance":provenance,
                "original":projections, "projections":projections,
                "samples":samples, "teams_map":teams_map,
            }
            st.session_state.pop("nba_auto_bets", None)
            st.success(f"Engine complete: {len(projections)} player projections and {trials:,} simulations.")
        except Exception as exc:
            st.error(f"NBA data or simulation failed: {exc}")
            st.info("The public data feeds may be unavailable. No synthetic statistics will be substituted.")

run = st.session_state.get("nba_auto_result")
if run is not None and run["run_key"] == run_key:
    projections = run["projections"]
    history = run["history"]
    provenance = run["provenance"]
    samples = run["samples"]
    st.caption(f"Player data as of {provenance['snapshot']} | Latest observed game {provenance['latest_game_date']} | "
               f"{provenance['rows']:,} historical player-game records | {'/'.join(provenance['seasons'])}")
    age_days = (target - date.fromisoformat(provenance["latest_game_date"])).days
    if age_days > 4:
        st.warning(f"Historical data is {age_days} days old relative to the target game. Confirm roster moves and current playing time.")

    projection_tab, props_tab, adjustments_tab = st.tabs(["📊 Player projections", "🎯 Props & parlays", "⚙️ Injuries / minutes"])
    with projection_tab:
        st.subheader("Player statistical predictions")
        st.write("Sort and filter player projections; these are estimates, not sportsbook offers.")
        display = projections[["player","team","opponent","minutes","Points","Rebounds","Assists",
                               "Threes","PRA","PR","PA","RA","current_season_games","sample_flag"]].copy()
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.download_button("Export player projections (CSV)", projections.to_csv(index=False).encode(),
                           "team_ai2_nba_projections.csv", mime="text/csv")
        st.caption("Values represent projected means. Limited = fewer than three current-season games or fewer than ten observed prior games.")

    with props_tab:
        st.subheader("Enter real sportsbook prop lines")
        st.caption("The app never invents a line or betting price. Type offered lines into the grid, or upload a CSV.")
        names = {f"{p.player} ({p.team})": str(p.player_id) for p in projections.itertuples(index=False)}
        defaults = list(names)[:12]
        chosen = st.multiselect("Players to compare", list(names), default=defaults)
        market_choices = st.multiselect("Markets", list(MARKETS),
                                        default=["Points","Rebounds","Assists","Threes","PRA"])
        upload = st.file_uploader("Optional: upload sportsbook lines CSV (player_id, market, line, side, odds)",
                                  type=["csv"], key=f"nba_market_{schedule_key}_{target}")
        if upload is not None:
            bet_grid = pd.read_csv(BytesIO(upload.getvalue()))
            st.caption("Using the sportsbook-line CSV you supplied.")
        else:
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
                key=f"nba_lines_{run_key}",
            )
        st.download_button("Download importable empty lines template",
                           b"player_id,market,line,side,odds\n", "nba_real_lines_template.csv", mime="text/csv")
        try:
            bets = prepare_bet_lines(bet_grid, projections)
            if bets.empty:
                st.info("Fill in at least one actual sportsbook line to see probabilities.")
            else:
                report,status = evaluate_bets(bets,samples)
                st.subheader("Modeled prop outcomes")
                shown = report[["label","sim_mean","win_probability","push_probability",
                                "odds","book_implied","model_edge"]].copy()
                for c in ["win_probability","push_probability","book_implied","model_edge"]:
                    shown[c] = (100 * shown[c]).round(1)
                shown["sim_mean"] = shown["sim_mean"].round(2)
                st.dataframe(shown, hide_index=True, use_container_width=True)
                st.caption("Win, push, and edge are percentages. Sportsbook implied probabilities include bookmaker margin. An apparent edge is NOT evidence of profitability until calibrated.")
                st.download_button("Export prop comparison CSV", report.to_csv(index=False).encode(),
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
                    new_samples = simulate_stat_lines(new_proj,history,trials=trials,seed=20261029)
                run["projections"] = new_proj
                run["samples"] = new_samples
                st.success("Updated projections and simulations. Review the other tabs before considering bets.")
            except Exception as exc:
                st.error(f"Adjustment could not be applied: {exc}")
elif run is not None:
    st.caption("The game, date or simulation settings changed. Run the engine again to use the new selection.")
