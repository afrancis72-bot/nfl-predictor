"""Team AI² Pro NBA research tab — first pregame props baseline.

The user may enter historical per-36 rates (from a verified source) and
expected minutes. This page does not invent player statistics, injury
adjustments, sportsbook lines, or odds.
"""
from pathlib import Path
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sports.nba.prop_model import PlayerRates, project_props

st.title("🏀 NBA · Player Props Lab")
st.caption("Team AI² Pro | Research-only foundation — not yet a calibrated betting model")
st.info(
    "Start with your own verified player rates. The tool creates a pregame "
    "statistical baseline without referencing sportsbook lines. Automated "
    "NBA data, injury redistribution and possession simulations are still in development."
)
with st.form("nba_baseline"):
    a, b = st.columns(2)
    player = a.text_input("Player", value="")
    team = b.text_input("Team", value="")
    games = a.number_input("Historical games sampled", min_value=1, max_value=82, value=10)
    past_mpg = b.number_input("Historical minutes per game", min_value=1., max_value=48., value=32., step=.5)
    minutes = st.number_input("Expected minutes tonight", min_value=0., max_value=48., value=34., step=.5)
    st.caption("Enter historical per-36-minute rates (not total per-game statistics).")
    c1, c2, c3, c4 = st.columns(4)
    points = c1.number_input("Points / 36", min_value=0., value=24., step=.5)
    rebounds = c2.number_input("Rebounds / 36", min_value=0., value=7., step=.5)
    assists = c3.number_input("Assists / 36", min_value=0., value=6., step=.5)
    threes = c4.number_input("Made 3s / 36", min_value=0., value=3., step=.5)
    submit = st.form_submit_button("Calculate independent baseline")
if submit:
    if not player.strip() or not team.strip():
        st.error("Enter the player's name and team.")
    else:
        stats = project_props(PlayerRates(
            player=player.strip(), team=team.strip(),
            minutes_per_game=past_mpg, points_per_36=points,
            rebounds_per_36=rebounds, assists_per_36=assists,
            threes_per_36=threes, sample_games=games,
        ), expected_minutes=minutes)
        st.subheader(f"Pregame baseline — {stats['player']}")
        cols = st.columns(5)
        for col, label, field in zip(
            cols, ["PTS", "REB", "AST", "3PM", "PRA"],
            ["points", "rebounds", "assists", "threes", "points_rebounds_assists"]
        ):
            col.metric(label, f"{stats[field]:.1f}")
        st.caption(stats["method"] + ". " + stats["limitations"])
