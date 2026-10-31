"""Streamlit PGA salary-band review embedded in the ACTUAL four-sport workspace."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from ai2_pga_salary_review import (
    TIERS,build_review,attach_comparison,rebuild_reviewed
)
from ai2_pga_oneclick import dk_roster_ids


def render_review_page(model, *, signature, lineup_count, max_exposure,
                       salary_floor, min_unique, strategy,
                       locks=(), base_exclusions=()):
    st.subheader("🔎 PGA Salary-Tier Recommendations & Veto Review")
    st.caption("An evidence-based analyst board for $6K–$10K+ golfers, before you approve entries. "
               "Uses the tournament already simulated in Tournament DFS; no additional salary upload.")
    if model is None:
        st.info("Start with **Tournament DFS · One Click** and run the tournament model. "
                "Then return here to review golfer recommendations before entering.")
        return

    if st.session_state.get("ai2_pga_review_event_id") != signature:
        st.session_state["ai2_pga_review_event_id"] = signature
        st.session_state["ai2_pga_review_decisions"] = {}
        st.session_state.pop("ai2_pga_review_portfolio",None)
        for tier in TIERS:
            st.session_state.pop(f"pga_review_editor_{tier}",None)

    metadata=model.get("metadata",{})
    ranking=build_review(model["projections"],metadata)
    if ranking.empty:
        st.warning("The current simulated field has no qualifying golfer salary bands.")
        return
    meta_src=str(metadata.get("source_mode","unverified golfer skill"))
    st.write(f"**{metadata.get('tournament','Current event')}** | "
             f"{len(ranking)} ranked golfers | {metadata.get('simulations',0):,} simulations "
             f"| seed {metadata.get('seed','unknown')}")
    if not metadata.get("independent_sg_match",0) and not metadata.get("otis_match",0):
        st.error("LOW CONFIDENCE: this build has no measured golfer skill and was based on DraftKings salary. "
                 "Do not treat these rankings as an independently researched edge.")
    elif not metadata.get("independent_sg_match",0):
        st.warning("Golfer performance uses OTIS research as a provisional model input. "
                   "No separate measured SG bank was supplied.")
    else:
        st.info("The ranking uses this run's simulated fantasy outcomes and actual measured stats "
                "where provided. Course effects and cut probabilities are not historically calibrated.")
    st.caption("Why these picks: 35% relative mean DK per $1,000 salary, "
               "35% relative P90 per $1,000 and 30% modeled make-cut chance, "
               "scored separately within each salary tier. These weights rank REVIEW PRIORITY, "
               "not new scoring probabilities.")
    tier=st.radio("DraftKings salary band",TIERS,horizontal=True,key="ai2_pga_salary_band")
    show_all=st.checkbox("Show entire salary band",key="ai2_pga_show_all_salary")
    displayed=ranking[ranking.tier.eq(tier)].copy()
    if not show_all: displayed=displayed.head(8)
    decisions=st.session_state.setdefault("ai2_pga_review_decisions",{})
    if displayed.empty:
        st.info("No golfers in this salary range.")
    else:
        displayed["Decision"]=displayed.player.map(decisions).fillna("Review")
        shown=["tier_rank","player","salary","mean_dk","p90_dk",
               "cut_pct","mean_per_1k","why","concerns","Decision"]
        revised=st.data_editor(
            displayed[shown],key=f"pga_review_editor_{tier}",hide_index=True,
            use_container_width=True,num_rows="fixed",
            disabled=[k for k in shown if k!="Decision"],
            column_config={
                "tier_rank":"Rank","player":"Golfer",
                "salary":st.column_config.NumberColumn("Salary",format="$%d"),
                "mean_dk":st.column_config.NumberColumn("Mean DK",format="%.1f"),
                "p90_dk":st.column_config.NumberColumn("P90 DK",format="%.1f"),
                "cut_pct":st.column_config.NumberColumn("Make cut %",format="%.1f"),
                "mean_per_1k":st.column_config.NumberColumn("Mean/$1K",format="%.1f"),
                "why":st.column_config.TextColumn("Why the model likes him",width="large"),
                "concerns":st.column_config.TextColumn("Concerns / veto considerations",width="large"),
                "Decision":st.column_config.SelectboxColumn(
                    "Your decision",options=["Review","Approve","Veto"],required=True)
            },
        )
        changed=False
        for row in revised.itertuples(index=False):
            if row.Decision in ("Review","Approve","Veto") and decisions.get(row.player,"Review")!=row.Decision:
                decisions[row.player]=row.Decision
                changed=True
        if changed:
            st.session_state.pop("ai2_pga_review_portfolio",None)

        pick=displayed.iloc[0]
        st.markdown(f"**Top model choice in {tier}: {pick.player}**")
        st.write("Evidence: "+pick["why"])
        st.write("Challenge: "+pick["concerns"])

    vetoes={name for name,decision in decisions.items() if decision=="Veto"}
    approvals={name for name,decision in decisions.items() if decision=="Approve"}
    st.write(f"**{len(approvals)} approved**, **{len(vetoes)} vetoed** this tournament. "
             "Approving means the golfer is ALLOWED, not forced into a lineup. "
             "Unreviewed golfers remain eligible. Vetoed golfers cannot enter a reviewed build.")
    if vetoes: st.caption("Your veto list: "+", ".join(sorted(vetoes)))
    st.download_button("Download all $6K–$10K+ rankings with evidence",
                       ranking.to_csv(index=False).encode("utf-8-sig"),
                       "team_ai2_pga_salary_tier_ranking.csv",mime="text/csv")
    with st.expander("Model differences versus OTIS and RunGood (comparison only)"):
        compare=attach_comparison(ranking,model.get("otis"),model.get("course_fit"))
        st.caption("An external course-fit rank is not a DraftKings-points rank; "
                   "disagreement calls for review, not automatic model adjustment.")
        columns=[c for c in ("player","salary","overall_mean_rank",
                      "OTIS rank","OTIS rank minus our mean DK rank",
                      "RRG course-fit rank","RRG course-fit rank minus our mean DK rank")
                 if c in compare]
        st.dataframe(compare[columns].sort_values("overall_mean_rank"),
                     hide_index=True,use_container_width=True)

    st.divider()
    st.subheader("Rebuild using your vetoes")
    excluded=set(base_exclusions)|vetoes
    locked=set(locks)
    if locked & excluded:
        st.error("You have locked and vetoed the same golfer. Remove the lock or veto before building.")
        return
    review_settings=(signature,tuple(sorted(excluded)),tuple(sorted(locked)),
                     lineup_count,max_exposure,salary_floor,min_unique,strategy)
    st.caption(f"Original tournament simulation and seed preserved. "
               f"{len(excluded)} golfers excluded, "
               f"{max_exposure:.0%} exposure limit, {lineup_count} lineups. "
               "A full re-simulation is not necessary to apply player vetoes.")
    if st.button("⛳ Build reviewed PGA portfolio",type="primary",
                 use_container_width=True,key="ai2_pga_rebuild_review"):
        try:
            with st.spinner("Building only legal lineups using approved player pool..."):
                rebuilt=rebuild_reviewed(
                    model,vetoes=excluded,locks=locked,
                    lineup_count=lineup_count,max_exposure=max_exposure,
                    salary_floor=salary_floor,min_unique=min_unique,
                    strategy=strategy)
            rebuilt["signature"]=review_settings
            st.session_state["ai2_pga_review_portfolio"]=rebuilt
            st.success(f"Built {len(rebuilt['summary'])} reviewed lineups. "
                       "The original lineup portfolio is unchanged.")
        except Exception as exc:
            st.session_state.pop("ai2_pga_review_portfolio",None)
            st.error("Reviewed lineup build blocked: "+str(exc))
    rebuilt=st.session_state.get("ai2_pga_review_portfolio")
    if not rebuilt: return
    if rebuilt.get("signature")!=review_settings:
        st.warning("Review settings changed. Rebuild before downloading new lineups.")
        return
    st.dataframe(rebuilt["summary"],use_container_width=True,hide_index=True)
    st.dataframe(rebuilt["exposure"],use_container_width=True,hide_index=True)
    st.download_button("Download REVIEWED lineups",
                       rebuilt["lineups"].to_csv(index=False).encode("utf-8-sig"),
                       "team_ai2_pga_reviewed_lineups.csv",mime="text/csv")
    try:
        ids=dk_roster_ids(rebuilt["lineups"],model["projections"])
        st.download_button("Download reviewed DK golfer IDs",ids,
                           "team_ai2_pga_reviewed_dk_rosters.csv",mime="text/csv")
    except ValueError as exc:
        st.caption("DK ID export unavailable: "+str(exc))
