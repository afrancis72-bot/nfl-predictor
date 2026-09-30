    st.dataframe(games, use_container_width=True, hide_index=True)
    st.markdown("#### Weekly DraftKings upload")
    up=st.file_uploader("Optional: upload a new DraftKings salary CSV for slate review", type=["csv"])
    if up is not None:
        dk=pd.read_csv(up)
        st.success(f"Loaded {len(dk):,} salary rows. This build keeps the current Week 4 model projections until the weekly model refresh is run.")
        st.dataframe(dk.head(50),use_container_width=True,hide_index=True)

elif view == "Player Projections":
    st.subheader("Player Projections")
    positions=st.multiselect("Position", ["QB","RB","WR","TE","DST"], default=["QB","RB","WR","TE","DST"])
    teams=st.multiselect("Team", sorted(pool.TeamAbbrev.unique()), default=[])
    q=st.text_input("Search player")
    eligible_only=st.toggle("Optimizer-eligible players only", value=True)
    x=pool[pool.Position.isin(positions)].copy()
    if eligible_only: x=x[x["optimizer_eligible"] == True]
    if teams: x=x[x.TeamAbbrev.isin(teams)]
    if q: x=x[x.Name.str.contains(q,case=False,na=False)]
    x["Value/1K"]=x["proj"]/(x["Salary"]/1000)
    show=x.sort_values("proj",ascending=False)[["Name","Position","TeamAbbrev","game","Salary","proj","ceiling","p95_use","ownership_pct","leverage_score","AvgPointsPerGame","optimizer_eligible","Value/1K"]]
    st.dataframe(show,use_container_width=True,hide_index=True,
                 column_config={"proj":st.column_config.NumberColumn("Mean",format="%.2f"),
                                "ceiling":st.column_config.NumberColumn("P90",format="%.2f"),
                                "p95_use":st.column_config.NumberColumn("P95",format="%.2f"),
                                "ownership_pct":st.column_config.NumberColumn("Own %",format="%.1f"),
                                "leverage_score":st.column_config.NumberColumn("Leverage",format="%.2f"),
                                "Value/1K":st.column_config.NumberColumn("Value/1K",format="%.2f")})

elif view == "Simulation":
    st.subheader("50,000-run Correlated Monte Carlo")
    st.caption("Current V8 model includes role-aware correlations, target share, aDOT and red-zone inputs.")
    metric=st.selectbox("Sort by",["mean","p90","p95","boom_25","boom_30"])
    pos=st.multiselect("Position",["QB","RB","WR","TE"],default=["QB","RB","WR","TE"])
    sx=mc[mc.Position.isin(pos)].sort_values(metric,ascending=False)
    st.dataframe(sx,use_container_width=True,hide_index=True)
    st.download_button("Download simulation CSV", mc.to_csv(index=False), "nfl_simulation_v8.csv","text/csv")

elif view == "Lineup Builder":
    st.subheader("DraftKings Portfolio Optimizer")
    a,b,c,d=st.columns(4)
    n_lineups=a.number_input("Lineups",1,150,15,1)
    max_exp=b.slider("Max exposure",0.05,1.0,0.30,0.05)
    min_unique=c.slider("Minimum unique players",1,8,4,1)
    min_salary=d.number_input("Minimum lineup salary",40000,50000,47500,100)
    e,f,g=st.columns(3)
    strategy=e.selectbox("Strategy",["GPP Ceiling","Balanced","Median"])
    own_weight=f.slider("Ownership fade weight",0.0,2.0,0.25,0.05)
    leverage_weight=g.slider("Leverage weight",0.0,2.0,0.35,0.05)

    eligible=pool[pool["optimizer_eligible"] == True].sort_values(["Position","proj"],ascending=[True,False])
    names=eligible["Name"].drop_duplicates().tolist()
    locks=st.multiselect("Lock players",names)
    excludes=st.multiselect("Exclude players",names)

    st.info("Default portfolio rules: 15 lineups • 30% max exposure • 4 minimum unique players • $47,500 salary floor • QB + pass-catcher stack • no offensive player against selected DST.")

    if st.button("Generate portfolio",type="primary"):
        with st.spinner("Generating diversified portfolio..."):
            lineups, expo, max_count=build_portfolio(
                pool,int(n_lineups),float(max_exp),int(min_unique),int(min_salary),50000,
                strategy,float(own_weight),float(leverage_weight),locks,excludes
            )
        if not lineups:
            st.error("No valid portfolio found. Relax locks/exclusions, uniqueness, exposure, or salary floor.")
        else:
            st.session_state["nfl_lineups"]=lineups
            st.success(f"Generated {len(lineups)} of {int(n_lineups)} requested lineups.")
            ldf=lineups_to_df(lineups)
            edf=exposure_df(lineups)
            st.markdown("#### Lineups")
            st.dataframe(ldf,use_container_width=True,hide_index=True)
            st.download_button("Download lineups",ldf.to_csv(index=False),"nfl_lineups.csv","text/csv")
            st.markdown("#### Exposure")
            st.dataframe(edf,use_container_width=True,hide_index=True)
            st.download_button("Download exposure",edf.to_csv(index=False),"nfl_exposure.csv","text/csv")

elif view == "Portfolio Analysis":
    st.subheader("Portfolio Analysis")
    lineups=st.session_state.get("nfl_lineups")
    if not lineups:
        st.info("Generate a portfolio in Lineup Builder first.")
    else:
        ldf=lineups_to_df(lineups)
        edf=exposure_df(lineups)
        sets=[set(x.Name) for x in lineups]
        max_overlap=0
        for i in range(len(sets)):
            for j in range(i+1,len(sets)):
                max_overlap=max(max_overlap,len(sets[i]&sets[j]))
        min_unique_actual=9-max_overlap if len(sets)>1 else 9
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Lineups",len(lineups))
        c2.metric("Max exposure",f"{edf['Exposure %'].max():.1f}%")
        c3.metric("Min pairwise unique",min_unique_actual)
        c4.metric("Salary range",f"${ldf.Salary.min():,}–${ldf.Salary.max():,}")
        st.markdown("#### Exposure")
        st.dataframe(edf,use_container_width=True,hide_index=True)
        st.markdown("#### Game exposure")
        rows=[]
        for i,l in enumerate(lineups,1):
            for game,n in l.groupby("game").size().items():
                rows.append({"Lineup":i,"Game":game,"Players":int(n)})
        gx=pd.DataFrame(rows)
        summary=gx.groupby("Game",as_index=False).agg(Lineups=("Lineup","nunique"),Max_players_in_one_lineup=("Players","max"))
        summary["Lineup exposure %"]=100*summary["Lineups"]/len(lineups)
        st.dataframe(summary.sort_values("Lineup exposure %",ascending=False),use_container_width=True,hide_index=True)
