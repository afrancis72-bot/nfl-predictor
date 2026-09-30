"""NFL Predictor Pro portfolio utilities.
The Streamlit app contains the interactive optimizer; this module provides
portable QC helpers for downloaded portfolios.
"""
import pandas as pd

def portfolio_qc(lineups_csv):
    df = pd.read_csv(lineups_csv) if not isinstance(lineups_csv, pd.DataFrame) else lineups_csv.copy()
    if "Players" not in df.columns:
        raise ValueError("Expected a Players column.")
    sets=[set(str(x).split(" | ")) for x in df["Players"]]
    max_overlap=0
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            max_overlap=max(max_overlap,len(sets[i]&sets[j]))
    return {
        "lineups":len(df),
        "min_salary":int(df["Salary"].min()),
        "max_salary":int(df["Salary"].max()),
        "max_pairwise_overlap":max_overlap,
        "min_pairwise_unique":9-max_overlap if len(df)>1 else 9,
    }
