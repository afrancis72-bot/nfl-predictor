"""Regression tests for the experimental classic slate input contract."""
import pandas as pd
import pytest
from classic_slate_research import prepare_classic_slate, matchup_pool

def sample():
    return pd.DataFrame([
        {"Name":"Starter A","ID":101,"Position":"QB","Roster Position":"QB","Salary":6200,"Game Info":"CAR@PHI 10/18/2026 01:00PM ET","TeamAbbrev":"CAR"},
        {"Name":"Starter B","ID":102,"Position":"QB","Roster Position":"QB","Salary":4000,"Game Info":"CAR@PHI 10/18/2026 01:00PM ET","TeamAbbrev":"PHI"},
        {"Name":"Receiver A","ID":103,"Position":"WR","Roster Position":"WR/FLEX","Salary":5300,"Game Info":"CAR@PHI 10/18/2026 01:00PM ET","TeamAbbrev":"CAR"},
        {"Name":"Defense B","ID":104,"Position":"DST","Roster Position":"DST","Salary":2800,"Game Info":"CAR@PHI 10/18/2026 01:00PM ET","TeamAbbrev":"PHI"},
    ])

def test_matchup_and_manual_exclusions():
    eligible, offense, audit, excluded = prepare_classic_slate(sample(), {"Starter B"})
    assert len(eligible)==3
    assert excluded["Name"].tolist()==["Starter B"]
    assert audit.iloc[0]["matchup"]=="CAR@PHI"
    assert set(matchup_pool(offense,"CAR@PHI")["Name"])=={"Starter A","Receiver A"}
    assert "Defense B" not in set(offense["Name"])

def test_duplicate_id_rejected():
    data=sample()
    data.loc[1,"ID"]=101
    with pytest.raises(ValueError,match="Duplicate"):
        prepare_classic_slate(data)

def test_wrong_team_rejected():
    data=sample()
    data.loc[0,"TeamAbbrev"]="NYJ"
    with pytest.raises(ValueError,match="Team mismatch"):
        prepare_classic_slate(data)

def test_malformed_game_rejected():
    data=sample()
    data.loc[0,"Game Info"]="unparseable"
    with pytest.raises(ValueError,match="Unrecognized"):
        prepare_classic_slate(data)
