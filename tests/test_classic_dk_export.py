import pandas as pd
import pytest
from classic_dk_export import export_classic_ids,fill_classic_reserved_entries

def sample():
    slots=["QB","RB","RB","WR","WR","WR","TE","FLEX","DST"]
    positions=["QB","RB","RB","WR","WR","WR","TE","WR","DST"]
    ids=list(range(101,110))
    picks=pd.DataFrame({"lineup":[1]*9,"slot":slots,"ID":ids})
    salary=pd.DataFrame({"ID":ids,"Position":positions,"Salary":[5500]*9,"TeamAbbrev":["SEA"]*9})
    return picks,salary

def test_export_and_preserve_existing_entry():
    picks,salary=sample()
    ids=export_classic_ids(picks,salary)
    assert ids.loc[0,"RB2"]=="103"
    assert ids.loc[0,"Salary"]==49500
    header="Entry ID,Contest Name,QB,RB,RB,WR,WR,WR,TE,FLEX,DST"
    original="555,Contest,901,902,903,904,905,906,907,908,909"
    blank="556,Contest,,,,,,,,,"
    csv=header+"\\n"+original+"\\n"+blank+"\\n"
    result=fill_classic_reserved_entries(csv,ids)
    assert original in result
    assert "556,Contest,101,102,103,104,105,106,107,108,109" in result

def test_wrong_position_rejected():
    picks,salary=sample()
    salary.loc[salary.ID==109,"Position"]="RB"
    with pytest.raises(ValueError,match="DST"):
        export_classic_ids(picks,salary)

def test_missing_reserved_rows_rejected():
    picks,salary=sample()
    ids=export_classic_ids(picks,salary)
    with pytest.raises(ValueError,match="Insufficient"):
        fill_classic_reserved_entries("Entry ID,QB,RB,RB,WR,WR,WR,TE,FLEX,DST\\n",ids)
