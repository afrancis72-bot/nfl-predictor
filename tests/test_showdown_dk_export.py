"""Regression checks for Team AI² Showdown DK exports."""
import io
import unittest
import pandas as pd
from showdown_dk_export import export_showdown_ids, fill_reserved_entries

class ShowdownExportTests(unittest.TestCase):
    def setUp(self):
        players=[("Alpha","HOU"),("Bravo","HOU"),("Charlie","HOU"),("Delta","JAX"),("Echo","JAX"),("Foxtrot","JAX")]
        rows=[]
        for i,(name,team) in enumerate(players):
            rows.extend([
                {"Name":name,"TeamAbbrev":team,"ID":str(100+i),"Salary":12000,"Roster Position":"CPT","Status":""},
                {"Name":name,"TeamAbbrev":team,"ID":str(200+i),"Salary":8000,"Roster Position":"FLEX","Status":""},
            ])
        self.salary=pd.DataFrame(rows)
        self.lineups=pd.DataFrame([{"CPT":"Alpha","FLEX1":"Bravo","FLEX2":"Charlie","FLEX3":"Delta","FLEX4":"Echo","FLEX5":"Foxtrot"}])

    def test_salary_cap(self):
        with self.assertRaisesRegex(ValueError,"exceeds"):
            export_showdown_ids(self.lineups,self.salary)
    def test_block_out(self):
        self.salary.loc[self.salary.Name=="Alpha","Status"]="OUT"
        with self.assertRaisesRegex(ValueError,"blocked DK status"):
            export_showdown_ids(self.lineups,self.salary)
    def test_template_preserves_manual(self):
        self.salary.loc[self.salary["Roster Position"]=="FLEX","Salary"]=7000
        self.salary.loc[self.salary["Roster Position"]=="CPT","Salary"]=10500
        ids=export_showdown_ids(self.lineups,self.salary)
        template="Entry ID,Contest Name,Contest ID,Entry Fee,CPT,FLEX,FLEX,FLEX,FLEX,FLEX,,Instructions\\n1,Game,12,0.5,999,998,997,996,995,994,,\\n2,Game,12,0.5,,,,,,,,\\n"
        out=fill_reserved_entries(template,ids)
        rows=list(__import__("csv").reader(io.StringIO(out)))
        self.assertEqual(rows[1][4:10],["999","998","997","996","995","994"])
        self.assertEqual(rows[2][4:10],["100","201","202","203","204","205"])
    def test_ambiguous_team(self):
        duplicate=self.salary.iloc[[0]].copy()
        duplicate["TeamAbbrev"]="JAX"
        with self.assertRaisesRegex(ValueError,"multiple teams"):
            export_showdown_ids(self.lineups,pd.concat([self.salary,duplicate],ignore_index=True))
if __name__=="__main__": unittest.main()
