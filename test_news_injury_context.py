import unittest
from news_injury_context import merge_confirmed_out

class InjuryContextTests(unittest.TestCase):
    def test_duplicate_alias_is_one_player(self):
        old=[{"team":"LA","player":"Example Receiver"}]
        new=[{"team":"LAR","player":"Example Receiver"}]
        result=merge_confirmed_out(old,new)
        self.assertEqual(result,[{"team":"LAR","player":"Example Receiver"}])
    def test_switch_does_not_include_auto_when_off(self):
        result=merge_confirmed_out([], [{"team":"NYJ","player":"Example Back"}],enabled=False)
        self.assertEqual(result,[])
    def test_team_mismatch_not_silently_merged(self):
        result=merge_confirmed_out([{"team":"NYJ","player":"Example Back"}],
                                  [{"team":"NYG","player":"Example Back"}])
        self.assertEqual(len(result),2)
    def test_stable_order(self):
        result=merge_confirmed_out([{"team":"TB","player":"B Player"},
                                    {"team":"ATL","player":"A Player"}],[])
        self.assertEqual([p["team"] for p in result],["ATL","TB"])
if __name__=="__main__":
    unittest.main()
