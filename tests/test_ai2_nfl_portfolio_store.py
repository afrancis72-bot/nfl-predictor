"""Portfolio restore must be scoped to exact saved slate contents."""
import tempfile
import unittest
from ai2_nfl_portfolio_store import slate_fingerprint, save_portfolio, load_latest_portfolio

class PortfolioStoreTests(unittest.TestCase):
    def test_restores_same_slate_not_replaced_slate(self):
        a = ("salary.csv", b"ID,Salary\n1,5000", "time")
        b = ("proj.csv", b"ID,proj\n1,12", "time")
        changed = ("proj.csv", b"ID,proj\n1,14", "time")
        old_key = slate_fingerprint(a, b)
        new_key = slate_fingerprint(a, changed)
        self.assertNotEqual(old_key, new_key)
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(load_latest_portfolio(directory, old_key))
            save_portfolio(directory, old_key, [{"players": [{"ID": "1"}], "salary": 5000}])
            self.assertEqual(load_latest_portfolio(directory, old_key)["candidates"][0]["salary"], 5000)
            self.assertIsNone(load_latest_portfolio(directory, new_key))
    def test_requires_inputs(self):
        with self.assertRaises(ValueError):
            slate_fingerprint(None, None)

if __name__ == "__main__":
    unittest.main()
