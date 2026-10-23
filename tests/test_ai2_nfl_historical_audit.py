"""Tests for multi-slate historical distribution evaluation."""
import unittest
import numpy as np
import pandas as pd
from ai2_nfl_historical_audit import audit_historical_slates


def slate(slate_id, actual=15., frozen=True):
    return dict(slate_id=slate_id, frozen_before_kickoff=frozen,
                players=pd.DataFrame({"ID": ["1"], "Name": ["Receiver"],
                                      "Position": ["WR"], "slate_date": ["2026-10-18"]}),
                draws=np.linspace(0, 30, 200).reshape(-1, 1),
                actuals=pd.DataFrame({"ID": ["1"], "actual_dk": [actual],
                                      "slate_date": ["2026-10-18"]}))


class HistoricalAuditTests(unittest.TestCase):
    def test_aggregates_two_independent_slates(self):
        per_slate, players, overall = audit_historical_slates([slate("week6"), slate("week7")])
        self.assertEqual(len(per_slate), 2)
        self.assertEqual(len(players), 2)
        self.assertEqual(overall["slates"], 2)
        self.assertAlmostEqual(overall["mean_absolute_error"], 0)

    def test_refuses_unfrozen_predictions(self):
        with self.assertRaises(ValueError):
            audit_historical_slates([slate("week6", frozen=False)])

    def test_refuses_duplicate_slates(self):
        with self.assertRaises(ValueError):
            audit_historical_slates([slate("week6"), slate("week6")])

    def test_refuses_missing_observations(self):
        bad = slate("week6")
        bad["actuals"] = bad["actuals"].iloc[0:0]
        with self.assertRaises(ValueError):
            audit_historical_slates([bad])


if __name__ == "__main__":
    unittest.main()
