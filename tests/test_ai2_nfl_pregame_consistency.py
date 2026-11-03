"""Pregame consistency must distinguish repeatable value from one-game spikes."""
import unittest
import pandas as pd

from ai2_nfl_pregame_consistency import (
    build_pregame_role_evidence, historical_value_replay,
)


def fixture():
    rows = []
    # Corum-like backup: five successive meaningful games.
    for week in range(1, 7):
        rows.append(dict(season=2026, week=week, player_name="Steady Backup",
                         recent_team="LAR", position="RB",
                         carries=9, targets=3, rushing_yards=53,
                         rushing_tds=0, receptions=3,
                         receiving_yards=45, receiving_tds=0))
    # Earlier lead rusher loses opportunity in the two games before the
    # test slate; this is observable, but NOT proof he was injured.
    for week in range(1, 6):
        carries = 18 if week <= 3 else 3
        rows.append(dict(season=2026, week=week,
                         player_name="Competing Lead Runner", recent_team="LAR",
                         position="RB", carries=carries, targets=2,
                         rushing_yards=carries * 4, rushing_tds=0,
                         receptions=1, receiving_yards=5, receiving_tds=0))
    # Singletary-like four-game role: one major increase, little else.
    for week, carries, yards in zip(range(2, 6), [2, 3, 17, 2], [5, 12, 55, 8]):
        rows.append(dict(season=2026, week=week, player_name="One Spike",
                         recent_team="NYG", position="RB",
                         carries=carries, targets=1, rushing_yards=yards,
                         rushing_tds=0, receptions=1,
                         receiving_yards=4, receiving_tds=0))
    # Hutchinson-like wideout: two target spikes but no repeated 5+ role.
    for week, targets in zip(range(1, 6), [3, 4, 9, 2, 5]):
        rows.append(dict(season=2026, week=week, player_name="Uneven Receiver",
                         recent_team="HOU", position="WR",
                         carries=0, targets=targets, rushing_yards=0,
                         rushing_tds=0, receptions=min(targets, 4),
                         receiving_yards=targets * 5,
                         receiving_tds=0))
    # Existing featured player can remain a regular-price candidate, but
    # does not count as a validated cheap value.
    salary = [
        dict(ID="c", Name="Steady Backup", TeamAbbrev="LAR",
             Position="RB", Salary=5100),
        dict(ID="s", Name="One Spike", TeamAbbrev="NYG",
             Position="RB", Salary=4600),
        dict(ID="h", Name="Uneven Receiver", TeamAbbrev="HOU",
             Position="WR", Salary=4000),
        dict(ID="x", Name="Unknown Value", TeamAbbrev="DET",
             Position="WR", Salary=4000),
        dict(ID="p", Name="Full-Price Player", TeamAbbrev="LAR",
             Position="RB", Salary=7000),
    ]
    return pd.DataFrame(rows), salary


class PregameConsistencyTests(unittest.TestCase):
    def test_repeatable_backup_passes_spikes_do_not(self):
        history, salary = fixture()
        results = build_pregame_role_evidence(history, salary, 2026, 6)
        self.assertEqual(results["c"]["Role gate"], "PASS")
        self.assertEqual(results["s"]["Role gate"], "BLOCK")
        self.assertEqual(results["h"]["Role gate"], "BLOCK")
        self.assertEqual(results["x"]["Role gate"], "BLOCK")
        self.assertEqual(results["p"]["Role gate"], "NOT_LOW_SALARY")
        self.assertGreaterEqual(results["c"]["Meaningful opportunities"], 4)
        self.assertIn("Competing Lead Runner: DECLINING OBSERVED ROLE",
                      results["c"]["Same-position teammate pregame trends"])

    def test_does_not_use_target_week_outcomes_to_establish_prior_role(self):
        history, salary = fixture()
        original = build_pregame_role_evidence(history, salary, 2026, 6)
        changed = history.copy()
        mask = changed["week"].eq(6) & changed["player_name"].eq("Steady Backup")
        changed.loc[mask, "rushing_yards"] = 500
        changed.loc[mask, "carries"] = 40
        replay = build_pregame_role_evidence(changed, salary, 2026, 6)
        self.assertEqual(original, replay)

    def test_historical_replay_includes_failures_not_just_winners(self):
        history, salary = fixture()
        result = historical_value_replay(history, salary, 2026, 6)
        byname = {r["Player"]:r for r in result}
        self.assertIn("Steady Backup", byname)
        self.assertIn("One Spike", byname)
        self.assertEqual(byname["Steady Backup"]["Pregame role gate"], "PASS")
        self.assertTrue(byname["Steady Backup"]["Outcome available"])
        self.assertFalse(byname["One Spike"]["Outcome available"])


if __name__ == "__main__":
    unittest.main()
