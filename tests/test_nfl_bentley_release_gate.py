"""Fail-closed acceptance for NFL Classic UI, without touching other sports."""
import pathlib
import unittest

from ai2_nfl_release_gate import (
    nfl_classic_release_status,
    assert_nfl_classic_ready_for_entries,
    CONTRACT_ID, REQUIREMENTS, STATUS_NOT_READY,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


class NFLBentleyReleaseTests(unittest.TestCase):
    def test_all_unvalidated_requirements_block_entry(self):
        status=nfl_classic_release_status()
        self.assertEqual(status["contract_id"],CONTRACT_ID)
        self.assertEqual(status["status"],STATUS_NOT_READY)
        self.assertFalse(status["ready_for_entries"])
        self.assertTrue(status["must_not_build_lineups"])
        self.assertEqual(status["checks_total"],9)
        self.assertEqual(status["checks_passed"],0)
        self.assertEqual(len(status["blockers"]),9)
        ids={row["id"] for row in status["blockers"]}
        self.assertIn("ROLE_SPECIFIC_DEFENSE",ids)
        self.assertIn("POSSESSION_WORLD",ids)
        self.assertIn("STAT_OUTCOME_CALIBRATION",ids)
        self.assertIn("FULL_REGRESSION_AND_DEPLOYMENT",ids)
        for row in REQUIREMENTS:
            self.assertTrue(row["criterion"])
            self.assertTrue(row["pass_proof"])
            self.assertTrue(row["evidence"])
            self.assertNotEqual(row["status"],"PASS")

    def test_server_side_release_guard_rejects_any_new_entry_build(self):
        with self.assertRaisesRegex(ValueError,
                                    "NOT READY FOR ENTRY.*ROLE_SPECIFIC_DEFENSE"):
            assert_nfl_classic_ready_for_entries()

    def test_same_unified_nfl_ui_disables_button_and_stale_results(self):
        source=(ROOT/"unified_app.py").read_text()
        self.assertIn('if sport == "NFL":',source)
        self.assertIn('release_gate=nfl_classic_release_status()',source)
        self.assertIn('disabled=not release_ready',source)
        self.assertIn('assert_nfl_classic_ready_for_entries()',source)
        self.assertIn('if release_ready else None',source)
        self.assertIn('st.session_state.get("ai2_oneclick_result")',source)
        self.assertIn('with st.expander("Protected approved lineups and previous entries")',source)
        self.assertIn('if sport == "PGA":',source)
        self.assertIn('if sport == "NBA":',source)
        self.assertIn('if sport == "NASCAR":',source)
        self.assertIn('NFL CLASSIC — NOT READY FOR ENTRY',source)
        self.assertIn('release_gate["blockers"]',source)

    def test_showdown_is_not_accidentally_approved_with_classic(self):
        report=nfl_classic_release_status()
        self.assertIn("SECONDARY",report["showdown_status"])
        self.assertFalse(report["ready_for_entries"])
        self.assertTrue(report["user_final_approval_required_even_if_ready"])

    def test_fail_closed_contract_has_no_single_user_toggle(self):
        source=(ROOT/"ai2_nfl_release_gate.py").read_text()
        self.assertNotIn("os.environ",source)
        self.assertNotIn("st.session_state",source)
        self.assertNotIn("force_ready",source)
        self.assertNotIn("override_ready",source)
        self.assertIn('x["status"] != "PASS"',source)


if __name__=="__main__":
    unittest.main()
