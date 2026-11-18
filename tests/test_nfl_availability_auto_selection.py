"""No pretend second NFL engine: auto-select only with verified data."""
import pathlib
import unittest

from ai2_nfl_availability_auto import (
    select_nfl_engine_from_sources, report_nfl_availability_effect)

ROOT=pathlib.Path(__file__).resolve().parents[1]


class NFLAutomaticAvailabilityTests(unittest.TestCase):
    def test_empty_archive_uses_original_engine_no_checkbox_or_radio(self):
        mode,why=select_nfl_engine_from_sources({
            "snapshots_by_game":{},
            "report":{"game_snapshot_count":0}})
        self.assertEqual(mode,"existing")
        self.assertEqual(why,"NO_VERIFIED_GAME_DAY_SOURCE")
        x=report_nfl_availability_effect({
            "football_only_players_added":0,
            "newly_applied_verified_out":0,
        },mode)
        self.assertEqual(x["engine_input_effect"],
                         "EXISTING_MODEL_NO_EXTRA_VERIFIED_INPUTS")
        self.assertFalse(x["changed_model_player_inputs"])
        self.assertFalse(x["independent_predictive_accuracy_proven"])
        source=(ROOT/"unified_app.py").read_text()
        self.assertNotIn("nfl_engine_choice",source)
        self.assertNotIn("NFL simulation engine",source)
        self.assertIn("select_nfl_engine_from_sources(",source)
        self.assertIn("report_nfl_availability_effect(",source)
        self.assertIn("build_oneclick_portfolio(",source)
        self.assertIn('f"BUILD MY {requested_count} LINEUPS"',source)

    def test_positive_archive_is_not_claimed_predictively_better(self):
        mode,why=select_nfl_engine_from_sources({
            "snapshots_by_game":{"KC@ATL":{"status":"RESEARCH_TEST"}}})
        self.assertEqual(mode,"availability_aware")
        self.assertEqual(why,"VERIFIED_GAME_DAY_SOURCE_PRESENT")
        no_change=report_nfl_availability_effect({
            "football_only_players_added":0,
            "newly_applied_verified_out":0,
            "applied_verified_out_to_priced_players":0},mode)
        self.assertEqual(no_change["engine_input_effect"],
                         "VERIFIED_SOURCE_WITH_NO_NEW_SIMULATION_INPUTS")
        self.assertFalse(no_change["changed_model_player_inputs"])

    def test_new_off_or_teammate_reports_actual_effect(self):
        x=report_nfl_availability_effect({
            "football_only_players_added":2,
            "newly_applied_verified_out":1,
            "applied_verified_out_to_priced_players":1,
        },"availability_aware")
        self.assertEqual(x["engine_input_effect"],
                         "VERIFIED_STATUS_CHANGED_SIMULATION_INPUTS")
        self.assertTrue(x["changed_model_player_inputs"])
        self.assertIn("2 internal non-DK teammates",x["engine_input_effect_explanation"])
        self.assertIn("1 newly identified",x["engine_input_effect_explanation"])
        self.assertFalse(x["independent_predictive_accuracy_proven"])

    def test_corrupt_evidence_never_enables_mode(self):
        for ctx in (None,{},{"snapshots_by_game":None},
                    {"snapshots_by_game":[]}):
            with self.assertRaisesRegex(ValueError,"source-backed|game map"):
                select_nfl_engine_from_sources(ctx)
        with self.assertRaisesRegex(ValueError,"negative"):
            report_nfl_availability_effect({
                "football_only_players_added":-1}, "availability_aware")
        with self.assertRaisesRegex(ValueError,"Unknown NFL"):
            report_nfl_availability_effect({},"fake")

    def test_previous_approved_lineups_are_unaffected_by_mode_choice(self):
        from ai2_nfl_release_gate import nfl_classic_release_status
        status=nfl_classic_release_status()
        self.assertFalse(status["ready_for_entries"])
        self.assertEqual(status["checks_passed"],0)
        ui=(ROOT/"unified_app.py").read_text()
        self.assertIn('snapshot_draft, load_draft',ui)
        self.assertIn('snapshot_approved, load_approved',ui)
        self.assertNotIn("nfl_draft_review_ack",ui)


if __name__=="__main__":
    unittest.main()
