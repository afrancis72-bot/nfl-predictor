"""Persistence regression checks: restart, archive, isolation, empty uploads."""
import tempfile
import unittest
from ai2_local_storage import save_file, load_file, list_history

class LocalStorageTests(unittest.TestCase):
    def test_survives_new_connections_and_archives_replacements(self):
        with tempfile.TemporaryDirectory() as directory:
            save_file(directory, "NFL", "DraftKings salaries", "week7.csv", b"first")
            self.assertEqual(load_file(directory, "NFL", "DraftKings salaries")[1], b"first")
            save_file(directory, "NFL", "DraftKings salaries", "week8.csv", b"second")
            self.assertEqual(load_file(directory, "NFL", "DraftKings salaries")[1], b"second")
            archive = list_history(directory, "NFL", "DraftKings salaries")
            self.assertEqual(len(archive), 1)
            self.assertEqual(archive[0][1:3], ("week7.csv", b"first"))
            self.assertIsNone(load_file(directory, "PGA", "DraftKings salaries"))
    def test_empty_upload_cannot_replace_existing(self):
        with tempfile.TemporaryDirectory() as directory:
            save_file(directory, "NFL", "Projections", "valid.csv", b"valid")
            with self.assertRaises(ValueError):
                save_file(directory, "NFL", "Projections", "empty.csv", b"")
            self.assertEqual(load_file(directory, "NFL", "Projections")[1], b"valid")

if __name__ == "__main__":
    unittest.main()
