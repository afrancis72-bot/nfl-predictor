"""Tests for strict verification-flag conversion."""
import unittest
from ai2_nfl_flags import verified_flag

class TestFlags(unittest.TestCase):
    def test_explicit_true(self):
        self.assertTrue(verified_flag(True))
        self.assertTrue(verified_flag("true"))
        self.assertTrue(verified_flag(" TRUE "))

    def test_explicit_false(self):
        self.assertFalse(verified_flag(False))
        self.assertFalse(verified_flag("false"))

    def test_unknown_fails_closed(self):
        for value in (None, "", "yes", "1", 1, [], {}):
            with self.subTest(value=value):
                self.assertFalse(verified_flag(value))

if __name__ == "__main__":
    unittest.main()
