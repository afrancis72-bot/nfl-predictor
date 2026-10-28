import unittest
from ai2_nfl_eligibility import eligible_nfl_rows

class TestGuardrails(unittest.TestCase):
    def test_empty_pool(self):
        self.assertEqual(eligible_nfl_rows([]), [])

if __name__ == '__main__':
    unittest.main()
