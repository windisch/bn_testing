import unittest

from bn_testing.helpers import _generate_int_suffixes


class TestIntSuffixes(unittest.TestCase):

    def test_zero_n(self):
        self.assertListEqual(
            _generate_int_suffixes('X', 0),
            []
        )
