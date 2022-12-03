import unittest
import numpy as np

from bn_testing.helpers import (
    _generate_int_suffixes,
    abslog,
)


class TestIntSuffixes(unittest.TestCase):

    def test_zero_n(self):
        self.assertListEqual(
            _generate_int_suffixes('X', 0),
            []
        )


class TestLog(unittest.TestCase):

    def test_log(self):

        self.assertEqual(
            abslog(-4),
            np.log(1+4),
        )

        self.assertEqual(
            abslog(0),
            np.log(1),
        )
