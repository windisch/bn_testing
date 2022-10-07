"""
Helpers
"""

import numpy as np


def _make_random_state(random):

    if random is None:
        return np.random.RandomState()
    else:
        return random


def _generate_int_suffixes(prefix, n):
    return [
        "{}{}".format(
            prefix,
            str(i).zfill(len(str(n)))) for i in range(n)
    ]
