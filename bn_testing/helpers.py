"""
Helpers
"""


def _generate_int_suffixes(prefix, n):
    return [
        "{}{}".format(
            prefix,
            str(i).zfill(len(str(n)))) for i in range(n)
    ]
