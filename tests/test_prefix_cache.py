from vaayu.prefix_cache import longest_common_prefix


def test_lcp_identical():
    assert longest_common_prefix([1, 2, 3], [1, 2, 3]) == 3


def test_lcp_partial():
    assert longest_common_prefix([1, 2, 3, 9], [1, 2, 3, 8, 8]) == 3


def test_lcp_none():
    assert longest_common_prefix([5, 1], [4, 1]) == 0


def test_lcp_empty():
    assert longest_common_prefix([], [1, 2]) == 0
    assert longest_common_prefix([1, 2], []) == 0


def test_lcp_one_is_prefix_of_other():
    assert longest_common_prefix([1, 2], [1, 2, 3, 4]) == 2
