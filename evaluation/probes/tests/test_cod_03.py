from candidate import search_rotated
def test_found():
    assert search_rotated([4,5,6,7,0,1,2], 0) == 4
def test_absent():
    assert search_rotated([4,5,6,7,0,1,2], 3) == -1
def test_single():
    assert search_rotated([1], 1) == 0
def test_unrotated():
    assert search_rotated([1,2,3,4,5], 4) == 3
def test_pivot_first():
    assert search_rotated([5,1,2,3,4], 5) == 0
