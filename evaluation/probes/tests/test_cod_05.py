from candidate import second_largest
def test_basic():
    assert second_largest([3,1,4,1,5]) == 4
def test_dupes_at_top():
    assert second_largest([5,5,3]) == 3
def test_all_same():
    assert second_largest([2,2,2]) is None
def test_too_short():
    assert second_largest([1]) is None
def test_no_mutation():
    src = [3,1,2]
    second_largest(src)
    assert src == [3,1,2]
