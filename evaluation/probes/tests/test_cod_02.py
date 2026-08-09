from candidate import fizzbuzz_sum
def test_small():
    # 3,5,6,9,10,12 -> 45 ; 15 excluded (div by both)
    assert fizzbuzz_sum(15) == 45
def test_one():
    assert fizzbuzz_sum(1) == 0
def test_thirty():
    # 3,5,6,9,10,12,18,20,21,24,25,27 -> 180 (15 and 30 excluded)
    assert fizzbuzz_sum(30) == 180
