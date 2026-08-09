from candidate import top_k_frequent
def test_basic():
    assert top_k_frequent(["i","love","code","i","love","fun"], 2) == ["i","love"]
def test_tiebreak_lexicographic():
    assert top_k_frequent(["b","a","c"], 2) == ["a","b"]
def test_k_all():
    assert top_k_frequent(["x","y","x"], 2) == ["x","y"]
