from candidate import reverse_words
def test_basic():
    assert reverse_words("the sky is blue") == "blue is sky the"
def test_whitespace():
    assert reverse_words("  the sky   is blue ") == "blue is sky the"
def test_single():
    assert reverse_words("word") == "word"
def test_empty():
    assert reverse_words("   ") == ""
