from candidate import eval_expr
def test_precedence():
    assert eval_expr("2+3*4") == 14
def test_parens():
    assert eval_expr("(2+3)*4") == 20
def test_div_trunc():
    assert eval_expr("7/2") == 3
def test_nested():
    assert eval_expr("((1+2)*(3+4))-5") == 16
def test_spaces():
    assert eval_expr(" 10 - 2 * 3 ") == 4
