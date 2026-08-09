def eval_expr(s: str) -> int:
    t, i = [], 0
    s = s.replace(" ", "")
    while i < len(s):
        if s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            t.append(int(s[i:j])); i = j
        else:
            t.append(s[i]); i += 1
    pos = 0
    def expr():
        nonlocal pos
        v = term()
        while pos < len(t) and t[pos] in "+-":
            op = t[pos]; pos += 1
            r = term()
            v = v + r if op == "+" else v - r
        return v
    def term():
        nonlocal pos
        v = factor()
        while pos < len(t) and t[pos] in "*/":
            op = t[pos]; pos += 1
            r = factor()
            v = v * r if op == "*" else int(v / r)
        return v
    def factor():
        nonlocal pos
        if t[pos] == "(":
            pos += 1
            v = expr()
            pos += 1
            return v
        v = t[pos]; pos += 1
        return v
    return expr()
