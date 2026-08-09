from collections import Counter
def top_k_frequent(words, k):
    c = Counter(words)
    return [w for w, _ in sorted(c.items(), key=lambda p: (-p[1], p[0]))][:k]
