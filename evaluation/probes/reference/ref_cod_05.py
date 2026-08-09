def second_largest(nums):
    u = sorted(set(nums), reverse=True)
    return u[1] if len(u) >= 2 else None
