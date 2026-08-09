def fizzbuzz_sum(n: int) -> int:
    return sum(i for i in range(1, n+1)
               if (i % 3 == 0) != (i % 5 == 0))
