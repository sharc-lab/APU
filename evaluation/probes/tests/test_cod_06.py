from candidate import LRUCache
def test_basic():
    c = LRUCache(2)
    c.put(1,1); c.put(2,2)
    assert c.get(1) == 1
    c.put(3,3)              # evicts key 2
    assert c.get(2) == -1
    assert c.get(3) == 3
def test_update_counts_as_use():
    c = LRUCache(2)
    c.put(1,1); c.put(2,2)
    c.put(1,10)             # refresh key 1
    c.put(3,3)              # evicts key 2
    assert c.get(1) == 10
    assert c.get(2) == -1
def test_capacity_one():
    c = LRUCache(1)
    c.put(1,1); c.put(2,2)
    assert c.get(1) == -1
    assert c.get(2) == 2
