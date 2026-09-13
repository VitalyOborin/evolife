from evolife.innovation import InnovationDatabase


def test_innovation_reuses_for_known_pair():
    db = InnovationDatabase()
    a = db.innovation_for(1, 2)
    b = db.innovation_for(1, 2)
    assert a == b
    assert a == 0


def test_innovation_assigns_distinct_for_distinct_pairs():
    db = InnovationDatabase()
    a = db.innovation_for(1, 2)
    b = db.innovation_for(2, 1)
    c = db.innovation_for(1, 3)
    assert len({a, b, c}) == 3


def test_innovation_is_deterministic_across_reset():
    db1 = InnovationDatabase()
    db1.innovation_for(1, 2)
    db1.innovation_for(2, 3)
    db1.innovation_for(1, 2)
    db2 = InnovationDatabase()
    db2.innovation_for(1, 2)
    db2.innovation_for(2, 3)
    db2.innovation_for(1, 2)
    assert db1._next == db2._next
    assert db1._by_pair == db2._by_pair
