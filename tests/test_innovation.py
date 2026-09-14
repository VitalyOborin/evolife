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
    assert db1._next_conn == db2._next_conn
    assert db1._by_pair == db2._by_pair


def test_split_reuses_node_and_edges_for_same_historical_connection():
    db = InnovationDatabase()
    orig = db.innovation_for(0, 3)
    a = db.split_for(orig, 0, 3)
    b = db.split_for(orig, 0, 3)
    assert a == b
    assert a.new_node_id == 4


def test_split_of_different_connections_gets_different_node_ids():
    db = InnovationDatabase()
    i1 = db.innovation_for(0, 3)
    i2 = db.innovation_for(1, 3)
    a = db.split_for(i1, 0, 3)
    b = db.split_for(i2, 1, 3)
    assert a.new_node_id != b.new_node_id
    assert a.in_innovation != b.in_innovation
