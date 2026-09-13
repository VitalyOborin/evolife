from evolife.genome import Activation, Genome, NodeGene, NodeType


def test_genome_empty_is_valid():
    g = Genome()
    assert list(g.sensors()) == []
    assert list(g.motors()) == []
    assert list(g.active_connections()) == []


def test_make_default_genome_shape():
    from evolife.brain import Brain

    g = Brain.make_default_genome()
    sensor_ids = {n.id for n in g.sensors()}
    motor_ids = {n.id for n in g.motors()}
    hidden_ids = {n.id for n in g.nodes.values() if n.type is NodeType.HIDDEN}
    assert len(sensor_ids) == 5
    assert len(motor_ids) == 4
    assert len(hidden_ids) == 4
    assert sensor_ids.isdisjoint(motor_ids)
    assert sensor_ids.isdisjoint(hidden_ids)


def test_node_types_match_activation():
    from evolife.brain import Brain

    g = Brain.make_default_genome()
    sensors = g.sensors()
    motors = g.motors()
    assert all(s.activation is Activation.LINEAR for s in sensors)
    assert all(m.activation is Activation.TANH for m in motors)


def test_active_connections_only_enabled():
    g = Genome()
    g.nodes[0] = NodeGene(id=0, type=NodeType.SENSOR)
    g.nodes[1] = NodeGene(id=1, type=NodeType.MOTOR)
    # No connections yet; iteration should be empty.
    assert list(g.active_connections()) == []
