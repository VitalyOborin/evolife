import numpy as np

from evolife.brain import Brain
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
    assert len(sensor_ids) == 3
    assert len(motor_ids) == 2
    # Phase 1.5: founder carries 1 hidden node (feedforward, no recurrence).
    assert len(hidden_ids) == 1
    # 3 sensor->hidden + 2 hidden->motor = 5 connections.
    assert len(g.connections) == 5
    assert sensor_ids.isdisjoint(motor_ids)


def test_node_types_match_activation():
    from evolife.brain import Brain
    from evolife.genome import NodeType

    g = Brain.make_default_genome()
    sensors = g.sensors()
    motors = g.motors()
    assert all(s.activation is Activation.LINEAR for s in sensors)
    # Both motors are TANH (zero-centered turn and locomotion).
    assert motors[0].activation is Activation.TANH
    assert motors[1].activation is Activation.TANH
    assert motors[0].bias == 0.0


def test_founder_locomotion_bias_is_diverse():
    rng = np.random.default_rng(1)
    biases = [
        Brain.make_default_genome(rng=rng).motors()[1].bias for _ in range(40)
    ]
    assert min(biases) < 0.0 < max(biases)


def test_active_connections_only_enabled():
    g = Genome()
    g.nodes[0] = NodeGene(id=0, type=NodeType.SENSOR)
    g.nodes[1] = NodeGene(id=1, type=NodeType.MOTOR)
    # No connections yet; iteration should be empty.
    assert list(g.active_connections()) == []


def test_fingerprint_is_stable():
    g = Brain.make_default_genome()
    h1 = g.fingerprint()
    h2 = g.fingerprint()
    assert h1 == h2
    assert len(h1) == 16


def test_fingerprint_changes_with_weights():
    g = Brain.make_default_genome()
    h1 = g.fingerprint()
    # Perturb one weight.
    key = next(iter(g.connections))
    g.connections[key].weight += 1.0
    h2 = g.fingerprint()
    assert h1 != h2


def test_fingerprint_changes_with_topology():
    g = Brain.make_default_genome()
    h1 = g.fingerprint()
    # Disable one connection.
    key = next(iter(g.connections))
    g.connections[key].enabled = False
    h2 = g.fingerprint()
    assert h1 != h2
