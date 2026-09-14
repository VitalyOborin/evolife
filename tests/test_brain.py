import numpy as np
import pytest

from evolife.brain import Brain, _ITERATIONS
from evolife.config import N_MOTORS, N_SENSORS
from evolife.genome import Activation, ConnectionGene, NodeGene, NodeType


def test_brain_uses_one_iteration_per_world_tick():
    assert _ITERATIONS == 1


def test_default_brain_forward_shape():
    g = Brain.make_default_genome()
    brain = Brain(g)
    sensors = np.zeros(N_SENSORS, dtype=np.float32)
    out = brain.forward(sensors)
    assert out.shape == (N_MOTORS,)
    assert np.all(out >= -1.0)
    assert np.all(out <= 1.0)


def test_proto_brain_turn_is_quiet_without_smell():
    """Turn bias is 0, so no smell → no spinning."""
    rng = np.random.default_rng(0)
    g = Brain.make_default_genome(rng=rng)
    brain = Brain(g)
    out = brain.forward(np.zeros(N_SENSORS, dtype=np.float32))
    assert abs(float(out[0])) < 0.05


def test_zero_locomotion_bias_rests_without_smell():
    g = Brain.make_default_genome(rng=np.random.default_rng(0))
    g.motors()[1].bias = 0.0
    out = Brain(g).forward(np.zeros(N_SENSORS, dtype=np.float32))
    assert abs(float(out[1])) < 0.05


def test_brain_rejects_wrong_sensor_shape():
    g = Brain.make_default_genome()
    brain = Brain(g)
    with pytest.raises(ValueError):
        brain.forward(np.zeros(N_SENSORS + 1, dtype=np.float32))


def test_brain_forward_deterministic_with_frozen_weights():
    """A fresh brain called twice with the same sensors must give the
    same output. (Persistent state means subsequent calls would not be
    identical — that's a feature, not a bug.)"""
    g = Brain.make_default_genome()
    brain = Brain(g)
    sensors = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out1 = brain.forward(sensors)
    brain.reset_state()
    out2 = brain.forward(sensors)
    np.testing.assert_allclose(out1, out2, atol=1e-6)


def test_brain_with_no_connections_returns_zeros():
    """An organism with no connections must not crash; motors are zero."""
    g = Brain.make_default_genome()
    g.connections.clear()
    brain = Brain(g)
    out = brain.forward(np.ones(N_SENSORS, dtype=np.float32))
    np.testing.assert_array_equal(out, np.zeros(N_MOTORS, dtype=np.float32))


def test_brain_reflects_disabled_connection():
    """Disabling a connection must change the forward output."""
    g = Brain.make_default_genome()
    sensors = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out_before = Brain(g).forward(sensors)
    for c in g.connections.values():
        c.enabled = False
    out_after = Brain(g).forward(sensors)
    assert not np.allclose(out_before, out_after)


def test_brain_persistent_state_changes_output_across_ticks():
    """A recurrent hidden node must carry state from one tick to the next."""
    g = Brain.make_default_genome()
    hidden_id = max(g.nodes.keys()) + 1
    g.nodes[hidden_id] = NodeGene(
        id=hidden_id, type=NodeType.HIDDEN, activation=Activation.TANH
    )
    sensor_id = next(n.id for n in g.nodes.values() if n.type is NodeType.SENSOR)
    motor_id = next(n.id for n in g.nodes.values() if n.type is NodeType.MOTOR)
    g.connections[100] = ConnectionGene(
        innovation=100, in_node=sensor_id, out_node=hidden_id, weight=2.0, enabled=True
    )
    g.connections[101] = ConnectionGene(
        innovation=101, in_node=hidden_id, out_node=hidden_id, weight=1.5, enabled=True
    )
    g.connections[102] = ConnectionGene(
        innovation=102, in_node=hidden_id, out_node=motor_id, weight=1.0, enabled=True
    )
    brain = Brain(g)
    sensors_a = np.array([0.9, 0.0, 0.0], dtype=np.float32)
    sensors_b = np.zeros(N_SENSORS, dtype=np.float32)

    for _ in range(10):
        brain.forward(sensors_a)
    state_after_a = brain.state.copy()

    for _ in range(3):
        brain.forward(sensors_b)
    state_after_b = brain.state.copy()

    assert not np.allclose(state_after_a, state_after_b)


def test_brain_handles_no_connections():
    """A genome with no connections must still run, returning zeros."""
    g = Brain.make_default_genome()
    g.connections.clear()
    brain = Brain(g)
    out = brain.forward(np.ones(N_SENSORS, dtype=np.float32))
    np.testing.assert_array_equal(out, np.zeros(N_MOTORS, dtype=np.float32))


def test_brain_supports_arbitrary_topology():
    """Brain must execute a hidden node and a recurrent edge without crashing."""
    g = Brain.make_default_genome()
    hidden_id = max(g.nodes.keys()) + 1
    g.nodes[hidden_id] = NodeGene(
        id=hidden_id, type=NodeType.HIDDEN, activation=Activation.TANH
    )
    sensor_id = next(n.id for n in g.nodes.values() if n.type is NodeType.SENSOR)
    motor_id = next(n.id for n in g.nodes.values() if n.type is NodeType.MOTOR)
    g.connections[100] = ConnectionGene(
        innovation=100, in_node=sensor_id, out_node=hidden_id, weight=1.5, enabled=True
    )
    g.connections[101] = ConnectionGene(
        innovation=101, in_node=hidden_id, out_node=motor_id, weight=1.0, enabled=True
    )
    g.connections[102] = ConnectionGene(
        innovation=102, in_node=hidden_id, out_node=hidden_id, weight=0.8, enabled=True
    )
    brain = Brain(g)
    sensors = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    for _ in range(5):
        out = brain.forward(sensors)
    assert out.shape == (N_MOTORS,)
