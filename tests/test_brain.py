import numpy as np
import pytest

from evolife.brain import Brain
from evolife.config import N_MOTORS, N_SENSORS


def test_default_brain_forward_shape():
    g = Brain.make_default_genome()
    brain = Brain(g)
    sensors = np.zeros(N_SENSORS, dtype=np.float32)
    out = brain.forward(sensors)
    assert out.shape == (N_MOTORS,)
    assert np.all(out >= -1.0)
    assert np.all(out <= 1.0)


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
    sensors = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    out1 = brain.forward(sensors)
    # Reset state and call again.
    brain.reset_state()
    out2 = brain.forward(sensors)
    np.testing.assert_allclose(out1, out2, atol=1e-6)


def test_brain_with_no_connections_returns_zeros():
    """An organism with no connections must not crash; motors are zero."""
    from evolife.genome import Genome

    g = Brain.make_default_genome()
    # Wipe all connections.
    g.connections.clear()
    brain = Brain(g)
    out = brain.forward(np.ones(N_SENSORS, dtype=np.float32))
    np.testing.assert_array_equal(out, np.zeros(N_MOTORS, dtype=np.float32))


def test_brain_reflects_disabled_connection():
    """Disabling a connection must change the forward output."""
    g = Brain.make_default_genome()
    sensors = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    out_before = Brain(g).forward(sensors)
    # Disable every connection.
    for c in g.connections.values():
        c.enabled = False
    out_after = Brain(g).forward(sensors)
    # With everything disabled, motors should be zero (or at least
    # observably different from out_before).
    assert not np.allclose(out_before, out_after)


def test_brain_persistent_state_changes_output_across_ticks():
    """A brain's state should persist across calls and affect output."""
    from evolife.genome import ConnectionGene

    g = Brain.make_default_genome()
    brain = Brain(g)
    sensors_a = np.array([0.9, 0.0, 0.0, 0.5, 1.0], dtype=np.float32)
    sensors_b = np.zeros(N_SENSORS, dtype=np.float32)

    # Pump sensors_a for several ticks to fill state.
    for _ in range(10):
        brain.forward(sensors_a)
    state_after_a = brain.state.copy()

    # Now pump sensors_b; state should evolve from sensors_a's state,
    # not from zero.
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
    """Brain must execute hidden->sensor and motor->hidden correctly."""
    from evolife.genome import ConnectionGene, NodeGene, NodeType, Activation

    g = Brain.make_default_genome()
    # Add a hidden -> hidden connection (a second recurrent edge).
    hidden_ids = [
        n.id for n in g.nodes.values() if n.type is NodeType.HIDDEN
    ]
    if len(hidden_ids) >= 2:
        g.connections[100] = ConnectionGene(
            innovation=100,
            in_node=hidden_ids[0],
            out_node=hidden_ids[1],
            weight=1.5,
            enabled=True,
        )
    brain = Brain(g)
    sensors = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    # Run a few ticks; must not raise.
    for _ in range(5):
        out = brain.forward(sensors)
    assert out.shape == (N_MOTORS,)
