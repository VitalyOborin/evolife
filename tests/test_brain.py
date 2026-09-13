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
    g = Brain.make_default_genome()
    brain = Brain(g)
    sensors = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    out1 = brain.forward(sensors)
    out2 = brain.forward(sensors)
    np.testing.assert_array_equal(out1, out2)


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
