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
    # All motor outputs are in [-1, 1] by construction.
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
