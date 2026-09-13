"""Brain — a thin executable view over a frozen Genome.

In v0 the topology is fixed at construction time (8→4→4-ish, controlled by
config). In v1 the topology will be read from the genome and built once per
organism birth. Either way, Brain is stateless across organisms: each
organism owns its own Brain instance.
"""

from __future__ import annotations

import numpy as np

from .config import N_HIDDEN, N_MOTORS, N_SENSORS
from .genome import Activation, Genome, NodeGene, NodeType


class Brain:
    """Feedforward brain built from a Genome.

    For v0 we ignore hidden-node activation and bias and just use tanh on
    the hidden layer; full activation dispatch comes with v1.
    """

    def __init__(self, genome: Genome, n_hidden: int = N_HIDDEN) -> None:
        self.genome = genome
        self.n_hidden = n_hidden

        sensors = sorted(
            (n for n in genome.nodes.values() if n.type is NodeType.SENSOR),
            key=lambda n: n.id,
        )
        motors = sorted(
            (n for n in genome.nodes.values() if n.type is NodeType.MOTOR),
            key=lambda n: n.id,
        )
        if len(sensors) != N_SENSORS:
            raise ValueError(
                f"Genome has {len(sensors)} sensors, expected {N_SENSORS}"
            )
        if len(motors) != N_MOTORS:
            raise ValueError(
                f"Genome has {len(motors)} motors, expected {N_MOTORS}"
            )

        self._sensor_ids = [n.id for n in sensors]
        self._motor_ids = [n.id for n in motors]

        # Build dense matrices for v0. Hidden layer is fixed-size; we
        # project sensor -> hidden -> motor via two weight matrices.
        rng = np.random.default_rng(0)
        self.w_sensory_to_hidden = rng.normal(
            0, 1.0, size=(N_SENSORS, n_hidden)
        ).astype(np.float32)
        self.w_hidden_to_motor = rng.normal(
            0, 1.0, size=(n_hidden, N_MOTORS)
        ).astype(np.float32)

    def forward(self, sensors: np.ndarray) -> np.ndarray:
        """Compute motor outputs from sensor inputs.

        sensors: shape (N_SENSORS,), dtype float32.
        returns: shape (N_MOTORS,), values clipped to a sane range.
        """
        if sensors.shape != (N_SENSORS,):
            raise ValueError(
                f"Expected sensor vector of shape ({N_SENSORS},), "
                f"got {sensors.shape}"
            )
        hidden = np.tanh(self.w_sensory_to_hidden.T @ sensors)
        motors = np.tanh(self.w_hidden_to_motor.T @ hidden)
        return np.clip(motors, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def make_default_genome(
        n_sensors: int = N_SENSORS,
        n_hidden: int = N_HIDDEN,
        n_motors: int = N_MOTORS,
    ) -> Genome:
        """Construct a minimal NEAT-shaped genome matching the v0 layout."""
        g = Genome()
        next_id = 0
        for _ in range(n_sensors):
            g.nodes[next_id] = NodeGene(
                id=next_id, type=NodeType.SENSOR, activation=Activation.LINEAR
            )
            next_id += 1
        for _ in range(n_hidden):
            g.nodes[next_id] = NodeGene(
                id=next_id, type=NodeType.HIDDEN, activation=Activation.TANH
            )
            next_id += 1
        for _ in range(n_motors):
            g.nodes[next_id] = NodeGene(
                id=next_id, type=NodeType.MOTOR, activation=Activation.TANH
            )
            next_id += 1
        return g
