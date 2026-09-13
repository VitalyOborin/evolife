"""Brain — a thin executable view over a frozen Genome.

Brain reads the active connections from the Genome and evaluates the
network each forward pass. Topology and weights both come from the
genome; the brain owns no state of its own.

In v0 the topology was hardcoded in numpy matrices and the genome was
ignored at forward time. In v1 the brain is driven entirely by the
genome, so structural mutations (add_node, add_connection) immediately
change behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import N_HIDDEN, N_MOTORS, N_SENSORS
from .genome import (
    Activation,
    ConnectionGene,
    Genome,
    NodeGene,
    NodeType,
)


@dataclass
class _CompiledNet:
    """A pre-sorted view of one genome, ready for forward evaluation."""

    node_ids: list[int]
    node_types: list[NodeType]
    node_activations: list[Activation]
    node_biases: np.ndarray  # shape (n_nodes,)
    in_idx: np.ndarray  # shape (n_conns,)
    out_idx: np.ndarray  # shape (n_conns,)
    weights: np.ndarray  # shape (n_conns,)


class Brain:
    """A genome-driven feedforward brain.

    The genome is the source of truth. Each forward pass rebuilds the
    compiled view because in v1 we expect structural mutations to be
    frequent and topology to change at birth time. For higher tick rates
    we will cache the compiled view and only rebuild when the genome
    changes — that's a v2 optimisation.
    """

    def __init__(self, genome: Genome) -> None:
        self.genome = genome

    def forward(self, sensors: np.ndarray) -> np.ndarray:
        """Compute motor outputs from sensor inputs.

        sensors: shape (N_SENSORS,), dtype float32.
        returns: shape (N_MOTORS,), values in [-1, 1].
        """
        if sensors.shape != (N_SENSORS,):
            raise ValueError(
                f"Expected sensor vector of shape ({N_SENSORS},), "
                f"got {sensors.shape}"
            )

        net = _compile(self.genome, sensors)
        if net.in_idx.size == 0:
            # No connections at all: return zeros on motors. The organism
            # does not move, does not eat, does not reproduce. It dies.
            return np.zeros(N_MOTORS, dtype=np.float32)

        # Forward pass in numpy.
        values = np.zeros(len(net.node_ids), dtype=np.float32)
        values[: N_SENSORS] = sensors

        # Order: sensors (already set), then hidden/motor nodes in
        # insertion order. _compile returns nodes in deterministic order:
        # sensors by id, then the rest by id. We accumulate into `values`
        # as we go.
        for conn_idx in range(net.in_idx.size):
            i = int(net.in_idx[conn_idx])
            o = int(net.out_idx[conn_idx])
            w = float(net.weights[conn_idx])
            values[o] += values[i] * w

        # Apply activation + bias for non-sensor nodes.
        for n, (typ, act, bias) in enumerate(
            zip(net.node_types, net.node_activations, net.node_biases),
            start=0,
        ):
            if typ is NodeType.SENSOR:
                continue
            values[n] = _activate(act, values[n] + bias)

        # Extract motor outputs.
        motor_ids = [n.id for n in self.genome.motors()]
        motor_idx = [net.node_ids.index(mid) for mid in motor_ids]
        return np.clip(values[motor_idx], -1.0, 1.0).astype(np.float32)

    @staticmethod
    def make_default_genome(
        n_sensors: int = N_SENSORS,
        n_hidden: int = N_HIDDEN,
        n_motors: int = N_MOTORS,
    ) -> Genome:
        """Construct a minimal NEAT-shaped genome matching the v0 layout.

        v0 had hardcoded dense matrices; v1 actually wires sensors to
        hidden to motors through ConnectionGenes so structural mutations
        have effect.
        """
        g = Genome()
        sensor_ids: list[int] = []
        for _ in range(n_sensors):
            nid = len(g.nodes)
            g.nodes[nid] = NodeGene(
                id=nid, type=NodeType.SENSOR, activation=Activation.LINEAR
            )
            sensor_ids.append(nid)
        hidden_ids: list[int] = []
        for _ in range(n_hidden):
            nid = len(g.nodes)
            g.nodes[nid] = NodeGene(
                id=nid, type=NodeType.HIDDEN, activation=Activation.TANH
            )
            hidden_ids.append(nid)
        motor_ids: list[int] = []
        for _ in range(n_motors):
            nid = len(g.nodes)
            g.nodes[nid] = NodeGene(
                id=nid, type=NodeType.MOTOR, activation=Activation.TANH
            )
            motor_ids.append(nid)

        # Connections: sensor -> hidden (one per pair), hidden -> motor
        # (one per pair). Innovation numbers count up from 0.
        innov = 0
        rng = np.random.default_rng(0)
        for s in sensor_ids:
            for h in hidden_ids:
                g.connections[innov] = ConnectionGene(
                    innovation=innov,
                    in_node=s,
                    out_node=h,
                    weight=float(rng.normal(0, 1.0)),
                    enabled=True,
                )
                innov += 1
        for h in hidden_ids:
            for m in motor_ids:
                g.connections[innov] = ConnectionGene(
                    innovation=innov,
                    in_node=h,
                    out_node=m,
                    weight=float(rng.normal(0, 1.0)),
                    enabled=True,
                )
                innov += 1
        g.max_innovation = innov
        return g


# --- helpers ---------------------------------------------------------------


def _compile(genome: Genome, sensors: np.ndarray) -> _CompiledNet:
    """Materialise a per-forward executable view of the genome."""
    sensors_sorted = sorted(
        (n for n in genome.nodes.values() if n.type is NodeType.SENSOR),
        key=lambda n: n.id,
    )
    others = [
        n for n in genome.nodes.values() if n.type is not NodeType.SENSOR
    ]
    others.sort(key=lambda n: n.id)
    ordered = sensors_sorted + others

    node_ids = [n.id for n in ordered]
    node_types = [n.type for n in ordered]
    node_activations = [n.activation for n in ordered]
    node_biases = np.array([n.bias for n in ordered], dtype=np.float32)

    in_list: list[int] = []
    out_list: list[int] = []
    w_list: list[float] = []
    id_to_pos = {nid: i for i, nid in enumerate(node_ids)}
    for conn in genome.active_connections():
        if conn.in_node not in id_to_pos or conn.out_node not in id_to_pos:
            continue
        in_list.append(id_to_pos[conn.in_node])
        out_list.append(id_to_pos[conn.out_node])
        w_list.append(conn.weight)

    return _CompiledNet(
        node_ids=node_ids,
        node_types=node_types,
        node_activations=node_activations,
        node_biases=node_biases,
        in_idx=np.array(in_list, dtype=np.int64),
        out_idx=np.array(out_list, dtype=np.int64),
        weights=np.array(w_list, dtype=np.float32),
    )


def _activate(act: Activation, x: float) -> float:
    if act is Activation.LINEAR:
        return x
    if act is Activation.TANH:
        return float(np.tanh(x))
    if act is Activation.SIGMOID:
        # Numerically safe sigmoid.
        if x >= 0:
            z = np.exp(-x)
            return float(1.0 / (1.0 + z))
        z = np.exp(x)
        return float(z / (1.0 + z))
    if act is Activation.RELU:
        return float(max(0.0, x))
    return x
