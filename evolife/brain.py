"""Recurrent brain with persistent state.

v2: each organism owns a Brain whose state (node activations) persists
across ticks. Forward pass:

  1. Load sensors into state[:N_SENSORS].
  2. Compute delta for each non-sensor node as
        delta[j] = bias[j] + sum_{(i -> j) active} state[i] * w
  3. Apply activation to delta[j], write back as state[j].
  4. Read motors.

This correctly supports arbitrary topology: cycles, hidden->sensor
(its value is overwritten next tick by sensor input — fine), motor->hidden,
etc.

The compiled view (node ordering, sparse indices, weights, biases) is
cached and only rebuilt when the genome changes — i.e. once at birth.
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

_ACT_LINEAR = 0
_ACT_TANH = 1
_ACT_SIGMOID = 2
_ACT_RELU = 3

# Number of refinement iterations per forward pass. 1 is enough for
# strictly feed-forward networks; arbitrary topologies (cycles,
# motor->hidden, etc.) need more for information to propagate. 3 is a
# good default for small (~20 node) brains.
_ITERATIONS = 3


def _activation_id(a: Activation) -> int:
    return {
        Activation.LINEAR: _ACT_LINEAR,
        Activation.TANH: _ACT_TANH,
        Activation.SIGMOID: _ACT_SIGMOID,
        Activation.RELU: _ACT_RELU,
    }[a]


@dataclass
class _CompiledNet:
    """Pre-sorted view of one genome, ready for forward evaluation."""

    n_nodes: int
    is_sensor: np.ndarray  # bool, shape (n_nodes,)
    act_ids: np.ndarray  # int8, shape (n_nodes,)
    biases: np.ndarray  # float32, shape (n_nodes,)
    in_idx: np.ndarray  # int64, shape (n_conns,)
    out_idx: np.ndarray  # int64, shape (n_conns,)
    weights: np.ndarray  # float32, shape (n_conns,)
    motor_idx: np.ndarray  # int64, shape (n_motors,)


class Brain:
    """A recurrent, persistent, genome-driven brain."""

    def __init__(self, genome: Genome) -> None:
        self.genome = genome
        self._compiled: _CompiledNet | None = None
        self._compiled_hash: str | None = None
        self.state: np.ndarray | None = None

    def forward(self, sensors: np.ndarray) -> np.ndarray:
        if sensors.shape != (N_SENSORS,):
            raise ValueError(
                f"Expected sensor vector of shape ({N_SENSORS},), "
                f"got {sensors.shape}"
            )

        net = self._ensure_compiled()

        # Initialise state on first use or after topology change.
        if self.state is None or self.state.shape != (net.n_nodes,):
            self.state = np.zeros(net.n_nodes, dtype=np.float32)

        # Sensors overwrite the first N_SENSORS positions every tick.
        # This is the "perception" boundary: sensor values come from
        # the world, not from internal state.
        self.state[:N_SENSORS] = sensors

        if net.in_idx.size > 0:
            # Iteratively refine non-sensor activations.
            # For feed-forward topology 1 iteration is enough; for
            # arbitrary topologies (cycles, motor->hidden) we run a few
            # iterations so information propagates. Sensors stay fixed
            # at the values just written; only non-sensor positions
            # update.
            non_sensor_mask = ~net.is_sensor
            for _ in range(_ITERATIONS):
                # delta[j] = bias[j] + sum_i(state[i] * w) over (i -> j)
                delta = np.zeros(net.n_nodes, dtype=np.float32)
                np.add.at(
                    delta,
                    net.out_idx,
                    self.state[net.in_idx] * net.weights,
                )
                delta += net.biases
                self.state[non_sensor_mask] = _activate_vec(
                    net.act_ids[non_sensor_mask], delta[non_sensor_mask]
                )

        motor_vals = self.state[net.motor_idx]
        return np.clip(motor_vals, -1.0, 1.0).astype(np.float32)

    def reset_state(self) -> None:
        self.state = None

    def _ensure_compiled(self) -> _CompiledNet:
        h = self.genome.fingerprint()
        if self._compiled is not None and self._compiled_hash == h:
            return self._compiled
        self._compiled = _compile(self.genome)
        self._compiled_hash = h
        self.reset_state()
        return self._compiled

    @staticmethod
    def make_default_genome(
        n_sensors: int = N_SENSORS,
        n_hidden: int = N_HIDDEN,
        n_motors: int = N_MOTORS,
    ) -> Genome:
        """Construct a minimal NEAT-shaped genome with one recurrent edge.

        The recurrent edge between two hidden nodes gives the brain
        actual memory from birth. Without it the network is feed-forward
        and cannot maintain state across ticks — the experiment would
        collapse to a single-tick reaction.
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
        # Motor 0 (turn rate) is signed -> TANH. Motors 1..N (move
        # speed, eat_attempt, reproduce_attempt) are non-negative
        # intensities -> SIGMOID. Using TANH for non-negative motors
        # would silently zero out half the output range (clip(0,1)
        # below), so we set the activation to match the semantics.
        for i in range(n_motors):
            nid = len(g.nodes)
            act = Activation.TANH if i == 0 else Activation.SIGMOID
            g.nodes[nid] = NodeGene(
                id=nid, type=NodeType.MOTOR, activation=act
            )
            motor_ids.append(nid)

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
        if len(hidden_ids) >= 2:
            a, b = hidden_ids[0], hidden_ids[1]
            g.connections[innov] = ConnectionGene(
                innovation=innov,
                in_node=a,
                out_node=b,
                weight=float(rng.normal(0, 0.5)),
                enabled=True,
            )
            innov += 1
        g.max_innovation = innov
        return g


# --- helpers ---------------------------------------------------------------


def _compile(genome: Genome) -> _CompiledNet:
    sensors_sorted = sorted(
        (n for n in genome.nodes.values() if n.type is NodeType.SENSOR),
        key=lambda n: n.id,
    )
    others = [
        n for n in genome.nodes.values() if n.type is not NodeType.SENSOR
    ]
    others.sort(key=lambda n: n.id)
    ordered = sensors_sorted + others

    n_nodes = len(ordered)
    is_sensor = np.array(
        [n.type is NodeType.SENSOR for n in ordered], dtype=bool
    )
    act_ids = np.array(
        [_activation_id(n.activation) for n in ordered], dtype=np.int8
    )
    biases = np.array([n.bias for n in ordered], dtype=np.float32)

    in_list: list[int] = []
    out_list: list[int] = []
    w_list: list[float] = []
    id_to_pos = {nid: i for i, nid in enumerate([n.id for n in ordered])}
    for conn in genome.active_connections():
        if conn.in_node not in id_to_pos or conn.out_node not in id_to_pos:
            continue
        in_list.append(id_to_pos[conn.in_node])
        out_list.append(id_to_pos[conn.out_node])
        w_list.append(conn.weight)

    motor_idx = [
        id_to_pos[m.id] for m in genome.motors() if m.id in id_to_pos
    ]

    return _CompiledNet(
        n_nodes=n_nodes,
        is_sensor=is_sensor,
        act_ids=act_ids,
        biases=biases,
        in_idx=np.array(in_list, dtype=np.int64),
        out_idx=np.array(out_list, dtype=np.int64),
        weights=np.array(w_list, dtype=np.float32),
        motor_idx=np.array(motor_idx, dtype=np.int64),
    )


def _activate_vec(act_ids: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorised activation. act_ids is per-element int8."""
    out = np.empty_like(x, dtype=np.float32)
    # Tanh and sigmoid can be applied in one shot for their values, but
    # we still need to mask by activation id because different elements
    # may have different activations.
    tanh_mask = act_ids == _ACT_TANH
    sig_mask = act_ids == _ACT_SIGMOID
    relu_mask = act_ids == _ACT_RELU
    linear_mask = act_ids == _ACT_LINEAR
    out[tanh_mask] = np.tanh(x[tanh_mask]).astype(np.float32)
    # Numerically safe sigmoid.
    if sig_mask.any():
        xv = x[sig_mask]
        pos = xv >= 0
        out[sig_mask] = np.where(
            pos,
            1.0 / (1.0 + np.exp(-xv)),
            np.exp(xv) / (1.0 + np.exp(xv)),
        ).astype(np.float32)
    out[relu_mask] = np.maximum(0.0, x[relu_mask]).astype(np.float32)
    out[linear_mask] = x[linear_mask]
    return out
