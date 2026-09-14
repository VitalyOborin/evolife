"""Brain — persistent-state executor that supports arbitrary topology.

v2: each organism owns a Brain whose state (node activations) persists
across ticks. The executor supports cycles, hidden->sensor (its value
is overwritten next tick by sensor input), motor->hidden, etc.

Crucially, v2.2 founders are PURE FEEDFORWARD: no hidden nodes, no
recurrent edges. Recurrent dynamics can only arise via structural
mutation (add_connection or add_node). This is intentional: we want
recurrent circuitry to be an EMERGENT property of evolution, not a
gift from the developer. If a lineage ever evolves `sensor -> hidden
-> motor` plus a cycle, that is a milestone, not a baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    BIAS_MAX,
    INITIAL_LOCOMOTION_BIAS_SIGMA,
    INITIAL_WEIGHT_SIGMA,
    N_HIDDEN,
    N_MOTORS,
    N_SENSORS,
)
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

# One brain update per world tick. Extra iterations push a recurrent
# net toward its attractor inside a single physical step. v2.2 keeps
# this at 1: founder networks are feedforward so attractor dynamics
# do not apply, and we want each tick to correspond to one world
# observation, not several. Motor lag of one tick is intended.
_ITERATIONS = 1


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
    """A persistent, genome-driven brain.

    Topology is read from the genome; the executor handles arbitrary
    connectivity including cycles, but v2.2 founders start feedforward.
    Recurrence is an emergent property, not a default.
    """

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
        rng: np.random.Generator | None = None,
    ) -> Genome:
        """Construct a proto-brain: sensors wired straight to motors.

        Default v2.2 topology is 3 smell sensors, 0 hidden, 2 motors,
        6 connections. Weights are tiny. Turn bias is 0; locomotion
        bias is sampled N(0, INITIAL_LOCOMOTION_BIAS_SIGMA) so the
        founding population mixes sitters and roamers. Sensory input
        then modulates that basal drive.

        Hidden neurons and recurrent edges are not gifted; they can
        appear later via structural mutation.

        If `rng` is provided, initial weights are sampled from it so
        each founder differs. If `rng` is None, a fresh default RNG
        is used.
        """
        if rng is None:
            rng = np.random.default_rng()
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
        # Both motors are TANH. Turn bias stays 0 so founders do not
        # spin; locomotion bias is sampled so basal activity is a
        # heritable gene, not a forced walk or a locked rest.
        for i in range(n_motors):
            nid = len(g.nodes)
            if i == 0:
                bias = 0.0
            else:
                bias = float(
                    np.clip(
                        rng.normal(0.0, INITIAL_LOCOMOTION_BIAS_SIGMA),
                        -BIAS_MAX,
                        BIAS_MAX,
                    )
                )
            g.nodes[nid] = NodeGene(
                id=nid,
                type=NodeType.MOTOR,
                activation=Activation.TANH,
                bias=bias,
            )
            motor_ids.append(nid)

        layer_in = sensor_ids
        layer_out = hidden_ids if hidden_ids else motor_ids
        innov = 0
        sigma = INITIAL_WEIGHT_SIGMA
        for src in layer_in:
            for dst in layer_out:
                g.connections[innov] = ConnectionGene(
                    innovation=innov,
                    in_node=src,
                    out_node=dst,
                    weight=float(rng.normal(0, sigma)),
                    enabled=True,
                )
                innov += 1
        if hidden_ids:
            for h in hidden_ids:
                for m in motor_ids:
                    g.connections[innov] = ConnectionGene(
                        innovation=innov,
                        in_node=h,
                        out_node=m,
                        weight=float(rng.normal(0, sigma)),
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
