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


# Phase 4 lifetime synaptic plasticity. Phase 3 (and earlier) ignore
# these and structural mutation alone drives all weight change.
PLASTICITY_ALPHA: float = 0.0   # local Hebbian term (pre * post)
PLASTICITY_BETA: float = 0.0    # reward-modulated term (RPE * pre * post)
PLASTICITY_WEIGHT_CLAMP: float = 5.0  # keep weights bounded

# Phase 4.1 eligibility-trace reward-modulated Hebbian plasticity
# (Miconi 2017 rHebb). When PLASTICITY_TRACE is True, each forward
# pass accumulates a per-edge eligibility trace e_ij, and episodes
# commit the accumulated trace weighted by reward-prediction-error.
# See PHASE4_LIT.md for the design rationale.
PLASTICITY_TRACE: bool = False      # master switch for Phase 4.1
ELIGIBILITY_DECAY: float = 0.95     # lambda in e(t) = lam * e(t-1) + ...
ELIGIBILITY_BASELINE_EMA: float = 0.99  # EMA for the per-edge <pre*post>
SUPERLINEAR_POWER: int = 3          # S(x) = sign(x) * |x|^k, Miconi's non-linearity
PLASTICITY_RATE: float = 0.005      # eta in Delta_w = eta (R - Rb) e
REWARD_BASELINE_EMA: float = 0.99   # EMA decay for R_b (organism-level)


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
        # Phase 4: pre-activations from the last forward pass, kept
        # for plasticity updates. None until forward() runs.
        self._last_pre: np.ndarray | None = None
        self._last_post: np.ndarray | None = None
        # Phase 4.1: per-edge eligibility trace, per-edge Hebbian
        # baseline EMA, organism-level reward baseline, and accumulated
        # episode reward. Allocated lazily so the default (Phase 3)
        # brain has zero per-tick overhead.
        self._e_trace: np.ndarray | None = None
        self._e_baseline: np.ndarray | None = None
        self._r_baseline: float = 0.0
        self._episode_reward_acc: float = 0.0

    def forward(self, sensors: np.ndarray) -> np.ndarray:
        net = self._ensure_compiled()

        # Sensor count comes from the *compiled* net, not the global
        # config. Phase 3 founder has 7 sensor nodes (6 smell + 1
        # feedback); legacy founder has 3. The expected sensor vector
        # length equals the number of sensor nodes in the topology.
        n_sensor = int(net.is_sensor.sum())
        if sensors.shape != (n_sensor,):
            raise ValueError(
                f"Expected sensor vector of shape ({n_sensor},), "
                f"got {sensors.shape}"
            )

        # Initialise state on first use or after topology change.
        if self.state is None or self.state.shape != (net.n_nodes,):
            self.state = np.zeros(net.n_nodes, dtype=np.float32)

        # Sensors overwrite the first n_sensor positions every tick.
        # This is the "perception" boundary: sensor values come from
        # the world, not from internal state.
        # Snapshot the *previous* state as pre-activations BEFORE
        # we overwrite sensors (recurrent edges can read hidden->hidden
        # from prior tick).
        pre = self.state.copy()
        self.state[:n_sensor] = sensors

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

        # Cache pre/post activations for Phase 4 plasticity. pre was
        # the state BEFORE this forward pass wrote sensors; the post
        # is the post-activation state right now.
        self._last_pre = pre
        self._last_post = self.state.copy()

        # Phase 4.1: eligibility-trace accumulation. No-op if the
        # trace is disabled (Phase 3 default). Allocated lazily so
        # the no-plasticity path stays zero-cost.
        if PLASTICITY_TRACE and net.in_idx.size > 0:
            self._accumulate_trace(pre, self.state)

        motor_vals = self.state[net.motor_idx]
        return np.clip(motor_vals, -1.0, 1.0).astype(np.float32)

    def apply_plasticity(self, reward_signal: float) -> None:
        """Phase 4 lifetime synaptic plasticity update.

        Applies a local Hebbian term plus a reward-modulated term
        to every active edge in the compiled network. Reward signal
        is the intake_feedback at the current tick (e.g. +1.0 or
        -1.0 or 0.0 if no recent eat).

        Updates the in-place compiled weights AND the genome's
        connection weight for next reproduction, then clamps to
        PLASTICITY_WEIGHT_CLAMP. No-op if PLASTICITY_ALPHA and
        PLASTICITY_BETA are both zero (Phase 3 default) or if
        forward() hasn't run yet.
        """
        if PLASTICITY_ALPHA == 0.0 and PLASTICITY_BETA == 0.0:
            return
        if self._last_pre is None or self._last_post is None:
            return
        net = self._ensure_compiled()
        if net.in_idx.size == 0:
            return
        pre_vals = self._last_pre[net.in_idx]
        post_vals = self._last_post[net.out_idx]
        target = post_vals
        local = pre_vals * (post_vals - target)
        modulation = PLASTICITY_BETA * reward_signal * pre_vals * post_vals
        delta = (PLASTICITY_ALPHA * local + modulation).astype(np.float32)
        # Apply in-place to compiled weights.
        new_w = net.weights + delta
        np.clip(new_w, -PLASTICITY_WEIGHT_CLAMP, PLASTICITY_WEIGHT_CLAMP,
                out=new_w)
        net.weights = new_w
        # Mirror back into genome.connections for inheritance.
        # The compiled net's (in_idx, out_idx) are positional; the genome
        # connections are keyed by innovation id. Cache the mapping.
        cache = getattr(self, "_edge_to_innov", None)
        if cache is None or len(cache) != net.in_idx.size:
            cache = {}
            for k, c in self.genome.connections.items():
                cache[(c.in_node, c.out_node)] = k
            self._edge_to_innov = cache
        # Vectorise the genome update via numpy indexing is hard because
        # cache is heterogeneous; this loop runs once per tick and is
        # acceptable for Phase 4's opt-in plasticity.
        for i in range(net.in_idx.size):
            k = cache.get((int(net.in_idx[i]), int(net.out_idx[i])))
            if k is not None:
                self.genome.connections[k].weight = float(new_w[i])

    # --- Phase 4.1 eligibility-trace plasticity (Miconi 2017 rHebb) -----

    def _ensure_trace(self, n_edges: int) -> None:
        """Allocate per-edge trace + baseline if absent."""
        if self._e_trace is None or self._e_trace.size != n_edges:
            self._e_trace = np.zeros(n_edges, dtype=np.float32)
            self._e_baseline = np.zeros(n_edges, dtype=np.float32)

    def _accumulate_trace(self, pre_full: np.ndarray, post_full: np.ndarray) -> None:
        """One step of Miconi-style eligibility-trace accumulation.

        e_ij(t) = lambda * e_ij(t-1) + S(pre_i * (post_j - <pre*post>_ema))
        where S(x) = sign(x) * |x|^SUPERLINEAR_POWER and the per-edge
        baseline tracks the running mean of pre*post. The decay lambda
        and EMA coefficient come from module constants; this method is
        called once per forward() pass when PLASTICITY_TRACE is on.
        """
        net = self._ensure_compiled()
        self._ensure_trace(net.in_idx.size)
        pre_vals = pre_full[net.in_idx].astype(np.float32)
        post_vals = post_full[net.out_idx].astype(np.float32)
        raw = pre_vals * post_vals
        # Update per-edge baseline EMA: b <- c * b + (1 - c) * raw
        self._e_baseline *= ELIGIBILITY_BASELINE_EMA
        self._e_baseline += (1.0 - ELIGIBILITY_BASELINE_EMA) * raw
        centered = raw - self._e_baseline
        # Supralinear: keep sign, raise magnitude. Small |x| damped, large
        # co-activation amplified. Use np.copysign for vectorised sign.
        k = SUPERLINEAR_POWER
        nonlin = np.copysign(np.abs(centered) ** k, centered).astype(np.float32)
        self._e_trace *= ELIGIBILITY_DECAY
        self._e_trace += nonlin

    def accumulate_episode_reward(self, r: float) -> None:
        """Add a tick-level reward signal to the running episode total.

        Called from the world's eat handler; the running sum is what
        gets committed at episode boundaries (reproduction or death).
        """
        self._episode_reward_acc += float(r)

    def begin_episode(self) -> None:
        """Start a new lifetime episode. Resets eligibility trace and
        the per-episode reward accumulator. Keeps _r_baseline (the
        organism-level reward baseline is preserved across episodes
        so a long-lived individual learns RPE-style).
        """
        if self._e_trace is not None:
            self._e_trace.fill(0.0)
        self._episode_reward_acc = 0.0

    def commit_episode(self) -> None:
        """End-of-episode plasticity commit (Miconi 2017).

        Delta_w_ij = PLASTICITY_RATE * (R - R_b) * e_ij(T)
        then update R_b <- EMA(R_b, R), clamp weights, mirror back to
        genome for inheritance, reset trace. No-op if trace disabled,
        no edges, or accumulated reward is exactly zero (degenerate
        case: starve at birth before eating anything).
        """
        if not PLASTICITY_TRACE:
            return
        R = float(self._episode_reward_acc)
        Rb = float(self._r_baseline)
        # Update reward baseline first — this is independent of the
        # trace state. Even on degenerate episodes (no edges, no
        # forward call yet) we want R_b to track running reward.
        self._r_baseline = REWARD_BASELINE_EMA * Rb + (1.0 - REWARD_BASELINE_EMA) * R
        net = self._ensure_compiled()
        if (
            self._e_trace is None
            or net.in_idx.size == 0
            or self._e_trace.size != net.in_idx.size
        ):
            # No edges or trace not yet allocated: nothing to learn.
            self._episode_reward_acc = 0.0
            return
        rpe = R - Rb
        if rpe == 0.0:
            # No learning signal this episode. Baseline already updated
            # above; just reset trace and accumulator.
            if self._e_trace is not None:
                self._e_trace.fill(0.0)
            self._episode_reward_acc = 0.0
            return
        delta = (PLASTICITY_RATE * rpe * self._e_trace).astype(np.float32)
        new_w = net.weights + delta
        np.clip(new_w, -PLASTICITY_WEIGHT_CLAMP, PLASTICITY_WEIGHT_CLAMP,
                out=new_w)
        net.weights = new_w
        # Mirror to genome.connections so inherited offspring inherit
        # the lifetime-shaped weights. Same edge cache as Phase 4.
        cache = getattr(self, "_edge_to_innov", None)
        if cache is None or len(cache) != net.in_idx.size:
            cache = {}
            for k, c in self.genome.connections.items():
                cache[(c.in_node, c.out_node)] = k
            self._edge_to_innov = cache
        for i in range(net.in_idx.size):
            k = cache.get((int(net.in_idx[i]), int(net.out_idx[i])))
            if k is not None:
                self.genome.connections[k].weight = float(new_w[i])
        # Reward baseline already updated above. Reset trace for next
        # episode; keep _r_baseline.
        self._e_trace.fill(0.0)
        self._episode_reward_acc = 0.0

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
        6 connections. Locomotion bias is a weak prior
        N(0, INITIAL_LOCOMOTION_BIAS_SIGMA); sensory weights are larger
        so smell can flip rest ↔ move. Turn bias stays 0.

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
