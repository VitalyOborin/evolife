import numpy as np
import pytest

import evolife.brain as brain_mod
from evolife.brain import Brain, _ITERATIONS
from evolife.config import N_MOTORS, N_SENSORS, locomotion_speed
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


def _set_sensor_to_locomotion_weights(g, weight: float) -> None:
    """Phase 1.5 founder has N_HIDDEN=1. To make a smell probe reliably
    drive the locomotion motor, set BOTH sides of the path:
      sensor->hidden = +1.0   (so full smell saturates the hidden to +1)
      hidden->motor[1] = weight  (the locomotion contribution)

    Hidden bias defaults to 0 so a zero sensor input keeps the hidden
    at 0, and the hidden->motor edge contributes nothing. With this
    wiring, full=ones vs zero=zero produces clean step-function control.
    """
    loc_id = g.motors()[1].id
    hidden_ids = {
        n.id for n in g.nodes.values() if n.type.value == "hidden"
    }
    for c in g.connections.values():
        if c.out_node in hidden_ids:
            c.weight = 1.0
        elif c.in_node in hidden_ids and c.out_node == loc_id:
            c.weight = weight


def test_smell_can_stop_a_weak_roamer():
    # Feedforward founder + persistent state: hidden is updated from the
    # *previous* tick's sensor input. So one tick of full smell saturates
    # the hidden node; the NEXT tick the hidden->motor edge actually
    # delivers its weight. Two ticks are needed to see the effect.
    g = Brain.make_default_genome(rng=np.random.default_rng(0))
    g.motors()[0].bias = 0.0
    g.motors()[1].bias = 0.08
    _set_sensor_to_locomotion_weights(g, -0.20)
    brain = Brain(g)
    none = np.zeros(N_SENSORS, dtype=np.float32)
    full = np.ones(N_SENSORS, dtype=np.float32)
    # Baseline roamer: zero smell -> locomotion > 0.
    assert locomotion_speed(float(brain.forward(none)[1])) > 0.0
    brain.reset_state()
    # Apply full smell: hidden saturates to ~1 on tick 1, takes effect
    # on tick 2.
    brain.forward(full)
    out = brain.forward(full)
    # motor[1] = 0.08 + (-0.20) * tanh(3) ~= 0.08 - 0.20 = -0.12,
    # after tanh still negative -> locomotion_speed == 0.
    assert locomotion_speed(float(out[1])) == 0.0


def test_smell_can_start_a_weak_sitter():
    g = Brain.make_default_genome(rng=np.random.default_rng(0))
    g.motors()[0].bias = 0.0
    g.motors()[1].bias = -0.07
    _set_sensor_to_locomotion_weights(g, 0.20)
    brain = Brain(g)
    none = np.zeros(N_SENSORS, dtype=np.float32)
    full = np.ones(N_SENSORS, dtype=np.float32)
    # Baseline sitter: zero smell -> motor[1] = -0.07, after tanh
    # negative -> locomotion_speed == 0.
    assert locomotion_speed(float(brain.forward(none)[1])) == 0.0
    brain.reset_state()
    # Full smell tick 1 primes hidden; tick 2 hidden->motor[1] = +0.20
    # -> motor[1] = -0.07 + 0.20 ~= 0.13, after tanh positive.
    brain.forward(full)
    out = brain.forward(full)
    assert locomotion_speed(float(out[1])) > 0.0


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


# --- Phase 4.1: eligibility-trace rHebb -------------------------------


@pytest.fixture
def trace_brain():
    """A 2-sensor → hidden → motor brain with PLASTICITY_TRACE on.

    Returns a brain whose compiled net has exactly 3 edges:
    sensor0→hidden, sensor1→hidden, hidden→motor. Pre-activations
    are easy to reason about: with sensor inputs (1, 0), the hidden
    node receives w_in*1 + w_in*0 = w_in on its only non-zero edge.
    """
    saved = {
        "PLASTICITY_TRACE": brain_mod.PLASTICITY_TRACE,
        "ELIGIBILITY_DECAY": brain_mod.ELIGIBILITY_DECAY,
        "SUPERLINEAR_POWER": brain_mod.SUPERLINEAR_POWER,
        "PLASTICITY_RATE": brain_mod.PLASTICITY_RATE,
    }
    brain_mod.PLASTICITY_TRACE = True
    brain_mod.ELIGIBILITY_DECAY = 0.9  # faster decay for test brevity
    brain_mod.SUPERLINEAR_POWER = 3
    brain_mod.PLASTICITY_RATE = 0.1
    try:
        g = Brain.make_default_genome()
        # Strip down to exactly 2 sensors + 1 hidden + 1 motor = 3 edges.
        # The default founder has 3 sensors → hidden and hidden → 2 motors.
        # We zero out the third sensor edge and second motor edge.
        edges_to_keep = []
        for c in g.connections.values():
            in_id = c.in_node
            out_id = c.out_node
            in_type = g.nodes[in_id].type.value
            out_type = g.nodes[out_id].type.value
            if in_type == "sensor" and out_type == "hidden":
                # keep only the first 2 of 3
                edges_to_keep.append(c.innovation)
        # Walk a list and trim.
        kept_innov = edges_to_keep[:2]
        for k in list(g.connections.keys()):
            if k not in kept_innov:
                if g.nodes[g.connections[k].out_node].type.value == "motor":
                    del g.connections[k]
        yield Brain(g)
    finally:
        brain_mod.PLASTICITY_TRACE = saved["PLASTICITY_TRACE"]
        brain_mod.ELIGIBILITY_DECAY = saved["ELIGIBILITY_DECAY"]
        brain_mod.SUPERLINEAR_POWER = saved["SUPERLINEAR_POWER"]
        brain_mod.PLASTICITY_RATE = saved["PLASTICITY_RATE"]


def test_phase4_1_trace_no_op_when_disabled():
    """PLASTICITY_TRACE=False must short-circuit before allocating arrays."""
    saved = brain_mod.PLASTICITY_TRACE
    brain_mod.PLASTICITY_TRACE = False
    try:
        g = Brain.make_default_genome()
        brain = Brain(g)
        sensors = np.ones(N_SENSORS, dtype=np.float32)
        brain.forward(sensors)
        # No-op when disabled: trace arrays stay unallocated.
        assert brain._e_trace is None
        brain.commit_episode()  # also no-op
        assert brain._e_trace is None
    finally:
        brain_mod.PLASTICITY_TRACE = saved


def test_phase4_1_begin_episode_resets_trace(trace_brain):
    """begin_episode must zero the eligibility trace and reward accumulator."""
    # Drive a few forward passes first.
    for _ in range(5):
        trace_brain.forward(np.ones(N_SENSORS, dtype=np.float32))
    # Trace should be non-zero.
    assert trace_brain._e_trace is not None
    assert np.any(trace_brain._e_trace != 0.0)
    trace_brain.accumulate_episode_reward(2.0)
    assert trace_brain._episode_reward_acc == 2.0
    trace_brain.begin_episode()
    assert np.all(trace_brain._e_trace == 0.0)
    assert trace_brain._episode_reward_acc == 0.0


def test_phase4_1_commit_episode_3factor_rule(trace_brain):
    """RPE=R-Rb drives Δw = η*(R-Rb)*e. Positive RPE should strengthen
    edges with positive eligibility; negative RPE should weaken them."""
    initial_w = trace_brain.genome.connections[0].weight
    for _ in range(10):
        trace_brain.forward(np.ones(N_SENSORS, dtype=np.float32))
    # Force a positive episode reward well above the baseline.
    trace_brain.accumulate_episode_reward(5.0)
    # Snapshot trace so we can compare Δw direction.
    e_pre = trace_brain._e_trace.copy()
    # Commit positive RPE.
    trace_brain.commit_episode()
    # Weights must have moved. Direction depends on sign of (R - Rb)*e.
    # R_b defaults to 0 so rpe = 5; weight change = 0.1 * 5 * e_pre.
    # But weights are clamped to PLASTICITY_WEIGHT_CLAMP. Verify weights
    # moved in the expected direction at least for the first edge.
    innov_first = 0
    new_w_first = trace_brain.genome.connections[innov_first].weight
    expected_delta = 0.1 * 5.0 * e_pre[0]
    assert abs(new_w_first - (initial_w + expected_delta)) < 1e-5


def test_phase4_1_commit_resets_baseline(trace_brain):
    """After a commit, _r_baseline moves toward the episode reward."""
    trace_brain._r_baseline = 0.0
    trace_brain.accumulate_episode_reward(3.0)
    trace_brain.commit_episode()
    # EMA coefficient is 0.99, so baseline should be 0.99*0 + 0.01*3 = 0.03.
    assert abs(trace_brain._r_baseline - 0.03) < 1e-6


def test_phase4_1_commit_resets_trace_and_reward(trace_brain):
    """After commit, trace and reward accumulator both go to zero."""
    for _ in range(3):
        trace_brain.forward(np.ones(N_SENSORS, dtype=np.float32))
    trace_brain.accumulate_episode_reward(2.5)
    assert trace_brain._episode_reward_acc != 0.0
    trace_brain.commit_episode()
    assert trace_brain._episode_reward_acc == 0.0
    assert np.all(trace_brain._e_trace == 0.0)


def test_phase4_1_superlinear_damps_small_signal(trace_brain):
    """A small raw pre*post should produce a much smaller |e| increment
    than a large one — the Miconi supralinear amplification."""
    # Drive a forward pass first so _last_pre/_last_post exist with
    # the right shape (matches the compiled net's node count).
    sensors = np.ones(N_SENSORS, dtype=np.float32)
    trace_brain.forward(sensors)
    pre = trace_brain._last_pre.copy()
    post = trace_brain._last_post.copy()
    # Set a small pre*post on all nodes: e.g. all activations = 0.1
    pre[:] = 0.1
    post[:] = 0.1
    trace_brain._accumulate_trace(pre, post)
    small_max = float(np.max(np.abs(trace_brain._e_trace)))
    # Reset trace and try a large pre*post: activations = 0.9
    trace_brain._e_trace.fill(0.0)
    pre[:] = 0.9
    post[:] = 0.9
    trace_brain._accumulate_trace(pre, post)
    big_max = float(np.max(np.abs(trace_brain._e_trace)))
    # 0.9^3 ~= 0.729 vs 0.1^3 = 0.001. Big should be much larger.
    assert big_max > small_max * 100, (
        f"supralinear not amplifying: small={small_max} big={big_max}"
    )
