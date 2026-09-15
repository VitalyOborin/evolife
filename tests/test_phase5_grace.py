"""Tests for the Phase 5 grace period after a season flip.

The grace period softens the negative-food penalty linearly from
PHASE3_GRACE_PENALTY (at flip) to the full PHASE3_FOOD_*_NEGATIVE_ENERGY
(at end of PHASE3_GRACE_TICKS). This gives the brain time to adapt
to the new sign of reward without immediate mass starvation.

Validation covers:
- Defaults are 200 ticks / -0.5 soft penalty.
- Setting PHASE3_GRACE_TICKS=0 disables the mechanism (Phase 3/4 compat).
- grace_ticks_remaining resets to GRACE_TICKS on each season flip.
- Linear interpolation: t=GRACE_TICKS -> soft, t=0 -> hard, t=GRACE/2 -> mid.
- Positive food is unaffected by grace.
"""
from __future__ import annotations

import pytest

import evolife.config as cfg
import evolife.phase3 as ph3
from evolife.phase3 import MemoryEcologyWorld


@pytest.fixture(autouse=True)
def _restore_grace_defaults():
    """Snapshot and restore PHASE3_GRACE_* so tests don't leak config."""
    saved = (
        cfg.PHASE3_GRACE_TICKS,
        ph3.PHASE3_GRACE_TICKS,
        cfg.PHASE3_GRACE_PENALTY,
        ph3.PHASE3_GRACE_PENALTY,
    )
    yield
    (
        cfg.PHASE3_GRACE_TICKS,
        ph3.PHASE3_GRACE_TICKS,
        cfg.PHASE3_GRACE_PENALTY,
        ph3.PHASE3_GRACE_PENALTY,
    ) = saved


def test_phase5_grace_defaults_match_spec():
    """Spec: 200 ticks of soft penalty, soft = -0.5."""
    assert cfg.PHASE3_GRACE_TICKS == 200
    assert cfg.PHASE3_GRACE_PENALTY == -0.5


def test_phase5_grace_ticks_init_zero():
    """grace_ticks_remaining starts at 0 (no grace until first flip)."""
    w = MemoryEcologyWorld(seed=1, mode="hidden_season")
    assert w.grace_ticks_remaining == 0


def test_phase5_grace_disabled_when_zero():
    """PHASE3_GRACE_TICKS=0 must be a clean no-op."""
    cfg.PHASE3_GRACE_TICKS = 0
    ph3.PHASE3_GRACE_TICKS = 0
    w = MemoryEcologyWorld(seed=2, mode="hidden_season")
    # Step past SEASON_LENGTH to force a flip.
    from evolife.config import SEASON_LENGTH
    for _ in range(SEASON_LENGTH + 1):
        w.step()
    # grace was never armed because GRACE_TICKS == 0.
    assert w.grace_ticks_remaining == 0


def test_phase5_grace_resets_on_flip():
    """After SEASON_LENGTH ticks, grace_ticks_remaining resets to GRACE_TICKS."""
    cfg.PHASE3_GRACE_TICKS = 100
    ph3.PHASE3_GRACE_TICKS = 100
    w = MemoryEcologyWorld(seed=3, mode="hidden_season")
    from evolife.config import SEASON_LENGTH
    for _ in range(SEASON_LENGTH + 1):
        w.step()
    # Right after the flip, grace was set to 100, then decremented once
    # by the same step that incremented ticks_in_season.
    assert w.grace_ticks_remaining == 100 - 1


def test_phase5_grace_decrements_each_tick():
    """Each tick decrements grace_ticks_remaining by 1."""
    cfg.PHASE3_GRACE_TICKS = 50
    ph3.PHASE3_GRACE_TICKS = 50
    w = MemoryEcologyWorld(seed=4, mode="hidden_season")
    from evolife.config import SEASON_LENGTH
    # Get past the flip.
    for _ in range(SEASON_LENGTH + 1):
        w.step()
    grace_after_flip = w.grace_ticks_remaining
    # Step 10 more ticks; grace should drop by 10.
    for _ in range(10):
        w.step()
    assert w.grace_ticks_remaining == grace_after_flip - 10


def test_phase5_grace_linear_interpolation_endpoints():
    """At grace=GRACE -> soft_penalty; at grace=0 -> full penalty."""
    # Set up a controlled scenario with one organism near food.
    cfg.PHASE3_GRACE_TICKS = 100
    cfg.PHASE3_GRACE_PENALTY = -0.5
    cfg.PHASE3_FOOD_A_NEGATIVE_ENERGY = -3.0
    cfg.PHASE3_FOOD_B_NEGATIVE_ENERGY = -3.0
    ph3.PHASE3_GRACE_TICKS = 100
    ph3.PHASE3_GRACE_PENALTY = -0.5
    ph3.PHASE3_FOOD_A_NEGATIVE_ENERGY = -3.0
    ph3.PHASE3_FOOD_B_NEGATIVE_ENERGY = -3.0
    from evolife.config import SEASON_LENGTH, EAT_RADIUS

    w = MemoryEcologyWorld(seed=5, mode="hidden_season")
    # Force season=1 (where food_a is negative).
    w.season = 1
    # Step just past SEASON_LENGTH so the world doesn't flip back to 0.
    # We need to skip the flip logic; instead, manually place the org
    # on top of food_a and call _resolve_eat_phase3 with grace set.
    org = w.organisms[0]
    org.x, org.y = w.food_a[0].x, w.food_a[0].y
    # Snapshot food list; we'll repopulate after each test.
    food_snapshot = list(w.food_a)
    # Case 1: grace just opened -> reward should equal soft_penalty.
    w.grace_ticks_remaining = 100  # == GRACE_TICKS
    # Re-add food so the eat resolves.
    w.food_a.append(food_snapshot[0])
    org.energy = 50.0
    w._resolve_eat_phase3()
    e_after_soft = org.energy
    # soft penalty: -0.5. Org started at 50, so energy = 50 + (-0.5) = 49.5.
    assert abs(e_after_soft - 49.5) < 1e-5, (
        f"At grace=GRACE: expected -0.5, got energy delta {(e_after_soft - 50.0)}"
    )

    # Case 2: grace at end -> reward should equal full penalty.
    w.food_a.append(food_snapshot[0])
    org.x, org.y = w.food_a[0].x, w.food_a[0].y
    org.energy = 50.0
    w.grace_ticks_remaining = 0
    w._resolve_eat_phase3()
    e_after_hard = org.energy
    # full penalty: -3.0. Energy = 50 + (-3) = 47.
    assert abs(e_after_hard - 47.0) < 1e-5, (
        f"At grace=0: expected -3.0, got energy delta {(e_after_hard - 50.0)}"
    )

    # Case 3: grace halfway -> reward should be midpoint.
    w.food_a.append(food_snapshot[0])
    org.x, org.y = w.food_a[0].x, w.food_a[0].y
    org.energy = 50.0
    w.grace_ticks_remaining = 50  # halfway
    w._resolve_eat_phase3()
    e_mid = org.energy
    # midpoint: -0.5*0.5 + -3.0*0.5 = -1.75. Energy = 50 + (-1.75) = 48.25.
    assert abs(e_mid - 48.25) < 1e-5, (
        f"At grace=GRACE/2: expected -1.75, got energy delta {(e_mid - 50.0)}"
    )


def test_phase5_grace_does_not_affect_positive_food():
    """Positive food reward is independent of grace state."""
    cfg.PHASE3_GRACE_TICKS = 100
    cfg.PHASE3_GRACE_PENALTY = -0.5
    cfg.PHASE3_FOOD_A_POSITIVE_ENERGY = 25.0
    ph3.PHASE3_GRACE_TICKS = 100
    ph3.PHASE3_GRACE_PENALTY = -0.5
    ph3.PHASE3_FOOD_A_POSITIVE_ENERGY = 25.0

    w = MemoryEcologyWorld(seed=6, mode="hidden_season")
    w.season = 0  # food_a is positive in season 0.
    org = w.organisms[0]
    food_snapshot = list(w.food_a)
    # With grace fully open.
    w.food_a.append(food_snapshot[0])
    org.x, org.y = w.food_a[0].x, w.food_a[0].y
    org.energy = 50.0
    w.grace_ticks_remaining = 100
    w._resolve_eat_phase3()
    assert abs(org.energy - 75.0) < 1e-5  # 50 + 25
