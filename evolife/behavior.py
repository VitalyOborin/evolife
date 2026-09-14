"""Per-tick behavioral accumulators and descriptors.

Observational only. These describe phenotype; they are not fitness.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .config import EXPLORE_BIN, TURN_ACTION_THRESHOLD

if TYPE_CHECKING:
    from .organism import Organism

N_SMELL_BINS = 3
N_SENSOR_STATES = 27  # 3^3
N_ACTIONS = 4  # REST, FORWARD, TURN_LEFT, TURN_RIGHT
REST, FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3


def note_act(
    org: "Organism",
    sensors,
    turn: float,
    speed: float,
    width: int,
    height: int,
) -> None:
    """Update locomotion + sensory-coupling stats for one tick."""
    org.note_locomotion(speed)
    org.distance_sum += speed
    org.abs_turn_sum += abs(turn)

    bx = int(org.x) // EXPLORE_BIN
    by = int(org.y) // EXPLORE_BIN
    nx = max(1, width // EXPLORE_BIN)
    org.explored_bins.add(by * nx + bx)

    left, _front, right = float(sensors[0]), float(sensors[1]), float(sensors[2])
    asym = right - left
    org.steer_n += 1
    org.steer_sum_x += asym
    org.steer_sum_y += turn
    org.steer_sum_xy += asym * turn
    org.steer_sum_x2 += asym * asym
    org.steer_sum_y2 += turn * turn

    s_state = _sensor_state(sensors)
    action = _action(turn, speed)
    org.action_counts[s_state * N_ACTIONS + action] += 1


def descriptors(org: "Organism") -> dict[str, float]:
    age = max(org.age, 1)
    moving = max(org.ticks_moving, 1)
    rest_ticks = max(age - org.ticks_moving, 0)
    n_move_bouts = max((org.movement_transitions + 1) // 2, 1)
    n_rest_bouts = max((org.movement_transitions + 2) // 2, 1)
    food = org.food_eaten
    return {
        "moving_fraction": org.ticks_moving / age,
        "transition_rate": 1000.0 * org.movement_transitions / age,
        "mean_speed": org.speed_sum / age,
        "mean_speed_while_moving": org.speed_sum / moving if org.ticks_moving else 0.0,
        "mean_rest_bout": rest_ticks / n_rest_bouts,
        "mean_move_bout": org.ticks_moving / n_move_bouts,
        "longest_rest": float(org.longest_rest),
        "longest_move": float(org.longest_move),
        "food_rate": 1000.0 * food / age,
        "distance_per_food": (
            org.distance_sum / food if food else org.distance_sum
        ),
        "time_to_first_food": float(
            org.time_to_first_food if org.time_to_first_food is not None else age
        ),
        "mean_abs_turn": org.abs_turn_sum / age,
        "turns_per_distance": org.abs_turn_sum / max(org.distance_sum, 1e-9),
        "exploration_rate": 1000.0 * len(org.explored_bins) / age,
        "steering_alignment": steering_alignment(org),
        "state_dependence": state_dependence(org),
    }


def steering_alignment(org: "Organism") -> float:
    n = org.steer_n
    if n < 2:
        return 0.0
    mx = org.steer_sum_x / n
    my = org.steer_sum_y / n
    cov = org.steer_sum_xy / n - mx * my
    vx = org.steer_sum_x2 / n - mx * mx
    vy = org.steer_sum_y2 / n - my * my
    denom = math.sqrt(max(vx, 0.0) * max(vy, 0.0))
    if denom < 1e-12:
        return 0.0
    return float(max(-1.0, min(1.0, cov / denom)))


def state_dependence(org: "Organism") -> float:
    """H(Action | Sensors). ~0 for a deterministic feed-forward reflex."""
    counts = org.action_counts
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for s in range(N_SENSOR_STATES):
        row = counts[s * N_ACTIONS : (s + 1) * N_ACTIONS]
        n_s = sum(row)
        if n_s == 0:
            continue
        p_s = n_s / total
        h_as = 0.0
        for n_a in row:
            if n_a == 0:
                continue
            p = n_a / n_s
            h_as -= p * math.log2(p)
        h += p_s * h_as
    return float(h)


def _sensor_state(sensors) -> int:
    l = _bin(float(sensors[0]))
    f = _bin(float(sensors[1]))
    r = _bin(float(sensors[2]))
    return l * 9 + f * 3 + r


def _bin(v: float) -> int:
    if v < 0.33:
        return 0
    if v < 0.66:
        return 1
    return 2


def _action(turn: float, speed: float) -> int:
    if speed <= 0.0:
        return REST
    if turn > TURN_ACTION_THRESHOLD:
        return TURN_LEFT
    if turn < -TURN_ACTION_THRESHOLD:
        return TURN_RIGHT
    return FORWARD
