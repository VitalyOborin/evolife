"""Memory advantage metric.

Runs a Phase 3 founder genome twice on the same scenario + seed:

  NORMAL    : brain state persists across ticks (default behaviour).
  ABLATED   : brain.reset_state() is called before every tick.

memory_advantage = performance(NORMAL) - performance(ABLATED)

If state doesn't help, the difference is ~0. If state matters for the
task, NORMAL wins.

We use a minimal Phase 3 scenario: uniform spawn of FoodA and FoodB,
no extra structure. We track only total_eaten and total_positive_eaten
(the latter only counts eats that gave positive reward). Positive
eaten is what natural selection acts on, and it is the cleanest
proxy for "did the brain do the right thing?".

Phase 3 scenarios are kept intentionally simple so the metric is
about state, not navigation. The MemoryEcologyWorld is reused as-is.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys

import numpy as np

from evolife.brain import Brain
from evolife.phase3 import MemoryEcologyWorld


def run_episode_phase3(
    genome,
    *,
    n_ticks: int,
    seed: int,
    reset_state: bool,
    world_kwargs: dict | None = None,
) -> dict:
    """One episode of a frozen genome in a MemoryEcologyWorld.

    If `reset_state` is True, brain.reset_state() is called every tick
    before forward(). Returns a dict with summary metrics.
    """
    world = MemoryEcologyWorld(seed=seed, **(world_kwargs or {}))
    brain = Brain(genome)
    org = world.organisms[0]  # founder slot
    org.brain = brain
    org.genome = genome

    # Energy bookkeeping for this episode.
    initial_energy = org.energy
    final_energy = org.energy
    food_eaten = 0
    positive_eaten = 0
    negative_eaten = 0
    reward_sum = 0.0
    distance = 0.0
    last_pos = (org.x, org.y)

    for _ in range(n_ticks):
        if reset_state:
            brain.reset_state()
        sensors = world._sensors_for(org)
        motors = brain.forward(sensors)
        turn = float(motors[0]) * 1.0  # MAX_TURN_RATE handled by world normally
        move = float(motors[1])
        org.heading = (org.heading + turn) % (2 * math.pi)
        org.x = (org.x + np.cos(org.heading) * move) % world.width
        org.y = (org.y + np.sin(org.heading) * move) % world.height
        # Update distance.
        dx = org.x - last_pos[0]
        dy = org.y - last_pos[1]
        dx -= world.width * np.round(dx / world.width)
        dy -= world.height * np.round(dy / world.height)
        distance += float(np.hypot(dx, dy))
        last_pos = (org.x, org.y)
        # Try to eat.
        ate = False
        for lst in (world.food_a, world.food_b):
            if not lst:
                continue
            # nearest
            fx = np.array([f.x for f in lst])
            fy = np.array([f.y for f in lst])
            ddx = fx - org.x
            ddy = fy - org.y
            ddx -= world.width * np.round(ddx / world.width)
            ddy -= world.height * np.round(ddy / world.height)
            dist = np.hypot(ddx, ddy)
            idx = int(np.argmin(dist))
            if dist[idx] <= 4.0:  # EAT_RADIUS default
                food_eaten += 1
                if isinstance(lst[idx], type(lst[0])) and lst[idx].__class__.__name__ == "FoodA":
                    reward = (
                        25.0 if world.season == 0 else -10.0
                    )
                else:
                    reward = (
                        25.0 if world.season == 1 else -10.0
                    )
                reward_sum += reward
                if reward > 0:
                    positive_eaten += 1
                else:
                    negative_eaten += 1
                lst.pop(idx)
                ate = True
                break
        # Drain a little to make the episode finite-cost.
        org.energy -= 0.02

    final_energy = org.energy
    return {
        "food_eaten": food_eaten,
        "positive_eaten": positive_eaten,
        "negative_eaten": negative_eaten,
        "reward_sum": reward_sum,
        "distance": distance,
        "final_energy": final_energy,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--genome-json", required=True,
                   help="Path to a Genome JSON serialised by Genome.to_dict()")
    p.add_argument("--n-ticks", type=int, default=200)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--out", type=str, default="evolife_memory_advantage.csv")
    args = p.parse_args()

    import json
    from evolife.genome import Genome
    with open(args.genome_json) as fh:
        g = Genome.from_dict(json.load(fh))

    rows = []
    for seed in range(args.n_seeds):
        norm = run_episode_phase3(g, n_ticks=args.n_ticks, seed=seed, reset_state=False)
        abl = run_episode_phase3(g, n_ticks=args.n_ticks, seed=seed, reset_state=True)
        rows.append({
            "seed": seed,
            "normal_food": norm["food_eaten"],
            "ablate_food": abl["food_eaten"],
            "food_advantage": norm["food_eaten"] - abl["food_eaten"],
            "normal_positive": norm["positive_eaten"],
            "ablate_positive": abl["positive_eaten"],
            "positive_advantage": norm["positive_eaten"] - abl["positive_eaten"],
            "normal_reward": norm["reward_sum"],
            "ablate_reward": abl["reward_sum"],
            "reward_advantage": norm["reward_sum"] - abl["reward_sum"],
        })

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")

    # Print aggregate.
    n = len(rows)
    print(f"Per-seed (n={n}):")
    print(f"{'seed':>4} {'N_food':>6} {'A_food':>6} {'adv_f':>6} "
          f"{'N_pos':>5} {'A_pos':>5} {'adv_p':>5}")
    for r in rows:
        print(f"{r['seed']:>4} {r['normal_food']:>6} {r['ablate_food']:>6} "
              f"{r['food_advantage']:>+6} {r['normal_positive']:>5} "
              f"{r['ablate_positive']:>5} {r['positive_advantage']:>+5}")
    mean_food_adv = sum(r["food_advantage"] for r in rows) / n
    mean_pos_adv = sum(r["positive_advantage"] for r in rows) / n
    mean_reward_adv = sum(r["reward_advantage"] for r in rows) / n
    print()
    print(f"Mean food_advantage      = {mean_food_adv:+.3f}")
    print(f"Mean positive_advantage  = {mean_pos_adv:+.3f}")
    print(f"Mean reward_advantage    = {mean_reward_adv:+.3f}")


if __name__ == "__main__":
    sys.exit(main())
