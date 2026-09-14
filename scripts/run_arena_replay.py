"""Ancestor vs Descendant replay.

Picks two frozen genomes (typically `ancestor_generation=0` and
`descendant_generation` from a living world), runs them side-by-side
in identical ArenaWorlds with the same seed, and visualises their
trajectories.

Two output modes:
  --ascii   : write an ANSI-coloured grid to stdout (terminal-friendly).
  --png     : write a side-by-side PNG with trail lines (requires
              matplotlib).
  --both    : both.

The trail length is capped to `--trail` ticks (default 300). Food is
shown as `*`, current position as `A` (ancestor) and `D` (descendant).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evolife.arena import (
    ArenaWorld,
    brain_shape,
    uniform_scenario,
)
from evolife.organism import Organism
from evolife.world import World


def run_with_trail(
    genome, scenario_kind: str, seed: int, n_ticks: int
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], int]:
    """Run an arena episode and return (trail, food_trail, food_eaten).

    `trail` is the organism's (x, y) per tick. `food_trail` is a
    snapshot of food positions at each tick — useful for relocating
    scenarios. For uniform scenarios we only need a single food
    snapshot.
    """
    spec = uniform_scenario(seed) if scenario_kind == "uniform" else None
    if spec is None:
        from evolife.arena import SCENARIO_BUILDERS

        spec = SCENARIO_BUILDERS[scenario_kind](seed)
    arena = ArenaWorld(spec, genome)
    arena._scenario = spec
    trail = [(arena.organisms[0].x, arena.organisms[0].y)]
    food_snapshot = [(f.x, f.y) for f in arena.food]
    for _ in range(n_ticks):
        if not arena.organisms or not arena.organisms[0].alive:
            break
        arena.step()
        arena._maybe_relocate_food(spec)
        org = arena.organisms[0]
        trail.append((org.x, org.y))
    return trail, food_snapshot, org.food_eaten


def render_ascii(
    trail_a, trail_d, food, width: int, height: int,
    downsample: int, trail_len: int
) -> str:
    """Render two trails side by side on an ASCII grid.

    The grid is `width // downsample` columns wide and the same for
    height. Both trails are clipped to the last `trail_len` ticks.
    """
    grid_w = max(8, width // downsample)
    grid_h = max(8, height // downsample)
    cols_a = [[" "] * grid_w for _ in range(grid_h)]
    cols_d = [[" "] * grid_w for _ in range(grid_h)]

    def cell(trail, cols, marker):
        for i, (x, y) in enumerate(trail[-trail_len:]):
            cx = int(x // downsample) % grid_w
            cy = int(y // downsample) % grid_h
            cols[cy][cx] = marker
        # Food positions (only on the left grid to keep output compact).
        return cols

    cell(trail_a, cols_a, "a")
    cell(trail_d, cols_d, "d")

    # Food on both panels.
    food_chars_a = [[" "] * grid_w for _ in range(grid_h)]
    food_chars_d = [[" "] * grid_w for _ in range(grid_h)]
    for fx, fy in food:
        cx = int(fx // downsample) % grid_w
        cy = int(fy // downsample) % grid_h
        food_chars_a[cy][cx] = "*"
        food_chars_d[cy][cx] = "*"

    lines = []
    lines.append(f"=== ANCESTOR (gen 0) | DESCENDANT (gen N) ===")
    lines.append(f"world {width}x{height}  grid {grid_w}x{grid_h}  "
                 f"trail last {trail_len} ticks")
    lines.append("legend: '.' empty  '*' food  'a'/'A' ancestor  'd'/'D' descendant")
    lines.append("")
    # Header.
    pad = " " * 4
    header_a = "ANCESTOR"
    header_d = "DESCENDANT"
    lines.append(f"{pad}{header_a:<{grid_w}}  {header_d:<{grid_w}}")
    for row in range(grid_h):
        line_a = "".join(
            food_chars_a[row][c] if food_chars_a[row][c] != " " else cols_a[row][c]
            for c in range(grid_w)
        )
        line_d = "".join(
            food_chars_d[row][c] if food_chars_d[row][c] != " " else cols_d[row][c]
            for c in range(grid_w)
        )
        lines.append(f"{pad}{line_a}  {line_d}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--evo-ticks", type=int, default=20_000)
    p.add_argument("--ancestor-gen", type=int, default=0)
    p.add_argument("--descendant-gen", type=int, default=20)
    p.add_argument("--scenario", type=str, default="uniform")
    p.add_argument("--n-ticks", type=int, default=600)
    p.add_argument("--trail", type=int, default=300)
    p.add_argument("--out", type=Path, default=Path("evolife_arena_replay.json"))
    p.add_argument("--ascii", action="store_true", default=True)
    p.add_argument("--png", action="store_true", default=False)
    args = p.parse_args()

    print(f"[replay] running evolution: ticks={args.evo_ticks}, seed={args.seed}")
    world = World(seed=args.seed)
    for _ in range(args.evo_ticks):
        world.step()
    max_gen = world.max_generation()
    print(f"[replay] evolution done: final_pop={world.population()} maxGen={max_gen}")

    by_gen: dict[int, Organism] = {}
    for org in world.organisms:
        if not org.alive or org.genome is None:
            continue
        if org.generation not in by_gen:
            by_gen[org.generation] = org

    if args.ancestor_gen not in by_gen:
        # Fall back to the lowest available generation.
        ancestor_g = min(by_gen)
        print(
            f"[replay] no ancestor at gen {args.ancestor_gen}; "
            f"falling back to gen {ancestor_g}"
        )
    else:
        ancestor_g = args.ancestor_gen
    desc_g = min(args.descendant_gen, max(by_gen))
    if desc_g not in by_gen:
        print(f"[replay] no descendant at gen {desc_g}, available: "
              f"{sorted(by_gen.keys())}")
        return

    anc = by_gen[ancestor_g].genome
    desc = by_gen[desc_g].genome
    print(
        f"[replay] ancestor (gen {ancestor_g}): "
        f"brain_shape={brain_shape(anc)}"
    )
    print(
        f"[replay] descendant (gen {desc_g}): "
        f"brain_shape={brain_shape(desc)}"
    )

    seed = args.seed
    print(f"[replay] running {args.scenario} arena for {args.n_ticks} ticks "
          f"at seed={seed}")
    trail_a, food, food_eaten_a = run_with_trail(
        anc, args.scenario, seed, args.n_ticks
    )
    trail_d, _, food_eaten_d = run_with_trail(
        desc, args.scenario, seed, args.n_ticks
    )
    print(
        f"[replay] food_eaten: ancestor={food_eaten_a}  "
        f"descendant={food_eaten_d}"
    )

    payload = {
        "scenario": args.scenario,
        "seed": seed,
        "ancestor": {
            "generation": ancestor_g,
            "trail": trail_a,
            "food_eaten": food_eaten_a,
        },
        "descendant": {
            "generation": desc_g,
            "trail": trail_d,
            "food_eaten": food_eaten_d,
        },
        "food": food,
    }
    args.out.write_text(json.dumps(payload))
    print(f"[replay] wrote {args.out}")

    if args.ascii:
        from evolife.config import WORLD_HEIGHT, WORLD_WIDTH

        text = render_ascii(
            trail_a, trail_d, food,
            WORLD_WIDTH, WORLD_HEIGHT,
            downsample=8, trail_len=args.trail,
        )
        print()
        print(text)


if __name__ == "__main__":
    main()
