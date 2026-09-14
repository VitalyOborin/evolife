"""Regenerate a warm-start parent genome for Phase 3.2.

Runs the Phase 1.5 world for a few thousand ticks, lets evolution
discover navigation, then dumps a parent's genome as JSON. We pick a
parent whose child is the most-fit at end-of-run (i.e. the genome
that successfully reproduced, not just any founder).
"""
from __future__ import annotations

import json
import sys

from evolife.metrics import Metrics
from evolife.world import World


def main() -> None:
    seed = 1
    ticks = 4000
    world = World(seed=seed)
    metrics = Metrics(f"evolife_regen_warm_seed{seed}.sqlite")

    # Track per-lineage best (most-reproduced) founder genome.
    lineage_kids: dict[int, int] = {}

    for t in range(ticks):
        world.step()
        # Count offspring per founder_lineage_id.
        for o in world.organisms:
            if o.alive and o.founder_lineage_id is not None:
                lineage_kids[o.founder_lineage_id] = lineage_kids.get(o.founder_lineage_id, 0) + 1
        if t > 0 and t % 500 == 0:
            pop = sum(1 for o in world.organisms if o.alive)
            print(f"t={world.tick:>5} pop={pop:>3} max_gen={max((o.generation for o in world.organisms), default=0)}", flush=True)
        metrics.record_archive_milestones(world.archive)
        metrics.record_cycle_carriers(world.archive)
    metrics.close()

    # Pick founder with most descendants.
    pop_final = sum(1 for o in world.organisms if o.alive)
    print(f"final pop={pop_final} lineages={len(lineage_kids)}")
    if not lineage_kids:
        print("Extinction — warm-start will need a cold fallback.")
        sys.exit(0)
    best_founder_id = max(lineage_kids.items(), key=lambda kv: kv[1])[0]
    print(f"best founder lineage_id={best_founder_id} descendants={lineage_kids[best_founder_id]}")

    # Save parent genome = the founder itself (not its kids — we want a
    # baseline navigable genome, not a recurrent one).
    founder = next(o for o in world.organisms if o.founder_lineage_id == best_founder_id)
    out = "evolife_regen_warm_parent.json"
    with open(out, "w") as fh:
        json.dump(founder.genome.to_dict(), fh)
    print(f"saved {out}: nodes={len(founder.genome.nodes)} conns={len(founder.genome.connections)}")


if __name__ == "__main__":
    main()