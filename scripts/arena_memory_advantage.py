"""Memory advantage: causal ablation via SCC edge removal.

Per CANON, "memory advantage" must be a *causal* metric: the only
difference between NORMAL and ABLATED must be the recurrent edges.
We find recurrent edges via strongly connected components (SCC) of
the genome's directed graph. Edges within any cycle are removed in
ABLATED; everything else (weights, biases, structure) is identical.

We run both versions on the same MemoryEcologyWorld through real
world.step() so seasons, smell recompute, intake feedback, and
reproduction are all live. The metric is reward-sum over the
episode; positive-fraction is reported as a secondary signal.

Genome can be supplied either:
  --genome-json   path to JSON of Genome.to_dict()
  --sample-from  sqlite path (phase 3 archive); uses lineage-median
                  genome at sample_tick (or nearest lower)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np

from evolife.brain import Brain
from evolife.genome import Genome, NodeType
from evolife.phase3 import (
    MemoryEcologyWorld,
    make_phase3_founder_from_warmstart,
)


def find_recurrent_connection_ids(genome: Genome) -> set[int]:
    """Return innovation ids of edges that participate in a cycle
    (Tarjan SCC, simplified for small graphs).
    """
    # Build adjacency list of enabled connections.
    adj: dict[int, list[int]] = {nid: [] for nid in genome.nodes}
    for c in genome.connections.values():
        if c.enabled:
            adj.setdefault(c.in_node, []).append(c.out_node)
            adj.setdefault(c.out_node, [])

    # Tarjan SCC.
    index_counter = [0]
    stack: list[int] = []
    on_stack: set[int] = set()
    indices: dict[int, int] = {}
    lowlinks: dict[int, int] = {}
    sccs: list[list[int]] = []

    def strongconnect(v: int) -> None:
        indices[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in adj.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[w])
        if lowlinks[v] == indices[v]:
            comp: list[int] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in list(genome.nodes):
        if v not in indices:
            strongconnect(v)

    # Edges inside an SCC of size >= 2 are recurrent.
    recurrent_nodes: set[int] = set()
    for comp in sccs:
        if len(comp) >= 2:
            recurrent_nodes.update(comp)
    recurrent_edges: set[int] = set()
    for cid, c in genome.connections.items():
        if c.enabled and c.in_node in recurrent_nodes and c.out_node in recurrent_nodes:
            recurrent_edges.add(cid)
    return recurrent_edges


def ablated_genome(genome: Genome, recurrent_edges: set[int]) -> Genome:
    """Return a copy of genome with recurrent edges disabled (not removed,
    so the topology is otherwise unchanged).
    """
    g = Genome.from_dict(genome.to_dict())
    for cid in recurrent_edges:
        if cid in g.connections:
            g.connections[cid].enabled = False
    return g


def run_episode(
    genome: Genome,
    *,
    n_ticks: int,
    seed: int,
    mode: str = "hidden_season",
    visible_season_sensor: bool = False,
    warm_genome: Genome | None = None,
    ablate: bool = False,
) -> dict:
    """One episode. The input `genome` may have any topology
    (typically 3-sensor Phase 1.5 or 7-sensor Phase 3). We convert it
    to a Phase 3 founder via the standard warm-start graft so the
    brain topology matches what world._sensors_for emits.
    """
    # Build the Phase 3 founder that the input genome would produce.
    phase3_genome = make_phase3_founder_from_warmstart(
        rng=np.random.default_rng(seed),
        warm_genome=genome,
        visible_season_sensor=visible_season_sensor,
    )
    if ablate:
        recurrent = find_recurrent_connection_ids(phase3_genome)
        phase3_genome = ablated_genome(phase3_genome, recurrent)
    world = MemoryEcologyWorld(
        seed=seed, mode=mode,
        visible_season_sensor=visible_season_sensor,
        warm_genome=warm_genome,
    )
    founder = world.organisms[0]
    founder.genome = phase3_genome
    founder.brain = Brain(phase3_genome)
    founder.intake_feedback = 0.0
    founder.intake_feedback_ttl = 0
    founder.positive_eats = 0
    founder.negative_eats = 0
    world.organisms = [founder]

    initial_energy = founder.energy
    for _ in range(n_ticks):
        world.step()
    final_energy = founder.energy
    return {
        "positive_eats": founder.positive_eats,
        "negative_eats": founder.negative_eats,
        "reward_sum": (
            founder.positive_eats * 25 - founder.negative_eats * 3
        ),
        "ticks_alive": n_ticks,
        "final_energy": final_energy,
        "initial_energy": initial_energy,
    }


def load_genome(path: str) -> Genome:
    with open(path) as fh:
        return Genome.from_dict(json.load(fh))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--genome-json", required=True)
    p.add_argument("--warm-json", default=None,
                   help="Warm-start genome (Phase 1.5 navigator) for the world bootstrap")
    p.add_argument("--n-ticks", type=int, default=200)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--mode", type=str, default="hidden_season",
                   choices=("static_dual", "visible_season", "hidden_season"))
    p.add_argument("--visible-season-sensor", action="store_true")
    p.add_argument("--out", type=str, default="evolife_memory_advantage.csv")
    args = p.parse_args()

    g = load_genome(args.genome_json)
    warm = load_genome(args.warm_json) if args.warm_json else None

    recurrent = find_recurrent_connection_ids(g)
    print(f"genome: nodes={len(g.nodes)} conns={len(g.connections)} "
          f"recurrent_edges={len(recurrent)}")
    if recurrent:
        for cid in sorted(recurrent):
            c = g.connections[cid]
            print(f"  recurrent: {c.in_node} -> {c.out_node} (innov={cid}, weight={c.weight:.3f})")

    rows = []
    for seed in range(args.n_seeds):
        norm = run_episode(g, n_ticks=args.n_ticks, seed=seed,
                           mode=args.mode,
                           visible_season_sensor=args.visible_season_sensor,
                           warm_genome=warm, ablate=False)
        abl = run_episode(g, n_ticks=args.n_ticks, seed=seed,
                          mode=args.mode,
                          visible_season_sensor=args.visible_season_sensor,
                          warm_genome=warm, ablate=True)
        rows.append({
            "seed": seed,
            "normal_positive": norm["positive_eats"],
            "ablate_positive": abl["positive_eats"],
            "positive_advantage": norm["positive_eats"] - abl["positive_eats"],
            "normal_negative": norm["negative_eats"],
            "ablate_negative": abl["negative_eats"],
            "negative_advantage": norm["negative_eats"] - abl["negative_eats"],
            "normal_reward": norm["reward_sum"],
            "ablate_reward": abl["reward_sum"],
            "reward_advantage": norm["reward_sum"] - abl["reward_sum"],
        })

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")

    # Aggregate.
    n = len(rows)
    mean_pos = sum(r["positive_advantage"] for r in rows) / n
    mean_neg = sum(r["negative_advantage"] for r in rows) / n
    mean_reward = sum(r["reward_advantage"] for r in rows) / n
    print()
    print(f"Mean positive_advantage = {mean_pos:+.3f}")
    print(f"Mean negative_advantage = {mean_neg:+.3f}")
    print(f"Mean reward_advantage   = {mean_reward:+.3f}")


if __name__ == "__main__":
    sys.exit(main())
