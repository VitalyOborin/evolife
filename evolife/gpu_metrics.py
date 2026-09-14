"""GPU-side metrics adapter.

`Metrics` (in metrics.py) was designed around the CPU World's
`world.organisms` and `world.species_manager` interfaces. GpuWorld has
neither — organisms live as GPU tensors, and species tracking is not
implemented on GPU. This module provides the same Metrics API surface
backed by cheap, batched CPU syncs from the GPU tensors.

Per-tick cost:
- record_gpu_world: 2 scalar CPU syncs.
- record_gpu_organisms: one (~N, k) sync per call, where k is small
  (parent_id, generation, food_eaten, age, energy). Done at
  METRICS_ORGANISM_EVERY cadence only, so the overhead is bounded.
- record_gpu_species: optional; reads founder_lineage_id tensors.
- flush_gpu_events: walks world.events.events (only Birth is currently
  emitted by GpuWorld — Death / Eat / Reproduction are not).

This is intentionally minimal: we keep the same SQLite schema and
column semantics so downstream analysis scripts (test_metrics,
test_behavior) work for both CPU and GPU runs.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import numpy as np
import torch

from .config import (
    METRICS_ORGANISM_EVERY,
    METRICS_SPECIES_EVERY,
    METRICS_WORLD_EVERY,
    N_HIDDEN,
    N_MOTORS,
    N_SENSORS,
)
from .metrics import Metrics

if TYPE_CHECKING:
    from .gpu_world import GpuWorld


_FOUNDER_NODES = N_SENSORS + N_HIDDEN + N_MOTORS


class GpuMetrics(Metrics):
    """Subclass of Metrics that knows how to read GpuWorld tensors.

    Inherits the same SQLite schema and write paths. Adds four methods
    that mirror the CPU ones:

    - record_gpu_world(world)
    - record_gpu_organisms(world)
    - record_gpu_species(world)
    - flush_gpu_events(world)

    Use these instead of the CPU record_* methods when running on GPU.
    """

    def __init__(self, path: str = "evolife_metrics_gpu.sqlite") -> None:
        super().__init__(path=path)

    # ---- world -------------------------------------------------------------

    def record_gpu_world(self, world: "GpuWorld") -> None:
        if world.tick % METRICS_WORLD_EVERY != 0:
            return
        pop = world.population()
        mean_e = world.mean_energy()
        food_count = int(world._food_count)
        self._conn.execute(
            "INSERT OR REPLACE INTO world_snapshots VALUES (?, ?, ?, ?)",
            (world.tick, pop, mean_e, food_count),
        )
        self._conn.commit()

    # ---- organisms ---------------------------------------------------------

    def record_gpu_organisms(self, world: "GpuWorld") -> None:
        if world.tick % METRICS_ORGANISM_EVERY != 0:
            return
        alive_idx = world.alive_mask.nonzero(as_tuple=False).squeeze(1)
        n = int(alive_idx.shape[0])
        if n == 0:
            return
        # Pull only the columns we need.
        parent_id_cpu = world.parent_id[alive_idx].to("cpu").numpy()
        energy_cpu = world.energy[alive_idx].to("cpu").numpy()
        age_cpu = world.age[alive_idx].to("cpu").numpy()
        generation_cpu = world.generation[alive_idx].to("cpu").numpy()
        food_eaten_cpu = world.food_eaten[alive_idx].to("cpu").numpy()
        # Genome: count enabled connections per organism (cheap on GPU).
        enabled_cpu = world.gene_enabled[alive_idx].to("cpu").numpy()
        conn_count_cpu = enabled_cpu.sum(axis=1).astype(np.int64)
        # Organism id: use parent_id for the absolute slot if not tracked.
        # GpuWorld doesn't keep a unique id per slot; use the alive index.
        rows = []
        for i in range(n):
            rows.append(
                (
                    world.tick,
                    int(alive_idx[i].item()),
                    int(parent_id_cpu[i]),
                    int(age_cpu[i]),
                    float(energy_cpu[i]),
                    float(energy_cpu[i]),  # peak_energy unknown → use energy
                    0,  # children count not tracked on GPU
                    _FOUNDER_NODES,  # genome_nodes (sensors + motors only)
                    int(conn_count_cpu[i]),  # genome_conns (active)
                    int(generation_cpu[i]),
                    0,  # founder_lineage_id — see record_gpu_species
                    int(food_eaten_cpu[i]),
                    0,  # movement_transitions — unknown on GPU
                    0,  # ticks_moving
                    0.0,  # speed_sum
                    0,  # longest_rest
                    0,  # longest_move
                    0.0,  # bias_over_weights — unknown
                    1,  # alive
                )
            )
        self._conn.executemany(
            "INSERT OR REPLACE INTO organism_snapshots ("
            "tick, organism_id, parent_id, age, energy, peak_energy, "
            "children, genome_nodes, genome_conns, generation, "
            "founder_lineage_id, food_eaten, movement_transitions, "
            "ticks_moving, speed_sum, longest_rest, longest_move, "
            "bias_over_weights, alive"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    # ---- species -----------------------------------------------------------

    def record_gpu_species(self, world: "GpuWorld") -> None:
        """Naive line_id-based species tracker.

        GpuWorld doesn't have a SpeciesManager. We synthesise species
        buckets from `parent_id` chains: each founder has parent_id=-1
        and gets a fresh species_id. Children inherit their parent's
        species_id (we cache the mapping by following parent_id
        pointers back to a founder).
        """
        if world.tick % METRICS_SPECIES_EVERY != 0:
            return
        if not hasattr(self, "_line_to_species"):
            self._line_to_species: dict[int, int] = {}
            self._species_buckets: dict[
                int, dict
            ] = {}
            self._next_species_id = 0
        # Sync alive slots.
        alive_idx = world.alive_mask.nonzero(as_tuple=False).squeeze(1)
        if alive_idx.numel() == 0:
            return
        parent_id_cpu = world.parent_id[alive_idx].to("cpu").numpy()
        generation_cpu = world.generation[alive_idx].to("cpu").numpy()
        # Update species buckets for each alive organism.
        now = world.tick
        rows = []
        seen_species: set[int] = set()
        for i in range(int(alive_idx.shape[0])):
            slot = int(alive_idx[i].item())
            species_id = self._resolve_species_id(slot, int(parent_id_cpu[i]))
            seen_species.add(species_id)
            bucket = self._species_buckets.setdefault(
                species_id,
                {
                    "born_tick": now,
                    "last_seen_tick": now,
                    "peak_population": 0,
                    "parent_species_id": None,
                    "founder_organism_id": slot,
                    "extinct": False,
                    "established": False,
                    "member_count": 0,
                },
            )
            bucket["last_seen_tick"] = now
            bucket["member_count"] += 1
            if bucket["member_count"] > bucket["peak_population"]:
                bucket["peak_population"] = bucket["member_count"]
        # Mark unseen species as extinct.
        for sp_id, bucket in self._species_buckets.items():
            if sp_id not in seen_species and not bucket["extinct"]:
                bucket["extinct"] = True
                bucket["last_seen_tick"] = now
        # Write snapshots only for species that exist.
        for sp_id, bucket in self._species_buckets.items():
            if bucket["member_count"] == 0 and bucket["extinct"]:
                continue
            rows.append(
                (
                    now,
                    sp_id,
                    bucket["member_count"],
                    bucket["peak_population"],
                    bucket["born_tick"],
                    bucket["last_seen_tick"],
                    bucket["parent_species_id"],
                    int(bucket["extinct"]),
                    int(bucket["established"]),
                    None,  # no representative_genome_hash on GPU
                )
            )
            # Reset per-tick count for next iteration.
            bucket["member_count"] = 0
        if not rows:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO species_snapshots ("
            "tick, species_id, member_count, peak_population, born_tick, "
            "last_seen_tick, parent_species_id, extinct, established, "
            "representative_genome_hash"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def _resolve_species_id(self, slot: int, parent_id: int) -> int:
        """Walk parent_id chain back to a founder (parent_id==-1) and
        return / assign a stable species_id."""
        if not hasattr(self, "_line_to_species"):
            self._line_to_species = {}
            self._species_buckets = {}
            self._next_species_id = 0
        # Walk via cached mapping first.
        if slot in self._line_to_species:
            return self._line_to_species[slot]
        # New organism: inherit parent's species if known, else new.
        if parent_id == -1:
            sp_id = self._next_species_id
            self._next_species_id += 1
            self._line_to_species[slot] = sp_id
            return sp_id
        if parent_id in self._line_to_species:
            sp_id = self._line_to_species[parent_id]
            self._line_to_species[slot] = sp_id
            return sp_id
        # Parent unknown to cache (e.g. parent died); treat as new species.
        sp_id = self._next_species_id
        self._next_species_id += 1
        self._line_to_species[slot] = sp_id
        return sp_id

    # ---- events ------------------------------------------------------------

    def flush_gpu_events(self, world: "GpuWorld") -> None:
        """Persist events. Currently GpuWorld only emits Birth events.

        Death/Eat/Reproduction are not emitted by GpuWorld by design
        (see the file-level docstring of gpu_world.py for the
        performance trade-off). For now we just persist what exists.
        """
        rows = []
        for e in world.events.events:
            f = e.fields
            rows.append(
                (
                    e.tick,
                    e.kind.value,
                    f.get("org_id"),
                    f.get("parent_id"),
                    f.get("child_id"),
                    f.get("cause"),
                    f.get("genome_hash") or f.get("child_genome_hash"),
                    f.get("x"),
                    f.get("y"),
                )
            )
        if not rows:
            return
        self._conn.executemany(
            "INSERT INTO events (tick, kind, org_id, parent_id, child_id, "
            "cause, genome_hash, x, y) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
