"""Observational NEAT-style speciation.

Species are a taxonomy: they do not affect survival or reproduction.
A child is assigned a species_id at birth by distance to living
representatives. IDs are stable; we never recluster the whole population.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .config import (
    COMPAT_C_BIAS,
    COMPAT_C_DISJOINT,
    COMPAT_C_EXCESS,
    COMPAT_C_WEIGHT,
    SPECIES_REPRESENTATIVE_EVERY,
    SPECIES_THRESHOLD,
)
from .genome import Genome


def compatibility_distance(a: Genome, b: Genome) -> float:
    """NEAT-style distance. Excess/disjoint are not divided by gene count."""
    innovs_a = set(a.connections)
    innovs_b = set(b.connections)
    if not innovs_a and not innovs_b:
        return _bias_distance(a, b)

    matching = innovs_a & innovs_b
    only_a = innovs_a - innovs_b
    only_b = innovs_b - innovs_a
    max_a = max(innovs_a) if innovs_a else -1
    max_b = max(innovs_b) if innovs_b else -1
    if max_a > max_b:
        excess = sum(1 for i in only_a if i > max_b)
        disjoint = len(only_a) - excess + len(only_b)
    elif max_b > max_a:
        excess = sum(1 for i in only_b if i > max_a)
        disjoint = len(only_b) - excess + len(only_a)
    else:
        excess = 0
        disjoint = len(only_a) + len(only_b)

    if matching:
        w_diff = sum(
            abs(a.connections[i].weight - b.connections[i].weight)
            for i in matching
        ) / len(matching)
    else:
        w_diff = 0.0

    return (
        COMPAT_C_EXCESS * excess
        + COMPAT_C_DISJOINT * disjoint
        + COMPAT_C_WEIGHT * w_diff
        + COMPAT_C_BIAS * _bias_distance(a, b)
    )


def _bias_distance(a: Genome, b: Genome) -> float:
    ids = set(a.nodes) & set(b.nodes)
    if not ids:
        return 0.0
    return sum(abs(a.nodes[i].bias - b.nodes[i].bias) for i in ids) / len(ids)


@dataclass
class Species:
    id: int
    representative: Genome
    born_tick: int
    last_seen_tick: int
    member_count: int = 0
    peak_population: int = 0
    parent_species_id: int | None = None
    founder_organism_id: int = 0
    extinct: bool = False


@dataclass
class SpeciesEvent:
    tick: int
    species_id: int
    kind: str  # ORIGIN | EXTINCTION
    parent_species_id: int | None = None
    founder_organism_id: int | None = None
    representative_genome_hash: str = ""


@dataclass
class SpeciesManager:
    """Persistent species taxonomy. Assignment happens at birth only."""

    threshold: float = SPECIES_THRESHOLD
    species: dict[int, Species] = field(default_factory=dict)
    events: list[SpeciesEvent] = field(default_factory=list)
    _next_id: int = 0

    def assign(
        self,
        genome: Genome,
        tick: int,
        organism_id: int,
        parent_species_id: int | None = None,
    ) -> int:
        if (
            parent_species_id is not None
            and parent_species_id in self.species
            and not self.species[parent_species_id].extinct
            and compatibility_distance(
                genome, self.species[parent_species_id].representative
            )
            <= self.threshold
        ):
            return parent_species_id

        for sid, sp in self.species.items():
            if sp.extinct:
                continue
            if sid == parent_species_id:
                continue
            if compatibility_distance(genome, sp.representative) <= self.threshold:
                return sid

        return self._originate(
            genome, tick, organism_id, parent_species_id
        )

    def sync(self, organisms, tick: int) -> None:
        counts: dict[int, int] = {}
        for org in organisms:
            if not org.alive:
                continue
            counts[org.species_id] = counts.get(org.species_id, 0) + 1
        for sid, sp in self.species.items():
            n = counts.get(sid, 0)
            sp.member_count = n
            if n > 0:
                sp.last_seen_tick = tick
                if n > sp.peak_population:
                    sp.peak_population = n
            elif not sp.extinct:
                sp.extinct = True
                self.events.append(
                    SpeciesEvent(
                        tick=tick,
                        species_id=sid,
                        kind="EXTINCTION",
                        parent_species_id=sp.parent_species_id,
                        representative_genome_hash=sp.representative.fingerprint(),
                    )
                )

    def maybe_refresh_representatives(self, organisms, tick: int) -> None:
        if tick == 0 or tick % SPECIES_REPRESENTATIVE_EVERY != 0:
            return
        members: dict[int, list] = {}
        for org in organisms:
            if not org.alive or org.genome is None:
                continue
            members.setdefault(org.species_id, []).append(org)
        for sid, group in members.items():
            sp = self.species.get(sid)
            if sp is None or len(group) == 1:
                if sp is not None and group:
                    sp.representative = copy.deepcopy(group[0].genome)
                continue
            best = group[0]
            best_sum = float("inf")
            for cand in group:
                total = sum(
                    compatibility_distance(cand.genome, other.genome)
                    for other in group
                    if other is not cand
                )
                if total < best_sum:
                    best_sum = total
                    best = cand
            sp.representative = copy.deepcopy(best.genome)

    def living(self) -> list[Species]:
        return [s for s in self.species.values() if not s.extinct]

    def _originate(
        self,
        genome: Genome,
        tick: int,
        organism_id: int,
        parent_species_id: int | None,
    ) -> int:
        sid = self._next_id
        self._next_id += 1
        sp = Species(
            id=sid,
            representative=copy.deepcopy(genome),
            born_tick=tick,
            last_seen_tick=tick,
            member_count=1,
            peak_population=1,
            parent_species_id=parent_species_id,
            founder_organism_id=organism_id,
        )
        self.species[sid] = sp
        self.events.append(
            SpeciesEvent(
                tick=tick,
                species_id=sid,
                kind="ORIGIN",
                parent_species_id=parent_species_id,
                founder_organism_id=organism_id,
                representative_genome_hash=genome.fingerprint(),
            )
        )
        return sid


def speciate(genomes: list[Genome]) -> dict[int, list[Genome]]:
    """Legacy helper: one-shot clustering. Prefer SpeciesManager."""
    mgr = SpeciesManager()
    buckets: dict[int, list[Genome]] = {}
    for i, g in enumerate(genomes):
        sid = mgr.assign(g, tick=0, organism_id=i)
        buckets.setdefault(sid, []).append(g)
    return buckets
