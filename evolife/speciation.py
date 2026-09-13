"""Speciation.

In v0 we have a single species — the compatibility distance is unused but
the function is defined so v1 can flip a config flag and have speciation
without touching world.py.
"""

from __future__ import annotations

from .genome import Genome


def compatibility_distance(a: Genome, b: Genome) -> float:
    """Stub NEAT-style compatibility distance.

    Defined in v1. For now any two genomes have distance 0.
    """
    return 0.0


def speciate(genomes: list[Genome]) -> dict[int, list[Genome]]:
    """Group genomes into species buckets.

    v0 always returns a single bucket keyed by 0. v1 will return multiple
    buckets based on compatibility_distance.
    """
    return {0: list(genomes)}
