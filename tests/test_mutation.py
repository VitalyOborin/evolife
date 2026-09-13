import numpy as np

from evolife.brain import Brain
from evolife.mutation import (
    mutate_add_connection,
    mutate_add_node,
    mutate_toggle_connection,
    mutate_weights,
)


def test_mutate_weights_is_pure_with_rate_zero():
    g = Brain.make_default_genome()
    mutated = mutate_weights(g, np.random.default_rng(0), rate=0.0)
    # With rate=0 the function should short-circuit and return the same
    # object (per its docstring: returns the input unchanged).
    assert mutated is g


def test_mutate_weights_changes_some_weights_with_rate_one():
    g = Brain.make_default_genome()
    # Give every connection a baseline weight so perturbation is observable.
    # The default genome has no connections yet, so we exercise the function
    # with a synthetic genome instead.
    from evolife.genome import ConnectionGene

    g.connections[1] = ConnectionGene(
        innovation=1, in_node=0, out_node=5, weight=1.0, enabled=True
    )
    mutated = mutate_weights(g, np.random.default_rng(0), rate=1.0)
    assert mutated is not g
    # With rate=1 and perturb_rate=0.9, virtually all weights change.
    assert mutated.connections[1].weight != 1.0


def test_structural_mutations_are_stubs():
    g = Brain.make_default_genome()
    rng = np.random.default_rng(0)
    for fn in (mutate_add_node, mutate_add_connection, mutate_toggle_connection):
        try:
            fn(g, rng)
        except NotImplementedError:
            continue
        else:  # pragma: no cover - we expect NotImplementedError
            raise AssertionError(f"{fn.__name__} should be a v1 stub")
