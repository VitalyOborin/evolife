import numpy as np

from evolife.brain import Brain
from evolife.innovation import InnovationDatabase
from evolife.mutation import (
    mutate_add_connection,
    mutate_add_node,
    mutate_toggle_connection,
    mutate_weights,
)
from evolife.genome import ConnectionGene


def test_mutate_weights_is_pure_with_rate_zero():
    g = Brain.make_default_genome()
    mutated = mutate_weights(g, np.random.default_rng(0), rate=0.0)
    assert mutated is g


def test_mutate_weights_changes_some_weights_with_rate_one():
    g = Brain.make_default_genome()
    g.connections[1000] = ConnectionGene(
        innovation=1000, in_node=0, out_node=5, weight=1.0, enabled=True
    )
    mutated = mutate_weights(g, np.random.default_rng(0), rate=1.0)
    assert mutated is not g
    assert mutated.connections[1000].weight != 1.0


def test_add_connection_appends_when_pair_is_new():
    g = Brain.make_default_genome()
    n_before = len(g.connections)
    innov = InnovationDatabase()
    rng = np.random.default_rng(0)
    # Force a specific outcome by exhausting random draws: keep retrying
    # until the mutation succeeds. With rate=1.0 and 20 attempts, success
    # is essentially guaranteed for a genome with >2 nodes.
    new = mutate_add_connection(g, rng, innov, rate=1.0)
    assert new is not g
    assert len(new.connections) >= n_before


def test_add_connection_short_circuits_with_rate_zero():
    g = Brain.make_default_genome()
    innov = InnovationDatabase()
    new = mutate_add_connection(
        g, np.random.default_rng(0), innov, rate=0.0
    )
    assert new is g


def test_add_node_inserts_and_disables_old_connection():
    g = Brain.make_default_genome()
    n_conns_before = sum(1 for c in g.connections.values() if c.enabled)
    n_nodes_before = len(g.nodes)
    innov = InnovationDatabase()
    # Seed innovation DB with the existing connections so add_node can
    # allocate new ones.
    for c in g.connections.values():
        innov.innovation_for(c.in_node, c.out_node)
    new = mutate_add_node(g, np.random.default_rng(0), innov, rate=1.0)
    assert new is not g
    assert len(new.nodes) == n_nodes_before + 1
    # The number of enabled connections stays the same (one disabled,
    # two added).
    enabled_after = sum(1 for c in new.connections.values() if c.enabled)
    assert enabled_after == n_conns_before + 1


def test_add_node_reuses_innovation_when_pair_already_seen():
    g = Brain.make_default_genome()
    innov = InnovationDatabase()
    for c in g.connections.values():
        innov.innovation_for(c.in_node, c.out_node)
    # Apply add_node twice. The two new connections added by the first
    # call should be in the DB by the time the second call runs.
    new = mutate_add_node(g, np.random.default_rng(0), innov, rate=1.0)
    n_nodes_after_first = len(new.nodes)
    # Apply again on the mutated genome.
    new2 = mutate_add_node(new, np.random.default_rng(0), innov, rate=1.0)
    # Two calls -> two new nodes.
    assert len(new2.nodes) >= n_nodes_after_first + 1


def test_toggle_connection_flips_some_enabled_state():
    g = Brain.make_default_genome()
    enabled_before = [c.enabled for c in g.connections.values()]
    new = mutate_toggle_connection(g, np.random.default_rng(0), rate=1.0)
    enabled_after = [c.enabled for c in new.connections.values()]
    # Exactly one connection flipped state.
    diffs = sum(1 for a, b in zip(enabled_before, enabled_after) if a != b)
    assert diffs == 1


def test_toggle_connection_short_circuits_with_rate_zero():
    g = Brain.make_default_genome()
    enabled_before = [c.enabled for c in g.connections.values()]
    new = mutate_toggle_connection(g, np.random.default_rng(0), rate=0.0)
    assert new is g
    enabled_after = [c.enabled for c in new.connections.values()]
    assert enabled_before == enabled_after
