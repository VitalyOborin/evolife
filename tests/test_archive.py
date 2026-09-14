"""Tests for the Archive (observability milestones, not selection)."""

from evolife.archive import Archive, MilestoneKind


def test_archive_starts_empty():
    a = Archive()
    assert a.milestones == []
    assert a.summary()["milestone_count"] == 0


def test_archive_maybe_fire_fires_once():
    a = Archive()
    a.maybe_fire(MilestoneKind.FIRST_HIDDEN_NODE, tick=10, value=1)
    a.maybe_fire(MilestoneKind.FIRST_HIDDEN_NODE, tick=20, value=1)
    assert len(a.milestones) == 1
    assert a.milestones[0].tick == 10


def test_archive_record_max_only_updates_on_increase():
    a = Archive()
    a.record_max(MilestoneKind.MAX_BRAIN_NODES, tick=10, value=5)
    a.record_max(MilestoneKind.MAX_BRAIN_NODES, tick=20, value=3)
    a.record_max(MilestoneKind.MAX_BRAIN_NODES, tick=30, value=7)
    assert len(a.milestones) == 2
    assert [m.tick for m in a.milestones] == [10, 30]


def test_archive_summary_has_firsts_and_maxes():
    a = Archive()
    a.maybe_fire(MilestoneKind.FIRST_HIDDEN_NODE, tick=5, value=1)
    a.record_max(MilestoneKind.MAX_BRAIN_NODES, tick=10, value=8)
    summary = a.summary()
    assert "first_hidden_node" in summary["firsts"]
    assert summary["max_values"]["max_brain_nodes"] == 8
