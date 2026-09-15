"""Visualization helpers and a thin Visualizer wrapper around LabVisualizer.

Species colours, click picking, and the inspector text live here so tests
can exercise them without opening a window. The actual lab window — torus,
smell field, probe rays, side chart — is `evolife.viz_lab.LabVisualizer`.
"""

from __future__ import annotations

import colorsys
import math

from .organism import Organism
from .speciation import Species
from .world import World

NEWBORN_COLOR = (148, 148, 156)


class Visualizer:
    """Back-compat wrapper. Prefer LabVisualizer for new code."""

    def __init__(self, world: World) -> None:
        from .viz_lab import LabVisualizer

        self.world = world
        self._lab = LabVisualizer(world)

    def render(self) -> None:
        self._lab.render()

    def close(self) -> None:
        self._lab.close()

    @property
    def selected_species_id(self) -> int | None:
        return self._lab.selected_species_id

    @selected_species_id.setter
    def selected_species_id(self, value: int | None) -> None:
        self._lab.selected_species_id = value


def species_draw_color(species_id: int, established: bool) -> tuple[int, int, int]:
    """Unique color for established species; one gray for all newborns."""
    if not established:
        return NEWBORN_COLOR
    hue = (int(species_id) * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def organism_at(
    organisms, x: float, y: float, radius: float
) -> Organism | None:
    """Nearest living organism within radius, or None."""
    best: Organism | None = None
    best_d = radius
    for org in organisms:
        if not org.alive:
            continue
        d = math.hypot(org.x - x, org.y - y)
        if d <= best_d:
            best_d = d
            best = org
    return best


def inspector_lines(
    sp: Species,
    tick: int,
    own: dict[str, float] | None,
    other_id: int | None,
    other: dict[str, float] | None,
) -> list[str]:
    """Plain-text species card. Comparison column is the parent if living."""
    age = (sp.last_seen_tick if sp.extinct else tick) - sp.born_tick
    status = "extinct" if sp.extinct else (
        "established" if sp.established else "newborn"
    )
    parent = "none" if other_id is None else f"species {other_id}"
    lines = [
        f"Species {sp.id}  {status}",
        f"origin: tick {sp.born_tick:,}",
        f"parent: {parent}",
        f"age: {age:,} ticks",
        f"population: {sp.member_count}",
        f"peak population: {sp.peak_population}",
        "",
        "genome",
        f"nodes: {_fmt_stat(own, 'genome_nodes', 'avg')}",
        f"connections: {_fmt_stat(own, 'genome_conns', 'avg')}",
    ]
    if other_id is None:
        return lines
    left = f"S{other_id}"
    right = f"S{sp.id}"
    lines.extend(
        [
            "",
            f"behavior vs species {other_id}",
            "",
            f"{'':<18}{left:>8}{right:>8}",
            _cmp_row("moving", "moving_fraction", "pct", other, own),
            _cmp_row("food/1000", "food_rate", "1", other, own),
            _cmp_row("transitions", "transition_rate", "1", other, own),
            _cmp_row("exploration", "exploration_rate", "1", other, own),
            _cmp_row("steering align", "steering_alignment", "align", other, own),
            _cmp_row("state dependence", "state_dependence", "align", other, own),
        ]
    )
    return lines


def _cmp_row(
    label: str,
    key: str,
    kind: str,
    left: dict[str, float] | None,
    right: dict[str, float] | None,
) -> str:
    return f"{label:<18}{_fmt_cell(left, key, kind):>8}{_fmt_cell(right, key, kind):>8}"


def _fmt_stat(stats: dict[str, float] | None, key: str, suffix: str) -> str:
    if stats is None or key not in stats:
        return "—"
    return f"{stats[key]:.1f} {suffix}"


def _fmt_cell(stats: dict[str, float] | None, key: str, kind: str) -> str:
    if stats is None or key not in stats:
        return "—"
    v = stats[key]
    if kind == "pct":
        return f"{v * 100:.0f}%"
    if kind == "align":
        return f"{v:.2f}"
    return f"{v:.1f}"
