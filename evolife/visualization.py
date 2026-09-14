"""Visualization — Pygame top-down renderer.

Renders the world each frame. World drives the simulation; visualisation
only reads state. Click an organism (or an established-species label)
to inspect whether a genetic species also lives differently.
"""

from __future__ import annotations

import colorsys
import math

import pygame

from .behavior import mean_descriptors
from .config import (
    FOOD_RADIUS,
    FPS_TARGET,
    ORGANISM_RADIUS,
)
from .organism import Organism
from .speciation import Species
from .world import World

NEWBORN_COLOR = (148, 148, 156)
_CLICK_RADIUS = 12
_PANEL_WIDTH = 300
_LINE_H = 16


class Visualizer:
    """Pygame top-down renderer for the world."""

    def __init__(self, world: World) -> None:
        self.world = world
        pygame.init()
        self.screen = pygame.display.set_mode(
            (world.width, world.height)
        )
        pygame.display.set_caption("EvoLife v2.2")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 14)
        self.small = pygame.font.SysFont("Consolas", 13)
        self.selected_species_id: int | None = None
        self._hud_hits: list[tuple[pygame.Rect, int]] = []
        self._inspector_rect: pygame.Rect | None = None

    def render(self) -> None:
        self._handle_events()

        self.screen.fill((0, 0, 0))

        for food in self.world.food:
            pygame.draw.circle(
                self.screen,
                (60, 200, 60),
                (int(food.x), int(food.y)),
                FOOD_RADIUS,
            )

        selected = self.selected_species_id
        for org in self.world.organisms:
            if not org.alive:
                continue
            color = species_draw_color(
                org.species_id, self._is_established(org.species_id)
            )
            pos = (int(org.x), int(org.y))
            pygame.draw.circle(self.screen, color, pos, ORGANISM_RADIUS)
            if selected is not None and org.species_id == selected:
                pygame.draw.circle(
                    self.screen, (255, 255, 255), pos, ORGANISM_RADIUS + 2, 1
                )

        self._draw_hud()
        if selected is not None:
            self._draw_inspector(selected)
        else:
            self._inspector_rect = None

        pygame.display.flip()
        self.clock.tick(FPS_TARGET)

    def close(self) -> None:
        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._on_click(event.pos)

    def _on_click(self, pos: tuple[int, int]) -> None:
        if self._inspector_rect is not None and self._inspector_rect.collidepoint(pos):
            return
        for rect, sid in self._hud_hits:
            if rect.collidepoint(pos):
                self.selected_species_id = sid
                return
        org = organism_at(self.world.organisms, pos[0], pos[1], _CLICK_RADIUS)
        self.selected_species_id = None if org is None else org.species_id

    def _is_established(self, species_id: int) -> bool:
        sp = self.world.species_manager.species.get(species_id)
        return bool(sp is not None and sp.established)

    def _draw_hud(self) -> None:
        self._hud_hits = []
        n_est = self.world.n_established_species()
        n_new = self.world.n_species() - n_est
        hud = self.font.render(
            f"tick={self.world.tick}  pop={self.world.population()}  "
            f"food={len(self.world.food)}  "
            f"meanE={self.world.mean_energy():.1f}  "
            f"maxGen={self.world.max_generation()}  "
            f"lin={self.world.n_lineages()}  "
            f"est={n_est}  new={n_new}",
            True,
            (255, 255, 255),
        )
        self.screen.blit(hud, (4, 4))

        established = sorted(
            self.world.species_manager.established_living(),
            key=lambda s: (-s.member_count, s.id),
        )
        x = 4
        y = 22
        for sp in established[:6]:
            label = f"s{sp.id}:{sp.member_count}"
            color = species_draw_color(sp.id, True)
            surf = self.font.render(label, True, color)
            rect = surf.get_rect(topleft=(x, y))
            self.screen.blit(surf, rect)
            self._hud_hits.append((rect.inflate(4, 2), sp.id))
            x = rect.right + 10
        if n_new > 0:
            leftover = self.font.render(
                f"new:{n_new}", True, NEWBORN_COLOR
            )
            self.screen.blit(leftover, (x, y))

    def _draw_inspector(self, species_id: int) -> None:
        sp = self.world.species_manager.species.get(species_id)
        if sp is None:
            self._inspector_rect = None
            return
        own = mean_descriptors(
            o for o in self.world.organisms if o.species_id == species_id
        )
        other_id = sp.parent_species_id
        other = None
        if other_id is not None:
            other = mean_descriptors(
                o for o in self.world.organisms if o.species_id == other_id
            )
        lines = inspector_lines(
            sp, self.world.tick, own, other_id, other
        )
        pad = 8
        width = _PANEL_WIDTH
        height = pad * 2 + _LINE_H * len(lines)
        x = self.world.width - width - 4
        y = 42
        rect = pygame.Rect(x, y, width, height)
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((12, 12, 18, 210))
        self.screen.blit(panel, rect.topleft)
        pygame.draw.rect(self.screen, (80, 80, 96), rect, 1)
        for i, line in enumerate(lines):
            surf = self.small.render(line, True, (230, 230, 236))
            self.screen.blit(surf, (x + pad, y + pad + i * _LINE_H))
        self._inspector_rect = rect


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
