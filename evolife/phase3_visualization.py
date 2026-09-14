"""Phase 3 visualization - dual-resource Memory Ecology.

Extends the legacy Visualizer with:
  - FoodA drawn in green, FoodB drawn in orange (so the brain's choice
    is visually traceable)
  - HUD shows season, ticks_in_season, positive_eats, negative_eats,
    recurrent_organism_fraction
  - Color of an organism reflects whether its brain has any recurrent
    edges (white border if yes)

The base Visualizer draws organisms and species cards; we only
override the food loop, the HUD, and add the recurrent check.
"""
from __future__ import annotations

import colorsys

import pygame

from .config import (
    FOOD_RADIUS,
    FPS_TARGET,
    ORGANISM_RADIUS,
)
from .phase3 import MemoryEcologyWorld
from .visualization import (
    NEWBORN_COLOR,
    Visualizer,
    species_draw_color,
)


_FOOD_A_COLOR = (60, 200, 60)    # green
_FOOD_B_COLOR = (220, 140, 40)   # orange
_RECURRENT_BORDER = (240, 240, 100)


class Phase3Visualizer(Visualizer):
    def __init__(self, world: MemoryEcologyWorld) -> None:
        super().__init__(world)
        mode = getattr(world, "mode", "hidden_season")
        # BitGenerator has bit_generator.state, which exposes the seed
        # tuple. Use a short fingerprint instead of trying to recover
        # the original int seed.
        seed_state = getattr(world.rng, "bit_generator", None)
        seed_label = type(seed_state).__name__ if seed_state is not None else "?"
        pygame.display.set_caption(
            f"EvoLife Phase 3 :: {mode} :: rng={seed_label}"
        )

    def render(self) -> None:
        self._handle_events()

        self.screen.fill((0, 0, 0))

        # Food A.
        for food in self.world.food_a:
            pygame.draw.circle(
                self.screen, _FOOD_A_COLOR,
                (int(food.x), int(food.y)), FOOD_RADIUS,
            )
        # Food B.
        for food in self.world.food_b:
            pygame.draw.circle(
                self.screen, _FOOD_B_COLOR,
                (int(food.x), int(food.y)), FOOD_RADIUS,
            )

        # Organisms, with a yellow ring around any that have recurrent
        # edges in their brain.
        selected = self.selected_species_id
        for org in self.world.organisms:
            if not org.alive:
                continue
            color = species_draw_color(
                org.species_id, self._is_established(org.species_id)
            )
            pos = (int(org.x), int(org.y))
            pygame.draw.circle(self.screen, color, pos, ORGANISM_RADIUS)
            if _has_recurrent_edges(org):
                pygame.draw.circle(
                    self.screen, _RECURRENT_BORDER, pos,
                    ORGANISM_RADIUS + 2, 1,
                )
            if selected is not None and org.species_id == selected:
                pygame.draw.circle(
                    self.screen, (255, 255, 255), pos,
                    ORGANISM_RADIUS + 3, 1,
                )

        self._draw_hud_phase3()
        if selected is not None:
            self._draw_inspector(selected)
        else:
            self._inspector_rect = None

        pygame.display.flip()
        self.clock.tick(FPS_TARGET)

    def _draw_hud_phase3(self) -> None:
        self._hud_hits = []
        w = self.world
        n_alive = sum(1 for o in w.organisms if o.alive)
        n_rec = sum(1 for o in w.organisms if o.alive and _has_recurrent_edges(o))
        rec_frac = (n_rec / n_alive) if n_alive else 0.0
        season = getattr(w, "season", 0)
        ticks_in_season = getattr(w, "ticks_in_season", 0)
        season_enabled = getattr(w, "season_enabled", False)
        pos_total = sum(getattr(o, "positive_eats", 0) for o in w.organisms)
        neg_total = sum(getattr(o, "negative_eats", 0) for o in w.organisms)
        n_est = w.n_established_species()
        n_new = w.n_species() - n_est

        season_label = (
            f"season={season} t={ticks_in_season}"
            if season_enabled
            else "static"
        )
        hud = self.font.render(
            f"mode={getattr(w, 'mode', '?')}  {season_label}  "
            f"tick={w.tick}  pop={n_alive}  "
            f"recurrent={n_rec} ({rec_frac * 100:.0f}%)  "
            f"foodA={len(w.food_a)} foodB={len(w.food_b)}  "
            f"pos={pos_total} neg={neg_total}  "
            f"maxGen={w.max_generation()}  "
            f"est={n_est} new={n_new}",
            True,
            (255, 255, 255),
        )
        self.screen.blit(hud, (4, 4))

        # Recurrent-organism count takes priority in the species list
        # for Phase 3 because it's the hypothesis indicator.
        established = sorted(
            w.species_manager.established_living(),
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
            leftover = self.font.render(f"new:{n_new}", True, NEWBORN_COLOR)
            self.screen.blit(leftover, (x, y))


def _has_recurrent_edges(org) -> bool:
    """True if org.genome has any cycle in its directed graph.

    Uses Tarjan strongly-connected components; returns as soon as
    one SCC of size >= 2 is found.
    """
    if org.genome is None:
        return False
    g = org.genome
    adj: dict[int, list[int]] = {}
    for c in g.connections.values():
        if not c.enabled:
            continue
        adj.setdefault(c.in_node, []).append(c.out_node)
        adj.setdefault(c.out_node, [])
    index_counter = [0]
    stack: list[int] = []
    on_stack: set[int] = set()
    indices: dict[int, int] = {}
    lowlinks: dict[int, int] = {}
    found = [False]

    def strongconnect(v: int) -> None:
        if found[0]:
            return
        indices[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in adj.get(v, []):
            if found[0]:
                return
            if w not in indices:
                strongconnect(w)
                if found[0]:
                    return
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
            if len(comp) >= 2:
                found[0] = True
                return

    for v in list(g.nodes):
        if found[0]:
            break
        if v not in indices:
            strongconnect(v)
    return found[0]
