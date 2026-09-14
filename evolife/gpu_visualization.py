"""GPU-side visualization adapter.

Visualizer (in visualization.py) is built around CPU World's
Organism/Species objects. GpuWorld exposes positions and headings as
torch tensors with no per-organism Python object. GpuVisualizer
draws directly from those tensors each frame.

Trade-offs vs the CPU visualizer:
- No species colour or inspector (GpuWorld has no SpeciesManager).
- Colour comes from `parent_id` lineage: each founder (-1 parent)
  gets a unique hue; children inherit theirs. This gives the same
  visual feel as species colouring but reflects lineage, not speciation.
- Per-frame CPU sync of (alive_idx, x, y, parent_id) — fine for
  visualisation; the cost is the pygame loop, not the sync.
"""

from __future__ import annotations

import colorsys

import pygame
import torch

from .config import FOOD_RADIUS, FPS_TARGET, ORGANISM_RADIUS
from .gpu_world import GpuWorld


NEWBORN_COLOR = (148, 148, 156)
_PANEL_WIDTH = 300
_LINE_H = 16


class GpuVisualizer:
    """Pygame renderer for GpuWorld.

    Mirrors Visualizer.render() but reads positions from GPU tensors
    and uses parent_id lineage for colouring.
    """

    def __init__(self, world: GpuWorld) -> None:
        self.world = world
        pygame.init()
        self.screen = pygame.display.set_mode(
            (world.width, world.height)
        )
        pygame.display.set_caption("EvoLife v2.2 [GPU]")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 14)
        self.small = pygame.font.SysFont("Consolas", 13)
        self._line_to_hue: dict[int, float] = {}
        self._next_hue_seed = 0

    def render(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit

        self.screen.fill((0, 0, 0))

        # Food positions: GpuWorld stores them on GPU as a tensor.
        n_food = int(self.world._food_count)
        if n_food > 0:
            food_pos = self.world.food_pos[:n_food].to("cpu").numpy()
            for fx, fy in food_pos:
                pygame.draw.circle(
                    self.screen,
                    (60, 200, 60),
                    (int(fx), int(fy)),
                    FOOD_RADIUS,
                )

        # Organisms.
        alive_idx = self.world.alive_mask.nonzero(as_tuple=False).squeeze(1)
        if alive_idx.numel() > 0:
            pos = self.world.positions[alive_idx].to("cpu").numpy()
            parent_id = self.world.parent_id[alive_idx].to("cpu").numpy()
            slots = alive_idx.to("cpu").numpy()
            for i in range(int(alive_idx.shape[0])):
                hue = self._hue_for(int(parent_id[i]))
                color = _hue_to_rgb(hue)
                x, y = int(pos[i, 0]), int(pos[i, 1])
                pygame.draw.circle(
                    self.screen, color, (x, y), ORGANISM_RADIUS
                )

        # HUD.
        hud = self.font.render(
            f"[GPU]  tick={self.world.tick}  pop={self.world.population()}  "
            f"food={n_food}  meanE={self.world.mean_energy():.1f}  "
            f"maxGen={self.world.max_generation()}",
            True,
            (255, 255, 255),
        )
        self.screen.blit(hud, (4, 4))

        pygame.display.flip()
        self.clock.tick(FPS_TARGET)

    def close(self) -> None:
        pygame.quit()

    def _hue_for(self, parent_id: int) -> float:
        """Stable hue per founder lineage.

        Walks parent_id chain back to a founder (parent_id == -1).
        Founders get a unique hue from a golden-angle sequence so
        neighbouring lineages are visually distinct.
        """
        # Walk via cache first.
        if parent_id == -1:
            hue = (self._next_hue_seed * 0.6180339887) % 1.0
            self._next_hue_seed += 1
            return hue
        if parent_id in self._line_to_hue:
            return self._line_to_hue[parent_id]
        # Cache miss (parent died): assign a fresh hue.
        hue = (self._next_hue_seed * 0.6180339887) % 1.0
        self._next_hue_seed += 1
        self._line_to_hue[parent_id] = hue
        return hue


def _hue_to_rgb(hue: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))
