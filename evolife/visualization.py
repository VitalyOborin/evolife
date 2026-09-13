"""Visualization — Pygame top-down renderer.

Renders the world each frame. World drives the simulation; visualisation
only reads state.
"""

from __future__ import annotations

import pygame

from .config import (
    FOOD_RADIUS,
    FPS_TARGET,
    ORGANISM_RADIUS,
)
from .world import World


class Visualizer:
    """Pygame top-down renderer for the world."""

    def __init__(self, world: World) -> None:
        self.world = world
        pygame.init()
        self.screen = pygame.display.set_mode(
            (world.width, world.height)
        )
        pygame.display.set_caption("EvoLife v0")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 14)

    def render(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit

        self.screen.fill((0, 0, 0))

        for food in self.world.food:
            pygame.draw.circle(
                self.screen,
                (60, 200, 60),
                (int(food.x), int(food.y)),
                FOOD_RADIUS,
            )

        for org in self.world.organisms:
            if not org.alive:
                continue
            pygame.draw.circle(
                self.screen,
                (220, 220, 240),
                (int(org.x), int(org.y)),
                ORGANISM_RADIUS,
            )

        hud = self.font.render(
            f"tick={self.world.tick}  pop={self.world.population()}  "
            f"food={len(self.world.food)}  "
            f"meanE={self.world.mean_energy():.1f}",
            True,
            (255, 255, 255),
        )
        self.screen.blit(hud, (4, 4))

        pygame.display.flip()
        self.clock.tick(FPS_TARGET)

    def close(self) -> None:
        pygame.quit()
