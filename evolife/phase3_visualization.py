"""Phase 3 visualization — LabVisualizer draws dual smell + A/B food."""

from __future__ import annotations

from .phase3 import MemoryEcologyWorld
from .viz_lab import LabVisualizer


class Phase3Visualizer:
    def __init__(self, world: MemoryEcologyWorld) -> None:
        self.world = world
        self._lab = LabVisualizer(world)

    def render(self) -> None:
        self._lab.render()

    def close(self) -> None:
        self._lab.close()
