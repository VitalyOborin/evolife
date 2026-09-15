"""GPU visualizer — thin wrapper around LabVisualizer.

LabVisualizer reads GpuWorld tensors (positions, headings, smell grid)
and colourizes the smell field on the same CUDA device.
"""

from __future__ import annotations

from .gpu_world import GpuWorld
from .viz_lab import LabVisualizer


class GpuVisualizer:
    """Pygame renderer for GpuWorld."""

    def __init__(self, world: GpuWorld) -> None:
        self.world = world
        self._lab = LabVisualizer(world)

    def render(self) -> None:
        self._lab.render()

    def close(self) -> None:
        self._lab.close()
