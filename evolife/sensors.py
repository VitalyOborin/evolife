"""Smell field — a coarse diffusion grid computed once per tick.

Each food particle contributes a falloff (1 / (1 + dist/R)) to a single
2D numpy array. Organisms then sample three sectors (left/front/right)
at their position by averaging field values within each angular wedge.

The smell is the organism's ONLY sense of food. There is no GPS, no
"nearest food" oracle. The experiment is whether random recurrent
brains can learn to navigate by smell gradient.

Implementation: stack all food positions, splat each onto a padded
grid using the precomputed kernel, then take the central region as the
toroidal grid. Edge spillover is folded back via four strip copies.
"""

from __future__ import annotations

import math

import numpy as np

from .config import SMELL_FIELD_RADIUS


class SmellField:
    """Precomputed smell intensity grid."""

    def __init__(
        self,
        width: int,
        height: int,
        radius: int = SMELL_FIELD_RADIUS,
    ) -> None:
        self.width = width
        self.height = height
        self.radius = radius
        self.grid = np.zeros((height, width), dtype=np.float32)
        # Precompute the kernel.
        R = radius
        ys, xs = np.mgrid[-R:R + 1, -R:R + 1]
        self._kernel = 1.0 / (1.0 + np.hypot(xs, ys)).astype(np.float32)
        self._kernel_size = 2 * R + 1

    def recompute(self, food: list) -> None:
        """Zero the grid, then deposit each food particle's contribution.

        Splat each food's kernel onto a padded grid, then take the
        central region. Spillover past the world edge is folded back
        via four strip copies (toroidal wrap).
        """
        self.grid.fill(0.0)
        if not food:
            return
        R = self.radius
        H, W = self.height, self.width
        size = self._kernel_size

        # Wrap food positions into grid bounds.
        positions = np.array([(f.x, f.y) for f in food], dtype=np.float32)
        cx = np.round(positions[:, 0]).astype(np.int64) % W
        cy = np.round(positions[:, 1]).astype(np.int64) % H

        padded = np.zeros((H + 2 * R, W + 2 * R), dtype=np.float32)
        # Splat each food's kernel onto the padded grid.
        for fx, fy in zip(cx, cy):
            padded[fy : fy + size, fx : fx + size] += self._kernel

        # Take the central H x W region.
        central = padded[R:R + H, R:R + W].copy()
        # Fold spillover edges back in (toroidal wrap).
        central[-R:, :] += padded[:R, R:R + W]
        central[:R, :] += padded[H + R:, R:R + W]
        central[:, -R:] += padded[R:R + H, :R]
        central[:, :R] += padded[R:R + H, W + R:]
        self.grid = central

    def sample(
        self, x: float, y: float, heading: float, half_angle: float
    ) -> np.ndarray:
        """Sample smell intensity at three probe points: L, F, R.

        Each probe is a single field-cell sample at a fixed distance
        PROBE_DISTANCE ahead of the organism in the corresponding
        direction. This preserves gradient information: food at the
        probe point yields a high reading, food farther away yields a
        lower reading, food absent yields zero.

        Returns float32 array of shape (3,) with each value in [0, 1].
        """
        # Distance ahead of the organism at which we sample smell.
        # Must be <= SMELL_FIELD_RADIUS.
        PROBE_DISTANCE = 8.0
        cos_thresh = math.cos(half_angle)

        def probe(angle_offset: float) -> float:
            ang = heading + angle_offset
            sx = (x + PROBE_DISTANCE * math.cos(ang)) % self.width
            sy = (y + PROBE_DISTANCE * math.sin(ang)) % self.height
            v = float(self.grid[int(sy) % self.height, int(sx) % self.width])
            return v

        left_v = probe(-half_angle)
        front_v = probe(0.0)
        right_v = probe(+half_angle)
        # Soft saturation: 1 - exp(-k*v). With k=3, v=0.3 -> 0.59,
        # v=1.0 -> 0.95, v=10 -> 1.0 (asymptote). This prevents total
        # saturation in dense food regions while keeping low values
        # informative.
        k = 3.0
        return np.array(
            [
                1.0 - math.exp(-k * left_v),
                1.0 - math.exp(-k * front_v),
                1.0 - math.exp(-k * right_v),
            ],
            dtype=np.float32,
        )
