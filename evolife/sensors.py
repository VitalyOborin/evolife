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
        """Sample smell intensity in three sectors: left, front, right.

        Returns float32 array of shape (3,) with each value in [0, 1].
        Each sector is a +/-half_angle wedge around its direction.
        """
        R = self.radius
        cx = int(round(x))
        cy = int(round(y))
        x0 = max(0, cx - R)
        x1 = min(self.width, cx + R + 1)
        y0 = max(0, cy - R)
        y1 = min(self.height, cy + R + 1)
        if x0 >= x1 or y0 >= y1:
            return np.zeros(3, dtype=np.float32)

        sub = self.grid[y0:y1, x0:x1]
        ys, xs = np.mgrid[y0 - cy:y1 - cy, x0 - cx:x1 - cx]
        norm = np.hypot(xs, ys)
        valid = norm > 0.5
        ux = np.zeros_like(xs, dtype=np.float32)
        uy = np.zeros_like(ys, dtype=np.float32)
        ux[valid] = xs[valid] / norm[valid]
        uy[valid] = ys[valid] / norm[valid]

        cos_thresh = math.cos(half_angle)

        def sector_value(dx: float, dy: float) -> float:
            dot = ux * dx + uy * dy
            mask = valid & (dot >= cos_thresh)
            if not mask.any():
                return 0.0
            return float(sub[mask].sum())

        left_dir = (
            math.cos(heading - half_angle),
            math.sin(heading - half_angle),
        )
        front_dir = (math.cos(heading), math.sin(heading))
        right_dir = (
            math.cos(heading + half_angle),
            math.sin(heading + half_angle),
        )
        left_v = sector_value(*left_dir)
        front_v = sector_value(*front_dir)
        right_v = sector_value(*right_dir)
        # Squash into [0, 1] — kernel max is 1.0 at distance 0; summing
        # can exceed 1 in dense regions.
        return np.array(
            [
                min(1.0, left_v),
                min(1.0, front_v),
                min(1.0, right_v),
            ],
            dtype=np.float32,
        )
