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

from .config import PROBE_DISTANCE, SMELL_FIELD_RADIUS


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
        return _sample_three(self.grid, x, y, heading, half_angle, self.width, self.height)


def _sample_three(
    grid: np.ndarray,
    x: float,
    y: float,
    heading: float,
    half_angle: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Three-probe sample with soft saturation, used by both single and
    dual smell fields. Returns float32 array of shape (3,) in [0, 1].
    """
    k = 3.0

    def probe(angle_offset: float) -> float:
        ang = heading + angle_offset
        sx = (x + PROBE_DISTANCE * math.cos(ang)) % width
        sy = (y + PROBE_DISTANCE * math.sin(ang)) % height
        v = float(grid[int(sy) % height, int(sx) % width])
        return 1.0 - math.exp(-k * v)

    return np.array(
        [
            probe(-half_angle),
            probe(0.0),
            probe(+half_angle),
        ],
        dtype=np.float32,
    )


class DualSmellField:
    """Two independent SmellField instances, one per resource.

    Phase 3 uses this to give the brain distinguishable smell for FoodA
    and FoodB without changing sensor mechanics. Each field is recomputed
    independently from its own food list. `sample` returns a 6-vector:
    [a_left, a_front, a_right, b_left, b_front, b_right].
    """

    def __init__(
        self,
        width: int,
        height: int,
        radius: int = SMELL_FIELD_RADIUS,
    ) -> None:
        self.width = width
        self.height = height
        self.field_a = SmellField(width, height, radius)
        self.field_b = SmellField(width, height, radius)

    def recompute_split(self, food_a: list, food_b: list) -> None:
        """Recompute both fields from their respective food lists."""
        self.field_a.recompute(food_a)
        self.field_b.recompute(food_b)

    def sample(
        self, x: float, y: float, heading: float, half_angle: float
    ) -> np.ndarray:
        """Sample both fields and concatenate the (3,) results into (6,)."""
        a = self.field_a.sample(x, y, heading, half_angle)
        b = self.field_b.sample(x, y, heading, half_angle)
        return np.concatenate([a, b]).astype(np.float32)
