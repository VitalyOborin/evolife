"""Phase 6 -- social marker field (colony signalling).

Implements a generic environmental marker field per Choe & Chung (2011).
Organisms deposit a marker at their position when they eat; the marker
field diffuses and decays over time, providing an external memory
channel. Other organisms can sense the marker field through a probe
similar to the smell probe.

Design choices:

- **Generic**: the marker carries no information about food type. The
  semantics is "something was eaten here recently", nothing more.
  This is the test arm that isolates environmental memory from
  internal memory (recurrence).

- **Grid**: markers live on a coarser grid than the world. This makes
  diffusion a small convolution and gives organisms local smoothing
  without per-cell precision.

- **Diffusion + decay**: each tick M = decay * (alpha * blur(M) +
  (1 - alpha) * M) where alpha = COLONY_MARKER_DIFFUSION.

- **Backward compat**: when COLONY_MARKER_OFF is True (the default),
  emit() and step() are no-ops, the field stays at zero, and no sensor
  reads are added to organisms. All other phases behave unchanged.
"""
from __future__ import annotations

import numpy as np

from .config import (
    COLONY_MARKER_DECAY,
    COLONY_MARKER_DIFFUSION,
    COLONY_MARKER_GRID_SCALE,
    COLONY_MARKER_OFF,
)


class ColonyMarkerField:
    """Coarse 2D grid of social markers. Add-only API for callers."""

    def __init__(self, width: float, height: float):
        scale = COLONY_MARKER_GRID_SCALE
        self.grid_w = max(1, int(round(width * scale)))
        self.grid_h = max(1, int(round(height * scale)))
        self._m = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        # Pre-compute a separable Gaussian-ish kernel. Using a 1D kernel
        # reduces the per-tick convolution from 9 ops/cell to 6.
        self._k1d = np.array([0.25, 0.5, 0.25], dtype=np.float32)

    def emit(self, x: float, y: float, amount: float = 1.0) -> None:
        """Add `amount` of marker at world position (x, y).

        Out-of-bounds positions are wrapped modulo grid size, so the
        field is toroidal (matches the world wrap semantics).
        """
        if COLONY_MARKER_OFF:
            return
        gx = int(x * COLONY_MARKER_GRID_SCALE) % self.grid_w
        gy = int(y * COLONY_MARKER_GRID_SCALE) % self.grid_h
        self._m[gy, gx] += amount

    def step(self) -> None:
        """One tick of decay + diffusion. Called once per world step."""
        if COLONY_MARKER_OFF:
            return
        # Diffusion via separable 3x3 convolution (1D kernel applied
        # twice). Use np.roll for toroidal wrap without padding
        # artifacts. Two 1D convs are ~3x faster than one 2D conv.
        if COLONY_MARKER_DIFFUSION > 0.0:
            k = self._k1d
            # Horizontal pass: blur along axis=1.
            h0 = np.roll(self._m, -1, axis=1) * k[0]
            h1 = self._m * k[1]
            h2 = np.roll(self._m, 1, axis=1) * k[2]
            blurred_h = h0 + h1 + h2
            # Vertical pass: blur along axis=0.
            v0 = np.roll(blurred_h, -1, axis=0) * k[0]
            v1 = blurred_h * k[1]
            v2 = np.roll(blurred_h, 1, axis=0) * k[2]
            blurred = v0 + v1 + v2
            alpha = COLONY_MARKER_DIFFUSION
            self._m = alpha * blurred + (1.0 - alpha) * self._m
        # Decay (always applied).
        self._m *= COLONY_MARKER_DECAY

    def sample(self, x: float, y: float, radius: float = 1.0) -> float:
        """Return mean marker intensity in a (radius)-sized window."""
        if COLONY_MARKER_OFF:
            return 0.0
        scale = COLONY_MARKER_GRID_SCALE
        gx = x * scale
        gy = y * scale
        gx_lo = int(gx - radius) % self.grid_w
        gx_hi = int(gx + radius) + 1
        gy_lo = int(gy - radius) % self.grid_h
        gy_hi = int(gy + radius) + 1
        # Handle wrap-around slices.
        if gx_hi - gx_lo <= self.grid_w and gy_hi - gy_lo <= self.grid_h:
            return float(self._m[gy_lo:gy_hi, gx_lo:gx_hi].mean())
        # Wrap around edges -- take circular mean.
        total = 0.0
        count = 0
        for iy in range(gy_lo, gy_hi):
            for ix in range(gx_lo, gx_hi):
                total += float(self._m[iy % self.grid_h, ix % self.grid_w])
                count += 1
        return total / count if count else 0.0

    @property
    def grid(self) -> np.ndarray:
        """Expose the raw grid (read-only) for visualisation."""
        return self._m
