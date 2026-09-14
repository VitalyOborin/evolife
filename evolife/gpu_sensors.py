"""GPU-backed smell field and sensor sampling.

The CPU implementation splats each food particle one-by-one in a Python
loop. On the GPU we instead:

  1. Build a (1, 1, H, W) "food indicator" tensor with 1.0 at each
     food cell.
  2. Convolve with the falloff kernel (2R+1, 2R+1) under toroidal
     padding via circular convolution (F.conv2d with padding=2R).
     Output is (1, 1, H, W) of summed falloff contributions.

This single conv replaces the entire splat+foldback loop and runs in
microseconds for the 512x512 world. Sampling sensors becomes a
batched gather on the grid.

This module mirrors the CPU API (recompute, sample) so we can swap
smell fields transparently from GpuWorld.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import SMELL_FIELD_RADIUS


class GpuSmellField:
    """Diffusion-style smell field on GPU.

    Internally holds:
      - a CPU copy of the kernel for fallback / inspection
      - a GPU tensor `grid` of shape (1, 1, H, W) with current intensity

    recompute(food_positions) accepts a (N, 2) numpy/torch array of
    food (x, y) positions and produces the smell field via circular
    convolution.

    sample(positions, headings, half_angle) accepts a (N,) tensor of
    x, a (N,) tensor of y, a (N,) tensor of heading, and returns an
    (N, 3) float32 tensor of (L, F, R) probe intensities with the
    same soft saturation as the CPU sample().
    """

    PROBE_DISTANCE = 8.0

    def __init__(
        self,
        width: int,
        height: int,
        radius: int = SMELL_FIELD_RADIUS,
        device: torch.device | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.radius = radius
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        R = radius
        ys, xs = np.mgrid[-R:R + 1, -R:R + 1]
        kernel_np = (1.0 / (1.0 + np.hypot(xs, ys))).astype(np.float32)
        # F.conv2d wants (out_channels, in_channels, kH, kW).
        self._kernel = torch.from_numpy(kernel_np).reshape(
            1, 1, 2 * R + 1, 2 * R + 1
        ).to(self.device)
        self._grid = torch.zeros(
            (1, 1, height, width), dtype=torch.float32, device=self.device
        )

    @property
    def grid(self) -> torch.Tensor:
        return self._grid[0, 0]

    def to_cpu(self) -> np.ndarray:
        return self._grid[0, 0].cpu().numpy()

    def recompute(self, food: list) -> None:
        """Compute the smell field from a list of Food objects.

        Mirrors CPU SmellField.recompute API.
        """
        self._grid.zero_()
        if not food:
            return
        R = self.radius
        H, W = self.height, self.width
        positions = np.array(
            [(f.x, f.y) for f in food], dtype=np.float32
        )
        cx = np.round(positions[:, 0]).astype(np.int64) % W
        cy = np.round(positions[:, 1]).astype(np.int64) % H
        cx = np.clip(cx, 0, W - 1)
        cy = np.clip(cy, 0, H - 1)
        idx = cy * W + cx
        flat = torch.zeros(H * W, dtype=torch.float32, device=self.device)
        ones = torch.ones(len(idx), dtype=torch.float32, device=self.device)
        flat.scatter_add_(0, torch.from_numpy(idx).to(self.device), ones)
        food_grid = flat.reshape(1, 1, H, W)

        # Circular convolution: F.conv2d with circular padding on each side.
        padded = F.pad(food_grid, (R, R, R, R), mode="circular")
        out = F.conv2d(padded, self._kernel)
        self._grid = out

    def recompute_from_tensor(
        self,
        positions: torch.Tensor,
        alive_mask: torch.Tensor | None = None,
    ) -> None:
        """Recompute smell from a (F, 2) GPU tensor of food positions."""
        self._grid.zero_()
        if positions.numel() == 0:
            return
        R = self.radius
        H, W = self.height, self.width
        if alive_mask is not None:
            positions = positions[alive_mask]
        cx = torch.remainder(
            torch.round(positions[:, 0]).to(torch.int64), W
        )
        cy = torch.remainder(
            torch.round(positions[:, 1]).to(torch.int64), H
        )
        idx = cy * W + cx
        flat = torch.zeros(H * W, dtype=torch.float32, device=self.device)
        ones = torch.ones(idx.shape[0], dtype=torch.float32, device=self.device)
        flat.scatter_add_(0, idx, ones)
        food_grid = flat.reshape(1, 1, H, W)
        padded = F.pad(food_grid, (R, R, R, R), mode="circular")
        out = F.conv2d(padded, self._kernel)
        self._grid = out

    def sample_batch(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        headings: torch.Tensor,
        half_angle: float,
    ) -> torch.Tensor:
        """Sample smell for a batch of organisms.

        Returns (N, 3) float32 tensor on `self.device`.
        """
        D = self.PROBE_DISTANCE
        offsets = torch.tensor(
            [-half_angle, 0.0, +half_angle],
            dtype=torch.float32,
            device=self.device,
        )
        # angles: (N, 3) = headings[:, None] + offsets[None, :]
        angles = headings.unsqueeze(1) + offsets.unsqueeze(0)
        # probe_x: (N, 3) = (x + D*cos(angle)) % W
        # NOTE: float `%` can return values exactly equal to W (when the
        # divisor is W and the dividend lands on a boundary cell). We
        # wrap with a safe floor-mod so the result is strictly < W.
        probe_x = torch.remainder(
            xs.unsqueeze(1) + D * torch.cos(angles), self.width
        )
        probe_y = torch.remainder(
            ys.unsqueeze(1) + D * torch.sin(angles), self.height
        )
        ix = probe_x.long().clamp(min=0, max=self.width - 1)
        iy = probe_y.long().clamp(min=0, max=self.height - 1)
        # Gather values from grid (1, 1, H, W) -> (H, W).
        grid2d = self._grid[0, 0]
        # Linearise indices.
        lin = iy * self.width + ix  # (N, 3)
        flat = grid2d.reshape(-1)
        vals = flat[lin.reshape(-1)].reshape(ix.shape)  # (N, 3)
        # Soft saturation: 1 - exp(-k*v), k=3.
        return (1.0 - torch.exp(-3.0 * vals)).to(torch.float32)

    # Single-organism API for compatibility with the CPU code.
    def sample(
        self, x: float, y: float, heading: float, half_angle: float
    ) -> np.ndarray:
        xs = torch.tensor([x], dtype=torch.float32, device=self.device)
        ys = torch.tensor([y], dtype=torch.float32, device=self.device)
        headings = torch.tensor(
            [heading], dtype=torch.float32, device=self.device
        )
        out = self.sample_batch(xs, ys, headings, half_angle)
        return out[0].cpu().numpy()
