"""Lab visualizer — full torus, smell field, probe rays, side chart.

The simulation still drives itself. This module only reads world state
and paints it. Smell colormaps run on CUDA when torch.cuda is available.
"""

from __future__ import annotations

import colorsys
import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import pygame

from .config import (
    FOOD_ENERGY,
    FOOD_RADIUS,
    FPS_TARGET,
    ORGANISM_RADIUS,
    PROBE_DISTANCE,
    SMELL_HALF_ANGLE,
)
from .visualization import (
    NEWBORN_COLOR,
    organism_at,
    species_draw_color,
)

PANEL_WIDTH = 420
SERIES_CAP = 2400
_VOID = np.array([8, 10, 16], dtype=np.float32)
_SMELL_A = np.array([36, 196, 168], dtype=np.float32)
_SMELL_B = np.array([232, 148, 48], dtype=np.float32)
_FOOD_A = (110, 255, 190)
_FOOD_B = (255, 176, 64)
_FOOD = (110, 255, 190)
_POP_COLOR = (120, 210, 255)
_FOOD_COLOR = (90, 220, 140)
_MUT_COLOR = (255, 170, 70)
_PROBE_DIM = (70, 100, 130)
_PROBE_HOT = (190, 255, 245)
_BODY_DIM = (90, 88, 86)
_HEADING = (255, 230, 140)
_RECURRENT = (240, 240, 100)
_CLICK_RADIUS = 12

_PROBE_OFFSETS = (-SMELL_HALF_ANGLE, 0.0, SMELL_HALF_ANGLE)


def viz_torch_device():
    """CUDA if present, else CPU torch, else None (numpy colormap)."""
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def probe_point(
    x: float,
    y: float,
    heading: float,
    offset: float,
    width: float,
    height: float,
    distance: float = PROBE_DISTANCE,
) -> tuple[float, float, bool]:
    """Wrapped probe location and whether the segment crosses the seam."""
    ang = heading + offset
    px = (x + distance * math.cos(ang)) % width
    py = (y + distance * math.sin(ang)) % height
    wraps = abs(px - x) > distance + 1.0 or abs(py - y) > distance + 1.0
    return px, py, wraps


def genome_has_recurrent_cycle(genome) -> bool:
    """True if enabled edges contain a directed cycle of size >= 2."""
    if genome is None:
        return False
    adj: dict[int, list[int]] = {}
    for c in genome.connections.values():
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

    for v in list(genome.nodes):
        if found[0]:
            break
        if v not in indices:
            strongconnect(v)
    return found[0]


def colorize_smell_fields(
    fields: list[tuple[object, tuple[int, int, int]]],
    device=None,
) -> np.ndarray:
    """Map one or more smell grids to an HxWx3 uint8 heatmap.

    Each field is normalized to its own max, then tinted and added over
    a dark void. CUDA is used when `device` is a torch device.
    """
    if not fields:
        raise ValueError("colorize_smell_fields requires at least one field")
    if device is not None:
        return _colorize_torch(fields, device)
    return _colorize_numpy(fields)


def _colorize_numpy(
    fields: list[tuple[object, tuple[int, int, int]]],
) -> np.ndarray:
    first = np.asarray(fields[0][0], dtype=np.float32)
    rgb = np.broadcast_to(_VOID, first.shape + (3,)).copy()
    for grid, tint in fields:
        g = np.asarray(grid, dtype=np.float32)
        peak = float(g.max()) if g.size else 0.0
        if peak <= 1e-8:
            continue
        t = np.clip(g / peak, 0.0, 1.0) ** 0.5
        tint_arr = np.array(tint, dtype=np.float32)
        rgb = rgb + t[..., None] * (tint_arr - _VOID)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _colorize_torch(
    fields: list[tuple[object, tuple[int, int, int]]],
    device,
) -> np.ndarray:
    import torch

    rgb = None
    void = torch.tensor(_VOID, device=device, dtype=torch.float32)
    for grid, tint in fields:
        if hasattr(grid, "detach"):
            g = grid.detach().to(device=device, dtype=torch.float32)
        else:
            g = torch.from_numpy(
                np.ascontiguousarray(grid, dtype=np.float32)
            ).to(device)
        if rgb is None:
            rgb = void.view(1, 1, 3).expand(g.shape[0], g.shape[1], 3).clone()
        peak = g.amax().clamp(min=1e-8)
        t = (g / peak).clamp(0.0, 1.0).pow(0.5)
        tint_t = torch.tensor(tint, device=device, dtype=torch.float32)
        rgb = rgb + t.unsqueeze(-1) * (tint_t - void)
    assert rgb is not None
    return rgb.clamp(0, 255).to(torch.uint8).cpu().numpy()


def _lerp_color(
    a: tuple[int, int, int], b: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


@dataclass
class _OrgDraw:
    x: float
    y: float
    heading: float
    color: tuple[int, int, int]
    sensors: tuple[float, float, float]
    energy: float
    generation: int
    recurrent: bool
    species_id: int | None
    slot: int | None


class RollingSeries:
    """Tick-aligned pop / food / mutation history for the side chart."""

    def __init__(self, cap: int = SERIES_CAP) -> None:
        self.cap = cap
        self.pop: deque[float] = deque(maxlen=cap)
        self.food: deque[float] = deque(maxlen=cap)
        self.mutations: deque[float] = deque(maxlen=cap)

    def push(self, pop: float, food: float, mutations: float) -> None:
        self.pop.append(pop)
        self.food.append(food)
        self.mutations.append(mutations)

    def __len__(self) -> int:
        return len(self.pop)


class LabVisualizer:
    """Pygame lab: torus on the left, instrument panel on the right."""

    def __init__(self, world) -> None:
        self.world = world
        self.device = viz_torch_device()
        if hasattr(world, "device") and self.device is not None:
            # Prefer the world's device so a CUDA smell grid is not copied.
            self.device = world.device
        pygame.init()
        self.world_w = int(world.width)
        self.world_h = int(world.height)
        self.screen = pygame.display.set_mode(
            (self.world_w + PANEL_WIDTH, self.world_h)
        )
        pygame.display.set_caption(self._caption())
        self._torus = pygame.Surface((self.world_w, self.world_h))
        try:
            self._torus = self._torus.convert()
        except pygame.error:
            pass
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 14)
        self.small = pygame.font.SysFont("Consolas", 13)
        self.tiny = pygame.font.SysFont("Consolas", 12)
        self.series = RollingSeries()
        self.selected_species_id: int | None = None
        self.selected_slot: int | None = None
        self._line_to_hue: dict[int, float] = {}
        self._next_hue_seed = 0
        self._orgs: list[_OrgDraw] = []

    def _caption(self) -> str:
        kind = self._kind()
        dev = "CUDA" if self._cuda() else "CPU"
        if kind == "phase3":
            mode = getattr(self.world, "mode", "?")
            return f"EvoLife · {mode} · {dev}"
        return f"EvoLife · torus · {dev}"

    def _kind(self) -> str:
        if hasattr(self.world, "food_a"):
            return "phase3"
        if hasattr(self.world, "alive_mask"):
            return "gpu"
        return "classic"

    def _cuda(self) -> bool:
        return self.device is not None and str(self.device).startswith("cuda")

    def render(self) -> None:
        self._handle_events()
        kind = self._kind()
        smell = self._smell_rgb(kind)
        self._orgs = self._collect_orgs(kind)
        food_n = self._draw_torus(kind, smell)
        pop = self._population()
        mut = int(getattr(self.world, "tick_mutations", 0))
        self.series.push(float(pop), float(food_n), float(mut))
        self.screen.blit(self._torus, (0, 0))
        self._draw_panel(kind, pop, food_n, mut)
        pygame.display.flip()
        self.clock.tick(FPS_TARGET)

    def close(self) -> None:
        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._on_click(event.pos)

    def _on_click(self, pos: tuple[int, int]) -> None:
        x, y = pos
        if x >= self.world_w:
            return
        kind = self._kind()
        if kind == "gpu":
            best = None
            best_d = float(_CLICK_RADIUS)
            for org in self._orgs:
                d = math.hypot(org.x - x, org.y - y)
                if d <= best_d:
                    best_d = d
                    best = org
            self.selected_slot = None if best is None else best.slot
            self.selected_species_id = None
            return
        living = [
            o for o in getattr(self.world, "organisms", []) if o.alive
        ]
        hit = organism_at(living, x, y, _CLICK_RADIUS)
        self.selected_species_id = None if hit is None else hit.species_id
        self.selected_slot = None

    def _smell_rgb(self, kind: str) -> np.ndarray:
        device = self.device
        if kind == "phase3":
            return colorize_smell_fields(
                [
                    (self.world.dual_smell.field_a.grid, tuple(int(v) for v in _SMELL_A)),
                    (self.world.dual_smell.field_b.grid, tuple(int(v) for v in _SMELL_B)),
                ],
                device=device,
            )
        grid = self.world.smell.grid
        return colorize_smell_fields(
            [(grid, tuple(int(v) for v in _SMELL_A))],
            device=device,
        )

    def _collect_orgs(self, kind: str) -> list[_OrgDraw]:
        if kind == "gpu":
            return self._collect_gpu_orgs()
        out: list[_OrgDraw] = []
        established = self._established
        for org in self.world.organisms:
            if not org.alive:
                continue
            sensors = self._cpu_sensors(org, kind)
            color = species_draw_color(
                org.species_id, established(org.species_id)
            )
            e_t = max(0.0, min(1.0, float(org.energy) / (FOOD_ENERGY * 2.0)))
            body = _lerp_color(_BODY_DIM, color, 0.45 + 0.55 * e_t)
            out.append(
                _OrgDraw(
                    x=float(org.x),
                    y=float(org.y),
                    heading=float(org.heading),
                    color=body,
                    sensors=sensors,
                    energy=float(org.energy),
                    generation=int(org.generation),
                    recurrent=genome_has_recurrent_cycle(org.genome),
                    species_id=org.species_id,
                    slot=org.id,
                )
            )
        return out

    def _cpu_sensors(self, org, kind: str) -> tuple[float, float, float]:
        if kind == "phase3":
            vec = self.world.dual_smell.sample(
                org.x, org.y, org.heading, SMELL_HALF_ANGLE
            )
            a = vec[:3]
            b = vec[3:6]
            return (
                float(max(a[0], b[0])),
                float(max(a[1], b[1])),
                float(max(a[2], b[2])),
            )
        vec = self.world.smell.sample(
            org.x, org.y, org.heading, SMELL_HALF_ANGLE
        )
        return float(vec[0]), float(vec[1]), float(vec[2])

    def _collect_gpu_orgs(self) -> list[_OrgDraw]:
        w = self.world
        alive_idx = w.alive_mask.nonzero(as_tuple=False).squeeze(1)
        if alive_idx.numel() == 0:
            return []
        pos = w.positions[alive_idx].detach().to("cpu").numpy()
        head = w.headings[alive_idx].detach().to("cpu").numpy()
        energy = w.energy[alive_idx].detach().to("cpu").numpy()
        parent = w.parent_id[alive_idx].detach().to("cpu").numpy()
        gen = w.generation[alive_idx].detach().to("cpu").numpy()
        n_sens = min(3, int(w.brain_state.shape[1]))
        sensors = w.brain_state[alive_idx, :n_sens].detach().to("cpu").numpy()
        slots = alive_idx.detach().to("cpu").numpy()
        out: list[_OrgDraw] = []
        for i in range(int(alive_idx.shape[0])):
            hue = self._hue_for(int(parent[i]))
            r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.95)
            base = (int(r * 255), int(g * 255), int(b * 255))
            e_t = max(0.0, min(1.0, float(energy[i]) / (FOOD_ENERGY * 2.0)))
            body = _lerp_color(_BODY_DIM, base, 0.45 + 0.55 * e_t)
            sens = tuple(float(v) for v in sensors[i, :3])
            while len(sens) < 3:
                sens = sens + (0.0,)
            out.append(
                _OrgDraw(
                    x=float(pos[i, 0]),
                    y=float(pos[i, 1]),
                    heading=float(head[i]),
                    color=body,
                    sensors=(sens[0], sens[1], sens[2]),
                    energy=float(energy[i]),
                    generation=int(gen[i]),
                    recurrent=False,
                    species_id=None,
                    slot=int(slots[i]),
                )
            )
        return out

    def _hue_for(self, parent_id: int) -> float:
        if parent_id == -1:
            hue = (self._next_hue_seed * 0.6180339887) % 1.0
            self._next_hue_seed += 1
            return hue
        if parent_id in self._line_to_hue:
            return self._line_to_hue[parent_id]
        hue = (self._next_hue_seed * 0.6180339887) % 1.0
        self._next_hue_seed += 1
        self._line_to_hue[parent_id] = hue
        return hue

    def _established(self, species_id: int) -> bool:
        mgr = getattr(self.world, "species_manager", None)
        if mgr is None:
            return True
        sp = mgr.species.get(species_id)
        return bool(sp is not None and sp.established)

    def _population(self) -> int:
        if hasattr(self.world, "population"):
            return int(self.world.population())
        return sum(1 for o in self.world.organisms if o.alive)

    def _draw_torus(self, kind: str, smell: np.ndarray) -> int:
        # pygame.surfarray wants (W, H, 3).
        pygame.surfarray.blit_array(self._torus, np.transpose(smell, (1, 0, 2)))
        food_n = self._draw_food(kind)
        for org in self._orgs:
            self._draw_probes(org)
        for org in self._orgs:
            self._draw_body(org)
        self._draw_torus_hud()
        return food_n

    def _draw_food(self, kind: str) -> int:
        if kind == "phase3":
            n = 0
            for food in self.world.food_a:
                pygame.draw.circle(
                    self._torus, _FOOD_A, (int(food.x), int(food.y)), FOOD_RADIUS
                )
                n += 1
            for food in self.world.food_b:
                pygame.draw.circle(
                    self._torus, _FOOD_B, (int(food.x), int(food.y)), FOOD_RADIUS
                )
                n += 1
            return n
        if kind == "gpu":
            n_food = int(self.world._food_count)
            if n_food <= 0:
                return 0
            food_pos = self.world.food_pos[:n_food].detach().to("cpu").numpy()
            for fx, fy in food_pos:
                pygame.draw.circle(
                    self._torus, _FOOD, (int(fx), int(fy)), FOOD_RADIUS
                )
            return n_food
        for food in self.world.food:
            pygame.draw.circle(
                self._torus, _FOOD, (int(food.x), int(food.y)), FOOD_RADIUS
            )
        return len(self.world.food)

    def _draw_probes(self, org: _OrgDraw) -> None:
        for offset, value in zip(_PROBE_OFFSETS, org.sensors):
            px, py, wraps = probe_point(
                org.x, org.y, org.heading, offset, self.world_w, self.world_h
            )
            color = _lerp_color(_PROBE_DIM, _PROBE_HOT, float(value))
            if not wraps:
                pygame.draw.aaline(
                    self._torus,
                    color,
                    (org.x, org.y),
                    (px, py),
                )
            pygame.draw.circle(self._torus, color, (int(px), int(py)), 2)

    def _draw_body(self, org: _OrgDraw) -> None:
        pos = (int(org.x), int(org.y))
        pygame.draw.circle(self._torus, org.color, pos, ORGANISM_RADIUS)
        hx = org.x + math.cos(org.heading) * (ORGANISM_RADIUS + 4)
        hy = org.y + math.sin(org.heading) * (ORGANISM_RADIUS + 4)
        pygame.draw.aaline(self._torus, _HEADING, (org.x, org.y), (hx, hy))
        if org.recurrent:
            pygame.draw.circle(
                self._torus, _RECURRENT, pos, ORGANISM_RADIUS + 3, 1
            )
        selected = (
            (
                self.selected_species_id is not None
                and org.species_id == self.selected_species_id
            )
            or (
                self.selected_slot is not None
                and org.slot == self.selected_slot
            )
        )
        if selected:
            pygame.draw.circle(
                self._torus, (255, 255, 255), pos, ORGANISM_RADIUS + 4, 1
            )

    def _draw_torus_hud(self) -> None:
        label = self.tiny.render(
            f"torus {self.world_w}×{self.world_h}  probe {PROBE_DISTANCE:.0f}u",
            True,
            (160, 170, 180),
        )
        self._torus.blit(label, (6, self.world_h - 18))

    def _draw_panel(self, kind: str, pop: int, food_n: int, mut: int) -> None:
        x0 = self.world_w
        panel = pygame.Rect(x0, 0, PANEL_WIDTH, self.world_h)
        pygame.draw.rect(self.screen, (11, 14, 20), panel)
        pygame.draw.line(
            self.screen, (40, 48, 60), (x0, 0), (x0, self.world_h)
        )
        y = 12
        y = self._blit_text(self.font, self._caption_line(kind), (x0 + 14, y), (230, 236, 240))
        y += 6
        mean_e = 0.0
        if hasattr(self.world, "mean_energy"):
            mean_e = float(self.world.mean_energy())
        max_gen = 0
        if hasattr(self.world, "max_generation"):
            max_gen = int(self.world.max_generation())
        stats = [
            f"tick     {self.world.tick:>8}",
            f"pop      {pop:>8}",
            f"food     {food_n:>8}",
            f"mean E   {mean_e:>8.1f}",
            f"max gen  {max_gen:>8}",
            f"mut/tick {mut:>8}",
            f"births   {int(getattr(self.world, 'tick_births', 0)):>8}",
        ]
        if kind == "phase3":
            season_enabled = getattr(self.world, "season_enabled", False)
            if season_enabled:
                stats.append(
                    f"season   {int(self.world.season)}  "
                    f"t={int(self.world.ticks_in_season)}"
                )
            else:
                stats.append("season   static")
            n_rec = sum(1 for o in self._orgs if o.recurrent)
            stats.append(f"recurrent {n_rec}/{pop}")
        for line in stats:
            y = self._blit_text(self.small, line, (x0 + 14, y), (200, 208, 216))
        y += 10
        y = self._blit_text(self.tiny, "smell field · probe rays · food", (x0 + 14, y), (120, 130, 140))
        y += 4
        legend = [
            (_FOOD_A, "food / FoodA"),
            (_FOOD_B, "FoodB (phase 3)"),
            (tuple(int(v) for v in _SMELL_A), "smell A"),
            (tuple(int(v) for v in _SMELL_B), "smell B"),
            (_PROBE_HOT, "probe (bright = signal)"),
            (_HEADING, "heading"),
        ]
        if kind != "phase3":
            legend = [
                (_FOOD, "food"),
                (tuple(int(v) for v in _SMELL_A), "smell"),
                (_PROBE_HOT, "probe (bright = signal)"),
                (_HEADING, "heading"),
            ]
        for color, label in legend:
            pygame.draw.circle(self.screen, color, (x0 + 22, y + 7), 4)
            self.screen.blit(
                self.tiny.render(label, True, (180, 186, 194)),
                (x0 + 34, y),
            )
            y += 16
        y += 8
        y = self._draw_selection(x0, y)
        chart = pygame.Rect(x0 + 12, y + 8, PANEL_WIDTH - 24, self.world_h - y - 20)
        if chart.height > 80:
            _draw_chart(self.screen, self.series, chart, self.tiny)

    def _caption_line(self, kind: str) -> str:
        dev = "CUDA" if self._cuda() else "CPU"
        if kind == "phase3":
            return f"memory ecology  {dev}"
        if kind == "gpu":
            return f"gpu world  {dev}"
        return f"classic world  {dev}"

    def _draw_selection(self, x0: int, y: int) -> int:
        org = None
        if self.selected_slot is not None:
            for o in self._orgs:
                if o.slot == self.selected_slot:
                    org = o
                    break
        elif self.selected_species_id is not None:
            for o in self._orgs:
                if o.species_id == self.selected_species_id:
                    org = o
                    break
        if org is None:
            return self._blit_text(
                self.tiny, "click an organism on the torus",
                (x0 + 14, y), (110, 118, 128),
            )
        lines = [
            f"selected  gen {org.generation}  E={org.energy:.1f}",
            f"L {org.sensors[0]:.2f}  F {org.sensors[1]:.2f}  R {org.sensors[2]:.2f}",
        ]
        if org.species_id is not None:
            lines.insert(0, f"species {org.species_id}")
        for line in lines:
            y = self._blit_text(self.tiny, line, (x0 + 14, y), (220, 224, 230))
        return y

    def _blit_text(
        self,
        font,
        text: str,
        pos: tuple[int, int],
        color: tuple[int, int, int],
    ) -> int:
        surf = font.render(text, True, color)
        self.screen.blit(surf, pos)
        return pos[1] + surf.get_height() + 2


def _draw_chart(
    screen: pygame.Surface,
    series: RollingSeries,
    rect: pygame.Rect,
    font,
) -> None:
    pygame.draw.rect(screen, (16, 20, 28), rect)
    pygame.draw.rect(screen, (48, 56, 68), rect, 1)
    title = font.render("pop  food  mutations / tick", True, (170, 178, 186))
    screen.blit(title, (rect.x + 8, rect.y + 6))
    plot = pygame.Rect(rect.x + 8, rect.y + 24, rect.width - 16, rect.height - 42)
    pygame.draw.rect(screen, (10, 12, 18), plot)
    if len(series) < 2:
        empty = font.render("waiting for ticks…", True, (90, 96, 104))
        screen.blit(empty, (plot.x + 8, plot.y + 8))
        return
    n = len(series)
    step = max(1, n // plot.width)
    pop = list(series.pop)[::step]
    food = list(series.food)[::step]
    mut = list(series.mutations)[::step]
    m = min(len(pop), len(food), len(mut), plot.width)
    pop, food, mut = pop[:m], food[:m], mut[:m]
    _polyline(screen, plot, pop, _POP_COLOR)
    _polyline(screen, plot, food, _FOOD_COLOR)
    _polyline(screen, plot, mut, _MUT_COLOR)
    last_pop = series.pop[-1]
    last_food = series.food[-1]
    last_mut = series.mutations[-1]
    footer = font.render(
        f"pop {last_pop:.0f}   food {last_food:.0f}   mut {last_mut:.0f}",
        True,
        (190, 196, 204),
    )
    screen.blit(footer, (rect.x + 8, rect.bottom - 16))


def _polyline(
    screen: pygame.Surface,
    plot: pygame.Rect,
    values: list[float],
    color: tuple[int, int, int],
) -> None:
    if len(values) < 2:
        return
    peak = max(values) if values else 1.0
    peak = max(peak, 1.0)
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        x = plot.x + int(i * (plot.width - 1) / max(n - 1, 1))
        y = plot.bottom - 1 - int((v / peak) * (plot.height - 1))
        pts.append((x, y))
    pygame.draw.aalines(screen, color, False, pts)
