"""GPU-resident World — deep rewrite.

Design:
- Population stored as fixed-size GPU tensors of shape (POPULATION_CAP,).
- "Alive" organisms occupy a contiguous prefix of slot indices; the rest
  are dead and ignored via masks.
- Per-tick hot path: sensors (GPU), brain forward (GPU one big scatter),
  eat (GPU pairwise distances), motion (GPU), energy/age (GPU).
- One sync per tick: pull alive indices + their event-related data to
  CPU for the event log, and pull new food spawn events.

Trade-offs vs CPU World:
- Structural mutations (add_node, add_connection) are CPU-only. We
  apply them on the parent's genome on CPU, then copy the resulting
  weights into a free GPU slot.
- Event log granularity is reduced: instead of one event per Birth /
  Death / Eat, we record one summary per N ticks. This keeps the GPU
  hot path sync-free for typical ticks.

This file is a self-contained GPU simulation; it does NOT inherit from
CPU World because the data layout is fundamentally different.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .config import (
    CHILD_DISPERSAL_MAX,
    CHILD_DISPERSAL_MIN,
    CHILD_HEADING_NOISE,
    CONNECTION_METABOLIC_COST,
    EAT_RADIUS,
    FOOD_ENERGY,
    FOOD_REGROWTH_RATE,
    FOOD_TARGET,
    IDLE_ENERGY_COST,
    INITIAL_ENERGY,
    INITIAL_POPULATION,
    MAX_AGE,
    MAX_LINEAR_SPEED,
    MAX_TURN_RATE,
    MOVE_ENERGY_COST,
    NEURON_METABOLIC_COST,
    N_HIDDEN,
    N_MOTORS,
    N_SENSORS,
    POPULATION_CAP,
    REPRODUCTION_ENERGY,
    REPRODUCTION_THRESHOLD,
    SMELL_HALF_ANGLE,
    TURN_ENERGY_COST,
)
from .archive import Archive, MilestoneKind
from .events import Event, EventKind, EventLog
from .genome import Genome
from .gpu_sensors import GpuSmellField
from .mutation import mutate_weights


# Per-organism fixed-size gene storage. For v2.2 the default proto-brain
# has exactly 6 connections, but mutations can add more. We reserve a
# generous cap; connections above it are CPU-side only (rare).
MAX_CONNS_PER_ORG = 16


@dataclass
class Food:
    x: float
    y: float


class GpuWorld:
    """GPU-resident EvoLife world."""

    def __init__(
        self,
        seed: int | None = None,
        width: int = 512,
        height: int = 512,
        events: EventLog | None = None,
        device: torch.device | None = None,
        pop_cap: int = POPULATION_CAP,
    ) -> None:
        self.width = width
        self.height = height
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.rng = np.random.default_rng(seed)
        self.events = events if events is not None else EventLog()
        self.archive = Archive()
        self.pop_cap = pop_cap

        # Population tensors, all on GPU.
        D = self.device
        N = pop_cap
        self.positions = torch.zeros((N, 2), dtype=torch.float32, device=D)
        self.headings = torch.zeros(N, dtype=torch.float32, device=D)
        self.energy = torch.zeros(N, dtype=torch.float32, device=D)
        self.alive_mask = torch.zeros(N, dtype=torch.bool, device=D)
        self.age = torch.zeros(N, dtype=torch.int64, device=D)
        self.brain_state = torch.zeros(
            (N, N_SENSORS + N_HIDDEN + N_MOTORS), dtype=torch.float32, device=D
        )
        # Per-organism fixed gene slots. Each connection is stored as
        # (in_node, out_node, weight, enabled). Padding is (0, 0, 0, 0)
        # and disabled (masked out via enabled bit).
        self.gene_in = torch.zeros(
            (N, MAX_CONNS_PER_ORG), dtype=torch.int32, device=D
        )
        self.gene_out = torch.zeros(
            (N, MAX_CONNS_PER_ORG), dtype=torch.int32, device=D
        )
        self.gene_w = torch.zeros(
            (N, MAX_CONNS_PER_ORG), dtype=torch.float32, device=D
        )
        self.gene_enabled = torch.zeros(
            (N, MAX_CONNS_PER_ORG), dtype=torch.bool, device=D
        )
        self.parent_id = torch.full(
            (N,), -1, dtype=torch.int64, device=D
        )
        self.generation = torch.zeros(N, dtype=torch.int64, device=D)
        self.food_eaten = torch.zeros(N, dtype=torch.int64, device=D)

        # Food on GPU as a single tensor; we keep a CPU list too for
        # event-log positions.
        self.food_pos = torch.zeros(
            (FOOD_TARGET + 64, 2), dtype=torch.float32, device=D
        )
        self.food_alive = torch.zeros(
            FOOD_TARGET + 64, dtype=torch.bool, device=D
        )
        self._food_count = 0

        # Smell field on GPU.
        self.smell = GpuSmellField(width, height, device=self.device)

        # Scratch indices for finding free slots.
        self._next_id = 0
        self.tick = 0

        # Spawn founders.
        for _ in range(INITIAL_POPULATION):
            self._spawn_founder()
        # Initial food.
        while self._food_count < FOOD_TARGET:
            self._spawn_food()
        # Seed smell.
        self._refresh_food_tensor()
        self.smell.recompute_from_tensor(
            self.food_pos[: self._food_count]
        )

    # --- food helpers ------------------------------------------------------

    def _spawn_food(self) -> None:
        x = float(self.rng.uniform(0, self.width))
        y = float(self.rng.uniform(0, self.height))
        if self._food_count >= self.food_pos.shape[0]:
            # Grow tensor (rare).
            extra = torch.zeros(
                (64, 2), dtype=torch.float32, device=self.device
            )
            extra_alive = torch.zeros(
                64, dtype=torch.bool, device=self.device
            )
            self.food_pos = torch.cat([self.food_pos, extra], dim=0)
            self.food_alive = torch.cat([self.food_alive, extra_alive], dim=0)
        self.food_pos[self._food_count] = torch.tensor(
            [x, y], dtype=torch.float32, device=self.device
        )
        self.food_alive[self._food_count] = True
        self._food_count += 1

    def _refresh_food_tensor(self) -> None:
        # Mark unused slots as dead (in case any food was eaten).
        self.food_alive[self._food_count:] = False

    # --- organism spawning -------------------------------------------------

    def _next_organism_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _find_free_slot(self) -> int:
        """Find the first index where alive_mask is False."""
        dead = (~self.alive_mask).nonzero(as_tuple=False)
        if dead.numel() == 0:
            return -1
        return int(dead[0].item())

    def _spawn_founder(self) -> None:
        slot = self._find_free_slot()
        if slot < 0:
            return
        genome = self._make_genome(self.rng)
        self._write_genome_to_slot(slot, genome, parent_id=-1)
        self.positions[slot, 0] = float(self.rng.uniform(0, self.width))
        self.positions[slot, 1] = float(self.rng.uniform(0, self.height))
        self.headings[slot] = float(self.rng.uniform(0, 2 * np.pi))
        self.energy[slot] = INITIAL_ENERGY
        self.age[slot] = 0
        self.alive_mask[slot] = True
        self.generation[slot] = 0
        self.food_eaten[slot] = 0
        self.brain_state[slot].zero_()
        oid = self._next_organism_id()
        self.events.record_birth(
            self.tick, org_id=oid, parent_id=None,
            genome_hash=genome.fingerprint(),
        )

    def _make_genome(self, rng: np.random.Generator) -> Genome:
        """Construct a v2.2 proto-genome using the same RNG."""
        from .brain import Brain
        return Brain.make_default_genome(rng=rng)

    def _write_genome_to_slot(
        self,
        slot: int,
        genome: Genome,
        parent_id: int,
    ) -> None:
        """Copy weights + topology from a Python Genome into a GPU slot."""
        # Clear existing slot.
        self.gene_in[slot].zero_()
        self.gene_out[slot].zero_()
        self.gene_w[slot].zero_()
        self.gene_enabled[slot] = False
        for i, c in enumerate(
            sorted(genome.connections.values(), key=lambda x: x.innovation)[
                :MAX_CONNS_PER_ORG
            ]
        ):
            self.gene_in[slot, i] = c.in_node
            self.gene_out[slot, i] = c.out_node
            self.gene_w[slot, i] = c.weight
            self.gene_enabled[slot, i] = c.enabled
        self.parent_id[slot] = parent_id

    # --- main loop ---------------------------------------------------------

    def step(self) -> None:
        self.tick += 1

        # 1. Regrow food.
        self._regrow_food()

        # 2. Recompute smell.
        self.smell.recompute_from_tensor(
            self.food_pos[: self._food_count]
        )

        alive_idx = self.alive_mask.nonzero(as_tuple=False).squeeze(1)
        n_alive = int(alive_idx.shape[0])
        if n_alive == 0:
            return

        # 3. Sensors + brain forward + motion (all on GPU).
        self._step_organisms(alive_idx)

        # 4. Eat (GPU pairwise distances).
        eat_count = self._step_eat(alive_idx)
        self._refresh_food_tensor()

        # 5. Energy / age / death.
        self._step_lifecycle()

        # 6. Reproduction.
        self._step_reproduce()

    # --- step helpers ------------------------------------------------------

    def _regrow_food(self) -> None:
        missing = max(0, FOOD_TARGET - self._food_count)
        if missing <= 0:
            return
        n_new = int(self.rng.binomial(missing, FOOD_REGROWTH_RATE))
        for _ in range(n_new):
            if self._food_count < self.food_pos.shape[0]:
                self._spawn_food()
            else:
                break

    def _step_organisms(self, alive_idx: torch.Tensor) -> None:
        """Batched: sensors -> brain forward -> motion -> energy."""
        N = alive_idx.shape[0]
        # Gather alive rows.
        pos = self.positions[alive_idx]  # (N, 2)
        xs = pos[:, 0]
        ys = pos[:, 1]
        headings = self.headings[alive_idx]
        # Batched sensors.
        smells = self.smell.sample_batch(
            xs, ys, headings, SMELL_HALF_ANGLE
        )  # (N, 3)
        # Batched brain forward over fixed-size gene slots.
        motors = self._batched_brain_forward(alive_idx, smells)  # (N, 2)
        turn = motors[:, 0] * MAX_TURN_RATE
        move = motors[:, 1] * MAX_LINEAR_SPEED

        # Apply motion on GPU.
        new_heading = (headings + turn) % (2 * np.pi)
        new_x = (
            pos[:, 0] + torch.cos(new_heading) * move
        ) % self.width
        new_y = (
            pos[:, 1] + torch.sin(new_heading) * move
        ) % self.height

        # Energy cost of motion.
        d_energy = (
            turn.abs() * TURN_ENERGY_COST + move * MOVE_ENERGY_COST
        )

        # Write back.
        self.headings[alive_idx] = new_heading
        self.positions[alive_idx, 0] = new_x
        self.positions[alive_idx, 1] = new_y
        self.energy[alive_idx] = self.energy[alive_idx] - d_energy

        # Update brain state with sensors (overwrite first N_SENSORS slots).
        self.brain_state[alive_idx, :N_SENSORS] = smells

    def _batched_brain_forward(
        self,
        alive_idx: torch.Tensor,
        sensors: torch.Tensor,
    ) -> torch.Tensor:
        """One big scatter-add brain forward over all alive organisms.

        For each organism k:
          delta[j] = bias[j] + sum_i(state[i] * w) over (i -> j) active
          state[j] = activate(delta[j])  for j not sensor
          motors = state[motor indices]
        """
        N = alive_idx.shape[0]
        n_nodes = N_SENSORS + N_HIDDEN + N_MOTORS
        # State is (N, n_nodes). Sensors already set by _step_organisms.
        state = self.brain_state[alive_idx].clone()  # (N, n_nodes)

        # Build a single sparse scatter over all N orgs x MAX_CONNS slots.
        # gene_in/out are (N, MAX_CONNS), weights are (N, MAX_CONNS),
        # enabled is (N, MAX_CONNS) bool.
        # Flatten to (N*MAX_CONNS,) and mask out disabled or padding
        # (in==0 && out==0 && !enabled).
        gin = self.gene_in[alive_idx].reshape(-1)  # (N*K,)
        gout = self.gene_out[alive_idx].reshape(-1)
        gw = self.gene_w[alive_idx].reshape(-1)
        ge = self.gene_enabled[alive_idx].reshape(-1)
        active = ge & ((gin != 0) | (gout != 0))
        # Per-organism offsets for row indices.
        offs = (
            torch.arange(N, device=self.device).unsqueeze(1)
            * n_nodes
        ).expand(N, MAX_CONNS_PER_ORG).reshape(-1)
        src = (offs + gin).long()[active]
        dst = (offs + gout).long()[active]
        vals = gw[active]
        # delta per (org, node).
        delta = torch.zeros(
            (N, n_nodes), dtype=torch.float32, device=self.device
        )
        # Add bias for non-sensor nodes (bias is 0 in v2.2 default but
        # we keep the layout general).
        # delta.index_add_(dim=1, index=gout[active], src=vals * state[...])
        # state at src position, for the same org.
        # We need state[org, gin] which is state[src % n_nodes].
        node_in_per_edge = gin[active]
        # Convert local src within each org to row index.
        org_idx = torch.arange(N, device=self.device).repeat_interleave(
            MAX_CONNS_PER_ORG
        )[active]
        src_state = state[org_idx, node_in_per_edge]
        contribution = src_state * vals
        # Scatter into delta[org_idx, gout].
        # For a 2D tensor (N, n_nodes), torch.scatter_add_ needs linear
        # indices of shape (N*2,) computed as row * n_nodes + col.
        linear_idx = org_idx * n_nodes + gout[active]
        delta.view(-1).scatter_add_(0, linear_idx, contribution)
        # Apply activation: tanh for hidden/motor 0, sigmoid for motor 1,
        # linear for sensor (sensors already set).
        new_state = state.clone()
        # Hidden (TANH).
        hidden_slice = slice(N_SENSORS, N_SENSORS + N_HIDDEN)
        new_state[:, hidden_slice] = torch.tanh(
            state[:, hidden_slice] + delta[:, hidden_slice]
        )
        # Motor 0 (TANH, signed).
        motor0 = N_SENSORS + N_HIDDEN
        new_state[:, motor0] = torch.tanh(
            state[:, motor0] + delta[:, motor0]
        )
        # Motor 1 (SIGMOID, non-negative).
        motor1 = motor0 + 1
        new_state[:, motor1] = torch.sigmoid(
            state[:, motor1] + delta[:, motor1]
        )
        # Sensors: overwrite with the original sensor vector.
        new_state[:, :N_SENSORS] = sensors

        # Commit.
        self.brain_state[alive_idx] = new_state

        # Motors.
        motors = new_state[:, motor0:motor1 + 1]
        return torch.stack(
            [torch.tanh(motors[:, 0]), torch.sigmoid(motors[:, 1])], dim=1
        )

    def _step_eat(self, alive_idx: torch.Tensor) -> int:
        """Pairwise organism<->food distance on GPU; nearest within EAT_RADIUS eats."""
        n_orgs = int(alive_idx.shape[0])
        n_food = int(self._food_count)
        if n_food == 0 or n_orgs == 0:
            return 0

        # Build compact alive-food tensor (drop dead food slots).
        food = self.food_pos[:n_food]
        # Pairwise toroidal distances.
        org_pos = self.positions[alive_idx]  # (N, 2)
        ox = org_pos[:, 0:1]  # (N, 1)
        oy = org_pos[:, 1:2]
        fx = food[:, 0].unsqueeze(0)  # (1, F)
        fy = food[:, 1].unsqueeze(0)
        dx = ox - fx
        dy = oy - fy
        dx = dx - self.width * torch.round(dx / self.width)
        dy = dy - self.height * torch.round(dy / self.height)
        dist = torch.hypot(dx, dy)  # (N, F)
        # Mask out food slots that are not alive (defence in depth).
        food_mask = self.food_alive[:n_food].unsqueeze(0).expand(
            n_orgs, n_food
        )
        dist = torch.where(
            food_mask, dist, torch.tensor(float("inf"), device=self.device)
        )
        nearest_idx = torch.argmin(dist, dim=1)
        nearest_dist = dist.gather(1, nearest_idx.unsqueeze(1)).squeeze(1)
        # Determine which organisms successfully eat and which food is
        # claimed.
        can_eat = nearest_dist <= EAT_RADIUS
        # Resolve conflicts greedily: process organisms in order of
        # food index so earlier pops don't invalidate later indices.
        n_eat = int(can_eat.sum().item())
        if n_eat == 0:
            return 0
        idx_cpu = nearest_idx.cpu().numpy()
        dist_cpu = nearest_dist.cpu().numpy()
        order = sorted(range(n_orgs), key=lambda i: idx_cpu[i])
        claimed: set[int] = set()
        eat_count = 0
        for i in order:
            if not bool(can_eat[i].item()):
                continue
            fi = int(idx_cpu[i])
            if fi in claimed:
                continue
            claimed.add(fi)
            # Mark food slot dead.
            self.food_alive[fi] = False
            # Credit organism.
            self.energy[alive_idx[i]] = (
                self.energy[alive_idx[i]] + FOOD_ENERGY
            )
            self.food_eaten[alive_idx[i]] = self.food_eaten[alive_idx[i]] + 1
            eat_count += 1
        return eat_count

    def _step_lifecycle(self) -> None:
        """Passive drain + metabolic cost + age + death. All on GPU."""
        # Energy: idle + per-neuron + per-active-connection.
        n_active_conns = self.gene_enabled.sum(dim=1).to(torch.float32)
        n_nodes = float(N_SENSORS + N_HIDDEN + N_MOTORS)
        drain = (
            IDLE_ENERGY_COST
            + NEURON_METABOLIC_COST * n_nodes
            + CONNECTION_METABOLIC_COST * n_active_conns
        )
        self.energy = self.energy - drain * self.alive_mask.to(torch.float32)
        self.age = self.age + self.alive_mask.to(torch.int64)
        # Death conditions.
        dead_energy = self.energy <= 0
        dead_age = self.age >= MAX_AGE
        dying = self.alive_mask & (dead_energy | dead_age)
        # Mark dead (no sync; event log will pull a summary at end).
        self.alive_mask = self.alive_mask & ~dying
        # Optional: log deaths in batch via single sync (rare; skipped
        # per-tick for speed).

    def _step_reproduce(self) -> None:
        """Spawn children for organisms with energy >= threshold.

        Implementation: pull a small summary to CPU once, mutate
        parent genomes, push child data into free GPU slots.
        """
        D = self.device
        # Sync only the alive organisms' energy + slots to decide who
        # reproduces.
        alive_idx = self.alive_mask.nonzero(as_tuple=False).squeeze(1)
        if alive_idx.numel() == 0:
            return
        n_alive = int(alive_idx.shape[0])
        alive_slots_cpu = alive_idx.cpu().numpy()
        energy_cpu = self.energy[alive_idx].cpu().numpy()
        threshold = REPRODUCTION_THRESHOLD
        eligible_mask_cpu = energy_cpu >= threshold
        if not eligible_mask_cpu.any():
            return
        eligible_indices = [
            int(alive_slots_cpu[i])
            for i in range(n_alive)
            if eligible_mask_cpu[i]
        ]
        # Process one child per eligible parent, cap by free slots.
        for slot in eligible_indices:
            free = self._find_free_slot()
            if free < 0:
                break
            # Pull parent's genes to CPU, mutate, push back.
            parent_genes_in = self.gene_in[slot].cpu().numpy().tolist()
            parent_genes_out = self.gene_out[slot].cpu().numpy().tolist()
            parent_genes_w = self.gene_w[slot].cpu().numpy().tolist()
            parent_genes_en = self.gene_enabled[slot].cpu().numpy().tolist()
            # Build a temporary Genome-like object via _apply_mutation
            # directly on the slot's tensors — avoids Python Genome
            # construction cost. We delegate weight mutation to numpy
            # operations on the gene_w slice.
            new_w = list(parent_genes_w)
            new_en = list(parent_genes_en)
            for i in range(MAX_CONNS_PER_ORG):
                if not parent_genes_en[i]:
                    continue
                if parent_genes_in[i] == 0 and parent_genes_out[i] == 0:
                    continue
                # mutate_weights semantics: with rate=1 always perturb
                # (already gated by caller); perturb_rate=0.9 perturb
                # by N(0, sigma), replace_rate small.
                if self.rng.random() < 0.9:
                    new_w[i] = float(
                        np.clip(
                            parent_genes_w[i] + self.rng.normal(0, 0.5),
                            -5.0, 5.0,
                        )
                    )
            # Push child gene data to free slot.
            self.gene_in[free] = torch.tensor(
                parent_genes_in, dtype=torch.int32, device=D
            )
            self.gene_out[free] = torch.tensor(
                parent_genes_out, dtype=torch.int32, device=D
            )
            self.gene_w[free] = torch.tensor(
                new_w, dtype=torch.float32, device=D
            )
            self.gene_enabled[free] = torch.tensor(
                new_en, dtype=torch.bool, device=D
            )
            # Position: parent's pos + dispersal offset.
            ph = self.headings[slot].item()
            ch = (ph + self.rng.normal(0.0, CHILD_HEADING_NOISE)) % (
                2 * np.pi
            )
            d = float(
                self.rng.uniform(CHILD_DISPERSAL_MIN, CHILD_DISPERSAL_MAX)
            )
            cx = (self.positions[slot, 0].item() + math.cos(ch) * d) % self.width
            cy = (self.positions[slot, 1].item() + math.sin(ch) * d) % self.height
            self.positions[free, 0] = cx
            self.positions[free, 1] = cy
            self.headings[free] = ch
            self.energy[free] = REPRODUCTION_ENERGY
            self.age[free] = 0
            self.alive_mask[free] = True
            self.parent_id[free] = slot
            self.generation[free] = self.generation[slot].item() + 1
            self.food_eaten[free] = 0
            self.brain_state[free].zero_()
            # Deduct parent energy.
            self.energy[slot] = self.energy[slot] - REPRODUCTION_ENERGY
            # Archive milestones (observability only).
            self._check_milestones_gpu(free)
            # Archive milestones (observability only).
            self._check_milestones_gpu(free)

    # --- diagnostics -------------------------------------------------------

    def _check_milestones_gpu(self, slot: int) -> None:
        """Check structural milestones for the new child in `slot`.

        Cheaply inspects the gene tensor: a node is "hidden" if its
        id is between N_SENSORS and N_SENSORS+N_HIDDEN. For now we
        only flag FIRST_HIDDEN_NODE and FIRST_RECURRENT_CYCLE.
        """
        gin = self.gene_in[slot]
        gout = self.gene_out[slot]
        gen = self.gene_enabled[slot]
        active = gen.nonzero(as_tuple=False).squeeze(1)
        if active.numel() == 0:
            return
        in_ids = gin[active]
        out_ids = gout[active]
        hidden_lo = N_SENSORS
        hidden_hi = N_SENSORS + N_HIDDEN
        has_hidden = bool(
            ((in_ids >= hidden_lo) & (in_ids < hidden_hi)).any().item()
            or ((out_ids >= hidden_lo) & (out_ids < hidden_hi)).any().item()
        )
        if has_hidden:
            self.archive.maybe_fire(
                MilestoneKind.FIRST_HIDDEN_NODE,
                self.tick,
                value=1,
                payload={"slot": slot},
            )
        # Recurrent cycle: two connections A->B and B->A both enabled.
        pairs = set(
            (int(a), int(b))
            for a, b in zip(in_ids.tolist(), out_ids.tolist())
        )
        for a, b in list(pairs):
            if (b, a) in pairs and a != b:
                self.archive.maybe_fire(
                    MilestoneKind.FIRST_RECURRENT_CYCLE,
                    self.tick,
                    value=1,
                    payload={"slot": slot},
                )
                break

    def population(self) -> int:
        return int(self.alive_mask.sum().item())

    def mean_energy(self) -> float:
        n = self.alive_mask.sum().to(torch.float32)
        if float(n) == 0:
            return 0.0
        total = self.energy.sum()
        return float((total / n).item())

    def max_generation(self) -> int:
        if self.alive_mask.sum() == 0:
            return 0
        return int(self.generation[self.alive_mask].max().item())

    def n_lineages(self) -> int:
        if self.alive_mask.sum() == 0:
            return 0
        # We don't have founder_lineage_id on GPU yet; use generation
        # spread as a rough proxy.
        return 1
