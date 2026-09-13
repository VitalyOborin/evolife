"""EvoLife — a digital organism simulation.

V0 goals (this iteration):

- Continuous 2D world with food particles.
- Organisms with energy, age, position, direction, and a small neural brain.
- Reproduction with mutation. No explicit fitness function — fitness
  emerges as the number of surviving descendants.
- NEAT-shaped genome from day one (innovation numbers, structural
  mutations defined), but v0 only mutates weights.
- Headless + Pygame visualisation on CPU with ~500 organisms.
- Only food and starvation in v0. No poison, no day/night cycle.

Hard constraints:

- No LLM, no backpropagation, no RL, no datasets, no human labels.
- Single discrete tick. No global mutable state. Deterministic via seed.
- Metrics recorded from tick 1 even though we do not compute fitness.
"""

__version__ = "0.0.0"
