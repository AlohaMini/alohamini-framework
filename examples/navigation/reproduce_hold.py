"""Replay the three review scenarios with synthetic metric grids."""

from collections import Counter
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from alohamini.cerebellum.controller import Controller
from alohamini.cerebellum.obstacles import Cell, Grid
from alohamini.cerebellum.scenarios import synthetic_grid
from alohamini.cerebellum.scheduler import Tick
from alohamini.cerebellum.state import Feedback, Pose, Reference, ZERO, load_config


def step(core, grid):
    t = grid.stamp
    core.feedback(Feedback(t, t, ZERO))
    core.set_map(grid)
    core.set_reference(Reference(t, t, (Pose(1., 0., 0.),)))
    start = perf_counter()
    result = core.tick(Tick(t, 0., 0, .05))
    return result, (perf_counter() - start) * 1000


def run():
    cfg = load_config()
    core = Controller(cfg)
    reference = synthetic_grid(1.)
    cells = dict(reference.cells)
    cells[(10, 15)] = Cell.UNKNOWN
    print("UNKNOWN inside current footprint, then a fully known map")
    for i in range(4):
        grid = Grid(1 + i*.05, reference.source, reference.frame, reference.resolution,
                    reference.origin, cells if i < 3 else reference.cells)
        verdict, _ = step(core, grid)
        print(i, verdict.state, verdict.reason, verdict.rejected)

    core = Controller(cfg)
    print("\nFront block with safe lateral candidates")
    reasons = Counter()
    for i in range(5):
        grid = synthetic_grid(1 + i*.05, rectangles=((.25, -.05, .4, .05),), resolution=.01)
        verdict, _ = step(core, grid)
        reasons[verdict.reason] += 1
        lateral = sum(bool(c.command.y_m_s) for c in verdict.candidates)
        print(i, verdict.state, verdict.reason, len(verdict.candidates), lateral)
    print(dict(reasons))

    print("\nPlanner wall time with and without distant occupied cells")
    for occupied in (False, True):
        core = Controller(cfg)
        for i in range(3):
            rectangles = ((1.6, -1.5, 3.9, 1.4),) if occupied else ()
            grid = synthetic_grid(1 + i*.05, rectangles=rectangles, resolution=.05)
            verdict, elapsed_ms = step(core, grid)
            print(len(grid.occupied), f"{elapsed_ms:.2f} ms", verdict.state, verdict.reason)


if __name__ == "__main__":
    run()
