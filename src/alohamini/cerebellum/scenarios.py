"""Clearly labeled synthetic metric maps for offline validation."""

from .obstacles import Cell, Grid


def synthetic_grid(stamp, *, rectangles=(), unknown=(), resolution=.10):
    origin = (-1., -1.5)

    def overlaps(a, b):
        return a[0] < b[2] - 1e-9 and a[2] > b[0] + 1e-9 and a[1] < b[3] - 1e-9 and a[3] > b[1] + 1e-9

    cells = {}
    for i in range(round(5 / resolution)):
        for j in range(round(3 / resolution)):
            x, y = origin[0] + i * resolution, origin[1] + j * resolution
            box = x, y, x + resolution, y + resolution
            if any(overlaps(box, region) for region in rectangles):
                cell = Cell.OCCUPIED
            elif any(overlaps(box, region) for region in unknown):
                cell = Cell.UNKNOWN
            else:
                cell = Cell.FREE
            cells[i, j] = cell
    return Grid(stamp, "synthetic", "odom", resolution, origin, cells)
