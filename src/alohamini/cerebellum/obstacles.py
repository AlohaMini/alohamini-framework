"""Metric local occupancy and convex footprint checks."""

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
import math
from types import MappingProxyType


class Cell(IntEnum):
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 100


METRIC_SOURCES = {"synthetic", "manual_metric", "measured_metric"}


@dataclass(frozen=True)
class Grid:
    stamp: float
    source: str
    frame: str
    resolution: float
    origin: tuple[float, float]
    cells: dict

    def __post_init__(self):
        if (not all(map(math.isfinite, (self.stamp, self.resolution, *self.origin)))
                or self.resolution <= 0 or len(self.origin) != 2):
            raise ValueError("invalid grid geometry")
        clean = {}
        for key, value in self.cells.items():
            if len(key) != 2 or not all(type(i) is int for i in key):
                raise ValueError("grid indexes must be integer pairs")
            clean[key] = Cell(value)
        object.__setattr__(self, "cells", MappingProxyType(clean))
        object.__setattr__(self, "occupied", tuple(key for key, value in clean.items() if value == Cell.OCCUPIED))

        # Store disjoint horizontal runs. A query touches only nearby rows/runs.
        known = defaultdict(list)
        occupied = defaultdict(list)
        for (i, j), value in clean.items():
            if value != Cell.UNKNOWN:
                known[j].append(i)
            if value == Cell.OCCUPIED:
                occupied[j].append(i)
        # Split the exposed cells into horizontal and vertical runs. A long wall
        # then contributes a small number of boxes even when the query is close.
        horizontal = defaultdict(list)
        vertical = defaultdict(list)
        for i, j in self.occupied:
            if clean.get((i, j-1)) != Cell.OCCUPIED or clean.get((i, j+1)) != Cell.OCCUPIED:
                horizontal[j].append(i)
            elif clean.get((i-1, j)) != Cell.OCCUPIED or clean.get((i+1, j)) != Cell.OCCUPIED:
                vertical[i].append(j)
        object.__setattr__(self, "_surface_rows", _row_runs(horizontal))
        object.__setattr__(self, "_surface_cols", _row_runs(vertical))
        object.__setattr__(self, "_known", _row_runs(known))
        object.__setattr__(self, "_occupied", _row_runs(occupied))
        object.__setattr__(self, "_occupied_bounds", (
            min(occupied), max(occupied),
            min(i for i, _ in self.occupied), max(i for i, _ in self.occupied)
        ) if occupied else None)

    def unknown_cells(self, bounds):
        i0, i1, j0, j1 = bounds
        for j in range(j0, j1 + 1):
            cursor = i0
            for lo, hi in self.runs(self._known, j, i0, i1):
                for i in range(cursor, lo):
                    yield i, j
                cursor = hi + 1
            for i in range(cursor, i1 + 1):
                yield i, j

    @staticmethod
    def runs(rows, j, left, right):
        if j in rows:
            ends, runs = rows[j]
            for k in range(bisect_left(ends, left), len(runs)):
                lo, hi = runs[k]
                if lo > right:
                    break
                yield max(left, lo), min(right, hi)

    def surface_runs(self, bounds):
        i0, i1, j0, j1 = bounds
        limits = self._occupied_bounds
        if not limits:
            return
        j0, j1 = max(j0, limits[0]), min(j1, limits[1])
        i0, i1 = max(i0, limits[2]), min(i1, limits[3])
        if j0 > j1 or i0 > i1:
            return
        for j in range(j0, j1+1):
            for left, right in self.runs(self._surface_rows, j, i0, i1):
                yield left, j, right, j
        for i in range(i0, i1+1):
            for bottom, top in self.runs(self._surface_cols, i, j0, j1):
                yield i, bottom, i, top

    def occupied_within_radius(self, x, y, radius):
        """Closed disk against occupied cell areas, including edges and corners."""
        ox, oy = self.origin
        s = self.resolution
        r = radius + 1e-12  # Treat a rounded cell edge at the threshold as contact.
        left, right = math.floor((x-r-ox)/s), math.floor((x+r-ox)/s)
        for j in range(math.floor((y-r-oy)/s), math.floor((y+r-oy)/s)+1):
            bottom = oy + j*s
            dy = max(bottom-y, y-(bottom+s), 0.)
            for lo, hi in self.runs(self._occupied, j, left, right):
                dx = max(ox+lo*s-x, x-(ox+(hi+1)*s), 0.)
                if math.hypot(dx, dy) <= r:
                    return True
        return False

    def box(self, i, j):
        x, y = self.origin[0] + i * self.resolution, self.origin[1] + j * self.resolution
        return x, y, x + self.resolution, y + self.resolution


def _row_runs(rows):
    result = {}
    for j, indexes in rows.items():
        runs = []
        for i in sorted(indexes):
            if runs and i == runs[-1][1] + 1:
                runs[-1] = runs[-1][0], i
            else:
                runs.append((i, i))
        result[j] = tuple(hi for _, hi in runs), tuple(runs)
    return result


def footprint(vertices):
    poly = tuple(tuple(p) for p in vertices)
    if len(poly) < 3 or not all(len(p) == 2 and all(map(math.isfinite, p)) for p in poly):
        raise ValueError("expected a finite convex footprint polygon")
    winding = None
    for a, b in zip(poly, poly[1:] + poly[:1]):
        if a == b:
            raise ValueError("duplicate footprint vertices")
        for p in poly:
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if abs(cross) > 1e-12:
                if winding is not None and (cross > 0) != winding:
                    raise ValueError("footprint must be convex and ordered")
                winding = cross > 0
    if winding is None:
        raise ValueError("degenerate footprint")
    return poly


def _support(poly):
    axes = [(1., 0.), (0., 1.)]
    for a, b in zip(poly, poly[1:] + poly[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        axes.append((-dy / length, dx / length))
    return [(u, v, min(x*u + y*v for x, y in poly), max(x*u + y*v for x, y in poly)) for u, v in axes]


def _lower_distance(support, box):
    """Conservative Euclidean-distance lower bound from separating axes."""
    x0, y0, x1, y1 = box
    gap = 0.
    for u, v, low, high in support:
        cx0, cx1 = (x0, x1) if u >= 0 else (x1, x0)
        cy0, cy1 = (y0, y1) if v >= 0 else (y1, y0)
        gap = max(gap, low - (u*cx1 + v*cy1), u*cx0 + v*cy0 - high)
    return gap


def _point_segment(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    fraction = max(0., min(1., ((p[0]-a[0])*dx + (p[1]-a[1])*dy) / (dx*dx + dy*dy)))
    return math.hypot(p[0] - a[0] - fraction*dx, p[1] - a[1] - fraction*dy)


def _distance(poly, box, lower):
    if lower <= 1e-12:
        return 0.
    x0, y0, x1, y1 = box
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    return min(
        min(_point_segment(p, a, b) for p in corners for a, b in zip(poly, poly[1:] + poly[:1])),
        min(math.hypot(max(x0-x, 0., x-x1), max(y0-y, 0., y-y1)) for x, y in poly),
    )


def world_polygon(poly, pose):
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return tuple((pose.x + c*x - s*y, pose.y + s*x + c*y) for x, y in poly)


def _bounds(grid, support, radius):
    ox, oy = grid.origin
    res = grid.resolution
    return (math.floor((support[0][2] - radius - ox) / res - 1e-10),
            math.floor((support[0][3] + radius - ox) / res + 1e-10),
            math.floor((support[1][2] - radius - oy) / res - 1e-10),
            math.floor((support[1][3] + radius - oy) / res + 1e-10))


def initial_unknown(grid, poly, pose):
    world = world_polygon(poly, pose)
    support = _support(world)
    return frozenset(key for key in grid.unknown_cells(_bounds(grid, support, 0.))
                     if _lower_distance(support, grid.box(*key)) <= 0.)


def inspect_polygon(grid, world, margin, clearance_cap, allowed_unknown, *, unknown_policy="strict"):
    support = _support(world)
    bounds = _bounds(grid, support, margin)
    nearest = math.inf
    # Search far enough for both collision and clearance, including when margin > cap.
    radius = max(margin, clearance_cap)
    # If the polygon is entirely inside a solid, no exposed cell need be nearby.
    x, y = world[0]
    key = (math.floor((x-grid.origin[0])/grid.resolution),
           math.floor((y-grid.origin[1])/grid.resolution))
    if grid.cells.get(key) == Cell.OCCUPIED:
        return "occupied_footprint", 0.
    for i, j, last_i, last_j in grid.surface_runs(_bounds(grid, support, radius)):
        box = grid.box(i, j)[:2] + grid.box(last_i, last_j)[2:]
        lower = _lower_distance(support, box)
        if lower <= radius and lower < nearest:
            nearest = min(nearest, _distance(world, box, lower))
            if nearest <= margin:
                return "occupied_footprint", 0.
    if unknown_policy == "soft_cost":
        return None, min(nearest, clearance_cap) if grid.occupied else None
    if unknown_policy != "strict":
        raise ValueError("invalid unknown policy")
    for key in grid.unknown_cells(bounds):
        if key in allowed_unknown:
            continue
        box = grid.box(*key)
        lower = _lower_distance(support, box)
        if lower <= margin and _distance(world, box, lower) <= margin:
            return "unknown_footprint", None
    return None, min(nearest, clearance_cap) if grid.occupied else None


def convex_hull(vertices):
    points = sorted(set(vertices))

    def half(sequence):
        hull = []
        for x, y in sequence:
            while len(hull) >= 2:
                ax, ay = hull[-2]
                bx, by = hull[-1]
                if (bx-ax)*(y-ay) > (by-ay)*(x-ax):
                    break
                hull.pop()
            hull.append((x, y))
        return hull[:-1]

    return tuple(half(points) + half(reversed(points)))


def translation_hull(poly, start, end):
    a = world_polygon(poly, start)
    dx, dy = end.x - start.x, end.y - start.y
    return convex_hull(a + tuple((x + dx, y + dy) for x, y in a))


def translation_clear(grid, poly, start, end, margin, allowed_unknown, *, unknown_policy="strict"):
    hull = translation_hull(poly, start, end)
    return inspect_polygon(grid,hull,margin,0.,allowed_unknown,unknown_policy=unknown_policy)[0]


def unknown_area(grid, world, margin):
    """Unknown area in a convex envelope, using horizontal grid runs.

    Inflate its supporting planes by the margin and the half-cell extent.
    This conservative raster estimate is used only for ranking.
    """
    support = _support(world)
    left,right,bottom,top = _bounds(grid,support,margin)
    res=grid.resolution
    ox,oy=grid.origin
    area=0
    planes=[(u,v,lo-margin-res*(abs(u)+abs(v))/2,
             hi+margin+res*(abs(u)+abs(v))/2) for u,v,lo,hi in support]
    for j in range(bottom,top+1):
        y=oy+(j+.5)*res
        low,high=left,right
        for u,v,a,b in planes:
            if abs(u)<1e-12:
                if not a-1e-12<=v*y<=b+1e-12:
                    high=low-1
                    break
                continue
            x0,x1=(a-v*y)/u,(b-v*y)/u
            low=max(low,math.ceil((min(x0,x1)-ox)/res-.5-1e-10))
            high=min(high,math.floor((max(x0,x1)-ox)/res-.5+1e-10))
            if low>high:
                break
        if low<=high:
            known=sum(b-a+1 for a,b in grid.runs(grid._known,j,low,high))
            area+=high-low+1-known
    return area*res*res


def occupied_contacts(grid, poly, pose, margin):
    """Occupied cell keys intersecting the expanded footprint (exact narrow phase)."""
    world = world_polygon(poly, pose)
    support = _support(world)
    i0, i1, j0, j1 = _bounds(grid, support, margin)
    hits = set()
    for j in range(j0, j1 + 1):
        for left, right in grid.runs(grid._occupied, j, i0, i1):
            for i in range(left, right + 1):
                box = grid.box(i, j)
                lower = _lower_distance(support, box)
                if lower <= margin and _distance(world, box, lower) <= margin:
                    hits.add((i, j))
    return hits


def inspect_pose(grid, poly, pose, margin, clearance_cap, *, allowed_unknown=frozenset(), unknown_policy="strict"):
    return inspect_polygon(grid, world_polygon(poly, pose), margin, clearance_cap, allowed_unknown, unknown_policy=unknown_policy)
