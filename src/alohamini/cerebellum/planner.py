"""DWB-style short horizon sampling, including a bounded braking rollout."""

from collections import Counter
from dataclasses import dataclass
import math

from alohamini.schema import BodyVelocity

from .obstacles import Cell, Grid, footprint, initial_unknown, occupied_contacts, inspect_pose, translation_clear
from .obstacles import convex_hull, translation_hull, unknown_area, world_polygon
from .state import integrate, Pose, ZERO


def interpolate(a, b, fraction):
    return Pose(a.x + fraction * (b.x-a.x), a.y + fraction * (b.y-a.y),
                a.yaw + fraction * math.remainder(b.yaw-a.yaw, math.tau))


def components(v):
    return v.x_m_s, v.y_m_s, v.yaw_rad_s


def toward_zero(value, step):
    return math.copysign(max(0., abs(value) - step), value)


def brake(velocity, dt, limits):
    x, y, w = components(velocity)
    speed = math.hypot(x, y)
    scale = max(0., 1. - limits["brake_m_s2"] * dt / speed) if speed else 0.
    return BodyVelocity(x * scale, y * scale, toward_zero(w, limits["yaw_brake_rad_s2"] * dt))


def path_metrics(pose, path):
    """Closest segment distance, progress along path and reference heading."""
    best = math.inf
    progress, heading, passed = 0., path[-1].yaw, 0.
    for a, b in zip(path, path[1:]):
        dx, dy = b.x - a.x, b.y - a.y
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        f = min(1., max(0., ((pose.x-a.x)*dx + (pose.y-a.y)*dy) / length**2))
        distance = math.hypot(pose.x - a.x - f*dx, pose.y - a.y - f*dy)
        if distance < best:
            best = distance
            progress = passed + f*length
            delta = math.remainder(b.yaw - a.yaw, math.tau)
            heading = a.yaw + f*delta
        passed += length
    if not math.isfinite(best):
        best = math.hypot(pose.x-path[-1].x, pose.y-path[-1].y)
    return best, progress, abs(math.remainder(pose.yaw-heading, math.tau))


@dataclass(frozen=True)
class Candidate:
    command: BodyVelocity
    path: tuple
    clearance_m: float | None
    side: int
    terms: dict

    @property
    def score(self):
        return sum(self.terms.values())


class Rejections(Counter):
    def __init__(self):
        super().__init__()
        self.details = []

    def add(self, command, reason):
        self[reason] += 1
        self.details.append((*components(command), reason))


class Planner:
    def __init__(self, cfg, kinematics):
        self.p = cfg["planner"]
        self.policy = self.p.get("unknown_policy","strict")
        self.unknown_weight = self.p.get("unknown_weight",.35)
        self.limits = cfg["limits"]
        self.poly = footprint(cfg["footprint_m"])
        self.radius = max(math.hypot(x, y) for x, y in self.poly)
        self.kinematics = kinematics
        self.wheel_max = min(3000, self.limits["wheel_ticks_s"]) * math.tau / 4096
        self.side = 0
        self.side_since = 0.
        self.detour = ()
        self.metric_goal = None
        self.front = max(x for x, _ in self.poly)
        self.dock_offset = self.front + .75 * cfg["control"]["arrival_distance_m"]
        self.terminal_component = None

    def within_limits(self, v):
        x, y, w = components(v)
        l = self.limits
        return (-l["reverse_m_s"] <= x <= l["forward_m_s"] and abs(y) <= l["lateral_m_s"]
                and math.hypot(x, y) <= l["linear_m_s"] + 1e-10 and abs(w) <= l["yaw_rad_s"]
                and max(map(abs, self.kinematics.body_to_wheels(v))) <= self.wheel_max + 1e-10)

    def _segment(self, grid, start, velocity, dt, *, extra_m=0., allowed_unknown=frozenset()):
        p = self.p
        distance = dt * (math.hypot(*components(velocity)[:2]) + self.radius * abs(velocity.yaw_rad_s))
        steps = max(1, math.ceil(distance / p["sweep_resolution_m"]))
        padding = distance / (2 * steps) + extra_m
        clearance = []
        end = start
        for i in range(steps + 1):
            end = integrate(start, velocity, dt * i / steps)
            reason, gap = inspect_pose(grid, self.poly, end, p["margin_m"] + padding, p["clearance_cap_m"],
                                       allowed_unknown=allowed_unknown, unknown_policy=self.policy)
            if reason:
                return end, gap, reason
            if gap is not None:
                clearance.append(max(0., gap - padding))
        return end, min(clearance, default=None), None

    def approach(self, current, target, dt):
        """Interpolate within the convex body and wheel limits and the acceleration disk."""
        a, b = components(current), components(target)
        delta = tuple(y-x for x, y in zip(a, b))
        length = math.hypot(*delta[:2])
        scale = min(1., self.limits["accel_m_s2"]*dt/length) if length else 1.
        dw = self.limits["yaw_accel_rad_s2"]*dt
        result = BodyVelocity(a[0]+scale*delta[0], a[1]+scale*delta[1],
                              a[2]+max(-dw, min(dw, delta[2])))
        if self.within_limits(result):
            return result
        # A shared fraction also keeps the wheel speeds on the feasible line segment.
        scale = min(scale, dw/abs(delta[2])) if delta[2] else scale
        return BodyVelocity(*(x+scale*d for x, d in zip(a, delta)))

    def rollout(self, grid, start, velocity, *, horizon=None, initial_error_m=0.,
                target=None, first_dt=None, allowed_unknown=None, exposure_path=None):
        p = self.p
        if allowed_unknown is None:
            allowed_unknown = initial_unknown(grid, self.poly, start)
        horizon = p["horizon_s"] if horizon is None else horizon
        path = [start]
        if horizon == 0. and velocity == ZERO:
            reason, gap = inspect_pose(grid, self.poly, start, p["margin_m"] + initial_error_m,
                                       p["clearance_cap_m"], allowed_unknown=allowed_unknown,unknown_policy=self.policy)
            return tuple(path), gap, reason
        clearance = None
        pose, speed = start, velocity
        elapsed = 0.
        while elapsed < horizon - 1e-10 or speed != ZERO:
            tracking = elapsed < horizon - 1e-10
            dt = min(first_dt or p["step_s"], horizon) if not elapsed and tracking else p["step_s"]
            if tracking:
                dt = min(dt, horizon - elapsed)
                if elapsed and target is not None:
                    speed = self.approach(speed, target, dt)
            pose, gap, reason = self._segment(grid, pose, speed, dt, extra_m=initial_error_m,
                                              allowed_unknown=allowed_unknown)
            if reason:
                return tuple(path), gap, reason
            if gap is not None:
                clearance = gap if clearance is None else min(clearance, gap)
            if exposure_path is not None:
                exposure_path.append(pose)
            if tracking:
                path.append(pose)
            else:
                speed = brake(speed, dt, self.limits)
            elapsed += dt
        return tuple(path), clearance, None

    def terminal_reference(self, pose, reference, grid, allowed):
        """Clip against occupied components touching the terminal vehicle envelope."""
        end = reference[-1]
        key = (math.floor((end.x - grid.origin[0]) / grid.resolution),
               math.floor((end.y - grid.origin[1]) / grid.resolution))
        if self.policy=="strict" and grid.cells.get(key, Cell.UNKNOWN) == Cell.UNKNOWN:
            return reference, False
        if len(reference) == 1:
            reference = (pose, end)

        geometry = (end.x, end.y, pose.yaw)
        cache = self.terminal_component
        if cache is None or cache[0] is not grid or cache[1] != geometry:
            seeds = occupied_contacts(grid, self.poly, Pose(*geometry), self.p["margin_m"])
            if not seeds:
                return reference, False
            # Opposing banks of a narrow slot are a transit blockage. Do not
            # reinterpret the two walls as a single terminal target.
            if grid.cells.get(key) == Cell.FREE:
                before = next((a for a in reversed(reference[:-1]) if (a.x, a.y) != (end.x, end.y)), pose)
                dx, dy = end.x-before.x, end.y-before.y
                half_cell = grid.resolution * (abs(dx)+abs(dy)) / 2
                lateral = [dx*(grid.origin[1]+(j+.5)*grid.resolution-end.y)
                           - dy*(grid.origin[0]+(i+.5)*grid.resolution-end.x) for i,j in seeds]
                if min(lateral) < -half_cell and max(lateral) > half_cell and min(map(abs, lateral)) > half_cell:
                    return reference, False
            extent = math.ceil((self.radius + self.p["margin_m"] + .6) / grid.resolution)
            component = dict.fromkeys(seeds, Cell.OCCUPIED)
            frontier = list(seeds)
            while frontier:
                i, j = frontier.pop()
                for nbr in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                    if ((nbr[0]-key[0])**2 + (nbr[1]-key[1])**2 <= extent**2
                            and nbr not in component and grid.cells.get(nbr) == Cell.OCCUPIED):
                        component[nbr] = Cell.OCCUPIED
                        frontier.append(nbr)
            mask = Grid(grid.stamp, grid.source, grid.frame, grid.resolution, grid.origin, component)
            self.terminal_component = grid, geometry, mask
        else:
            mask = cache[2]

        def contact(layer, point):
            return inspect_pose(layer, self.poly, Pose(point.x, point.y, pose.yaw),
                                self.p["margin_m"], 0., allowed_unknown=allowed,unknown_policy=self.policy)[0]

        # On the component mask only OCCUPIED represents contact. The real map
        # decides whether the first pose clear of that surface can be used.
        for i in range(len(reference) - 2, -1, -1):
            a, b = reference[i:i+2]
            steps = max(1, math.ceil(math.hypot(b.x-a.x, b.y-a.y) / (grid.resolution / 2)))
            for k in reversed(range(steps)):
                lo, hi = k / steps, (k+1) / steps
                point = interpolate(a, b, lo)
                if contact(mask, point) == "occupied_footprint":
                    continue
                # Refine the boundary from its free side, to < 0.1 mm for a 5 cm grid.
                for _ in range(9):
                    mid = (lo + hi) / 2
                    if contact(mask, interpolate(a, b, mid)) == "occupied_footprint":
                        hi = mid
                    else:
                        lo = mid
                stop = interpolate(a, b, lo)
                if contact(grid, stop):
                    return reference, False
                return (*reference[:i+1], stop), True
        return reference, False

    def target_bypass(self, pose, target, grid, allowed):
        """Choose a collision-checked lateral leg and a return leg towards the target."""
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        end = Pose(target.x - c*self.dock_offset, target.y - s*self.dock_offset, pose.yaw)
        lateral = -s*(end.x-pose.x) + c*(end.y-pose.y)
        sign = 1 if lateral >= 0 else -1
        step = self.p["detour_step_m"]
        margin = self.p["margin_m"] + self.p["corner_tolerance_m"]
        def clear(a, b):
            return not translation_clear(grid, self.poly, a, b, margin, allowed, unknown_policy=self.policy)
        if clear(pose, end):
            return (end,)
        partial = None
        advance = min(self.p["guide_lookahead_m"]*.65, (end.x-pose.x)*c + (end.y-pose.y)*s)
        if advance >= .4:
            straight = Pose(pose.x+c*advance, pose.y+s*advance, pose.yaw)
            if clear(pose, straight):
                partial = (straight,)
        for side in (sign, -sign):
            for i in range(1, 1 + round(self.p["detour_width_m"]/step)):
                offset = side*i*step
                turn = Pose(pose.x - s*offset, pose.y + c*offset, pose.yaw)
                if not clear(pose, turn):
                    continue
                if clear(turn, end):
                    return (turn, end)
                # A chair arm or wall can occlude the eventual docking lane. Require
                # two fully checked legs with forward progress before taking its flank.
                if partial is None and advance >= .4 and side == sign and abs(offset) <= abs(lateral)+step:
                    flank = Pose(turn.x+c*advance, turn.y+s*advance, pose.yaw)
                    if clear(turn, flank) and math.hypot(target.x-flank.x,target.y-flank.y) < math.hypot(target.x-pose.x,target.y-pose.y):
                        partial = (turn, flank)
        return partial or ()

    def guide(self, pose, reference, grid, allowed, target_odom=None):
        p = self.p
        reference, clipped = self.terminal_reference(pose, reference, grid, allowed)
        if p.get("metric_target_detour"):
            target = target_odom
            keep = (target is not None and self.metric_goal is not None
                    and math.hypot(target.x-self.metric_goal.x, target.y-self.metric_goal.y) < .18)
            if target is not None and clipped:
                component = self.terminal_component[2]
                ahead = ((target.x-reference[-1].x)*math.cos(pose.yaw)
                         + (target.y-reference[-1].y)*math.sin(pose.yaw))
                separate = not component.occupied_within_radius(target.x, target.y, .15)
                keep = separate and (keep or ahead > self.front + p["margin_m"])
            if self.metric_goal is not None and not keep:
                self.detour = ()
                self.metric_goal = None
            if keep:
                self.metric_goal = target
                self.detour = self.target_bypass(pose, target, grid, allowed)
                if not self.detour:
                    return None, 0., "occupied_footprint"
                goal = self.detour[0]
                remaining = sum(math.hypot(b.x-a.x,b.y-a.y) for a,b in zip((pose,*self.detour),self.detour))
                return goal, remaining, None
        cross_track, along, _ = path_metrics(pose, reference)
        lengths = [math.hypot(b.x-a.x, b.y-a.y) for a, b in zip(reference, reference[1:])]
        total = sum(lengths)
        remaining = max(0., total - along) + cross_track
        if clipped and remaining < .001:
            self.detour = ()
            return None, remaining, "terminal_standoff"

        def point_at(arc):
            for a, b, length in zip(reference, reference[1:], lengths):
                if length > 1e-9 and arc < length:
                    return interpolate(a, b, arc / length)
                arc -= length
            return reference[-1]

        def blocked(a, b, extra=0.):
            return translation_clear(grid, self.poly, Pose(a.x, a.y, pose.yaw), b,
                                     p["margin_m"] + extra, allowed,unknown_policy=self.policy)

        arc = min(total, along + p["guide_lookahead_m"])
        goal = point_at(arc)
        # The cached destination is the selected rejoin on the *reference*, independent
        # of the nominal lookahead. Revalidate the complete suffix on every new map.
        if self.detour and path_metrics(self.detour[-1], reference)[0] <= p["corner_tolerance_m"]:
            while self.detour and math.hypot(pose.x-self.detour[0].x, pose.y-self.detour[0].y) < p["corner_tolerance_m"]:
                self.detour = self.detour[1:]
            if len(self.detour) == 1 and not blocked(pose, goal):
                self.detour = ()
                return goal, remaining, None
            if self.detour:
                points = (pose, *self.detour)
                if all(not blocked(a, b) for a, b in zip(points, points[1:])):
                    return self.detour[0], remaining, None
        self.detour = ()
        failure = blocked(pose, goal)
        if not failure:
            return goal, remaining, None

        signs = (self.side, -self.side) if self.side else (1, -1)
        initial_arc=arc
        while True:
            # No offset can make the final leg safe if its destination is obstructed.
            if not blocked(goal, goal):
                length = math.hypot(goal.x-pose.x, goal.y-pose.y)
                if length > 1e-8:
                    nx, ny = -(goal.y-pose.y)/length, (goal.x-pose.x)/length
                    routes={}
                    for step in range(1, 1 + int(p["detour_width_m"] / p["detour_step_m"])):
                        for sign in signs:
                            if sign in routes:
                                continue
                            shift = step * p["detour_step_m"] * sign
                            corner = Pose(pose.x+nx*shift, pose.y+ny*shift, pose.yaw)
                            bypass = Pose(goal.x+nx*shift, goal.y+ny*shift, pose.yaw)
                            if (not blocked(pose, corner) and not blocked(corner, bypass, p["corner_tolerance_m"])
                                    and not blocked(bypass, goal)):
                                if self.policy=="strict":
                                    self.detour = (corner, bypass, goal)
                                    return corner, remaining, None
                                route=(corner,bypass,goal)
                                exposure=sum(unknown_area(grid,translation_hull(self.poly,a,b),p["margin_m"])
                                             for a,b in zip((pose,corner,bypass),route))
                                routes[sign]=(exposure,step,route)
                        if len(routes)==2:
                            break
                    if routes:
                        self.detour=min(routes.values(),key=lambda item:item[:2])[2]
                        return self.detour[0],remaining,None
            if arc >= total:
                if self.policy=="soft_cost":
                    # LightNav may terminate its whole reference inside a red cluster.
                    # Take a verified two-leg path to its clear flank, then replan.
                    # Require substantial forward advancement as well as a clear turn.
                    flank=self.flank(pose,point_at(initial_arc),reference[-1],grid,blocked,signs)
                    if flank:
                        self.detour=flank
                        return flank[0],remaining,None
                return None, remaining, failure
            arc = min(total, arc + p["detour_step_m"])
            goal = point_at(arc)

    def flank(self,pose,goal,terminal,grid,blocked,signs):
        dx,dy=goal.x-pose.x,goal.y-pose.y
        length=math.hypot(dx,dy)
        if length<self.p["guide_lookahead_m"]*.5:
            return None
        choices=[]
        for sign in signs:
            for step in range(1,1+int(self.p["detour_width_m"]/self.p["detour_step_m"])):
                shift=sign*step*self.p["detour_step_m"]
                sx,sy=-dy/length*shift,dx/length*shift
                corner=Pose(pose.x+sx,pose.y+sy,pose.yaw)
                end=Pose(goal.x+sx,goal.y+sy,pose.yaw)
                if math.hypot(terminal.x-end.x,terminal.y-end.y)>=math.hypot(terminal.x-pose.x,terminal.y-pose.y):
                    continue
                if blocked(pose,corner) or blocked(corner,end,self.p["corner_tolerance_m"]):
                    continue
                area=sum(unknown_area(grid,translation_hull(self.poly,a,b),self.p["margin_m"])
                         for a,b in ((pose,corner),(corner,end)))
                choices.append((area,step,(corner,end)))
                break
        return min(choices,key=lambda item:item[:2])[2] if choices else None

    def velocity_targets(self, pose, goal, remaining):
        dx, dy = goal.x - pose.x, goal.y - pose.y
        distance = math.hypot(dx, dy)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        length = max(distance, 1e-12)
        ux, uy = (c*dx + s*dy) / length, (-s*dx + c*dy) / length
        # Both the end of the reference arc and the next detour corner need a taper.
        stopping_distance = min(distance, remaining)
        limits = self.limits
        # A constant-speed forecast occupies space before its braking tail starts.
        # Reserve the horizon and discrete step padding before the terminal boundary.
        horizon = self.p["horizon_s"] + 2 * self.p["step_s"]
        safe_speed = 2 * remaining / (horizon + math.sqrt(
            horizon*horizon + 2*remaining/limits["brake_m_s2"]))
        speed = min(limits["linear_m_s"], self.p["approach_gain_s_inv"] * stopping_distance,
                    safe_speed, math.sqrt(2 * limits["brake_m_s2"] * stopping_distance))
        yaw_error = math.remainder(goal.yaw - pose.yaw, math.tau)
        yaw = max(-limits["yaw_rad_s"], min(limits["yaw_rad_s"], yaw_error))

        targets = []
        planar = [(speed*ux, speed*uy), (speed, 0.), (-speed, 0.), (0., speed), (0., -speed)]
        if abs(yaw) > 1e-6:
            planar.append((0., 0.))
        for x, y in planar:
            wheels = self.kinematics.body_to_wheels(BodyVelocity(x, y, yaw))
            factor = max(1., x / limits["forward_m_s"], -x / limits["reverse_m_s"],
                         abs(y) / limits["lateral_m_s"], max(map(abs, wheels)) / self.wheel_max)
            candidate = BodyVelocity(x / factor, y / factor, yaw / factor)
            if candidate not in targets:
                targets.append(candidate)
        return targets, distance, yaw_error

    def project_command(self, command, measured, previous, dt):
        """Nearest point in the two equal XY acceleration disks and yaw intervals."""
        r = self.limits["accel_m_s2"] * dt
        dw = self.limits["yaw_accel_rad_s2"] * dt
        lo = max(measured.yaw_rad_s, previous.yaw_rad_s) - dw
        hi = min(measured.yaw_rad_s, previous.yaw_rad_s) + dw
        ax, ay = measured.x_m_s, measured.y_m_s
        bx, by = previous.x_m_s, previous.y_m_s
        dx, dy = bx-ax, by-ay
        separation = math.hypot(dx, dy)
        if lo > hi or separation > 2*r + 1e-12:
            return None
        x, y = command.x_m_s, command.y_m_s
        # approach() already placed command inside the measured disk. Project onto
        # the other disk; if this leaves the first disk, the nearest point is a vertex.
        distance = math.hypot(x-bx, y-by)
        if distance > r:
            x, y = bx + (x-bx)*r/distance, by + (y-by)*r/distance
            if math.hypot(x-ax, y-ay) > r + 1e-12 and separation > 0.:
                height = math.sqrt(max(0., r*r - separation*separation/4))
                side = 1 if dx*(command.y_m_s-ay) - dy*(command.x_m_s-ax) >= 0 else -1
                x = (ax+bx)/2 - side * dy/separation * height
                y = (ay+by)/2 + side * dx/separation * height
        projected = BodyVelocity(x, y, min(hi, max(lo, command.yaw_rad_s)))
        if self.within_limits(projected):
            return projected
        # Both centers may be legal yet a lens vertex can exceed a wheel bound.
        # Their midpoint remains in the acceleration lens; interpolate toward it.
        middle = BodyVelocity((ax+bx)/2, (ay+by)/2, (previous.yaw_rad_s+measured.yaw_rad_s)/2)
        if not self.within_limits(middle):
            return None
        a, b = components(middle), components(projected)
        low, high = 0., 1.
        for _ in range(20):
            f = (low+high)/2
            v = BodyVelocity(*(u+f*(w-u) for u, w in zip(a,b)))
            if self.within_limits(v):
                low, middle = f, v
            else:
                high = f
        return middle

    def choose(self, now, dt, pose, measured, previous, reference, grid, *,
               start_padding_m=0., previous_path=None, target_odom=None):
        p, limits = self.p, self.limits
        allowed = initial_unknown(grid, self.poly, pose)
        goal, remaining, failure = self.guide(pose, reference, grid, allowed, target_odom)
        rejected = Rejections()
        if failure == "terminal_standoff":
            return None, rejected, failure, ()
        if failure:
            rejected.add(ZERO, failure)
            why = "insufficient_observed_swept_space" if failure == "unknown_footprint" else "no_collision_free_motion"
            return None, rejected, why, ()
        targets, distance, yaw_error = self.velocity_targets(pose, goal, remaining)
        scoring_path = (pose, goal) if self.detour else reference
        options = []
        start_area=(unknown_area(grid,world_polygon(self.poly,pose),p["margin_m"])
                    if self.policy=="soft_cost" else 0.)
        for desired in targets:
            command = self.approach(measured, desired, dt)
            projected = self.project_command(command, measured, previous, dt)
            if projected is None:
                rejected.add(command, "acceleration_or_wheel_limit")
                continue
            command = projected
            if command == ZERO:
                continue
            delta = tuple(a-b for a, b in zip(components(command), components(measured)))
            error = dt*(math.hypot(*delta[:2])+self.radius*abs(delta[2]))
            exposure_path=[pose] if self.policy=="soft_cost" else None
            path, gap, failure = self.rollout(grid, pose, command, target=desired, first_dt=dt,
                                              initial_error_m=start_padding_m+error, allowed_unknown=allowed, exposure_path=exposure_path)
            if failure:
                rejected.add(command, failure)
                continue
            end = path[-1]
            progress = distance - math.hypot(goal.x - end.x, goal.y - end.y)
            heading = abs(yaw_error) - abs(math.remainder(goal.yaw - end.yaw, math.tau))
            # Clearance is used to rank feasible motions; the stationary comparison
            # uses geometric advancement only, independent of sweep interpolation padding.
            lateral = command.y_m_s
            side = (1 if lateral > 0 else -1) if abs(lateral) >= p["side_deadband_m_s"] else 0
            terms = {
                "progress": progress * p["progress_weight"],
                "heading": heading * p["heading_weight"],
                "path_error": -path_metrics(end, scoring_path)[0] * p["path_weight"],
                "clearance": (gap or 0.) * p["clearance_weight"],
                "smoothness": -p["smoothness_weight"] * sum(
                    (new-old)**2 for new, old in zip(components(command), components(previous))),
                "previous_path_error": -p["continuity_weight"] * path_metrics(end, previous_path)[0]
                if previous_path else 0.,
            }
            if exposure_path is not None:
                envelope=convex_hull(point for s in exposure_path for point in world_polygon(self.poly,s))
                added=max(0.,unknown_area(grid,envelope,p["margin_m"])-start_area)
                terms["unknown"]=-self.unknown_weight*added
            options.append(Candidate(command, path, gap, side, terms))
        improving = [v for v in options if v.terms["progress"] + v.terms["heading"] > 1e-9]
        if not improving:
            reason = "no_progress"
            if not options:
                if rejected["unknown_footprint"]:
                    reason = "insufficient_observed_swept_space"
                elif rejected["occupied_footprint"]:
                    reason = "no_collision_free_motion"
                else:
                    reason = "no_feasible_motion"
            return None, rejected, reason, tuple(options)
        winner = max(improving, key=lambda v: v.score)
        reason = "detour_subgoal" if self.detour else "best_score"
        if self.side and winner.side == -self.side:
            incumbents = [v for v in improving if v.side == self.side]
            if incumbents:
                best = max(incumbents, key=lambda v: v.score)
                if now - self.side_since < p["side_hold_s"] or winner.score - best.score < p["switch_advantage"]:
                    winner, reason = best, "direction_hysteresis"
        if winner.side != self.side:
            self.side, self.side_since = winner.side, now
        return winner, rejected, reason, tuple(options)
