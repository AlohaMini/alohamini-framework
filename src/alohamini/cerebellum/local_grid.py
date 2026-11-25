"""RGB-depth geometry for an offline, capture-time rolling occupancy map (SI units)."""

from dataclasses import dataclass
import math

import numpy as np

from .obstacles import Cell, Grid
from .state import Pose


PREVIEW = "uncalibrated_rgb_preview"


@dataclass(frozen=True)
class CalibrationSource:
    """Upstream calibration results; nominal/candidate or mismatched geometry stays preview."""
    intrinsics: dict
    extrinsics: dict

    def matches(self, K, T, shape):
        intr, ext = self.intrinsics, self.extrinsics
        try:
            return (str(intr["status"]).startswith("accepted_")
                    and str(ext["status"]).startswith("accepted_")
                    and bool(intr["camera_name"]) and intr["camera_name"] == ext["camera_name"]
                    and ext["parent_frame"] == "base_link"
                    and (intr["image_height"], intr["image_width"]) == shape
                    and intr["image_geometry"] == "rectified"
                    and np.array_equal(intr["rectification"]["K_new"], K)
                    and np.array_equal(np.asarray(intr["camera_matrix"]["data"]).reshape(3,3), K)
                    and not np.any(intr["distortion_coefficients"]["data"])
                    and np.array_equal(ext["T_parent_from_camera_optical"]["matrix"], T)
                    and np.array_equal(ext["T_base_from_camera_optical"], T))
        except (KeyError, TypeError, ValueError):
            return False


@dataclass(frozen=True)
class FrameFit:
    accepted: bool
    reason: str
    scale: float | None = None
    normal: tuple = ()
    floor_inliers: int = 0
    candidates: int = 0
    floor_p95_m: float | None = None


def _fit_floor(points, height, minimum, max_tilt_deg, *, height_band=False, gravity_seed=False):
    """RANSAC in gravity-oriented camera-relative coordinates, followed by plane refit."""
    min_normal_z = math.cos(math.radians(max_tilt_deg))
    count = len(points)
    if count < minimum:
        return FrameFit(False, "insufficient_floor_points", candidates=count)
    samples = points[np.linspace(0, count-1, min(1536, count), dtype=int)]
    triples = np.random.default_rng(42).integers(len(samples), size=(96, 3))
    if height_band:
        z = samples[triples, 2]
        triples = triples[np.all((z >= -1.75*height) & (z <= -.55*height), axis=1)]
    a, b, c = samples[triples].transpose(1, 0, 2)
    normals = np.cross(b-a, c-a)
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-12)
    normals *= np.where(normals[:, 2:3] < 0, -1., 1.)
    valid = (lengths > 1e-6) & (normals[:, 2] > min_normal_z)
    if not np.any(valid):
        return FrameFit(False, "no_ground_plane", candidates=count)
    normals, a = normals[valid], a[valid]
    offsets = np.sum(normals*a, axis=1)
    tolerance = .035
    if gravity_seed:
        plausible = (-height/.6 <= offsets) & (offsets <= -height/1.6)
        normals, offsets = normals[plausible], offsets[plausible]
        if not len(normals):
            return FrameFit(False, "no_metric_floor_hypothesis", candidates=count)
        tolerance = min(tolerance, .04*height)
    residual = np.abs(np.einsum("ij,kj->ik", samples, normals) - offsets)
    best = np.argmax(np.sum(residual < tolerance, axis=0))
    inliers = np.abs(points @ normals[best] - offsets[best]) < tolerance
    n = int(inliers.sum())
    if n < max(minimum, .45*count):
        return FrameFit(False, "insufficient_consensus", floor_inliers=n, candidates=count)
    cloud = points[inliers]
    center = cloud.mean(axis=0)
    spread, axes = np.linalg.eigh(np.cov((cloud-center).T))
    normal = axes[:, 0]
    normal *= 1. if normal[2] >= 0 else -1.
    if spread[1] < .0004 or normal[2] < min_normal_z:
        return FrameFit(False, "degenerate_or_tilted_plane", normal=tuple(normal), floor_inliers=n, candidates=count)
    offset = -float(center @ normal)
    scale = height/offset if offset > 1e-6 else math.inf
    if not .6 <= scale <= 1.6:
        return FrameFit(False, "scale_out_of_range", scale, tuple(normal), n, count)
    errors = np.abs(height + scale*(cloud @ normal))
    return FrameFit(True, "accepted", scale, tuple(normal), n, count, float(np.percentile(errors, 95)))


def aligned_points(camera_relative, translation, fit):
    """Rotate the fitted ground normal onto +z about the camera, then anchor scale."""
    n = np.asarray(fit.normal)
    v = np.cross(n, [0., 0., 1.])
    skew = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    rotation = np.eye(3) + skew + skew @ skew / (1.+n[2])
    return (camera_relative @ rotation.T)*fit.scale + translation


def _valid_geometry(K, T):
    return (K.shape == (3, 3) and T.shape == (4, 4)
            and np.isfinite(K).all() and np.isfinite(T).all()
            and np.allclose(K[2], [0., 0., 1.]) and min(K[0, 0], K[1, 1]) > 0
            and abs(np.linalg.det(K)) >= 1e-9 and np.allclose(T[3], [0., 0., 0., 1.])
            and np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(T[:3, :3]), 1., atol=1e-5))


def ground_ray_distance(u, v, K, T_base_from_optical, lift_height_m=0.):
    """Intersect an undistorted pixel ray with z=0; return ((x,y), range from base origin) in m."""
    K = np.asarray(K, dtype=float)
    T = np.asarray(T_base_from_optical, dtype=float)
    if not all(map(math.isfinite, (u, v, lift_height_m))) or not _valid_geometry(K, T):
        return None
    height = T[2, 3] + lift_height_m
    ray = T[:3, :3] @ np.linalg.solve(K, [u, v, 1.])
    if height <= 0 or ray[2] >= -1e-10 * np.linalg.norm(ray):
        return None
    x, y = T[:2, 3] - ray[:2] * (height / ray[2])
    return (float(x), float(y)), math.hypot(x, y)


class RollingLocalGrid:
    """One odometry epoch. T maps optical xyz (right/down/forward) to base at zero lift."""

    def __init__(self, resolution=.05, size_m=4., memory_s=1.75, min_floor_inliers=64, stride=3,
                 min_occupied_points=3, obstacle_min_z_m=.045):
        if not all(math.isfinite(x) and x > 0 for x in (resolution, size_m, memory_s)):
            raise ValueError("invalid window size, resolution or memory")
        if min_floor_inliers < 3 or type(stride) is not int or stride < 1:
            raise ValueError("invalid plane support or pixel stride")
        if type(min_occupied_points) is not int or min_occupied_points < 1:
            raise ValueError("min_occupied_points must be a positive integer")
        if not .045 <= obstacle_min_z_m < 1.25:
            raise ValueError("obstacle_min_z_m outside height bands")
        self.obstacle_min_z_m = obstacle_min_z_m
        self.min_occupied_points = min_occupied_points
        self.resolution = resolution
        self.width = round(size_m / resolution)
        if self.width < 1:
            raise ValueError("empty grid")
        self.memory_s = memory_s
        self.minimum = min_floor_inliers
        self.stride = stride
        self.reanchor()

    def reanchor(self, stamp=None):
        if stamp is not None and not math.isfinite(stamp):
            raise ValueError("invalid epoch boundary")
        self.epoch = stamp
        self.stamp = None
        self.pose = Pose(0., 0., 0.)
        self.source = PREVIEW
        self.evidence = {}  # odom cell -> (bounded log odds, last observed capture time)

    def _bounds(self, pose):
        return (math.floor(pose.x/self.resolution)-self.width//2,
                math.floor(pose.y/self.resolution)-self.width//2)

    def update(self, depth, K, T_base_from_optical, *, stamp, pose_odom, floor_mask=None,
               self_mask=None, lift_height_m=0., calibration=None, max_range_m: float | None = None):
        """Commit only accepted frames. Depth is optical-axis z in meters, not ray length."""
        if max_range_m is not None and (isinstance(max_range_m, bool)
                or not math.isfinite(max_range_m) or max_range_m <= 0):
            raise ValueError("max_range_m must be finite and positive")
        if not math.isfinite(stamp) or (self.epoch is not None and stamp < self.epoch
                or self.stamp is not None and stamp < self.stamp):
            return FrameFit(False, "stale_frame")
        if not all(map(math.isfinite, (pose_odom.x, pose_odom.y, pose_odom.yaw, lift_height_m))):
            return FrameFit(False, "invalid_pose")
        depth = np.asarray(depth)
        K = np.asarray(K, dtype=float)
        T = np.asarray(T_base_from_optical, dtype=float)
        if depth.ndim != 2 or depth.dtype.kind not in "fiu" or not _valid_geometry(K, T):
            return FrameFit(False, "invalid_camera_geometry")
        if any(np.shape(mask) != depth.shape for mask in (floor_mask,self_mask) if mask is not None):
            return FrameFit(False, "invalid_mask_geometry")
        translation = T[:3,3].copy()
        translation[2] += lift_height_m
        if translation[2] <= 0:
            return FrameFit(False, "invalid_camera_height")
        v, u = np.mgrid[0:depth.shape[0]:self.stride, 0:depth.shape[1]:self.stride]
        z = depth[v,u]
        valid = np.isfinite(z) & (z > 0)
        if self_mask is not None:
            valid &= ~np.asarray(self_mask, dtype=bool)[v,u]
        v,u,z = v[valid],u[valid],z[valid]
        pixels = np.column_stack((u,v,np.ones_like(v)))
        relative = (pixels @ np.linalg.inv(K).T * z[:,None]) @ T[:3,:3].T
        if floor_mask is None:
            height = translation[2]
            ground = ((v >= depth.shape[0]//2) & (relative[:,2] >= -1.75*height)
                      & (relative[:,2] <= -.55*height))
        else:
            ground = np.asarray(floor_mask, bool)[v,u]
        measured = isinstance(calibration, CalibrationSource) and calibration.matches(K, T, depth.shape)
        tilt_limit = 12. if measured else 18.
        # Keep an already-supported broad floor estimate stable across map updates.
        broad = (v >= depth.shape[0]//2) if floor_mask is None else ground
        fit = _fit_floor(relative[broad], translation[2], self.minimum, tilt_limit,
                         height_band=floor_mask is None)
        if floor_mask is None and not fit.accepted:
            # Restrict hypotheses to locally upward-facing patches. The RGB-depth
            # wall still participates in the final occupancy projection below.
            cloud = np.full((*valid.shape, 3), np.nan)
            cloud[valid] = relative
            normals = np.cross(cloud[2:-2,4:] - cloud[2:-2,:-4],
                               cloud[4:,2:-2] - cloud[:-4,2:-2])
            length = np.linalg.norm(normals, axis=2)
            level = np.zeros(valid.shape, dtype=bool)
            level[2:-2,2:-2] = ((length > 1e-6) &
                                 (abs(normals[:,:,2]) >= math.cos(math.radians(tilt_limit))*length))
            supported = ground & level[valid]
            if np.any(supported):
                near = supported & (v >= np.median(v[ground]))
                fit = _fit_floor(relative[near], height, self.minimum, tilt_limit, gravity_seed=True)
                if not fit.accepted:
                    fit = _fit_floor(relative[supported], height, self.minimum, tilt_limit, gravity_seed=True)
        if not fit.accepted:
            return fit

        base = aligned_points(relative, translation, fit)
        if max_range_m is not None:
            base = base[np.hypot(base[:, 0], base[:, 1]) <= max_range_m]
        c,s = math.cos(pose_odom.yaw), math.sin(pose_odom.yaw)
        xy = base[:,:2] @ np.array([[c,s],[-s,c]]) + [pose_odom.x, pose_odom.y]
        i0,j0 = self._bounds(pose_odom)
        cells = np.floor(xy/self.resolution)
        inside = ((cells[:,0] >= i0) & (cells[:,0] < i0+self.width)
                  & (cells[:,1] >= j0) & (cells[:,1] < j0+self.width))
        ij = cells[inside].astype(np.int64)
        h = base[inside,2]
        # Vote once per cell and frame. Sparse above-floor returns do not
        # become either obstacle or free evidence on their own.
        ids = (ij[:, 1]-j0)*self.width + ij[:, 0]-i0
        hits = np.bincount(ids[(h > self.obstacle_min_z_m) & (h <= 1.25)], minlength=self.width**2)
        floor_ids = np.unique(ids[(h >= -.05) & (h <= .045)])
        hit_ids = np.flatnonzero(hits >= self.min_occupied_points)
        free_ids = floor_ids[hits[floor_ids] < self.min_occupied_points]
        self.evidence = {key: entry for key, entry in self.evidence.items()
                         if stamp-entry[1] < self.memory_s and i0 <= key[0] < i0+self.width
                         and j0 <= key[1] < j0+self.width}
        # A supported hit establishes occupancy even after repeated floor
        # sightings; at saturation three observed free frames clear it.
        for observed_ids, increment in ((free_ids, -.6), (hit_ids, 1.6)):
            for cell_id in observed_ids.tolist():
                j, i = divmod(cell_id, self.width)
                key = (i+i0, j+j0)
                prior = self.evidence.get(key, (0., stamp))[0]
                self.evidence[key] = (max(-.8, min(1.5, prior+increment)), stamp)
        source = "measured_metric" if measured else PREVIEW
        self.source = source if self.stamp is None or self.source == source else PREVIEW
        self.stamp, self.pose = stamp, pose_odom
        return fit

    def export(self, now, pose_odom=None):
        """Preserve observation age across queries and rejected frames."""
        if not math.isfinite(now) or (self.stamp is not None and now < self.stamp):
            raise ValueError("export time precedes the latest accepted frame")
        i0,j0 = self._bounds(pose_odom or self.pose)
        cells = {}
        for (i, j), (score, stamp) in self.evidence.items():
            if now-stamp < self.memory_s and i0 <= i < i0+self.width and j0 <= j < j0+self.width:
                cells[i-i0, j-j0] = Cell.OCCUPIED if score > 0 else Cell.FREE
        return Grid(self.stamp if self.stamp is not None else now, self.source, "odom",
                    self.resolution, (i0*self.resolution,j0*self.resolution), cells)
