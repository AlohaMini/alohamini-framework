"""Forward RGB clearance estimate for the indoor AlohaMini camera mount."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Evidence:
    clear: bool
    reason: str
    floor_fraction: float
    nonfloor_area: int
    contrast_area: int
    left_clear: bool = False
    right_clear: bool = False
    near_blocked: bool = True
    preferred_turn: int = 0


class ClearanceGate:
    """Latch immediately on obstruction. Require fresh, sustained clear evidence."""
    def __init__(self):
        self.last_ns = 0
        self.clear_since_ns = 0
        self.clear_count = 0
        self.open = False

    def update(self, clear, capture_ns):
        if capture_ns <= self.last_ns or capture_ns - self.last_ns > 450_000_000:
            self.clear_count = 0
            self.open = False
        fresh = capture_ns > self.last_ns
        self.last_ns = max(capture_ns, self.last_ns)
        if not clear or not fresh:
            self.clear_count = 0
            self.open = False
        else:
            if not self.clear_count:
                self.clear_since_ns = capture_ns
            self.clear_count += 1
            self.open = self.clear_count >= 3 and capture_ns - self.clear_since_ns >= 500_000_000
        return self.open


def largest_patch(mask):
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if len(stats) > 1 else 0


class FloorDetector:
    """Image-space test, calibrated empirically for a fixed forward RGB view."""
    def __init__(self, model_dir):
        import onnxruntime as ort
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "preprocessor_config.json").read_text())
        labels = json.loads((model_dir / "config.json").read_text())["id2label"]
        self.floor_id = next(int(k) for k, label in labels.items() if label == "floor")
        self.mean = np.asarray(cfg["image_mean"], np.float32)
        self.std = np.asarray(cfg["image_std"], np.float32)
        self.size = cfg["size"]["width"], cfg["size"]["height"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_dir / "onnx/model.onnx"),
                                            sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        # Normalized trapezoid widens toward the base. No metric range is inferred.
        yy, xx = np.mgrid[:240, :320]
        y = (yy + 0.5) / 240
        self.corridor = (y >= 0.55) & (abs((xx + 0.5) / 320 - 0.5) <= 0.13 + (y - 0.55) * 0.30)

        x = (xx + 0.5) / 320 - 0.5
        offset = 0.12 + 0.34 * (1.0 - y)
        band = (y >= 0.55) & (y <= 0.95)
        width = 0.06 + 0.06 * (y - 0.55)
        self.left = band & (abs(x + offset) <= width)
        self.right = band & (abs(x - offset) <= width)
        self.near = (y >= 0.90) & (abs(x) <= 0.14)

    def inspect(self, jpeg):
        if not jpeg:
            return Evidence(False, "invalid_camera_geometry", 0.0, 0, 0)
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (480, 640):
            return Evidence(False, "invalid_camera_geometry", 0.0, 0, 0)
        small = cv2.resize(bgr, (320,240), interpolation=cv2.INTER_AREA)
        luma = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)[self.corridor]
        if np.mean((luma < 12) | (luma > 248)) > 0.70 or np.ptp(luma) < 5:
            return Evidence(False, "poor_visibility", 0.0, 0, 0)
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.size)
        tensor = ((rgb.astype(np.float32) / 255.0 - self.mean) / self.std).transpose(2,0,1)[None]
        logits = self.session.run(None, {self.input_name:tensor})[0][0]
        if not np.isfinite(logits).all():
            return Evidence(False, "invalid_segmentation", 0.0, 0, 0)
        weights = np.exp(logits - logits.max(axis=0))
        prob = cv2.resize(weights[self.floor_id] / weights.sum(axis=0), (320,240))
        # A small raised object can occupy only a few output pixels of SegFormer.
        # Compare each patch to its local floor appearance, independently of hue.
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        diff = lab.astype(np.float32) - cv2.medianBlur(lab, 25).astype(np.float32)
        diff[:,:,0] *= 0.55
        contrast = np.sum(diff * diff, axis=2) > 19**2
        nonfloor = prob < 0.55
        clear, reason, fraction, area, patch = region_evidence(self.corridor, nonfloor, contrast)
        if clear:
            return Evidence(True, reason, fraction, area, patch)
        left = region_evidence(self.left, nonfloor, contrast)[0]
        right = region_evidence(self.right, nonfloor, contrast)[0]
        near_blocked = not region_evidence(self.near, nonfloor, contrast)[0]
        # Image left corresponds to positive ROS yaw. Favor the side farther
        # from the visible obstruction when the model offers no turn.
        occupied = self.corridor & (nonfloor | contrast)
        left_count = np.count_nonzero(occupied[:, :160])
        right_count = np.count_nonzero(occupied[:, 160:])
        preferred = int(np.sign(right_count - left_count))
        return Evidence(False, reason, fraction, area, patch, left, right, near_blocked, preferred)


def region_evidence(region, nonfloor, contrast):
    occupied = nonfloor & region
    fraction = 1.0 - np.count_nonzero(occupied) / np.count_nonzero(region)
    nonfloor_area = largest_patch(occupied)
    binary = (contrast & region).astype(np.uint8)
    _, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # Short thin edges can have substantial area without a solid interior.
    # Keep large thin regions as well as smaller regions with a compact core.
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    core_labels = np.unique(components[distance >= 2.5])
    areas = stats[:, cv2.CC_STAT_AREA]
    contrast_area = int(areas[1:].max()) if len(areas) > 1 else 0
    compact_area = int(areas[core_labels].max()) if core_labels.size else 0
    if nonfloor_area >= 18 or fraction < 0.97:
        reason = "nonfloor_or_uncertain"
    elif compact_area >= 24 or contrast_area >= 64:
        reason = "local_obstruction"
    else:
        reason = "clear"
    return reason == "clear", reason, float(fraction), nonfloor_area, contrast_area


class GuardedPrediction:
    def __init__(self, model_dir=None):
        self.detector = FloorDetector(model_dir or os.environ.get(
            "ALOHAMINI_FLOOR_MODEL", Path.home() / "workspace/jjy/models/segformer-b0-ade"))
        self.gate = ClearanceGate()
        self.turn = 0
        self.encounter_ns = None
        self.worker = ThreadPoolExecutor(max_workers=1)

    def evaluate(self, predict, ws, seq, jpeg, instruction, stamp, stop_translation):
        # Run both models on the same capture so their latencies overlap.
        future = self.worker.submit(predict, ws, seq, jpeg, instruction)
        evidence = self.detector.inspect(jpeg)
        previously_open = self.gate.open
        self.gate.update(evidence.clear, stamp)
        if previously_open and not self.gate.open:
            stop_translation()
        return future.result(), evidence

    def constrain(self, cmd, evidence, max_yaw=0.5):
        if self.gate.open:
            self.turn = 0
            self.encounter_ns = None
            return "open"

        requested_x, requested_yaw = cmd.linear.x, cmd.angular.z
        cmd.linear.x = cmd.linear.y = cmd.angular.z = 0.0
        if evidence.clear:
            return "confirm_clear"
        if not (evidence.left_clear or evidence.right_clear):
            return "hold"
        if self.encounter_ns is None:
            self.encounter_ns = self.gate.last_ns
        # Commit to one visibly open side until clearance is confirmed.
        if not self.turn:
            if evidence.left_clear != evidence.right_clear:
                self.turn = 1 if evidence.left_clear else -1
            elif abs(requested_yaw) >= 0.08:
                self.turn = 1 if requested_yaw > 0 else -1
            else:
                self.turn = evidence.preferred_turn
        if not self.turn or not (evidence.left_clear if self.turn > 0 else evidence.right_clear):
            self.turn = 0
            return "hold"

        cmd.angular.z = self.turn * min(max_yaw, max(0.32, 1.25 * abs(requested_yaw)))
        # Bound cumulative approach during one obstruction, including a static
        # scene. Continue turning after this short arc until the view clears.
        if (not evidence.near_blocked and cmd.angular.z and
                self.gate.last_ns - self.encounter_ns < 1_000_000_000):
            cmd.linear.x = min(0.045, 0.25 * max(0.0, requested_x))
        return "arc" if cmd.linear.x > 0.0 else "turn"

    def close(self):
        self.worker.shutdown(wait=True)
