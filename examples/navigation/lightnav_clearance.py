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
        nonfloor = (prob < 0.55) & self.corridor
        fraction = 1.0 - nonfloor.sum() / self.corridor.sum()
        nonfloor_area = largest_patch(nonfloor)

        # A small raised object can occupy only a few output pixels of SegFormer.
        # Compare each patch to its local floor appearance, independently of hue.
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        diff = lab.astype(np.float32) - cv2.medianBlur(lab, 25).astype(np.float32)
        diff[:,:,0] *= 0.55
        contrast = (np.sum(diff * diff, axis=2) > 19**2) & self.corridor
        contrast_area = largest_patch(contrast)
        if nonfloor_area >= 18 or fraction < 0.97:
            reason = "nonfloor_or_uncertain"
        elif contrast_area >= 24:
            reason = "local_obstruction"
        else:
            reason = "clear"
        return Evidence(reason == "clear", reason, float(fraction), nonfloor_area, contrast_area)

class GuardedPrediction:
    def __init__(self, model_dir=None):
        self.detector = FloorDetector(model_dir or os.environ.get(
            "ALOHAMINI_FLOOR_MODEL", Path.home() / "workspace/jjy/models/segformer-b0-ade"))
        self.gate = ClearanceGate()
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

    def constrain(self, cmd, evidence):
        if not self.gate.open:
            cmd.linear.x = cmd.linear.y = 0.0
            if evidence.reason in ("invalid_camera_geometry", "poor_visibility", "invalid_segmentation"):
                cmd.angular.z = 0.0

    def close(self):
        self.worker.shutdown(wait=True)
