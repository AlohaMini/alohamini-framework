"""Write standard H.264/MP4 from Host-wire JPEGs using the policy's RGB convention."""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from itertools import chain
from pathlib import Path

import av
import numpy as np

from alohamini.datasets.experience.storage import atomic_json
from alohamini.datasets.image_statistics import ImageStatistics, sample_indices
from alohamini.datasets.images import decode_host_image
from alohamini.datasets.recording_video import encoder_settings
from alohamini.datasets.video import _encode_frames, file_sha256


def snapshot_directories(episode):
    observations = episode / "observations"
    steps = (
        sorted(observations.iterdir(), key=lambda p: int(p.name)) if observations.is_dir() else []
    )
    if any(path.name != str(i) for i, path in enumerate(steps)):
        raise ValueError(f"Non-contiguous observation indices in {episode}")
    terminal = episode / "terminal"
    if (terminal / "state.json").is_file():
        steps.append(terminal)
    return steps


def _camera_paths(directories):
    paths = {}
    for frame, directory in enumerate(directories):
        cameras = json.loads((directory / "cameras.json").read_bytes())
        if len(cameras) != len(set(cameras)) or any(
            not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in cameras
        ):
            raise ValueError(f"Invalid camera list in {directory}")
        if frame and set(cameras) != set(paths):
            raise ValueError(f"Camera set changed in {directory}")
        for i, camera in enumerate(cameras):
            paths.setdefault(camera, []).append(directory / f"{i}.jpg")
    return paths


def _encode_camera(camera, files, cache, fps):
    output = cache / f"{camera}.mp4"
    receipt = cache / f"{camera}.json"
    codec, options = encoder_settings("cpu")
    settings = {
        "codec": codec,
        "pix_fmt": "yuv420p",
        "options": options,
        "input": "host_decoded_rgb_v1",
    }
    if receipt.is_file() and output.is_file():
        saved = json.loads(receipt.read_bytes())
        if (
            saved.get("encoding") == settings
            and saved["frames"] == len(files)
            and saved["fps"] == fps
            and file_sha256(output) == saved["sha256"]
        ):
            return camera, saved

    rate = Fraction(str(fps)).limit_denominator(100_000)
    temporary = output.with_suffix(".tmp.mp4")

    def frames():
        for file in files:
            rgb = decode_host_image(file.read_bytes())
            yield av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")

    source = frames()
    first = next(source)
    _encode_frames(
        chain((first,), source),
        temporary,
        fps,
        (first.height, first.width, 3),
        codec=codec,
        pix_fmt=settings["pix_fmt"],
        options=options,
        strict_shape=True,
    )

    count = 0
    stats = ImageStatistics()
    samples = set(sample_indices(len(files)))
    checks = set(np.linspace(0, len(files) - 1, min(12, len(files)), dtype=int))
    absolute_error = swapped_error = pixels = 0
    with av.open(str(temporary)) as reader:
        stream = reader.streams.video[0]
        stream.codec_context.thread_count = 2
        if stream.average_rate != rate or stream.codec_context.pix_fmt != settings["pix_fmt"]:
            raise ValueError(f"{camera}: unexpected frame rate or pixel format")
        for index, frame in enumerate(reader.decode(stream)):
            if frame.pts is None or frame.pts * frame.time_base * rate != index:
                raise ValueError(f"{camera}: invalid video timestamp at frame {index}")
            if index in samples or index in checks:
                rgb = frame.to_ndarray(format="rgb24")
                if index in samples:
                    stats.update("observation.images." + camera, rgb, source="decoded_video")
                if index in checks:
                    actual = rgb.astype(np.int16)
                    original = decode_host_image(files[index].read_bytes()).astype(np.int16)
                    absolute_error += int(np.abs(actual - original).sum())
                    swapped_error += int(np.abs(actual - original[:, :, ::-1]).sum())
                    pixels += original.size
            count += 1
        if count != len(files) or not pixels:
            raise ValueError(f"{camera}: decoded {count} of {len(files)} frames")
        mae, bgr_mae = absolute_error / pixels, swapped_error / pixels
        if bgr_mae + 1 < mae:
            raise ValueError(
                f"{camera}: decoded video matches a red/blue swap better than the source RGB"
            )
        info = {
            "video.codec": stream.codec_context.name,
            "video.pix_fmt": stream.codec_context.pix_fmt,
            "video.height": stream.height,
            "video.width": stream.width,
            "video.fps": fps,
            "video.channels": 3,
            "video.is_depth_map": False,
            "has_audio": False,
        }
    os.replace(temporary, output)
    saved = dict(
        frames=count,
        fps=fps,
        bytes=output.stat().st_size,
        sha256=file_sha256(output),
        encoding=settings,
        color_check={"frames": len(checks), "rgb_mae": mae, "bgr_mae": bgr_mae},
        info=info,
        histogram=stats.payload(),
    )
    atomic_json(receipt, saved)
    return camera, saved


def prepare_videos(episode, cache, fps, *, workers=4):
    """Prepare all recorded cameras, including the operator's pre-reset terminal observation."""
    episode, cache = Path(episode), Path(cache)
    paths = _camera_paths(snapshot_directories(episode))
    if not paths:
        return {}
    cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as pool:
        futures = [
            pool.submit(_encode_camera, cam, files, cache, fps) for cam, files in paths.items()
        ]
        return dict(future.result() for future in futures)
