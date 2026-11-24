"""Read independent, video-reviewed task segments in episode-local action coordinates."""


def training_ranges(summary, step_count):
    """Return (first action, exclusive stop / next observation, local reward)."""
    annotation = summary.get("training_segments")
    if annotation is None:
        if summary.get("phase_labels") == "video-reviewed-placements-v1":
            raise ValueError("Reviewed episode is missing its training segments")
        return [(0, step_count, summary["reward"])] if step_count else []
    if not isinstance(annotation, dict) or annotation.get("version") != 1:
        raise ValueError("Unsupported training segment format")
    segments = annotation.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Training segments must be a list")
    result = []
    previous_stop = 0
    for segment in segments:
        if (
            not isinstance(segment, dict)
            or not isinstance(segment.get("name"), str)
            or not segment["name"].strip()
        ):
            raise ValueError("Training segment is missing a name")
        start, stop, reward = (
            segment.get(k) for k in ("start_frame", "end_frame_exclusive", "reward")
        )
        if (
            type(start) is not int
            or type(stop) is not int
            or not previous_stop <= start < stop <= step_count
        ):
            raise ValueError(f"Invalid or overlapping training interval [{start}, {stop})")
        if type(reward) is not int or reward not in (0, 1):
            raise ValueError("Training segment needs a binary outcome")
        result.append((start, stop, reward))
        previous_stop = stop
    return result
