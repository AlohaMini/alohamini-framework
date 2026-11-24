"""Match Host feedback to an acknowledged command using its clock and control epoch."""


def feedback_after_command(snapshot, result):
    """Compare Host clocks when available, otherwise require a post-receipt request."""
    safety, identity = snapshot.payload["_safety"], result["command"]
    if (
        safety.get("feedback_valid") is not True
        or safety.get("host_session_id") != identity["host_session_id"]
        or safety.get("control_epoch") != identity["control_epoch"]
    ):
        return False
    timing = snapshot.payload.get("_host_timing", {})
    cycle, completed = result.get("host_cycle"), result.get("completed_monotonic_s")
    if cycle is not None and completed is not None:
        sequence, sampled = (
            timing.get("state_sequence"),
            timing.get("state_sample_started_monotonic_s"),
        )
        return (
            type(sequence) is int
            and isinstance(sampled, (int, float))
            and sequence > cycle
            and sampled > completed
        )
    return snapshot.request_started_s > result.get("observed_s", result.get("time", float("inf")))
