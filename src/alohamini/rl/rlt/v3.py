"""RLT labels and replay boundaries for the scored-episode publisher."""

from alohamini.datasets.experience.v3 import DISPOSITIONS, ExperienceFormat
from alohamini.rl.rlt.episodes import (
    RUN_METADATA,
    replay_step,
    training_ranges,
    validate_training_episode,
)
from alohamini.rl.rlt.merge import ScoredEpisodeMerge


class RLTFormat(ExperienceFormat):
    prefix = "rlt"
    legacy_untagged = True
    receipt_field_key = "learned_names"
    dispositions = DISPOSITIONS
    merge = ScoredEpisodeMerge
    run_metadata = RUN_METADATA
    sources = {"HOLD": 0, "VLA": 1, "RLT": 2, "HUMAN": 3, "TERMINAL": 4}
    validate = staticmethod(validate_training_episode)
    eligible = staticmethod(replay_step)

    extra_vectors = ("rlt.step_reference",)
    extra_flags = ("rlt.has_reference", "rlt.critical")

    @staticmethod
    def extra_values(step, width):
        reference = step.get("reference")
        return {
            "rlt.critical": bool(step.get("critical")),
            "rlt.has_reference": reference is not None,
            "rlt.step_reference": reference if reference is not None else [0.0] * width,
        }

    @staticmethod
    def mark_boundaries(rows, summary, count):
        for start, stop, reward in training_ranges(summary, count):
            for i in range(stop - 1, start - 1, -1):
                if rows[i]["rlt.action_valid"]:
                    rows[i].update({"next.reward": float(reward), "next.done": True})
                    break
