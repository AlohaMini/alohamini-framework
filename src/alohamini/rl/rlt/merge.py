from alohamini.datasets.experience.merge import ScoredEpisodeMerge as ExperienceMerge


class ScoredEpisodeMerge(ExperienceMerge):
    directory = "meta/rlt"
    key = "alohamini_rlt"
    action_validity_fields = ("rlt.action_valid", "rlt.applied_mask")
