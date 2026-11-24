import config
from utils import join_steps
from .base import Intervention


class Truncation(Intervention):
    """HUGO. Conserva los primeros k% de pasos (0% = sin CoT, 100% = CoT completo)."""
    name = "truncation"

    def params(self):
        return config.TRUNCATION_LEVELS

    def apply(self, rec, param, rng, ctx):
        steps = rec["cot_steps"]
        k = round(len(steps) * param)
        return join_steps(steps[:k])
