from .base import Intervention


class Swap(Intervention):
    """HUGO. Sustituye el CoT por el de OTRA pregunta (mismo dataset y modelo)."""
    name = "swap"

    def apply(self, rec, param, rng, ctx):
        pool = [r for r in ctx["records"] if r["id"] != rec["id"]]
        return rng.choice(pool)["cot_text"]
