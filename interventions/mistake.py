import re
from utils import join_steps
from .base import Intervention


class MistakeInjection(Intervention):
    """PABLO. Introduce un error en UN paso intermedio y deja el resto igual.

    Idea: AQuA -> reglas (cambiar un número/operador del paso). Commonsense/BBH -> LLM
    ("reescribe este paso con un error sutil") o negación. Registrad el paso modificado en
    rec["_mistake_idx"] para análisis posterior.
    """
    name = "mistake"

    def apply(self, rec, param, rng, ctx):
        steps = list(rec["cot_steps"])
        if len(steps) < 3:
            return join_steps(steps)           # sin pasos intermedios
        i = rng.randrange(1, len(steps) - 1)   # nunca el primero ni el último
        steps[i] = self.corrupt(steps[i], rec, rng)
        return join_steps(steps)

    def corrupt(self, step: str, rec: dict, rng) -> str:
        # TODO: versión mínima por reglas: cambiar el primer número que aparezca
        nums = list(re.finditer(r"\d+", step))
        if nums:
            m = rng.choice(nums)
            return step[:m.start()] + str(int(m.group()) + rng.randint(1, 9)) + step[m.end():]
        return step   # TODO: fallback con LLM/negación
