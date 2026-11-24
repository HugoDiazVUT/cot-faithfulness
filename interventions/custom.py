from .base import Intervention


class CustomIntervention(Intervention):
    """PABLO. Intervención propia. Ideas: invertir el orden de los pasos, borrar solo el paso final,
    eliminar los números del razonamiento, sustituir el CoT por la respuesta correcta sin razonar..."""
    name = "custom"

    def apply(self, rec, param, rng, ctx):
        raise NotImplementedError
