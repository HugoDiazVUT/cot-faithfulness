from .base import Intervention

class StepReorder(Intervention):
    """Baraja el orden de los pasos intermedios del CoT para romper el flujo causal."""
    name = "reorder"

    def apply(self, rec, param, rng, ctx):
        steps = list(rec.get("cot_steps", []))
        
        # Si hay 2 pasos o menos, no hay pasos intermedios que reordenar
        if len(steps) <= 2:
            return "\n".join(steps)

        # Opción A: Barajar todos los pasos excepto el último (la conclusión)
        # Opción B (aquí): Mantener el primero y el último fijos, y barajar los del medio
        first_step = steps[0]
        middle_steps = steps[1:-1]
        last_step = steps[-1]

        # Asegurarnos de que el orden realmente cambie si hay al menos 2 pasos intermedios
        if len(middle_steps) >= 2:
            original_middle = list(middle_steps)
            while middle_steps == original_middle:
                rng.shuffle(middle_steps)
        else:
            # Si solo hay 3 pasos en total, barajamos todos menos el último
            prefix = steps[:-1]
            rng.shuffle(prefix)
            return "\n".join(prefix + [last_step])

        reordered_steps = [first_step] + middle_steps + [last_step]
        return "\n".join(reordered_steps)