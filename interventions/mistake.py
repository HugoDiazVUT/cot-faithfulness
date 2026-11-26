import re
from utils import join_steps
from .base import Intervention

class MistakeInjection(Intervention):
    """PABLO. Introduce un error en UN paso intermedio y deja el resto igual."""
    
    name = "mistake"

    def apply(self, rec, param, rng, ctx):
        steps = list(rec["cot_steps"])
        if len(steps) < 3:
            return join_steps(steps)  # Sin pasos suficientes para modificar un intermedio
        
        # Elegir un paso intermedio aleatorio (nunca el primero ni el último)
        i = rng.randrange(1, len(steps) - 1)
        
        # Guardar el índice del paso modificado para análisis posterior (requisito del proyecto)
        rec["_mistake_idx"] = i
        
        # Corromper el paso según el dataset o el contenido
        dataset_name = rec.get("dataset", "unknown")
        steps[i] = self.corrupt(steps[i], dataset_name, rng)
        
        return join_steps(steps)

    def corrupt(self, step: str, dataset: str, rng) -> str:
        # 1. Si es AQuA o contiene números, alteramos un valor numérico u operador
        nums = list(re.finditer(r"\d+", step))
        if nums and ("aqua" in dataset.lower() or rng.random() > 0.5):
            m = rng.choice(nums)
            val = int(m.group())
            new_val = val + rng.choice([-3, -2, -1, 1, 2, 3]) # Cambio sutil
            if new_val < 0: 
                new_val = abs(new_val) + 1
            return step[:m.start()] + str(new_val) + step[m.end():]
        
        # 2. Para CommonsenseQA / BBH (basado en negación o inversión lógica)
        # Reemplazos comunes de polaridad para introducir un error sutil de razonamiento
        replacements = [
            (r"\bcan be\b", "cannot be"),
            (r"\bcannot be\b", "can be"),
            (r"\bis\b", "is not"),
            (r"\bis not\b", "is"),
            (r"\bwill\b", "will not"),
            (r"\bwill not\b", "will"),
            (r"\bshould\b", "should not"),
            (r"\bshould not\b", "should")
        ]
        
        for pattern, repl in replacements:
            if re.search(pattern, step):
                return re.sub(pattern, repl, step, count=1)
                
        # Fallback si no encuentra patrones claros: añadir prefijo de contradicción
        return "Incorrectly, " + step