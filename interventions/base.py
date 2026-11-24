class Intervention:
    """Cada intervención recibe un registro del caché y devuelve el CoT modificado (str).

    - name:   nombre usado en los ficheros de resultados
    - params: lista de valores del parámetro a barrer (p. ej. fracciones de truncado)
    - apply:  rec, param, rng, ctx -> nuevo CoT.  ctx = {"tok": tokenizer|None, "records": [todos los rec]}
    """
    name = "base"

    def params(self):
        return [None]

    def apply(self, rec: dict, param, rng, ctx: dict) -> str:
        raise NotImplementedError
