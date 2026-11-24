from .base import Intervention


class Paraphrase(Intervention):
    """PABLO. Parafrasea el CoT preservando el contenido (cambia la forma, no la lógica).

    TODO: usar un LLM (Qwen3 o API) con temperatura baja; cachear las paráfrasis en
    cache/<modelo>/paraphrases_<dataset>.jsonl para no regenerar. Revisar a mano ~20 ejemplos.
    """
    name = "paraphrase"

    def apply(self, rec, param, rng, ctx):
        raise NotImplementedError
