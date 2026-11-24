from .base import Intervention

FILLER = " ."   # un token por repetición en el tokenizer de Qwen (comprobadlo)


class Filler(Intervention):
    """HUGO. Sustituye el CoT por tokens sin significado, mismo nº de tokens (teacher forcing: se inserta, no se pide)."""
    name = "filler"

    def apply(self, rec, param, rng, ctx):
        tok = ctx.get("tok")
        if tok is None:   # modo toy (sin tokenizer): palabras
            n = len(rec["cot_text"].split())
        else:
            n = len(tok(rec["cot_text"], add_special_tokens=False).input_ids)
        text = FILLER * n
        # TODO: verificar len(tok(text).input_ids) == n; si no, ajustar el filler
        return text
