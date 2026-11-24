"""Respuesta a partir de log-probabilidades de opción (sin parsear texto)."""
import torch


@torch.no_grad()
def score_options(tok, model, context: str, option_labels: list[str]) -> list[float]:
    """Score(O_i) = (1/N) * sum_t log P(token_t | context, tokens<t), N = nº de tokens del candidato.

    El candidato es f"{label})" (la letra y el paréntesis de cierre).
    TODO (optimización): si todas las etiquetas son un solo token, basta UN forward y leer los logits.
    """
    ctx_ids = tok(context, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    scores = []
    for label in option_labels:
        cand_ids = tok(f"{label})", add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        full = torch.cat([ctx_ids, cand_ids], dim=1)
        logits = model(full).logits[0, ctx_ids.shape[1] - 1:-1]       # predicen los tokens del candidato
        logp = torch.log_softmax(logits.float(), dim=-1)
        tok_lp = logp.gather(1, cand_ids[0].unsqueeze(1)).squeeze(1)
        scores.append(tok_lp.mean().item())                             # normalizado por N
    return scores


def predict(scores: list[float]) -> int:
    return max(range(len(scores)), key=scores.__getitem__)
