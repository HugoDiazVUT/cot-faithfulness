import numpy as np


def fraction_changed(rows) -> float:
    return float(np.mean([r["changed"] for r in rows]))


def truncation_curve(rows):
    """Devuelve (fracciones retenidas, acuerdo con la cadena completa)."""
    params = sorted({r["param"] for r in rows})
    agree = [1 - np.mean([r["changed"] for r in rows if r["param"] == p]) for p in params]
    return params, agree


def area_over_curve(rows) -> float:
    """AOC = integral de (1 - acuerdo) sobre la fracción retenida [0,1]. Alto = el CoT importa."""
    x, a = truncation_curve(rows)
    return float(np.trapz(1 - np.array(a), x))


def accuracy(rows) -> float:
    return float(np.mean([r["pred"] == r["gold"] for r in rows]))
