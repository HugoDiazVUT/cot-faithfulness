"""Carga de datasets a un formato común.

Example:
  id, dataset, question (texto con opciones ya formateadas),
  option_labels (["A","B",...]), gold (índice en option_labels)
"""
import random
from dataclasses import dataclass, asdict

from datasets import load_dataset

import config


@dataclass
class Example:
    id: str
    dataset: str
    question: str
    option_labels: list
    gold: int

    def to_dict(self):
        return asdict(self)


def _fmt(question, labels, texts):
    opts = "\n".join(f"({l}) {t}" for l, t in zip(labels, texts))
    return f"{question}\nOptions:\n{opts}"


def load_commonsenseqa(n, seed=config.SEED):
    ds = load_dataset("tau/commonsense_qa", split="validation")
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        labels = r["choices"]["label"]
        out.append(Example(r["id"], "commonsenseqa",
                           _fmt(r["question"], labels, r["choices"]["text"]),
                           labels, labels.index(r["answerKey"])))
    return out


def load_aqua(n, seed=config.SEED):
    # TODO: comprobar campos de divelab/aqua (question, options ["A)...",..], correct)
    ds = load_dataset("divelab/aqua", split="test")
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    out = []
    for i, r in enumerate(rows[:n]):
        labels = ["A", "B", "C", "D", "E"]
        texts = [o.split(")", 1)[-1].strip() for o in r["options"]]
        out.append(Example(f"aqua-{i}", "aqua", _fmt(r["question"], labels, texts),
                           labels, labels.index(r["correct"])))
    return out


def load_bbh(n, seed=config.SEED, subsets=config.BBH_SUBSETS):
    # TODO: el input de BBH ya incluye las opciones "(A) ..."; target = "(A)".
    # Balancead entre subsets (n // len(subsets) por subset) y detectad las etiquetas con regex.
    raise NotImplementedError


LOADERS = {"commonsenseqa": load_commonsenseqa, "aqua": load_aqua, "bbh": load_bbh}


def load(dataset: str, n: int = config.N_EXAMPLES):
    return LOADERS[dataset](n)
