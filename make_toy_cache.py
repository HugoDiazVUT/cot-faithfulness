"""Caché falso (sin GPU) para desarrollar intervenciones. python make_toy_cache.py"""
import random

import config
from utils import split_steps, write_jsonl

rng = random.Random(0)
rows = []
for i in range(20):
    cot = "\n".join(f"Step {j}: x = {rng.randint(1, 20)} * 2 gives {rng.randint(1, 40)}." for j in range(1, 6))
    rows.append({"id": f"toy-{i}", "dataset": "toy", "model": "toy", "question": "Toy question?",
                 "option_labels": list("ABCDE"), "gold": rng.randrange(5),
                 "generation": cot, "cot_text": cot, "cot_steps": split_steps(cot)})
write_jsonl(config.CACHE_DIR / "toy" / "toy.jsonl", rows)
print("ok:", len(rows))
