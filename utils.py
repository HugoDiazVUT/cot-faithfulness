import json
import re
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# Definición ÚNICA de "paso" (acordadla y no la cambiéis a mitad):
# salto de línea, o fin de frase seguido de mayúscula/número.
_STEP_SPLIT = re.compile(r"\n+|(?<=[.!?])\s+(?=[A-Z0-9])")


def split_steps(cot: str) -> list[str]:
    return [p.strip() for p in _STEP_SPLIT.split(cot) if p and p.strip()]


def join_steps(steps: list[str]) -> str:
    return "\n".join(steps)
