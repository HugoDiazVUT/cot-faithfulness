# Does the reasoning actually cause the answer? — Qwen3 CoT faithfulness

## Pipeline
`generate_cot.py` (genera y cachea CoT) → `run_interventions.py` (modifica CoT cacheado + puntúa por log-probs) → `metrics.py` / `plots.py`

Caché: `cache/<modelo>/<dataset>.jsonl`, un registro por ejemplo:
`id, dataset, model, question, option_labels, gold, generation, cot_text, cot_steps`
Resultados: `results/<modelo>/<dataset>/<intervencion>.jsonl`:
`id, intervention, param, logprobs, pred, gold, pred_full, changed`

## Reparto
| Hugo | Pablo |
|---|---|
| `data.py` (BBH incl.), `prompts.py`, `generate_cot.py`, `scoring.py` | `interventions/mistake.py`, `paraphrase.py`, `custom.py` |
| `interventions/truncation.py`, `filler.py`, `swap.py` | `bias.py` (few-shot sesgado + hint de usuario + detector de menciones) |
| `plots.py` (curvas de truncado, AOC) | `metrics.py` (tabla de fracción de respuestas cambiadas, gap de infidelidad) |
| Ejecutar 1.7B y 4B | Ejecutar 8B |

Juntos: poster, tabla final, preguntas orales.

## Desarrollo sin GPU
```
pip install -r requirements.txt
python make_toy_cache.py
pytest tests
```

## Convenciones
- Una sola definición de "paso": `utils.split_steps` (no cambiar a mitad del proyecto).
- Misma precisión (`config.DTYPE`), thinking de Qwen3 desactivado, semilla fija (`config.SEED`).
- Un intervención = una clase en `interventions/` registrada en `interventions/__init__.py`.
