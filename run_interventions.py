"""Aplica intervenciones al CoT cacheado y puntúa por log-probs de opción (sin regenerar).
python run_interventions.py --model qwen3-1.7b --dataset commonsenseqa --interventions truncation filler swap
Salida: results/<modelo>/<dataset>/<intervencion>.jsonl
"""
import argparse
import random

from tqdm import tqdm

import config
from interventions import REGISTRY
from models import load_model
from prompts import scoring_context
from scoring import predict, score_options
from utils import read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(config.MODELS))
    ap.add_argument("--dataset", required=True, choices=config.DATASETS)
    ap.add_argument("--interventions", nargs="+", default=["truncation"], choices=list(REGISTRY))
    args = ap.parse_args()

    tok, model = load_model(args.model)
    records = read_jsonl(config.CACHE_DIR / args.model / f"{args.dataset}.jsonl")
    rng = random.Random(config.SEED)
    ctx = {"tok": tok, "records": records}

    def run(rec, cot):
        lp = score_options(tok, model, scoring_context(tok, rec, cot), rec["option_labels"])
        return lp, predict(lp)

    # baseline: CoT completo
    base = {}
    for rec in tqdm(records, desc="baseline"):
        base[rec["id"]] = run(rec, rec["cot_text"])

    for name in args.interventions:
        iv = REGISTRY[name]()
        rows = []
        for param in iv.params():
            for rec in tqdm(records, desc=f"{name} {param}"):
                new_cot = iv.apply(rec, param, rng, ctx)
                lp, pred = run(rec, new_cot)
                rows.append({"id": rec["id"], "intervention": name, "param": param,
                             "logprobs": lp, "pred": pred, "gold": rec["gold"],
                             "pred_full": base[rec["id"]][1],
                             "changed": pred != base[rec["id"]][1]})
        write_jsonl(config.RESULTS_DIR / args.model / args.dataset / f"{name}.jsonl", rows)


if __name__ == "__main__":
    main()
