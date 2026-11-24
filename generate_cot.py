"""Genera el CoT UNA vez y lo cachea. Uso:
python generate_cot.py --model qwen3-1.7b --dataset commonsenseqa
"""
import argparse

import torch
from tqdm import tqdm

import config
from data import load
from models import load_model
from prompts import chat_prompt, strip_final_answer
from utils import split_steps, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(config.MODELS))
    ap.add_argument("--dataset", required=True, choices=config.DATASETS)
    ap.add_argument("--n", type=int, default=config.N_EXAMPLES)
    args = ap.parse_args()

    tok, model = load_model(args.model)
    examples = load(args.dataset, args.n)
    rows = []
    for ex in tqdm(examples):
        prompt = chat_prompt(tok, ex.question)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False)
        gen = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        cot = strip_final_answer(gen)
        rows.append({**ex.to_dict(), "model": args.model, "generation": gen,
                     "cot_text": cot, "cot_steps": split_steps(cot)})
    write_jsonl(config.CACHE_DIR / args.model / f"{args.dataset}.jsonl", rows)


if __name__ == "__main__":
    main()
