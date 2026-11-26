"""Genera el CoT UNA vez y lo cachea. Uso:
python generate_cot.py --model qwen3-1.7b --dataset commonsenseqa --n 300 --batch_size 16
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
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    tok, model = load_model(args.model)
    tok.padding_side = "left"            # necesario para generar por lotes
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    examples = load(args.dataset, args.n)
    out_path = config.CACHE_DIR / args.model / f"{args.dataset}.jsonl"
    rows = []
    for i in tqdm(range(0, len(examples), args.batch_size)):
        batch = examples[i:i + args.batch_size]
        prompts = [chat_prompt(tok, ex.question) for ex in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False,
                stop_strings=["Therefore, the answer is"], tokenizer=tok,
            )
        for ex, seq in zip(batch, out):
            gen = tok.decode(seq[enc.input_ids.shape[1]:], skip_special_tokens=True)
            cot = strip_final_answer(gen)
            rows.append({**ex.to_dict(), "model": args.model, "generation": gen,
                         "cot_text": cot, "cot_steps": split_steps(cot)})
        write_jsonl(out_path, rows)      # guarda tras cada lote