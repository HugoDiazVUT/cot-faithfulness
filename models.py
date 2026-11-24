import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


def load_model(model_key: str):
    name = config.MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=getattr(torch, config.DTYPE), device_map="auto"
    )
    model.eval()
    return tok, model
