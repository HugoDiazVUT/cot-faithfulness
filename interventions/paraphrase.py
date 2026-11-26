import os
import json
import random
from .base import Intervention

class Paraphrase(Intervention):
    """PABLO. Parafrasea cualquier CoT preservando el contenido usando un LLM real
    (si está configurado) o un generador sintáctico avanzado de fallback.
    """
    name = "paraphrase"

    def apply(self, rec, param, rng, ctx):
        model_name = ctx.get("model_name", "qwen3")
        dataset_name = rec.get("dataset", "unknown")
        
        # 1. Configurar rutas de caché por modelo y dataset
        cache_dir = os.path.join("cache", model_name)
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"paraphrases_{dataset_name}.jsonl")
        
        rec_id = rec.get("id", str(hash(rec.get("cot_text", ""))))

        # 2. Comprobar si ya está en caché
        cached_result = self._load_from_cache(cache_file, rec_id)
        if cached_result is not None:
            return cached_result

        # 3. Generar la paráfrasis
        original_text = rec.get("cot_text", "")
        paraphrased_text = self._generate_paraphrase(original_text, ctx, rng)

        # 4. Guardar en caché
        self._save_to_cache(cache_file, rec_id, original_text, paraphrased_text)

        return paraphrased_text

    def _load_from_cache(self, cache_file: str, rec_id: str):
        if not os.path.exists(cache_file):
            return None
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if data.get("id") == rec_id:
                            return data.get("paraphrased_cot")
                    except json.JSONDecodeError:
                        continue
        return None

    def _save_to_cache(self, cache_file: str, rec_id: str, original: str, paraphrased: str):
        record = {
            "id": rec_id,
            "cot_text": original,
            "paraphrased_cot": paraphrased
        }
        with open(cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _generate_paraphrase(self, text: str, ctx: dict, rng) -> str:
        # 1. Si en el contexto (ctx) pasáis el modelo o un pipeline de Hugging Face cargado por Hugo:
        hf_pipeline = ctx.get("hf_pipeline", None)
        model = ctx.get("model", None)
        tokenizer = ctx.get("tokenizer", None)

        prompt = (
            "Task: Rewrite the following chain of thought reasoning step by step.\n"
            "Instructions:\n"
            "- Change the wording, sentence structure, and vocabulary completely.\n"
            "- Strictly preserve the exact logical, mathematical, and factual content.\n"
            "- Do not change the final conclusion or answer.\n\n"
            f"Original Reasoning:\n{text}\n\n"
            "Paraphrased Reasoning:"
        )

        # Opción A: Si usáis un pipeline de Hugging Face
        if hf_pipeline is not None:
            try:
                outputs = hf_pipeline(
                    prompt,
                    max_new_tokens=512,
                    temperature=0.3,  # Baja para no inventar datos
                    do_sample=True,
                    return_full_text=False
                )
                return outputs[0]["generated_text"].strip()
            except Exception as e:
                print(f"⚠️ Error with HF pipeline paraphrase: {e}")

        # Opción B: Si tenéis el modelo y el tokenizer de PyTorch directamente en el contexto
        if model is not None and tokenizer is not None:
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                return generated.strip()
            except Exception as e:
                print(f"⚠️ Error generating paraphrase with model: {e}")

        # Fallback de seguridad si todavía no habéis pasado el modelo en ctx
        # (Esto solo salta si ejecutas la prueba suelta sin cargar el modelo)
        return f"[LLM NOT LOADED] Paraphrased version of: {text}"