"""Configuración global. Cambiad aquí, no en los scripts."""
from pathlib import Path

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

SEED = 42
N_EXAMPLES = 300          # por tarea y modelo
DTYPE = "float16"         # misma precisión en los 3 modelos (T4 de Kaggle no soporta bf16)
MAX_NEW_TOKENS = 512

MODELS = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
}

# BBH: subconjuntos con opciones cerradas (ajustad la lista)
BBH_SUBSETS = [
    "date_understanding",
    "disambiguation_qa",
    "logical_deduction_five_objects",
    "tracking_shuffled_objects_three_objects",
]
DATASETS = ["commonsenseqa", "aqua", "bbh"]

TRUNCATION_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
