import argparse
import difflib
import os
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# --- IMPORTACIONES ---
# Descomenta las de Hugo cuando estén implementadas:
# from interventions import Truncation, Filler, Swap
from interventions.mistake import MistakeInjection
from interventions.paraphrase import Paraphrase
from interventions.reorder import StepReorder
from utils import split_steps

# --- DATOS DE EJEMPLO ---
COT_NUMERIC = "Step 1: a = 2.\nStep 2: b = a * 3 = 6.\nStep 3: c = b + 1 = 7.\nStep 4: done."

COT_6_STEPS = (
    "Step 1: First, we identify that the user wants to organize a schedule for a weekly team meeting.\n"
    "Step 2: Next, we check the availability of all team members using their calendar profiles.\n"
    "Step 3: Then, we find that Tuesday at 10 AM has a common free slot for everyone.\n"
    "Step 4: After that, we reserve the main conference room for that specific time slot.\n"
    "Step 5: Following the reservation, we send out calendar invites containing the meeting agenda.\n"
    "Step 6: Therefore, the final outcome is a successfully scheduled team meeting for Tuesday morning."
)
OTHER = {"id": "b", "cot_text": "otro razonamiento", "cot_steps": ["otro razonamiento"]}


def get_rec_and_ctx(cot_type: str):
    if cot_type == "numeric":
        rec = {
            "id": "test_num_01",
            "cot_text": COT_NUMERIC,
            "cot_steps": split_steps(COT_NUMERIC),
            "dataset": "aqua",
        }
    else:
        rec = {
            "id": "test_non_num_01",
            "cot_text": COT_NON_NUMERIC,
            "cot_steps": split_steps(COT_NON_NUMERIC),
            "dataset": "commonsenseqa",
        }
    ctx = {"tok": None, "model_name": "qwen3", "client": None, "records": [rec, OTHER]}
    return rec, ctx


# --- TESTS UNITARIOS PARA PYTEST (HUGO - COMENTADOS) ---
# def test_truncation_extremes():
#     rec, ctx = get_rec_and_ctx("numeric")
#     t = Truncation()
#     assert t.apply(rec, 0.0, random.Random(0), ctx) == ""
#     assert t.apply(rec, 1.0, random.Random(0), ctx) == COT_NUMERIC


# def test_filler_same_length():
#     rec, ctx = get_rec_and_ctx("numeric")
#     out = Filler().apply(rec, None, random.Random(0), ctx)
#     assert len(out.split()) == len(COT_NUMERIC.split())


# def test_swap_differs():
#     rec, ctx = get_rec_and_ctx("numeric")
#     assert Swap().apply(rec, None, random.Random(0), ctx) == OTHER["cot_text"]


# --- TESTS DE PABLO (CON INSPECCIÓN VISUAL) ---
def run_mistake_test(cot_type="numeric", seed=1):
    rec, ctx = get_rec_and_ctx(cot_type)
    rec_copy = dict(rec)
    original_text = rec_copy["cot_text"]

    modified_text = MistakeInjection().apply(rec_copy, None, random.Random(seed), ctx)

    print("\n" + "=" * 50)
    print(f"🧪 MISTAKE INJECTION | CoT: {cot_type.upper()} | Seed: {seed}")
    print("=" * 50)
    print("--- 📄 ORIGINAL ---")
    print(original_text)
    print("\n--- ⚡ MODIFICADO ---")
    print(modified_text)
    print(f"\n👉 Índice del paso modificado: {rec_copy.get('_mistake_idx')}")
    print("\n--- 🔍 DIFF ---")
    for line in difflib.unified_diff(
        original_text.splitlines(), modified_text.splitlines(), fromfile="orig", tofile="mod", lineterm=""
    ):
        print(line)
    print("=" * 50 + "\n")

    diff_steps = [a != b for a, b in zip(split_steps(modified_text), split_steps(original_text))]
    assert sum(diff_steps) == 1, f"Se esperaba 1 paso modificado, se modificaron {sum(diff_steps)}"
    assert "_mistake_idx" in rec_copy


def run_paraphrase_test(cot_type="numeric"):
    rec, ctx = get_rec_and_ctx(cot_type)
    paraphraser = Paraphrase()

    out = paraphraser.apply(rec, None, random.Random(0), ctx)

    print("\n" + "=" * 50)
    print(f"🧪 PARAPHRASE | CoT: {cot_type.upper()} | ID: {rec['id']}")
    print("=" * 50)
    print("--- 📄 ORIGINAL ---")
    print(rec["cot_text"])
    print("\n--- ✍️ PARAFRASEADO ---")
    print(out)
    print("=" * 50 + "\n")

    assert out != rec["cot_text"], "La paráfrasis debe diferir del original"
    cache_path = os.path.join("cache", "qwen3", f"paraphrases_{rec['dataset']}.jsonl")
    assert os.path.exists(cache_path), "No se generó el archivo de caché"


# Funciones compatibles con pytest
def test_mistake_changes_only_one_step():
    run_mistake_test("numeric", seed=1)


def test_paraphrase_works():
    run_paraphrase_test("non_numeric")


def run_reorder_test(cot_type="numeric",seed=None):
    # Usamos nuestro ejemplo de 6 pasos
    rec = {
        "id": "test_6_steps",
        "cot_text": COT_6_STEPS,
        "cot_steps": split_steps(COT_6_STEPS),
        "dataset": "bbh"
    }
    ctx = {"model_name": "qwen3"}
    
    original_text = rec["cot_text"]
    rng = random.Random(seed)
    
    reorderer = StepReorder()
    modified_text = reorderer.apply(rec, None, rng, ctx)
    
    print("\n" + "="*50)
    print(f" STEP REORDER (6 PASOS) | Seed: {seed}")
    print("="*50)
    print("---  ORIGINAL ---")
    print(original_text)
    print("\n---  REORDENADO ---")
    print(modified_text)
    print("="*50 + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar pruebas de intervenciones de Pablo.")
    parser.add_argument(
        "--test",
        type=str,
        default="mistake",
        choices=["mistake", "paraphrase", "reorder"], 
        help="Prueba a ejecutar: 'mistake', 'paraphrase' o 'reorder'.",
    )
    parser.add_argument(
        "--cot",
        type=str,
        default="numeric",
        choices=["numeric", "non_numeric"],
        help="Tipo de CoT: 'numeric' o 'non_numeric'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla aleatoria (útil para variar los cambios en mistake).",
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)

    if args.test == "mistake":
        run_mistake_test(cot_type=args.cot, seed=args.seed)
    elif args.test == "paraphrase":
        run_paraphrase_test(cot_type=args.cot)
    elif args.test == "reorder":
        run_reorder_test(cot_type=args.cot, seed=args.seed)