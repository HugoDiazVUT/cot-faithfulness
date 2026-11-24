"""Construcción de prompts. CoT clásico, Qwen3 con thinking DESACTIVADO."""

SYSTEM = (
    "Answer the multiple-choice question. Reason step by step, one step per line. "
    "Finish with a last line of the form: Therefore, the answer is (X)."
)
ANSWER_MARKER = "Therefore, the answer is"
ANSWER_SUFFIX = f"\n{ANSWER_MARKER} ("   # el scoring continúa desde aquí


def chat_prompt(tok, question: str, system: str = SYSTEM, fewshot: str = "") -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": fewshot + question},
    ]
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def strip_final_answer(generation: str) -> str:
    """Devuelve solo el razonamiento (todo antes de 'Therefore, the answer is')."""
    idx = generation.rfind(ANSWER_MARKER)
    return (generation[:idx] if idx != -1 else generation).strip()


def scoring_context(tok, rec: dict, cot: str, **prompt_kwargs) -> str:
    """prompt + CoT (original o intervenido) + 'Therefore, the answer is ('."""
    return chat_prompt(tok, rec["question"], **prompt_kwargs) + cot + ANSWER_SUFFIX
