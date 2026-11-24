import random
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from interventions import Truncation, Filler, Swap, MistakeInjection
from utils import split_steps

COT = "Step 1: a = 2.\nStep 2: b = a * 3 = 6.\nStep 3: c = b + 1 = 7.\nStep 4: done."
REC = {"id": "a", "cot_text": COT, "cot_steps": split_steps(COT)}
OTHER = {"id": "b", "cot_text": "otro razonamiento", "cot_steps": ["otro razonamiento"]}
CTX = {"tok": None, "records": [REC, OTHER]}


def test_truncation_extremes():
    t = Truncation()
    assert t.apply(REC, 0.0, random.Random(0), CTX) == ""
    assert t.apply(REC, 1.0, random.Random(0), CTX) == COT


def test_filler_same_length():
    out = Filler().apply(REC, None, random.Random(0), CTX)
    assert len(out.split()) == len(COT.split())


def test_swap_differs():
    assert Swap().apply(REC, None, random.Random(0), CTX) == OTHER["cot_text"]


def test_mistake_changes_only_one_step():
    out = MistakeInjection().apply(REC, None, random.Random(1), CTX)
    diff = [a != b for a, b in zip(split_steps(out), REC["cot_steps"])]
    assert sum(diff) <= 1
