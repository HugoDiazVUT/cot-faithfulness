from .truncation import Truncation
from .filler import Filler
from .swap import Swap
from .mistake import MistakeInjection
from .paraphrase import Paraphrase
from .custom import CustomIntervention

REGISTRY = {c.name: c for c in [Truncation, Filler, Swap, MistakeInjection, Paraphrase, CustomIntervention]}
