import matplotlib.pyplot as plt

import config
from metrics import area_over_curve, truncation_curve
from utils import read_jsonl


def plot_truncation(datasets, models=tuple(config.MODELS), out="results/truncation_curves.png"):
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
    axes = [axes] if len(datasets) == 1 else axes
    for ax, ds in zip(axes, datasets):
        for m in models:
            p = config.RESULTS_DIR / m / ds / "truncation.jsonl"
            if not p.exists():
                continue
            rows = read_jsonl(p)
            x, a = truncation_curve(rows)
            ax.plot(x, a, marker="o", label=f"{m} (AOC={area_over_curve(rows):.2f})")
        ax.set_title(ds); ax.set_xlabel("fracción de CoT retenida"); ax.legend()
    axes[0].set_ylabel("acuerdo con la cadena completa")
    plt.tight_layout(); plt.savefig(out, dpi=200)
