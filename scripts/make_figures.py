"""Render scientific figures directly from the recorded weights and repair submission."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

from toy_weight_repair.experiment import forward

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/toy-symbol-repair-001"
OUT = ROOT / "figures"
INK, MUTED, TEAL, RED = "#172c3a", "#536878", "#008878", "#bc493a"
COLORS = ["#c2d8eb", "#a9d9ce", "#eac78d", "#d1bde3", "#efb7a6", "#b9d29a", "#a9cee4", "#dbb8c9"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "figure.facecolor": "#fbfcfd",
        "axes.facecolor": "#fbfcfd",
        "svg.hashsalt": "toy-repair",
    }
)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None})
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def load():
    cases = json.loads((EVIDENCE / "editor-input.json").read_text())
    answers = json.loads((EVIDENCE / "astra/answer.md").read_text())["cases"]
    results = json.loads((EVIDENCE / "results.json").read_text())["cases"]
    values = []
    for case in cases:
        answer = next(a for a in answers if a["id"] == case["id"])
        before = {k: np.array(v) for k, v in case["weights"].items()}
        after = {k: v.copy() for k, v in before.items()}
        for edit in answer["edits"]:
            after["W2"][edit["row"], edit["col"]] = edit["value"]
        values.append((case, answer, before, after))
    return values, results


def grid(ax, values, title, subtitle, practice, mode, errors=None, binary=False):
    cmap = ListedColormap(["#e7ecef", TEAL]) if binary else ListedColormap(COLORS)
    ax.imshow(values, cmap=cmap, vmin=0, vmax=1 if binary else 7)
    for a in range(8):
        for b in range(8):
            label = ("+" if values[a, b] else "·") if binary else str(values[a, b])
            color = "white" if binary and values[a, b] else INK
            ax.text(b, a, label, ha="center", va="center", color=color, fontsize=12)
            if errors is not None and errors[a, b]:
                ax.add_patch(
                    Rectangle(
                        (b - 0.46, a - 0.46), 0.92, 0.92, fill=False, edgecolor=RED, linewidth=1.5
                    )
                )
    for example in practice:
        a, b, m = example["input"]
        if m == mode:
            ax.plot(b + 0.30, a + 0.30, "o", ms=3.6, color="white", markeredgecolor=INK, mew=0.5)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xlabel("last symbol")
    ax.set_ylabel("first symbol")
    ax.set_xticks(np.arange(-0.5, 8), minor=True)
    ax.set_yticks(np.arange(-0.5, 8), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="both", length=0)
    ax.set_title(f"{title}\n", loc="left", fontsize=14, fontweight="bold", pad=12)
    ax.text(0, 1.025, subtitle, transform=ax.transAxes, color=MUTED, fontsize=10)


def behavior(values):
    case, answer, before, after = values[
        1
    ]  # first damaged case; fixed selection, no cherry-picking
    examples = [(a, b, 0) for a in range(8) for b in range(8)]
    previous = forward(before, examples).reshape(8, 8)
    repaired = forward(after, examples).reshape(8, 8)
    target = np.repeat(np.arange(8)[:, None], 8, axis=1)
    changes = (previous != repaired).astype(int)
    expected = np.zeros((8, 8), dtype=bool)
    for mode, symbol in answer["predicted_changed_slices"]:
        if mode == 0:
            expected[symbol] = True
    assert np.array_equal(changes, expected)
    other_examples = [(a, b, 1) for a in range(8) for b in range(8)]
    other = forward(after, other_examples).reshape(8, 8)
    assert np.array_equal(forward(before, other_examples).reshape(8, 8), other)
    fig, axes = plt.subplots(2, 2, figsize=(12, 11.2))
    fig.subplots_adjust(left=0.10, right=0.94, top=0.83, bottom=0.10, hspace=0.46, wspace=0.34)
    fig.text(
        0.08,
        0.955,
        "six weights. three broken routes. zero collateral damage.",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.920,
        "conditional copy · 632 parameters · case 2 · original astra submission",
        fontsize=12,
        color=MUTED,
    )
    grid(
        axes[0, 0],
        previous,
        "01 / before repair",
        "mode 0: copy first · red borders = wrong",
        case["practice"],
        0,
        errors=previous != target,
    )
    grid(
        axes[0, 1],
        repaired,
        "02 / after repair",
        "mode 0: every row now returns its own symbol",
        case["practice"],
        0,
    )
    grid(
        axes[1, 0],
        changes,
        "03 / predicted effect = actual effect",
        "only symbols 1, 2, 7 change · + = label changed",
        case["practice"],
        0,
        binary=True,
    )
    grid(
        axes[1, 1],
        other,
        "04 / other mode stays intact",
        "mode 1: copy last · all 64 predictions unchanged",
        case["practice"],
        1,
    )
    fig.text(
        0.08,
        0.035,
        "each cell = one input; numeral / color = predicted symbol. small dots = practice inputs;"
        " all other cells were held out.",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.08,
        0.015,
        "grids enumerate all 128 legal inputs; damaged-model hidden accuracy: 91/112 → 112/112."
        " constructed toy circuits, not a vision task.",
        fontsize=9,
        color=MUTED,
    )
    save(fig, "behavior")


def circuit(values):
    _, answer, before, after = values[1]
    changed_cols = sorted({e["col"] for e in answer["edits"]})
    # Recover detector labels from weights rather than hard-code a diagram.
    detectors = sorted((int(np.argmax(before["W1"][col, :8])), col) for col in changed_cols)
    symbols = [s for s, _ in detectors]
    y_positions = {symbol: 2 - i for i, symbol in enumerate(symbols)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.9))
    fig.subplots_adjust(left=0.06, right=0.96, top=0.73, bottom=0.15, wspace=0.20)
    fig.text(
        0.06, 0.94, "repair the destination. preserve the detector.", fontsize=23, fontweight="bold"
    )
    fig.text(
        0.06,
        0.88,
        "three removed edges + three restored edges = six individual W2 replacements",
        color=MUTED,
        fontsize=12,
    )
    for ax, weights, title, color in zip(
        axes,
        [before, after],
        ["before / a three-symbol cycle", "after / correct routing"],
        [RED, TEAL],
        strict=True,
    ):
        ax.set_xlim(-0.6, 3.5)
        ax.set_ylim(-0.5, 2.8)
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
        ax.text(0, 2.55, "active detector", ha="center", fontsize=10, color=MUTED)
        ax.text(2.8, 2.55, "output symbol", ha="center", fontsize=10, color=MUTED)
        for symbol, col in detectors:
            y = y_positions[symbol]
            dest = int(np.argmax(weights["W2"][:, col]))
            dest_y = y_positions[dest]
            ax.add_patch(
                FancyArrowPatch(
                    (0.28, y),
                    (2.52, dest_y),
                    arrowstyle="-|>",
                    mutation_scale=16,
                    color=color,
                    linewidth=2.5,
                    connectionstyle="arc3,rad=0.06",
                )
            )
            weight = weights["W2"][dest, col]
            ax.text(
                1.35,
                (y + dest_y) / 2 + 0.13,
                f"{weight:g}",
                ha="center",
                color=color,
                fontsize=11,
                bbox={"facecolor": "#fbfcfd", "edgecolor": "none", "pad": 1},
            )
            ax.scatter([0, 2.8], [y, y], s=1400, c=["#e5edf2", "#e5edf2"], zorder=3)
            ax.text(0, y, f"{symbol}", ha="center", va="center", fontsize=15, fontweight="bold")
            ax.text(0, y - 0.32, f"neuron {col}", ha="center", va="center", fontsize=9, color=MUTED)
            ax.text(2.8, y, str(symbol), ha="center", va="center", fontsize=15, fontweight="bold")
    fig.text(
        0.06,
        0.055,
        "case 2 · affected mode: copy first · labels on edges are actual weights."
        " all other weights stay fixed.",
        color=MUTED,
        fontsize=10,
    )
    fig.text(
        0.06,
        0.018,
        "only affected connections are shown; full 24-neuron / 8-output network is in the evidence.",
        color=MUTED,
        fontsize=9,
    )
    save(fig, "circuit")


def outcomes(results):
    fig, ax = plt.subplots(figsize=(11.5, 5.9))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.73, bottom=0.22)
    fig.text(
        0.08,
        0.94,
        "every damaged model repaired. healthy control untouched.",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.87,
        "one submission · no training · no execution feedback · 24.12 s recorded agent turn",
        fontsize=12,
        color=MUTED,
    )
    x = np.arange(4)
    before = [100 * r["baseline_correct"] / r["heldout_n"] for r in results]
    after = [100 * r["edited_correct"] / r["heldout_n"] for r in results]
    ax.bar(x - 0.17, before, 0.30, color="#8b9daa", label="before")
    ax.bar(x + 0.17, after, 0.30, color=TEAL, label="after")
    for i, result in enumerate(results):
        for offset, number in [
            (-0.17, result["baseline_correct"]),
            (0.17, result["edited_correct"]),
        ]:
            ax.text(
                i + offset,
                number / 112 * 100 + 2.2,
                f"{number}/112",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100], labels=["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("held-out accuracy")
    ax.set_xticks(
        x,
        [
            "healthy control\n0 edits",
            "damaged 1\n6 edits",
            "damaged 2\n6 edits",
            "damaged 3\n6 edits",
        ],
    )
    ax.yaxis.grid(True, color="#dfe6eb", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(
        handles=[Patch(color="#8b9daa", label="before"), Patch(color=TEAL, label="after")],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.20),
        frameon=False,
        ncol=2,
    )
    fig.text(
        0.08,
        0.09,
        "63 hidden errors corrected  /  0 regressions  /  4 of 4 exact behavioral-effect predictions",
        fontsize=13,
        fontweight="bold",
        color=TEAL,
    )
    fig.text(
        0.08,
        0.035,
        "exact counts over a fixed input set; no statistical error bars. four cases share one task family"
        " and one conversation.",
        fontsize=10,
        color=MUTED,
    )
    save(fig, "results")


def main():
    values, results = load()
    behavior(values)
    circuit(values)
    outcomes(results)
    print("rendered three figures as PNG, SVG, and PDF from the original evidence")


if __name__ == "__main__":
    main()
