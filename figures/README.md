# figures

all plots are generated from the published weights and astra's original answer. the generator
does not train, call an agent, or synthesize new experimental outcomes.

| figure | PNG | editable SVG | vector PDF |
|---|---|---|---|
| behavior before / after | [PNG](behavior.png) | [SVG](behavior.svg) | [PDF](behavior.pdf) |
| six-weight circuit repair | [PNG](circuit.png) | [SVG](circuit.svg) | [PDF](circuit.pdf) |
| all four cases | [PNG](results.png) | [SVG](results.svg) | [PDF](results.pdf) |

## captions

**behavior.** case 2, the first damaged fixture in recorded order. mode zero should return
the first symbol, so correct grids have constant rows. three rows initially route to the
wrong output. six output-weight replacements correct these rows. the effect map agrees
exactly with the pre-evaluation prediction; mode one retains all 64 labels. small white dots
are practice inputs, all other cells held out. colors are categorical; numerals are the
predicted symbols. the grids show all inputs, while the reported hidden score is 91/112
before and 112/112 after. this is a symbol task, not an image-input experiment.

**circuit.** only the three affected hidden detectors and their output destinations are
shown, using actual neuron indices and connection values. each wrong edge is removed and
its correct destination added. inputs, gating, other hidden neurons, and unrelated outputs
are omitted for clarity.

**results.** all three damaged fixtures improve from 91/112 to 112/112 hidden correct labels;
the healthy fixture stays at 112/112 with no edits. denominators are exact finite-set counts,
not independent statistical trials. all four cases share one conversation and task family.

## regenerate

from the repository root:

```sh
uv sync --locked --group plots
uv run --group plots python scripts/make_figures.py
```

the script checks the plotted changed-label mask against the original prospective prediction,
and verifies that the illustrated other switch mode is unchanged. it writes PNG at 180 dpi,
SVG, and PDF. the evidence remains untouched. figures are covered by the repository's MIT
license; retain the task scope and denominators when sharing.
