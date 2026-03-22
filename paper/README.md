# LOATO-Bench Paper

Publication-ready LaTeX paper (8–10 pages).

## Build

```bash
cd paper/
make pdf          # Produces main.pdf
make clean        # Remove build artifacts
```

Requires: `latexmk`, `pdflatex`, `bibtex` (included in TeX Live or MacTeX).

## Figures

Figures in `figures/` were generated on 2026-03-21 from the following commands:

```bash
# 4B-01: Core tables and heatmaps
uv run loato-bench analyze report

# 4B-04: Template homogeneity, UMAP, centroid distances
uv run loato-bench analyze template-homogeneity

# 5-00: Generalization gap summary
uv run python scripts/generate_generalization_gap_figure.py
```

Source artifacts in `analysis/` and `results/analysis/` (gitignored, reproducible via CLI).
