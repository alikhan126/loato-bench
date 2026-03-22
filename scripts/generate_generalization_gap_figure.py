"""Generate the generalization gap summary figure (LOATO-5-00).

Produces a grouped bar chart showing Standard CV, LOATO, and Direct→Indirect F1
for OpenAI-Small × MLP, with GPT-4o comparison on the transfer group.

Output: results/analysis/figures/generalization_gap_summary.{png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Style
plt.switch_backend("Agg")
sns.set_style("whitegrid")

# Data from paper tables
protocols = ["Standard CV", "LOATO\n(mean across folds)", "Direct→Indirect\nTransfer"]
classifier_f1 = [0.9974, 0.9758, 0.4130]
gpt4o_f1 = [None, None, 0.7105]  # Only shown for transfer

output_dir = Path("results/analysis/figures")
output_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

x = np.arange(len(protocols))
bar_width = 0.35

# Classifier bars
bars1 = ax.bar(
    x - bar_width / 2,
    classifier_f1,
    bar_width,
    label="OpenAI-Small × MLP",
    color="#2563eb",
    edgecolor="white",
    linewidth=0.5,
    zorder=3,
)

# GPT-4o bar (only on transfer)
gpt4o_vals = [0, 0, 0.7105]
bars2 = ax.bar(
    x[2] + bar_width / 2,
    [0.7105],
    bar_width,
    label="GPT-4o (zero-shot)",
    color="#f59e0b",
    edgecolor="white",
    linewidth=0.5,
    zorder=3,
)

# Value labels on bars
for bar, val in zip(bars1, classifier_f1):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#1e3a5f",
    )

ax.text(
    bars2[0].get_x() + bars2[0].get_width() / 2,
    bars2[0].get_height() + 0.015,
    "0.711",
    ha="center",
    va="bottom",
    fontsize=11,
    fontweight="bold",
    color="#92400e",
)

# Delta annotation: arrow from Standard CV to Transfer
ax.annotate(
    "",
    xy=(2 - bar_width / 2, 0.43),
    xytext=(0 - bar_width / 2, 0.95),
    arrowprops=dict(
        arrowstyle="->",
        color="#dc2626",
        lw=2,
        connectionstyle="arc3,rad=0.15",
    ),
)
ax.text(
    0.28,
    0.55,
    "ΔF1 = −0.584\n(58% collapse)",
    fontsize=10,
    color="#dc2626",
    fontweight="bold",
    ha="center",
    transform=ax.transData,
)

# Formatting
ax.set_ylabel("Macro F1", fontsize=12, fontweight="bold")
ax.set_title(
    "The Generalization Gap",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xticks(x)
ax.set_xticklabels(protocols, fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

# Add a subtle horizontal line at 0.5
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5, zorder=1)

ax.grid(axis="y", alpha=0.3, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()

# Save both formats
for ext in ("png", "pdf"):
    path = output_dir / f"generalization_gap_summary.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)
