"""Generate LOATO confusion matrix grids by replaying training with seed=42.

Standalone script (not integrated into CLI). For each embedding x classifier combo,
trains on each LOATO fold, collects predictions, and produces a 2x3 subplot grid
of confusion matrices (one per held-out category).

Validates F1 from the CM matches the stored result JSON (within 1e-3 tolerance).

Usage:
    uv run python scripts/generate_loato_confusion_matrices.py
    uv run python scripts/generate_loato_confusion_matrices.py \
        --embedding openai_small --classifier mlp
    uv run python scripts/generate_loato_confusion_matrices.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loato_bench.classifiers import (  # noqa: E402
    LogRegClassifier,
    MLPClassifier,
    SVMClassifier,
    XGBoostClassifier,
)
from loato_bench.embeddings.cache import EmbeddingCache  # noqa: E402
from loato_bench.utils.reproducibility import seed_everything  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

plt.switch_backend("Agg")
sns.set_style("whitegrid")

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "figures"

CATEGORY_NAMES: dict[str, str] = {
    "information_extraction": "C4: Info Extraction",
    "instruction_override": "C1: Instruction Override",
    "jailbreak_roleplay": "C2: Jailbreak / Roleplay",
    "obfuscation_encoding": "C3: Obfuscation / Encoding",
    "other": "C7: Other / Multi-Strategy",
    "social_engineering": "C5: Social Engineering",
}

CLASSIFIER_MAP: dict[str, type] = {
    "logreg": LogRegClassifier,
    "mlp": MLPClassifier,
    "svm": SVMClassifier,
    "xgboost": XGBoostClassifier,
}

# Default combos to generate (primary + secondary from plan)
DEFAULT_COMBOS: list[tuple[str, str]] = [
    ("openai_small", "mlp"),
    ("minilm", "xgboost"),
]


def load_data(embedding_name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load embeddings, labels, and LOATO splits."""
    # Labels
    df = pd.read_parquet(DATA_DIR / "processed" / "labeled_v1.parquet")
    labels = df["label"].values.astype(np.int64)

    # Embeddings
    cache = EmbeddingCache(embedding_name, base_dir=DATA_DIR / "embeddings")
    result = cache.load()
    if result is None:
        raise FileNotFoundError(f"No cached embeddings for {embedding_name}")
    embeddings, _ids = result

    # Splits
    with open(DATA_DIR / "splits" / "loato_splits.json") as f:
        splits = json.load(f)

    return embeddings, labels, splits


def load_stored_f1(embedding_name: str, classifier_name: str) -> dict[str, float]:
    """Load stored per-fold F1 from the result JSON."""
    path = RESULTS_DIR / f"loato_{embedding_name}_{classifier_name}.json"
    if not path.exists():
        logger.warning("No stored result at %s", path)
        return {}
    with open(path) as f:
        data = json.load(f)
    return {fold["held_out_category"]: fold["metrics"]["f1"]["value"] for fold in data["folds"]}


def make_classifier(name: str) -> object:
    """Instantiate classifier with default hyperparameters."""
    cls = CLASSIFIER_MAP[name]
    if name == "svm":
        return cls(pca_components=128)
    return cls()


def generate_cm_grid(
    embedding_name: str,
    classifier_name: str,
) -> Path:
    """Replay LOATO training, generate 2x3 confusion matrix grid."""
    seed_everything(42)
    embeddings, labels, splits = load_data(embedding_name)
    stored_f1s = load_stored_f1(embedding_name, classifier_name)

    folds = splits["folds"]
    n_folds = len(folds)
    nrows, ncols = 2, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 9), dpi=150)
    axes_flat = axes.flatten()

    emb_display = {
        "minilm": "MiniLM",
        "bge_large": "BGE-Large",
        "instructor": "Instructor",
        "openai_small": "OpenAI-Small",
        "e5_mistral": "E5-Mistral",
    }
    clf_display = {
        "logreg": "LogReg",
        "svm": "SVM",
        "xgboost": "XGBoost",
        "mlp": "MLP",
    }

    for i, fold in enumerate(folds):
        seed_everything(42)
        cat = fold["held_out_category"]
        train_idx = np.array(fold["train_indices"])
        test_idx = np.array(fold["test_indices"])

        X_train = embeddings[train_idx]
        y_train = labels[train_idx]
        X_test = embeddings[test_idx]
        y_test = labels[test_idx]

        logger.info(
            "Fold %d/%d: held_out=%s, train=%d, test=%d",
            i + 1,
            n_folds,
            cat,
            len(train_idx),
            len(test_idx),
        )

        clf = make_classifier(classifier_name)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Compute F1 and validate against stored
        fold_f1 = f1_score(y_test, y_pred, average="macro")
        stored = stored_f1s.get(cat)
        if stored is not None:
            diff = abs(fold_f1 - stored)
            if diff > 1e-3:
                logger.warning(
                    "F1 mismatch for %s: computed=%.6f, stored=%.6f, diff=%.6f",
                    cat,
                    fold_f1,
                    stored,
                    diff,
                )
            else:
                logger.info("  F1=%.4f (matches stored, diff=%.2e)", fold_f1, diff)
        else:
            logger.info("  F1=%.4f (no stored value to compare)", fold_f1)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        ax = axes_flat[i]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Injection"],
            yticklabels=["Benign", "Injection"],
            ax=ax,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
        )

        # Add percentages as secondary annotations
        total = cm.sum()
        for row in range(2):
            for col in range(2):
                count = cm[row, col]
                pct = count / total * 100
                ax.text(
                    col + 0.5,
                    row + 0.72,
                    f"({pct:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="gray",
                )

        cat_display = CATEGORY_NAMES.get(cat, cat)
        ax.set_title(f"Held out: {cat_display}\nF1 = {fold_f1:.4f}", fontsize=10)
        ax.set_ylabel("True" if i % ncols == 0 else "")
        ax.set_xlabel("Predicted" if i >= n_folds - ncols else "")

    # Hide unused axes
    for j in range(n_folds, nrows * ncols):
        axes_flat[j].set_visible(False)

    emb_disp = emb_display.get(embedding_name, embedding_name)
    clf_disp = clf_display.get(classifier_name, classifier_name)
    fig.suptitle(
        f"LOATO Confusion Matrices: {emb_disp} x {clf_disp}",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for ext in ("png", "pdf"):
        out_path = OUTPUT_DIR / f"confusion_matrices_{embedding_name}_{classifier_name}.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        saved_paths.append(out_path)
        logger.info("Saved: %s", out_path)
    plt.close(fig)

    return saved_paths[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LOATO confusion matrix grids")
    parser.add_argument("--embedding", type=str, help="Embedding model name")
    parser.add_argument("--classifier", type=str, help="Classifier name")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate for all 20 embedding x classifier combos",
    )
    args = parser.parse_args()

    if args.all:
        embeddings = ["minilm", "bge_large", "instructor", "openai_small", "e5_mistral"]
        classifiers = ["logreg", "svm", "xgboost", "mlp"]
        combos = [(e, c) for e in embeddings for c in classifiers]
    elif args.embedding and args.classifier:
        combos = [(args.embedding, args.classifier)]
    else:
        combos = DEFAULT_COMBOS

    for emb, clf in combos:
        logger.info("=== Generating CMs for %s x %s ===", emb, clf)
        generate_cm_grid(emb, clf)

    logger.info("Done! Generated %d confusion matrix grids.", len(combos))


if __name__ == "__main__":
    main()
