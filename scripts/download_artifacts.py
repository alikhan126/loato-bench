"""Download pre-computed artifacts from Hugging Face Hub.

This script downloads embeddings, experiment results, dataset files, and splits
from the HF dataset repo so you don't need to re-run the full pipeline.

Authentication:
    Set HF_TOKEN in your .env file or as an environment variable.
    You can also run `huggingface-cli login` to authenticate interactively.

Usage:
    uv run python scripts/download_artifacts.py          # Download everything
    uv run python scripts/download_artifacts.py --only embeddings
    uv run python scripts/download_artifacts.py --only results
    uv run python scripts/download_artifacts.py --only data
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

REPO_ID = "alikhan126/loato-bench-artifacts"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def download_artifacts(only: str | None = None) -> None:
    """Download artifacts from HF Hub to local project directories."""
    # Load .env for HF_TOKEN
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("HF_TOKEN")

    cache_dir = PROJECT_ROOT / ".hf_cache"

    print(f"Downloading from https://huggingface.co/datasets/{REPO_ID} ...")
    if token:
        print("  Using HF_TOKEN from environment")
    else:
        print("  No HF_TOKEN found — using anonymous access (may fail for private repos)")

    # Download the full snapshot (HF caches efficiently)
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(cache_dir),
        token=token,
    )
    local_path = Path(local_dir)

    copied = 0

    # Copy embeddings
    if only in (None, "embeddings"):
        src = local_path / "embeddings"
        if src.exists():
            for model_dir in sorted(src.iterdir()):
                if model_dir.is_dir():
                    dst = PROJECT_ROOT / "data" / "embeddings" / model_dir.name
                    dst.mkdir(parents=True, exist_ok=True)
                    for f in model_dir.iterdir():
                        shutil.copy2(f, dst / f.name)
                        copied += 1
                    print(f"  embeddings/{model_dir.name}/ -> data/embeddings/{model_dir.name}/")

    # Copy experiment results
    if only in (None, "results"):
        src = local_path / "results" / "experiments"
        if src.exists():
            dst = PROJECT_ROOT / "results" / "experiments"
            dst.mkdir(parents=True, exist_ok=True)
            for f in sorted(src.glob("*.json")):
                shutil.copy2(f, dst / f.name)
                copied += 1
            print(f"  results/experiments/ ({len(list(src.glob('*.json')))} files)")

    # Copy data files (parquets + splits)
    if only in (None, "data"):
        # Parquets
        src = local_path / "data" / "processed"
        if src.exists():
            dst = PROJECT_ROOT / "data" / "processed"
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob("*.parquet"):
                shutil.copy2(f, dst / f.name)
                copied += 1
                print(f"  data/processed/{f.name}")

        # Splits
        src = local_path / "data" / "splits"
        if src.exists():
            dst = PROJECT_ROOT / "data" / "splits"
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob("*.json"):
                shutil.copy2(f, dst / f.name)
                copied += 1
            print(f"  data/splits/ ({len(list(src.glob('*.json')))} files)")

    # Clean up cache
    shutil.rmtree(cache_dir, ignore_errors=True)

    print(f"\nDone! Copied {copied} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LOATO-Bench artifacts from HF Hub")
    parser.add_argument(
        "--only",
        choices=["embeddings", "results", "data"],
        default=None,
        help="Download only a specific category of artifacts",
    )
    args = parser.parse_args()
    download_artifacts(only=args.only)
