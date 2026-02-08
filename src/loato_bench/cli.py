"""LOATO-Bench CLI — Typer entrypoints for all pipeline stages."""

from __future__ import annotations

import logging

import typer
from rich.console import Console

from loato_bench.utils.config import DATA_DIR

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer(
    name="loato-bench",
    help="LOATO-Bench: Embedding-based prompt injection classifiers.",
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# Sub-command groups
# ---------------------------------------------------------------------------

data_app = typer.Typer(help="Data pipeline: download, harmonize, split.")
embed_app = typer.Typer(help="Compute embeddings for all models.")
train_app = typer.Typer(help="Train classifiers and run experiments.")
sweep_app = typer.Typer(help="Hyperparameter sweeps via W&B.")
analyze_app = typer.Typer(help="Analysis, visualization, and reporting.")

app.add_typer(data_app, name="data")
app.add_typer(embed_app, name="embed")
app.add_typer(train_app, name="train")
app.add_typer(sweep_app, name="sweep")
app.add_typer(analyze_app, name="analyze")


# ---------------------------------------------------------------------------
# data sub-commands
# ---------------------------------------------------------------------------


@data_app.command()
def download() -> None:
    """Download all raw datasets via HuggingFace."""
    from loato_bench.data import (
        DeepsetLoader,
        GenTelLoader,
        HackaPromptLoader,
        OpenPromptLoader,
        PINTLoader,
    )

    loaders = [
        ("Deepset", DeepsetLoader()),
        ("HackAPrompt", HackaPromptLoader()),
        ("GenTel-Bench", GenTelLoader(max_samples=10000)),
        ("PINT/Gandalf", PINTLoader()),
        ("Open-Prompt-Injection", OpenPromptLoader()),
    ]

    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    for name, loader in loaders:
        console.print(f"[bold green]Loading {name}...[/bold green]")
        try:
            samples = loader.load()
            console.print(f"  [dim]{len(samples)} samples loaded[/dim]")
            all_samples.extend(samples)
        except Exception as e:
            console.print(f"  [red]Failed: {e}[/red]")
            logger.exception("Failed to load %s", name)

    console.print(f"\n[bold]Total: {len(all_samples)} raw samples[/bold]")

    # Save raw samples as pickle for harmonize step
    import pickle

    raw_path = raw_dir / "all_samples.pkl"
    with open(raw_path, "wb") as f:
        pickle.dump(all_samples, f)
    console.print(f"[dim]Saved to {raw_path}[/dim]")


@data_app.command()
def harmonize() -> None:
    """Harmonize and deduplicate into unified parquet."""
    import pickle

    from loato_bench.data.harmonize import harmonize_samples

    raw_path = DATA_DIR / "raw" / "all_samples.pkl"
    if not raw_path.exists():
        console.print("[red]No raw data found. Run 'loato-bench data download' first.[/red]")
        raise typer.Exit(1)

    console.print("[bold green]Loading raw samples...[/bold green]")
    with open(raw_path, "rb") as f:
        samples = pickle.load(f)
    console.print(f"  [dim]{len(samples)} raw samples[/dim]")

    console.print("[bold green]Harmonizing...[/bold green]")
    df = harmonize_samples(samples)
    console.print(f"  [dim]{len(df)} samples after harmonization[/dim]")

    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "unified_dataset.parquet"
    df.to_parquet(out_path, index=False)
    console.print(f"[bold]Saved to {out_path}[/bold]")

    # Print summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Total samples: {len(df)}")
    console.print(f"  Sources: {df['source'].value_counts().to_dict()}")
    console.print(f"  Labels: {df['label'].value_counts().to_dict()}")
    console.print(f"  Languages: {df['language'].value_counts().to_dict()}")


@data_app.command()
def split() -> None:
    """Generate all evaluation splits (standard CV, LOATO, transfer)."""
    console.print("[bold green]Generating splits...[/bold green]")
    # TODO: Sprint 2A — invoke split generation
    console.print("[yellow]Not yet implemented.[/yellow]")


# ---------------------------------------------------------------------------
# embed sub-commands
# ---------------------------------------------------------------------------

EMBEDDING_MODELS = ["minilm", "bge_large", "instructor", "openai_small", "e5_mistral"]


@embed_app.command("run")
def embed_run(
    model: str | None = typer.Option(None, help="Embedding model name."),
    all_models: bool = typer.Option(False, "--all", help="Run all embedding models."),
) -> None:
    """Compute embeddings for one or all models."""
    import pandas as pd

    from loato_bench.embeddings import EmbeddingCache, compute_text_hash, get_embedding_model

    models = EMBEDDING_MODELS if all_models else ([model] if model else [])
    if not models:
        console.print("[red]Specify --model <name> or --all[/red]")
        raise typer.Exit(1)

    # Load processed dataset
    parquet_path = DATA_DIR / "processed" / "unified_dataset.parquet"
    if not parquet_path.exists():
        console.print("[red]No processed data. Run 'loato-bench data harmonize' first.[/red]")
        raise typer.Exit(1)

    df = pd.read_parquet(parquet_path)
    texts = df["text"].tolist()
    sample_ids = [str(i) for i in range(len(texts))]
    text_hash = compute_text_hash(texts)

    console.print(f"[dim]Loaded {len(texts)} texts from {parquet_path.name}[/dim]")

    for m in models:
        console.print(f"\n[bold green]Embedding model: {m}[/bold green]")
        cache = EmbeddingCache(m)

        if cache.is_valid(model_version=m, text_hash=text_hash):
            console.print(f"  [dim]Cache hit — skipping {m}[/dim]")
            result = cache.load()
            if result:
                emb, _ = result
                console.print(f"  [dim]{emb.shape[0]} samples × {emb.shape[1]} dims[/dim]")
            continue

        console.print("  [dim]Cache miss — computing embeddings...[/dim]")
        emb_model = get_embedding_model(m)
        embeddings = emb_model.encode(texts)
        cache.save(embeddings, sample_ids, model_version=m, text_hash=text_hash)
        console.print(
            f"  [bold]{embeddings.shape[0]} samples × {embeddings.shape[1]} dims → cached[/bold]"
        )


# ---------------------------------------------------------------------------
# train sub-commands
# ---------------------------------------------------------------------------

CLASSIFIERS = ["logreg", "svm", "xgboost", "mlp"]
EXPERIMENTS = ["standard_cv", "loato", "direct_indirect", "crosslingual"]


@train_app.command("run")
def train_run(
    embedding: str | None = typer.Option(None, help="Embedding model name."),
    classifier: str | None = typer.Option(None, help="Classifier name."),
    experiment: str = typer.Option("standard_cv", help="Experiment type."),
    all_combos: bool = typer.Option(False, "--all", help="Run all combos."),
) -> None:
    """Train classifiers on embeddings for a given experiment."""
    if all_combos:
        pairs = [(e, c) for e in EMBEDDING_MODELS for c in CLASSIFIERS]
    elif embedding and classifier:
        pairs = [(embedding, classifier)]
    else:
        console.print("[red]Specify --embedding + --classifier, or --all[/red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Experiment: {experiment} ({len(pairs)} runs)[/bold green]")
    for emb, clf in pairs:
        console.print(f"  {emb} × {clf}")
        # TODO: Sprint 2B / Sprint 3 — invoke training pipeline
    console.print("[yellow]Not yet implemented.[/yellow]")


# ---------------------------------------------------------------------------
# sweep sub-commands
# ---------------------------------------------------------------------------


@sweep_app.command("run")
def sweep_run(
    all_classifiers: bool = typer.Option(False, "--all", help="Sweep all classifiers."),
) -> None:
    """Run hyperparameter sweeps."""
    console.print("[bold green]Running hyperparameter sweeps...[/bold green]")
    # TODO: Sprint 2B — invoke W&B sweeps
    console.print("[yellow]Not yet implemented.[/yellow]")


# ---------------------------------------------------------------------------
# analyze sub-commands
# ---------------------------------------------------------------------------


@analyze_app.command()
def features(
    all_models: bool = typer.Option(False, "--all", help="Analyze all embedding models."),
) -> None:
    """Run SHAP feature importance analysis."""
    console.print("[bold green]Running feature analysis...[/bold green]")
    # TODO: Sprint 4B
    console.print("[yellow]Not yet implemented.[/yellow]")


@analyze_app.command("llm-baseline")
def llm_baseline(
    samples: int = typer.Option(500, help="Number of samples to evaluate."),
) -> None:
    """Run LLM zero-shot baseline evaluation."""
    console.print(f"[bold green]Running LLM baseline on {samples} samples...[/bold green]")
    # TODO: Sprint 4A
    console.print("[yellow]Not yet implemented.[/yellow]")


@analyze_app.command()
def report() -> None:
    """Generate markdown + LaTeX report tables."""
    console.print("[bold green]Generating report...[/bold green]")
    # TODO: Sprint 4B
    console.print("[yellow]Not yet implemented.[/yellow]")


# ---------------------------------------------------------------------------
# Entrypoint for pyproject.toml [project.scripts]
# ---------------------------------------------------------------------------


def app_entry() -> None:
    """Entry point called by `loato-bench` CLI."""
    app()
