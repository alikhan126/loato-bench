"""PromptGuard CLI — Typer entrypoints for all pipeline stages."""

from typing import Optional

import typer
from rich.console import Console

console = Console()
app = typer.Typer(
    name="promptguard",
    help="PromptGuard-Lite: Cross-attack generalization of embedding-based prompt injection classifiers.",
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
    """Download all raw datasets."""
    console.print("[bold green]Downloading datasets...[/bold green]")
    # TODO: Sprint 1A — invoke dataset loaders
    console.print("[yellow]Not yet implemented.[/yellow]")


@data_app.command()
def harmonize() -> None:
    """Harmonize and deduplicate into unified parquet."""
    console.print("[bold green]Harmonizing datasets...[/bold green]")
    # TODO: Sprint 1A — invoke harmonization pipeline
    console.print("[yellow]Not yet implemented.[/yellow]")


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
    model: Optional[str] = typer.Option(None, help="Embedding model name."),
    all_models: bool = typer.Option(False, "--all", help="Run all embedding models."),
) -> None:
    """Compute embeddings for one or all models."""
    models = EMBEDDING_MODELS if all_models else ([model] if model else [])
    if not models:
        console.print("[red]Specify --model <name> or --all[/red]")
        raise typer.Exit(1)
    for m in models:
        console.print(f"[bold green]Computing embeddings: {m}[/bold green]")
        # TODO: Sprint 1B — invoke embedding pipeline
    console.print("[yellow]Not yet implemented.[/yellow]")


# ---------------------------------------------------------------------------
# train sub-commands
# ---------------------------------------------------------------------------

CLASSIFIERS = ["logreg", "svm", "xgboost", "mlp"]
EXPERIMENTS = ["standard_cv", "loato", "direct_indirect", "crosslingual"]


@train_app.command("run")
def train_run(
    embedding: Optional[str] = typer.Option(None, help="Embedding model name."),
    classifier: Optional[str] = typer.Option(None, help="Classifier name."),
    experiment: str = typer.Option("standard_cv", help="Experiment type."),
    all_combos: bool = typer.Option(False, "--all", help="Run all embedding×classifier combos."),
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
    """Entry point called by `promptguard` CLI."""
    app()
