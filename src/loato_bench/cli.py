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
def eda(
    output_dir: str = typer.Option(
        "eda",
        "--output-dir",
        "-o",
        help="Output directory for figures and reports.",
    ),
    log_wandb: bool = typer.Option(True, "--log-wandb/--no-log-wandb", help="Log to W&B."),
) -> None:
    """Run exploratory data analysis on unified dataset."""
    import json
    import os
    from pathlib import Path

    from loato_bench.analysis import eda as eda_module
    from loato_bench.analysis import quality, visualization
    from loato_bench.data import taxonomy
    from loato_bench.tracking import wandb_utils
    from loato_bench.utils.config import RESULTS_DIR

    console.print("[bold green]Running Exploratory Data Analysis...[/bold green]")

    # Resolve output directory
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = RESULTS_DIR / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load unified dataset
    console.print("📊 Loading unified dataset...")
    try:
        df = eda_module.load_unified_dataset()
        console.print(f"  Loaded {len(df):,} samples")
    except FileNotFoundError:
        console.print(
            "[red]Error: unified_dataset.parquet not found. "
            "Run 'loato-bench data harmonize' first.[/red]"
        )
        raise typer.Exit(1)

    # 2. Compute statistics
    console.print("📈 Computing dataset statistics...")
    stats = eda_module.compute_dataset_statistics(df)
    text_props = eda_module.analyze_text_properties(df)
    label_dist = eda_module.analyze_label_distribution(df)
    source_dist = eda_module.analyze_source_distribution(df)
    lang_dist = eda_module.analyze_language_distribution(df)

    # 3. Quality analysis
    console.print("🔍 Running quality analysis...")
    gentel_issues = quality.detect_gentel_quality_issues(df)
    gentel_filter = quality.recommend_gentel_filtering(df)
    integrity_warnings = quality.validate_data_integrity(df)

    # 4. Taxonomy mapping
    console.print("🗂️  Applying taxonomy mapping (Tier 1 + 2)...")
    df_mapped = taxonomy.apply_taxonomy_mapping(df)
    category_coverage = taxonomy.compute_category_coverage(df_mapped)
    merge_recommendations = taxonomy.recommend_category_merges(df_mapped)

    # 5. Create visualizations
    console.print("📊 Creating visualizations...")
    figures_dir = output_path / "figures"
    figures_dir.mkdir(exist_ok=True)
    saved_figures = visualization.create_eda_dashboard(df_mapped, figures_dir)
    console.print(f"  Created {len(saved_figures)} figures in {figures_dir}")

    # 6. Save reports
    console.print("💾 Saving reports...")
    reports = {}

    # GenTel quality report
    gentel_report_path = output_path / "gentel_quality_report.json"
    with open(gentel_report_path, "w") as f:
        json.dump({"issues": gentel_issues, "filtering": gentel_filter}, f, indent=2)
    reports["gentel_quality_report.json"] = gentel_report_path

    # Taxonomy coverage report
    taxonomy_report_path = output_path / "taxonomy_coverage.json"
    with open(taxonomy_report_path, "w") as f:
        json.dump(
            {"coverage": category_coverage, "merges": merge_recommendations},
            f,
            indent=2,
        )
    reports["taxonomy_coverage.json"] = taxonomy_report_path

    # Data integrity report
    integrity_report_path = output_path / "data_integrity.json"
    with open(integrity_report_path, "w") as f:
        json.dump({"warnings": integrity_warnings}, f, indent=2)
    reports["data_integrity.json"] = integrity_report_path

    console.print(f"  Saved {len(reports)} reports to {output_path}")

    # 7. Log to W&B if enabled
    if log_wandb and os.getenv("WANDB_API_KEY"):
        console.print("📤 Logging to W&B...")
        import wandb

        run = wandb.init(
            project="loato-bench",
            name="eda",
            tags=["eda", "sprint-1a"],
            config={
                "total_samples": stats["total_samples"],
                "num_sources": stats["num_sources"],
            },
        )

        # Combine all stats
        all_stats = {
            "dataset": stats,
            "text_properties": text_props,
            "label_distribution": label_dist,
            "source_distribution": source_dist,
            "language_distribution": lang_dist,
            "gentel_issues": gentel_issues,
            "category_coverage": category_coverage,
        }

        # Log artifacts
        figure_paths = {fig.name: fig for fig in saved_figures}
        wandb_utils.log_eda_artifacts(run, all_stats, figure_paths, reports)

        wandb_utils.finish_run(run)
        console.print("  ✓ Logged to W&B")
    elif log_wandb:
        console.print("[yellow]  ⚠ WANDB_API_KEY not set, skipping W&B logging[/yellow]")

    console.print(f"\n✅ [bold green]EDA complete! Results saved to {output_path}[/bold green]")


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
