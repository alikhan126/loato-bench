"""LOATO-Bench CLI — Typer entrypoints for all pipeline stages."""

from __future__ import annotations

import logging

from rich.console import Console
import typer

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
def label(
    confidence_threshold: float = typer.Option(0.6, help="Min confidence to apply label."),
    max_calls: int | None = typer.Option(None, help="Max API calls (None = all)."),
    dry_run: bool = typer.Option(False, help="Preview only, no API calls."),
    output_dir: str | None = typer.Option(None, help="Output dir for labeling artifacts."),
    concurrency: int = typer.Option(8, help="Max concurrent API requests."),
    model: str | None = typer.Option(None, help="Override model (e.g. gpt-4.1-mini)."),
) -> None:
    """Label unlabeled injection samples using an OpenAI model."""
    import json
    from pathlib import Path

    import pandas as pd

    from loato_bench.data.harmonize import filter_gentel_samples
    from loato_bench.data.llm_labeler import label_samples, validate_distribution
    from loato_bench.data.taxonomy import apply_taxonomy_mapping
    from loato_bench.data.taxonomy_spec import OLD_SLUG_TO_NEW

    console.print("[bold green]Labeling unlabeled injection samples...[/bold green]")

    # 1. Load unified dataset
    parquet_path = DATA_DIR / "processed" / "unified_dataset.parquet"
    if not parquet_path.exists():
        console.print("[red]No processed data. Run 'loato-bench data harmonize' first.[/red]")
        raise typer.Exit(1)

    df = pd.read_parquet(parquet_path)
    console.print(f"  Loaded {len(df):,} samples")

    # 2. Apply GenTel filtering
    before = len(df)
    df = filter_gentel_samples(df)
    console.print(f"  GenTel filtering: {before:,} -> {len(df):,}")

    # 3. Apply Tier 1+2 taxonomy mapping
    df = apply_taxonomy_mapping(df, apply_tier3=False)
    total_inj = (df["label"] == 1).sum()
    mapped = df.loc[df["label"] == 1, "attack_category"].notna().sum()
    console.print(f"  Tier 1+2 mapping: {mapped}/{total_inj} injection samples mapped")

    # 4. Migrate old slugs
    mask = df["attack_category"].notna()
    df.loc[mask, "attack_category"] = df.loc[mask, "attack_category"].map(
        lambda x: OLD_SLUG_TO_NEW.get(x, x)
    )

    # 5. Run LLM labeling
    out_path = Path(output_dir) if output_dir else None
    df = label_samples(
        df,
        confidence_threshold=confidence_threshold,
        max_calls=max_calls,
        output_dir=out_path,
        dry_run=dry_run,
        concurrency=concurrency,
        model_override=model,
    )

    # 6. Validate distribution
    report = validate_distribution(df)
    if report["warnings"]:
        for w in report["warnings"]:
            console.print(f"  [yellow]Warning: {w}[/yellow]")
    else:
        console.print("  Distribution check passed.")

    # 7. Save labeled dataset
    labeled_path = DATA_DIR / "processed" / "labeled_v1.parquet"
    df.to_parquet(labeled_path, index=False)
    console.print(f"  Saved labeled dataset to {labeled_path}")

    # 8. Save coverage report
    labeling_dir = Path(output_dir) if output_dir else DATA_DIR / "labeling"
    labeling_dir.mkdir(parents=True, exist_ok=True)
    total_inj_final = (df["label"] == 1).sum()
    labeled_final = df.loc[df["label"] == 1, "attack_category"].notna().sum()
    coverage_pct = labeled_final / total_inj_final * 100 if total_inj_final > 0 else 0.0
    uncertain_count = (
        (df["label_source"] == "uncertain").sum() if "label_source" in df.columns else 0
    )

    coverage_report = {
        "total_injection": int(total_inj_final),
        "labeled": int(labeled_final),
        "coverage_pct": round(float(coverage_pct), 2),
        "uncertain_pool": int(uncertain_count),
        "category_counts": report.get("category_counts", {}),
    }
    coverage_path = labeling_dir / "coverage_report.json"
    with open(coverage_path, "w") as f:
        json.dump(coverage_report, f, indent=2)
    console.print(f"  Saved coverage report to {coverage_path}")

    # 9. Summary
    console.print("\n[bold green]Labeling complete![/bold green]")
    console.print(f"  Coverage: {labeled_final}/{total_inj_final} ({coverage_pct:.1f}%)")
    if report["category_counts"]:
        console.print(f"  LLM labels: {report['category_counts']}")


@data_app.command("label-batch")
def label_batch(
    model: str = typer.Option("gpt-4o", help="Model for batch labeling."),
    confidence_threshold: float = typer.Option(0.6, help="Min confidence to apply label."),
    output_dir: str | None = typer.Option(None, help="Output dir for labeling artifacts."),
    poll_interval: int = typer.Option(30, help="Seconds between batch status polls."),
) -> None:
    """Label unlabeled samples via OpenAI Batch API (cheaper, no rate limits)."""
    import json
    from pathlib import Path

    import pandas as pd

    from loato_bench.data.harmonize import filter_gentel_samples
    from loato_bench.data.llm_labeler import (
        create_batch_request_file,
        download_and_apply_batch_results,
        poll_batch,
        submit_batch,
        validate_distribution,
    )
    from loato_bench.data.taxonomy import apply_taxonomy_mapping
    from loato_bench.data.taxonomy_spec import OLD_SLUG_TO_NEW

    console.print(f"[bold green]Batch labeling with {model}...[/bold green]")

    # 1. Load + filter + map (same as `label`)
    parquet_path = DATA_DIR / "processed" / "unified_dataset.parquet"
    if not parquet_path.exists():
        console.print("[red]No processed data. Run 'loato-bench data harmonize' first.[/red]")
        raise typer.Exit(1)

    df = pd.read_parquet(parquet_path)
    console.print(f"  Loaded {len(df):,} samples")

    df = filter_gentel_samples(df)
    df = apply_taxonomy_mapping(df, apply_tier3=False)

    mask = df["attack_category"].notna()
    df.loc[mask, "attack_category"] = df.loc[mask, "attack_category"].map(
        lambda x: OLD_SLUG_TO_NEW.get(x, x)
    )

    # Ensure columns
    if "label_source" not in df.columns:
        df["label_source"] = pd.Series(dtype="object")
    if "confidence" not in df.columns:
        df["confidence"] = pd.Series(dtype="float64")
    has_cat = df["attack_category"].notna() & (df["label"] == 1)
    df.loc[has_cat & df["label_source"].isna(), "label_source"] = "tier1_2"

    out_path = Path(output_dir) if output_dir else DATA_DIR / "labeling"

    # 2. Create batch request file
    request_path, id_to_idx = create_batch_request_file(df, out_path, model=model)
    console.print(f"  Created {len(id_to_idx):,} batch requests")

    if not id_to_idx:
        console.print("[green]All samples already labeled![/green]")
        return

    # 3. Submit batch
    batch_id = submit_batch(request_path)
    console.print(f"  Batch submitted: {batch_id}")
    console.print(f"  Polling every {poll_interval}s...")

    # 4. Poll until done
    output_file_id = poll_batch(batch_id, poll_interval=poll_interval)
    console.print(f"  Batch complete! Output file: {output_file_id}")

    # 5. Download and apply
    df = download_and_apply_batch_results(
        output_file_id,
        df,
        out_path,
        confidence_threshold=confidence_threshold,
        model=model,
    )

    # 6. Validate + save (same as `label`)
    report = validate_distribution(df)
    if report["warnings"]:
        for w in report["warnings"]:
            console.print(f"  [yellow]Warning: {w}[/yellow]")

    labeled_path = DATA_DIR / "processed" / "labeled_v1.parquet"
    df.to_parquet(labeled_path, index=False)
    console.print(f"  Saved to {labeled_path}")

    total_inj = (df["label"] == 1).sum()
    labeled = df.loc[df["label"] == 1, "attack_category"].notna().sum()
    coverage_pct = labeled / total_inj * 100 if total_inj > 0 else 0.0

    coverage_report = {
        "total_injection": int(total_inj),
        "labeled": int(labeled),
        "coverage_pct": round(float(coverage_pct), 2),
        "category_counts": report.get("category_counts", {}),
        "model": model,
        "batch_id": batch_id,
    }
    coverage_path = out_path / "coverage_report.json"
    with open(coverage_path, "w") as f:
        json.dump(coverage_report, f, indent=2)

    console.print(
        f"\n[bold green]Done! Coverage: {labeled}/{total_inj} ({coverage_pct:.1f}%)[/bold green]"
    )
    if report["category_counts"]:
        console.print(f"  LLM labels: {report['category_counts']}")


@data_app.command()
def split(
    min_samples: int = typer.Option(150, help="Min samples per category for LOATO."),
    output_dir: str | None = typer.Option(None, help="Output directory for splits."),
) -> None:
    """Generate all evaluation splits from the labeled dataset."""
    from pathlib import Path

    import pandas as pd

    from loato_bench.data.splits import generate_all_splits

    console.print("[bold green]Generating splits...[/bold green]")

    labeled_path = DATA_DIR / "processed" / "labeled_v1.parquet"
    if not labeled_path.exists():
        console.print("[red]No labeled data. Run 'loato-bench data label' first.[/red]")
        raise typer.Exit(1)

    df = pd.read_parquet(labeled_path)
    console.print(f"  Loaded {len(df):,} samples from labeled_v1.parquet")

    total_inj = (df["label"] == 1).sum()
    mapped = df.loc[df["label"] == 1, "attack_category"].notna().sum()
    console.print(f"  Taxonomy coverage: {mapped}/{total_inj} injection samples")

    cats = df[df["label"] == 1]["attack_category"].value_counts()
    console.print(f"  Categories: {cats.to_dict()}")

    out = Path(output_dir) if output_dir else None
    saved = generate_all_splits(df, output_dir=out, min_loato_samples=min_samples)

    console.print(f"\n[bold green]Generated {len(saved)} split files:[/bold green]")
    for name, path in saved.items():
        console.print(f"  {name}: {path}")


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
