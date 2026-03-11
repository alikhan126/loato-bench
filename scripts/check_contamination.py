#!/usr/bin/env python3
"""LOATO-2A-05: Run contamination checks on all materialized splits.

Usage:
    uv run python scripts/check_contamination.py

Outputs:
    data/splits/contamination_report.json
    data/splits/contamination_flags.csv
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import sys

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loato_bench.data.contamination import check_all_splits  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run contamination checks and write outputs."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    if not splits_dir.exists():
        logger.error("Splits directory not found: %s", splits_dir)
        sys.exit(1)

    logger.info("Starting contamination checks on %s", splits_dir)
    report_entries, flags_df = check_all_splits(
        splits_dir,
        jaccard_threshold=0.8,
        cosine_threshold=0.95,
        k=5,
    )

    if not report_entries:
        logger.error("No split pairs found — nothing to check.")
        sys.exit(1)

    # Write contamination_report.json
    overall_pass = all(e["pass"] for e in report_entries)
    report = {
        "version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "jaccard_threshold": 0.8,
        "cosine_threshold": 0.95,
        "k_neighbors": 5,
        "remediation_threshold_pct": 1.0,
        "overall_pass": overall_pass,
        "n_splits_checked": len(report_entries),
        "splits": report_entries,
    }

    report_path = splits_dir / "contamination_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    logger.info("Wrote contamination report: %s", report_path)

    # Write contamination_flags.csv
    flags_path = splits_dir / "contamination_flags.csv"
    flags_df.to_csv(flags_path, index=False)
    logger.info("Wrote contamination flags: %s (%d rows)", flags_path, len(flags_df))

    # Summary
    logger.info("=" * 60)
    logger.info("CONTAMINATION CHECK SUMMARY")
    logger.info("=" * 60)
    for entry in report_entries:
        status = "PASS" if entry["pass"] else "FAIL"
        logger.info(
            "  %-25s  lex=%d  sem=%d  rate=%.4f%%  [%s]",
            entry["split_id"],
            entry["lexical_flagged"],
            entry["semantic_flagged"],
            entry["flagged_rate_pct"],
            status,
        )
    logger.info("=" * 60)
    logger.info("Overall: %s", "PASS" if overall_pass else "FAIL")

    if not overall_pass:
        logger.warning(
            "Some splits exceed 1%% contamination — re-examine before proceeding to Sprint 2B."
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
