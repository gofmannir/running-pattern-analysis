"""CLI entry point for experiment framework.

Usage:
    # Generate augmented datasets for all experiments
    uv run python -m rpa.experiment augment \
        --bucket gs://rpa-dataset \
        --base-split dataset_split.json \
        --suite-id aug_study_v1

    # Train a specific experiment
    uv run python -m rpa.experiment train \
        --bucket gs://rpa-dataset \
        --suite-id aug_study_v1 \
        --experiment baseline

    # Train all experiments
    uv run python -m rpa.experiment train-all \
        --bucket gs://rpa-dataset \
        --suite-id aug_study_v1

    # List experiment status
    uv run python -m rpa.experiment list \
        --bucket gs://rpa-dataset \
        --suite-id aug_study_v1

    # Show results summary
    uv run python -m rpa.experiment results \
        --bucket gs://rpa-dataset \
        --suite-id aug_study_v1
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from rpa.experiment.config import (
    EXPERIMENT_PRESETS,
    ExperimentSuiteConfig,
    get_preset_by_name,
)
from rpa.experiment.runner import ExperimentRunner


def cmd_augment(args: argparse.Namespace) -> int:
    """Run augmentation phase."""
    # Validate that either raw_input or base_split is provided
    raw_input = getattr(args, "raw_input", None)
    base_split = getattr(args, "base_split", "")

    if not raw_input and not base_split:
        logger.error("Either --raw-input or --base-split must be provided")
        return 1

    config = ExperimentSuiteConfig(
        suite_id=args.suite_id,
        gcs_bucket=args.bucket,
        base_split_json=base_split,
        raw_input_path=raw_input,
        versions_per_video=args.versions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.workers,
    )

    # Filter experiments if specified
    presets = None
    if args.experiments:
        presets = []
        for name in args.experiments:
            preset = get_preset_by_name(name)
            if preset:
                presets.append(preset)
            else:
                logger.error("Unknown experiment: {name}", name=name)
                return 1

    runner = ExperimentRunner(config, presets)

    if args.resume:
        runner.load_checkpoint()

    runner.run_augmentation_phase()
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train a specific experiment."""
    config = ExperimentSuiteConfig(
        suite_id=args.suite_id,
        gcs_bucket=args.bucket,
        base_split_json=args.base_split or "",
        versions_per_video=args.versions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    runner = ExperimentRunner(config)

    # Must load checkpoint to know what's been augmented
    if not runner.load_checkpoint():
        logger.error("No checkpoint found. Run augment phase first.")
        return 1

    if args.experiment not in runner.checkpoint.augmented_experiments:
        logger.error(
            "Experiment {name} has not been augmented. Run augment phase first.",
            name=args.experiment,
        )
        return 1

    force = getattr(args, "force", False)
    runner.run_training_phase(experiment_ids=[args.experiment], epochs=args.epochs, force=force)
    return 0


def cmd_train_all(args: argparse.Namespace) -> int:
    """Train all augmented experiments."""
    config = ExperimentSuiteConfig(
        suite_id=args.suite_id,
        gcs_bucket=args.bucket,
        base_split_json=args.base_split or "",
        versions_per_video=args.versions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    runner = ExperimentRunner(config)

    # Must load checkpoint to know what's been augmented
    if not runner.load_checkpoint():
        logger.error("No checkpoint found. Run augment phase first.")
        return 1

    runner.run_training_phase(epochs=args.epochs)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List experiment status."""
    config = ExperimentSuiteConfig(
        suite_id=args.suite_id,
        gcs_bucket=args.bucket,
        base_split_json="",
    )

    runner = ExperimentRunner(config)
    runner.load_checkpoint()
    runner.list_experiments()
    return 0


def cmd_results(args: argparse.Namespace) -> int:
    """Show results summary."""
    config = ExperimentSuiteConfig(
        suite_id=args.suite_id,
        gcs_bucket=args.bucket,
        base_split_json="",
    )

    runner = ExperimentRunner(config)
    if not runner.results_tracker.load():
        logger.error("No results found for suite: {id}", id=args.suite_id)
        return 1

    runner.show_results()

    # Optionally save to local file
    if args.output:
        from rpa.experiment.results import ResultsTracker

        local_tracker = ResultsTracker(args.output)
        local_tracker.results = runner.results_tracker.results
        local_tracker.save()
        logger.info("Results saved to: {path}.[csv|json]", path=args.output)

    return 0


def cmd_list_presets(_args: argparse.Namespace) -> int:
    """List all available experiment presets."""
    logger.info("=" * 70)
    logger.info("AVAILABLE EXPERIMENT PRESETS ({n} total)", n=len(EXPERIMENT_PRESETS))
    logger.info("=" * 70)
    logger.info("{:<5} {:<25} {}", "#", "Name", "Augmentations")
    logger.info("-" * 70)

    for i, preset in enumerate(EXPERIMENT_PRESETS, 1):
        aug_str = ", ".join(a.value for a in preset.enabled_augmentations) or "none"
        logger.info("{:<5} {:<25} {}", i, preset.name, aug_str)

    logger.info("=" * 70)
    return 0


def cmd_augment_status(args: argparse.Namespace) -> int:
    """Check augmentation status."""
    config = ExperimentSuiteConfig(
        suite_id=args.suite_id,
        gcs_bucket=args.bucket,
        base_split_json="",
    )

    runner = ExperimentRunner(config)
    if not runner.load_checkpoint():
        logger.info("No checkpoint found for suite: {id}", id=args.suite_id)
        return 0

    total = len(runner.presets)
    augmented = len(runner.checkpoint.augmented_experiments)

    logger.info("=" * 60)
    logger.info("AUGMENTATION STATUS: {id}", id=args.suite_id)
    logger.info("=" * 60)
    logger.info("Progress: {done}/{total} experiments augmented", done=augmented, total=total)
    logger.info("Started: {time}", time=runner.checkpoint.started_at or "N/A")
    logger.info("Last update: {time}", time=runner.checkpoint.last_updated or "N/A")

    if runner.checkpoint.current_experiment:
        logger.info(
            "Current: {name} ({phase})",
            name=runner.checkpoint.current_experiment,
            phase=runner.checkpoint.current_phase,
        )

    logger.info("-" * 60)
    logger.info("Completed experiments:")
    for name in runner.checkpoint.augmented_experiments:
        logger.info("  - {name}", name=name)

    remaining = [p.name for p in runner.presets if p.name not in runner.checkpoint.augmented_experiments]
    if remaining:
        logger.info("-" * 60)
        logger.info("Remaining experiments:")
        for name in remaining:
            logger.info("  - {name}", name=name)

    logger.info("=" * 60)
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """Show quick summary."""
    config = ExperimentSuiteConfig(
        suite_id=args.suite_id,
        gcs_bucket=args.bucket,
        base_split_json="",
    )

    runner = ExperimentRunner(config)
    runner.load_checkpoint()
    runner.results_tracker.load()

    total = len(runner.presets)
    augmented = len(runner.checkpoint.augmented_experiments)
    trained = len(runner.checkpoint.trained_experiments)

    logger.info("Suite: {id}", id=args.suite_id)
    logger.info("Augmented: {n}/{t}", n=augmented, t=total)
    logger.info("Trained: {n}/{t}", n=trained, t=total)

    runner.results_tracker.print_summary()
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Systematic augmentation experiment framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments
    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--bucket",
            type=str,
            required=True,
            help="GCS bucket for data and results (gs://...)",
        )
        p.add_argument(
            "--suite-id",
            type=str,
            required=True,
            help="Unique ID for this experiment suite",
        )

    def add_training_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--versions",
            type=int,
            default=25,
            help="Augmented versions per video (default: 25)",
        )
        p.add_argument(
            "--epochs",
            type=int,
            default=10,
            help="Training epochs per experiment (default: 10)",
        )
        p.add_argument(
            "--batch-size",
            type=int,
            default=8,
            help="Training batch size (default: 8)",
        )
        p.add_argument(
            "--lr",
            type=float,
            default=5e-5,
            help="Learning rate (default: 5e-5)",
        )

    # augment command
    augment_parser = subparsers.add_parser("augment", help="Generate augmented datasets")
    add_common_args(augment_parser)
    add_training_args(augment_parser)
    augment_parser.add_argument(
        "--raw-input",
        type=str,
        help="GCS path to raw videos folder (e.g., gs://bucket/raw/)",
    )
    augment_parser.add_argument(
        "--base-split",
        type=str,
        default="",
        help="Path to base dataset_split.json (only needed if --raw-input not provided)",
    )
    augment_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    augment_parser.add_argument(
        "--experiments",
        type=str,
        nargs="*",
        help="Specific experiment IDs to augment (default: all)",
    )
    augment_parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for augmentation (default: 8)",
    )
    augment_parser.set_defaults(func=cmd_augment)

    # augment-status command
    augment_status_parser = subparsers.add_parser("augment-status", help="Check augmentation status")
    add_common_args(augment_status_parser)
    augment_status_parser.set_defaults(func=cmd_augment_status)

    # train command
    train_parser = subparsers.add_parser("train", help="Train a specific experiment")
    add_common_args(train_parser)
    add_training_args(train_parser)
    train_parser.add_argument(
        "--base-split",
        type=str,
        help="Path to base dataset_split.json (optional, uses checkpoint)",
    )
    train_parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment ID to train",
    )
    train_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-training even if already trained",
    )
    train_parser.set_defaults(func=cmd_train)

    # train-all command
    train_all_parser = subparsers.add_parser("train-all", help="Train all augmented experiments")
    add_common_args(train_all_parser)
    add_training_args(train_all_parser)
    train_all_parser.add_argument(
        "--base-split",
        type=str,
        help="Path to base dataset_split.json (optional, uses checkpoint)",
    )
    train_all_parser.set_defaults(func=cmd_train_all)

    # list command
    list_parser = subparsers.add_parser("list", help="List experiment status")
    add_common_args(list_parser)
    list_parser.set_defaults(func=cmd_list)

    # results command
    results_parser = subparsers.add_parser("results", help="Show results summary")
    add_common_args(results_parser)
    results_parser.add_argument(
        "--output",
        type=str,
        help="Save results to local path (without extension)",
    )
    results_parser.set_defaults(func=cmd_results)

    # summary command
    summary_parser = subparsers.add_parser("summary", help="Show quick summary")
    add_common_args(summary_parser)
    summary_parser.set_defaults(func=cmd_summary)

    # list-presets command
    list_presets_parser = subparsers.add_parser("list-presets", help="List available experiment presets")
    list_presets_parser.set_defaults(func=cmd_list_presets)

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
