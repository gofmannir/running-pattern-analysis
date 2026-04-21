"""Experiment runner for systematic augmentation studies.

Orchestrates augmentation generation, training, and result collection
for all experiments in the suite.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from rpa.experiment.config import (
    EXPERIMENT_PRESETS,
    AugmentationPreset,
    ExperimentResult,
    ExperimentStatus,
    ExperimentSuiteConfig,
)
from rpa.experiment.results import (
    ResultsTracker,
    gcs_download_to_temp,
    gcs_list_blobs,
    gcs_read_text,
    gcs_upload_dir,
    gcs_write_text,
    is_gcs_path,
)
from rpa.train import TrainingMetrics


@dataclass
class SuiteCheckpoint:
    """Checkpoint for experiment suite progress."""

    suite_id: str
    started_at: str = ""
    last_updated: str = ""
    augmented_experiments: list[str] = field(default_factory=list)
    trained_experiments: list[str] = field(default_factory=list)
    current_experiment: str | None = None
    current_phase: str | None = None  # "augmenting" or "training"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "suite_id": self.suite_id,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "augmented_experiments": self.augmented_experiments,
            "trained_experiments": self.trained_experiments,
            "current_experiment": self.current_experiment,
            "current_phase": self.current_phase,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SuiteCheckpoint:
        """Create from dictionary."""
        return cls(
            suite_id=data.get("suite_id", ""),
            started_at=data.get("started_at", ""),
            last_updated=data.get("last_updated", ""),
            augmented_experiments=data.get("augmented_experiments", []),
            trained_experiments=data.get("trained_experiments", []),
            current_experiment=data.get("current_experiment"),
            current_phase=data.get("current_phase"),
        )


class ExperimentRunner:
    """Orchestrates the full experiment suite."""

    def __init__(
        self,
        config: ExperimentSuiteConfig,
        presets: list[AugmentationPreset] | None = None,
    ) -> None:
        """Initialize experiment runner.

        Args:
            config: Suite configuration.
            presets: List of experiments to run. If None, uses all EXPERIMENT_PRESETS.
        """
        self.config = config
        self.presets = presets or EXPERIMENT_PRESETS
        self.checkpoint = SuiteCheckpoint(suite_id=config.suite_id)
        self.results_tracker = ResultsTracker(self._results_path())

        # Initialize results for all experiments
        for preset in self.presets:
            result = ExperimentResult(
                experiment_id=preset.name,
                augmentations=[a.value for a in preset.enabled_augmentations],
                augmentation_description=preset.description,
                versions_per_video=config.versions_per_video,
            )
            self.results_tracker.add_result(result)

    def _base_path(self) -> str:
        """Get base path for suite data in GCS or local."""
        bucket = self.config.gcs_bucket.rstrip("/")
        return f"{bucket}/experiments/{self.config.suite_id}"

    def _checkpoint_path(self) -> str:
        """Get checkpoint file path."""
        return f"{self._base_path()}/checkpoint.json"

    def _results_path(self) -> str:
        """Get results file path (without extension)."""
        return f"{self._base_path()}/results"

    def _experiment_path(self, experiment_id: str) -> str:
        """Get path for a specific experiment."""
        return f"{self._base_path()}/{experiment_id}"

    def _augmented_path(self, experiment_id: str) -> str:
        """Get path for augmented data."""
        return f"{self._experiment_path(experiment_id)}/augmented"

    def _model_path(self, experiment_id: str) -> str:
        """Get path for trained model."""
        return f"{self._experiment_path(experiment_id)}/model"

    def _split_path(self, experiment_id: str) -> str:
        """Get path for experiment-specific split JSON."""
        return f"{self._experiment_path(experiment_id)}/split.json"

    def load_checkpoint(self) -> bool:
        """Load checkpoint from storage.

        Returns:
            True if checkpoint was loaded.
        """
        checkpoint_path = self._checkpoint_path()

        try:
            if is_gcs_path(checkpoint_path):
                content = gcs_read_text(checkpoint_path)
                if content is None:
                    return False
            else:
                path = Path(checkpoint_path)
                if not path.exists():
                    return False
                content = path.read_text()

            data = json.loads(content)
            self.checkpoint = SuiteCheckpoint.from_dict(data)

            # Also load results
            self.results_tracker.load()

            logger.info(
                "Loaded checkpoint: {augmented} augmented, {trained} trained",
                augmented=len(self.checkpoint.augmented_experiments),
                trained=len(self.checkpoint.trained_experiments),
            )
            return True
        except Exception as e:
            logger.warning("Failed to load checkpoint: {err}", err=e)
            return False

    def save_checkpoint(self) -> None:
        """Save checkpoint to storage with merge to prevent race conditions.

        When running multiple trainings in parallel, this reads the current
        checkpoint first and merges NEW experiments from GCS (not from memory).
        This prevents re-adding experiments that were intentionally removed.
        """
        checkpoint_path = self._checkpoint_path()

        # Read current checkpoint from storage and merge
        if is_gcs_path(checkpoint_path):
            existing_content = gcs_read_text(checkpoint_path)
            if existing_content:
                try:
                    existing = json.loads(existing_content)
                    # Only ADD experiments from GCS that we don't have
                    # Don't re-add our old experiments to GCS (they may have been removed)
                    existing_trained = set(existing.get("trained_experiments", []))
                    current_trained = set(self.checkpoint.trained_experiments)
                    # Add any NEW experiments from GCS to our list
                    new_from_gcs = existing_trained - current_trained
                    if new_from_gcs:
                        merged_trained = sorted(current_trained | new_from_gcs)
                        self.checkpoint.trained_experiments = merged_trained
                        logger.debug("Added {n} trained from GCS: {exps}", n=len(new_from_gcs), exps=new_from_gcs)

                    # Same for augmented
                    existing_aug = set(existing.get("augmented_experiments", []))
                    current_aug = set(self.checkpoint.augmented_experiments)
                    new_aug_from_gcs = existing_aug - current_aug
                    if new_aug_from_gcs:
                        merged_aug = sorted(current_aug | new_aug_from_gcs)
                        self.checkpoint.augmented_experiments = merged_aug

                except json.JSONDecodeError:
                    pass  # Use current checkpoint as-is

        self.checkpoint.last_updated = datetime.now(UTC).isoformat()
        content = json.dumps(self.checkpoint.to_dict(), indent=2)

        if is_gcs_path(checkpoint_path):
            gcs_write_text(content, checkpoint_path)
        else:
            path = Path(checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

        logger.debug("Saved checkpoint to {path}", path=checkpoint_path)

    def run_augmentation_phase(self, experiment_ids: list[str] | None = None) -> None:
        """Run augmentation phase for all (or specified) experiments.

        Args:
            experiment_ids: Optional list of specific experiments to augment.
                          If None, augments all pending experiments.
        """
        if not self.checkpoint.started_at:
            self.checkpoint.started_at = datetime.now(UTC).isoformat()

        presets_to_run = self.presets
        if experiment_ids:
            presets_to_run = [p for p in self.presets if p.name in experiment_ids]

        total = len(presets_to_run)
        for i, preset in enumerate(presets_to_run, 1):
            if preset.name in self.checkpoint.augmented_experiments:
                logger.info(
                    "[{i}/{total}] Skipping already augmented: {name}",
                    i=i,
                    total=total,
                    name=preset.name,
                )
                continue

            logger.info("=" * 60)
            logger.info(
                "[{i}/{total}] Augmenting: {name} ({desc})",
                i=i,
                total=total,
                name=preset.name,
                desc=preset.description,
            )
            logger.info("=" * 60)

            self.checkpoint.current_experiment = preset.name
            self.checkpoint.current_phase = "augmenting"
            self.save_checkpoint()

            try:
                self._run_augmentation(preset)
                self.checkpoint.augmented_experiments.append(preset.name)

                # Update result status
                result = self.results_tracker.get_result(preset.name)
                if result:
                    result.status = ExperimentStatus.AUGMENTED
                    result.dataset_path = self._augmented_path(preset.name)
                    self.results_tracker.add_result(result)

                self.save_checkpoint()
                self.results_tracker.save()

                logger.success("Augmentation complete for: {name}", name=preset.name)

            except Exception as e:
                logger.error("Augmentation failed for {name}: {err}", name=preset.name, err=e)
                result = self.results_tracker.get_result(preset.name)
                if result:
                    result.status = ExperimentStatus.FAILED
                    result.error_message = str(e)
                    self.results_tracker.add_result(result)
                self.results_tracker.save()

        self.checkpoint.current_experiment = None
        self.checkpoint.current_phase = None
        self.save_checkpoint()

        logger.info("=" * 60)
        logger.info("AUGMENTATION PHASE COMPLETE")
        logger.info(
            "Augmented: {n}/{total} experiments",
            n=len(self.checkpoint.augmented_experiments),
            total=total,
        )
        logger.info("=" * 60)

    def _list_gcs_videos(self, gcs_path: str) -> list[str]:
        """List all video files in a GCS path.

        Args:
            gcs_path: GCS path (gs://bucket/path/).

        Returns:
            List of video file paths.
        """
        import subprocess

        gcs_path = gcs_path.rstrip("/") + "/"
        result = subprocess.run(
            ["gcloud", "storage", "ls", gcs_path],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            logger.error("Failed to list GCS path {path}: {err}", path=gcs_path, err=result.stderr)
            return []

        videos = [
            line.strip()
            for line in result.stdout.strip().split("\n")
            if line.strip().endswith(".mp4")
        ]
        logger.info("Found {n} videos in {path}", n=len(videos), path=gcs_path)
        return videos

    def _run_augmentation(self, preset: AugmentationPreset) -> None:
        """Run augmentation for a single experiment.

        Args:
            preset: The experiment preset.
        """
        from rpa.augment import run_augmentation

        output_path = self._augmented_path(preset.name)
        checkpoint_path = f"{output_path}/.checkpoint.json"

        # Get video paths - either from raw_input_path or base_split
        if self.config.raw_input_path:
            train_videos = self._list_gcs_videos(self.config.raw_input_path)
        else:
            split_data = self._load_base_split()
            train_videos = split_data.get("train", [])

        if not train_videos:
            logger.error("No videos found (raw_input_path={raw}, base_split={split})",
                        raw=self.config.raw_input_path, split=self.config.base_split_json)
            return

        # Count original samples
        result = self.results_tracker.get_result(preset.name)
        if result:
            result.original_train_count = len(train_videos)
            result.augmented_train_count = len(train_videos) * self.config.versions_per_video
            self.results_tracker.add_result(result)

        # Create selective config for this experiment
        # For baseline (no augmentations), all probs are 0.0 (grayscale only)
        augment_config = preset.to_augment_config(self.config.versions_per_video)

        logger.info(
            "Augment config: flip={flip}, bc={bc}, blur={blur}, rot={rot}, scale={scale}, cutout={cutout}",
            flip=augment_config.flip_prob,
            bc=augment_config.brightness_contrast_prob,
            blur=augment_config.blur_prob,
            rot=augment_config.rotation_prob,
            scale=augment_config.scale_prob,
            cutout=augment_config.cutout_prob,
        )

        # Run augmentation on training videos only
        # Pass the specific video paths to process
        # Use configured workers, default to 8 if not set
        workers = self.config.num_workers if self.config.num_workers > 0 else 8

        results = run_augmentation(
            input_path=train_videos[0] if train_videos else "",  # Used for GCS mode detection
            output_path=output_path,
            config=augment_config,
            workers=workers,
            checkpoint_path=checkpoint_path,
            video_paths=train_videos,
        )

        logger.info(
            "Augmentation complete: {n} videos processed, {total} versions created",
            n=len(results),
            total=len(results) * self.config.versions_per_video,
        )

    def run_training_phase(
        self,
        experiment_ids: list[str] | None = None,
        epochs: int | None = None,
        force: bool = False,
    ) -> None:
        """Run training phase for all (or specified) experiments.

        Args:
            experiment_ids: Optional list of specific experiments to train.
                          If None, trains all augmented experiments.
            epochs: Override epochs from config.
            force: If True, re-train even if already in trained_experiments.
        """
        epochs = epochs or self.config.epochs

        # Only train experiments that have been augmented but not yet trained
        # Unless force=True, then allow re-training
        if force and experiment_ids:
            # Force mode: train specified experiments regardless of trained status
            trainable = [
                p
                for p in self.presets
                if p.name in self.checkpoint.augmented_experiments
                and p.name in experiment_ids
            ]
            # Remove from trained list so it can be re-added after training
            for exp_id in experiment_ids:
                if exp_id in self.checkpoint.trained_experiments:
                    self.checkpoint.trained_experiments.remove(exp_id)
                    logger.info("Force mode: removed {exp} from trained list", exp=exp_id)
        else:
            trainable = [
                p
                for p in self.presets
                if p.name in self.checkpoint.augmented_experiments
                and p.name not in self.checkpoint.trained_experiments
            ]
            if experiment_ids:
                trainable = [p for p in trainable if p.name in experiment_ids]

        if not trainable:
            logger.warning("No experiments ready for training")
            return

        total = len(trainable)
        for i, preset in enumerate(trainable, 1):
            logger.info("=" * 60)
            logger.info(
                "[{i}/{total}] Training: {name} ({desc})",
                i=i,
                total=total,
                name=preset.name,
                desc=preset.description,
            )
            logger.info("=" * 60)

            self.checkpoint.current_experiment = preset.name
            self.checkpoint.current_phase = "training"
            self.save_checkpoint()

            try:
                metrics = self._run_training(preset, epochs)

                self.checkpoint.trained_experiments.append(preset.name)

                # Update result with training metrics
                result = self.results_tracker.get_result(preset.name)
                if result and metrics:
                    result.train_acc = metrics.final_train_acc
                    result.val_acc = metrics.final_val_acc
                    result.test_acc = metrics.test_acc
                    result.train_loss = metrics.final_train_loss
                    result.val_loss = metrics.final_val_loss
                    result.test_loss = metrics.test_loss
                    result.best_val_acc = metrics.best_val_acc
                    result.status = ExperimentStatus.COMPLETED
                    result.model_path = self._model_path(preset.name)
                    self.results_tracker.add_result(result)

                self.save_checkpoint()
                self.results_tracker.save()

                logger.success("Training complete for: {name}", name=preset.name)

            except Exception as e:
                logger.error("Training failed for {name}: {err}", name=preset.name, err=e)
                result = self.results_tracker.get_result(preset.name)
                if result:
                    result.status = ExperimentStatus.FAILED
                    result.error_message = str(e)
                    self.results_tracker.add_result(result)
                self.results_tracker.save()

        self.checkpoint.current_experiment = None
        self.checkpoint.current_phase = None
        self.save_checkpoint()

        logger.info("=" * 60)
        logger.info("TRAINING PHASE COMPLETE")
        logger.info(
            "Trained: {n}/{total} experiments",
            n=len(self.checkpoint.trained_experiments),
            total=len(self.presets),
        )
        logger.info("=" * 60)

    def _run_training(self, preset: AugmentationPreset, epochs: int) -> TrainingMetrics | None:
        """Run training for a single experiment.

        Args:
            preset: The experiment preset.
            epochs: Number of epochs.

        Returns:
            TrainingMetrics from training, or None on failure.
        """
        import tempfile

        from rpa.train import AugmentationConfig, TrainConfig, train

        # Create experiment-specific split pointing to augmented data
        split_gcs_path = self._create_experiment_split(preset.name)
        model_gcs_path = self._model_path(preset.name)

        # Download split.json to local temp file if on GCS
        if is_gcs_path(split_gcs_path):
            split_local_path = gcs_download_to_temp(split_gcs_path)
            logger.info("Downloaded split to local: {path}", path=split_local_path)
        else:
            split_local_path = Path(split_gcs_path)

        # Use local temp directory for model output, upload to GCS after
        if is_gcs_path(model_gcs_path):
            model_temp_dir = Path(tempfile.mkdtemp(prefix="rpa_model_"))
            logger.info("Using temp model dir: {path}", path=model_temp_dir)
        else:
            model_temp_dir = Path(model_gcs_path)
            model_temp_dir.mkdir(parents=True, exist_ok=True)

        # Configure training with local paths
        # Default label remapping: 2 -> 1 for binary classification
        train_config = TrainConfig(
            split_json=split_local_path,
            output_dir=model_temp_dir,
            epochs=epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            num_workers=self.config.num_workers,
            enable_augmentation=False,  # Using pre-augmented data
            augmentation=AugmentationConfig(grayscale=True),  # Keep grayscale
            label_remap={2: 1},  # Binary classification: merge class 002 into 001
        )

        start_time = time.time()
        metrics = train(train_config)
        elapsed = time.time() - start_time

        # Upload model to GCS if needed
        if is_gcs_path(model_gcs_path):
            gcs_upload_dir(model_temp_dir, model_gcs_path)
            logger.info("Uploaded model to GCS: {path}", path=model_gcs_path)
            # Clean up temp directory
            import shutil

            shutil.rmtree(model_temp_dir, ignore_errors=True)

        # Clean up temp split file if downloaded
        if is_gcs_path(split_gcs_path) and split_local_path.exists():
            split_local_path.unlink()

        # Update training time in result
        result = self.results_tracker.get_result(preset.name)
        if result:
            result.training_time_seconds = elapsed
            self.results_tracker.add_result(result)

        return metrics

    def _load_base_split(self) -> dict:
        """Load the base dataset split JSON.

        Returns:
            Split data dictionary.
        """
        split_path = self.config.base_split_json

        if is_gcs_path(split_path):
            content = gcs_read_text(split_path)
            if content is None:
                msg = f"Base split not found: {split_path}"
                raise FileNotFoundError(msg)
            return json.loads(content)
        path = Path(split_path)
        if not path.exists():
            msg = f"Base split not found: {split_path}"
            raise FileNotFoundError(msg)
        return json.loads(path.read_text())

    def _get_preset(self, experiment_id: str) -> AugmentationPreset | None:
        """Get preset by experiment ID.

        Args:
            experiment_id: The experiment ID.

        Returns:
            The preset, or None if not found.
        """
        for preset in self.presets:
            if preset.name == experiment_id:
                return preset
        return None

    def _create_experiment_split(self, experiment_id: str) -> str:
        """Create a split JSON for an experiment pointing to augmented data.

        Uses actual augmented files from GCS/local instead of generating assumed paths.
        Only includes train videos that were actually augmented.

        Args:
            experiment_id: The experiment ID.

        Returns:
            Path to the created split JSON.
        """
        import re

        base_split = self._load_base_split()
        augmented_base = self._augmented_path(experiment_id)

        # Get actual augmented files
        if is_gcs_path(augmented_base):
            all_augmented = gcs_list_blobs(augmented_base, suffix=".mp4")
        else:
            aug_dir = Path(augmented_base)
            all_augmented = [str(f) for f in aug_dir.glob("*.mp4")] if aug_dir.exists() else []

        logger.info(
            "Found {n} augmented files in {path}",
            n=len(all_augmented),
            path=augmented_base,
        )

        # Build set of base video names from augmented files (strip version)
        # Augmented: name_v001_label.mp4 -> base: name_label.mp4
        augmented_bases: dict[str, list[str]] = {}
        for aug_path in all_augmented:
            filename = aug_path.rstrip("/").split("/")[-1]
            stem = Path(filename).stem
            # Remove _vNNN_ from filename to get base name
            base_name = re.sub(r"_v\d{3}_", "_", stem)
            augmented_bases.setdefault(base_name, []).append(aug_path)

        # Map train videos to their actual augmented versions
        train_augmented = []
        missing_count = 0
        for video_path in base_split.get("train", []):
            video_name = Path(video_path).stem
            if video_name in augmented_bases:
                train_augmented.extend(sorted(augmented_bases[video_name]))
            else:
                missing_count += 1

        if missing_count > 0:
            logger.warning(
                "{n} train videos had no augmented versions (skipped)",
                n=missing_count,
            )

        # Val and test use original (non-augmented) data
        split_data = {
            "train": train_augmented,
            "val": base_split.get("val", []),
            "test": base_split.get("test", []),
            "metadata": {
                "experiment_id": experiment_id,
                "original_train_count": len(base_split.get("train", [])),
                "augmented_train_count": len(train_augmented),
                "missing_train_count": missing_count,
                "base_split": self.config.base_split_json,
            },
        }

        split_path = self._split_path(experiment_id)
        content = json.dumps(split_data, indent=2)

        if is_gcs_path(split_path):
            gcs_write_text(content, split_path)
        else:
            path = Path(split_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

        logger.info("Created experiment split: {path}", path=split_path)
        return split_path

    def list_experiments(self) -> None:
        """Print status of all experiments."""
        logger.info("=" * 70)
        logger.info("EXPERIMENT STATUS")
        logger.info("=" * 70)
        logger.info(
            "{:<25} {:>12} {:>12} {:>12}",
            "Experiment",
            "Augmented",
            "Trained",
            "Status",
        )
        logger.info("-" * 70)

        for preset in self.presets:
            is_augmented = preset.name in self.checkpoint.augmented_experiments
            is_trained = preset.name in self.checkpoint.trained_experiments
            result = self.results_tracker.get_result(preset.name)
            status = result.status.value if result else "unknown"

            logger.info(
                "{:<25} {:>12} {:>12} {:>12}",
                preset.name,
                "yes" if is_augmented else "no",
                "yes" if is_trained else "no",
                status,
            )

        logger.info("=" * 70)

    def show_results(self) -> None:
        """Display results summary."""
        self.results_tracker.print_summary()
