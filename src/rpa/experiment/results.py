"""Results tracking and export for augmentation experiments.

Handles saving/loading experiment results to/from CSV and JSON formats,
with support for both local files and GCS.
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path

from loguru import logger

from rpa.experiment.config import ExperimentResult, ExperimentStatus


def is_gcs_path(path: str) -> bool:
    """Check if a path is a GCS path."""
    return path.startswith("gs://")


def gcs_write_text(content: str, gcs_path: str) -> None:
    """Write text content to GCS.

    Args:
        content: Text content to write.
        gcs_path: GCS path (gs://bucket/path/to/file).
    """
    from google.cloud import storage  # type: ignore[import-untyped]

    # Parse GCS path
    parts = gcs_path[5:].split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1]

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_string(content)
    logger.debug("Wrote to GCS: {path}", path=gcs_path)


def gcs_read_text(gcs_path: str) -> str | None:
    """Read text content from GCS.

    Args:
        gcs_path: GCS path (gs://bucket/path/to/file).

    Returns:
        File content as string, or None if not found.
    """
    from google.cloud import storage  # type: ignore[import-untyped]

    # Parse GCS path
    parts = gcs_path[5:].split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1]

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.download_as_text()
    except Exception:
        return None


def gcs_exists(gcs_path: str) -> bool:
    """Check if a GCS path exists.

    Args:
        gcs_path: GCS path (gs://bucket/path/to/file).

    Returns:
        True if the file exists.
    """
    from google.cloud import storage  # type: ignore[import-untyped]

    parts = gcs_path[5:].split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1]

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.exists()
    except Exception:
        return False


def gcs_download_to_temp(gcs_path: str) -> Path:
    """Download a GCS file to a local temp file.

    Args:
        gcs_path: GCS path (gs://bucket/path/to/file).

    Returns:
        Path to the local temp file.

    Raises:
        RuntimeError: If download fails.
    """
    from google.cloud import storage  # type: ignore[import-untyped]

    parts = gcs_path[5:].split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1]

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Create temp file with same suffix
        suffix = Path(blob_path).suffix or ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = Path(tmp_file.name)

        blob.download_to_filename(str(tmp_path))
        logger.debug("Downloaded {gcs} to {local}", gcs=gcs_path, local=tmp_path)
        return tmp_path
    except Exception as e:
        msg = f"Failed to download {gcs_path}: {e}"
        raise RuntimeError(msg) from e


def gcs_upload_dir(local_dir: Path, gcs_base_path: str) -> None:
    """Upload a local directory to GCS.

    Args:
        local_dir: Local directory to upload.
        gcs_base_path: GCS path prefix (gs://bucket/path/to/dir).
    """
    from google.cloud import storage  # type: ignore[import-untyped]

    parts = gcs_base_path[5:].split("/", 1)
    bucket_name = parts[0]
    base_blob_path = parts[1] if len(parts) > 1 else ""

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    count = 0
    for local_file in local_dir.rglob("*"):
        if local_file.is_file():
            relative_path = local_file.relative_to(local_dir)
            blob_path = f"{base_blob_path}/{relative_path}" if base_blob_path else str(relative_path)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(local_file))
            count += 1

    logger.info("Uploaded {n} files from {local} to {gcs}", n=count, local=local_dir, gcs=gcs_base_path)


def gcs_list_blobs(gcs_prefix: str, suffix: str = "") -> list[str]:
    """List GCS blob paths under a prefix.

    Args:
        gcs_prefix: GCS path prefix (gs://bucket/path/to/dir).
        suffix: Optional suffix filter (e.g. ".mp4").

    Returns:
        List of full GCS paths matching the prefix.
    """
    from google.cloud import storage  # type: ignore[import-untyped]

    parts = gcs_prefix[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    results = []
    for blob in blobs:
        if suffix and not blob.name.endswith(suffix):
            continue
        results.append(f"gs://{bucket_name}/{blob.name}")

    logger.debug("Listed {n} blobs under {prefix}", n=len(results), prefix=gcs_prefix)
    return results


class ResultsTracker:
    """Track and export experiment results.

    Supports saving results to both CSV and JSON formats,
    with support for local files and GCS paths.
    """

    def __init__(self, base_path: str) -> None:
        """Initialize results tracker.

        Args:
            base_path: Base path for results files (without extension).
                      Can be local path or GCS path (gs://...).
        """
        self.base_path = base_path
        self.results: list[ExperimentResult] = []

    def add_result(self, result: ExperimentResult) -> None:
        """Add or update a result.

        If a result with the same experiment_id exists, it will be updated.

        Args:
            result: The experiment result to add/update.
        """
        # Update existing or append
        for i, r in enumerate(self.results):
            if r.experiment_id == result.experiment_id:
                self.results[i] = result
                logger.debug(
                    "Updated result for experiment: {id}",
                    id=result.experiment_id,
                )
                return

        self.results.append(result)
        logger.debug("Added result for experiment: {id}", id=result.experiment_id)

    def get_result(self, experiment_id: str) -> ExperimentResult | None:
        """Get a result by experiment ID.

        Args:
            experiment_id: The experiment ID to look up.

        Returns:
            The experiment result, or None if not found.
        """
        for result in self.results:
            if result.experiment_id == experiment_id:
                return result
        return None

    def to_csv(self) -> str:
        """Export results to CSV format.

        Returns:
            CSV content as string.
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        headers = [
            "experiment_id",
            "augmentations",
            "original_samples",
            "augmented_samples",
            "versions_per_video",
            "train_acc",
            "val_acc",
            "test_acc",
            "best_val_acc",
            "train_loss",
            "val_loss",
            "test_loss",
            "training_time_s",
            "status",
        ]
        writer.writerow(headers)

        # Data rows
        for r in self.results:
            aug_str = "+".join(r.augmentations) if r.augmentations else "none"
            row = [
                r.experiment_id,
                aug_str,
                r.original_train_count,
                r.augmented_train_count,
                r.versions_per_video,
                f"{r.train_acc:.2f}" if r.train_acc is not None else "",
                f"{r.val_acc:.2f}" if r.val_acc is not None else "",
                f"{r.test_acc:.2f}" if r.test_acc is not None else "",
                f"{r.best_val_acc:.2f}" if r.best_val_acc is not None else "",
                f"{r.train_loss:.4f}" if r.train_loss is not None else "",
                f"{r.val_loss:.4f}" if r.val_loss is not None else "",
                f"{r.test_loss:.4f}" if r.test_loss is not None else "",
                f"{r.training_time_seconds:.0f}" if r.training_time_seconds is not None else "",
                r.status.value,
            ]
            writer.writerow(row)

        return output.getvalue()

    def to_json(self) -> str:
        """Export results to JSON format.

        Returns:
            JSON content as string.
        """
        data = {
            "results": [r.model_dump() for r in self.results],
            "summary": {
                "total_experiments": len(self.results),
                "completed": len([r for r in self.results if r.status == ExperimentStatus.COMPLETED]),
                "pending": len([r for r in self.results if r.status == ExperimentStatus.PENDING]),
                "failed": len([r for r in self.results if r.status == ExperimentStatus.FAILED]),
            },
        }
        return json.dumps(data, indent=2, default=str)

    def save(self) -> None:
        """Save results to CSV and JSON files with merge for parallel safety.

        When running multiple trainings in parallel, this reads the current
        results from GCS and only adds NEW completed experiments from GCS.
        This prevents re-adding results that were intentionally reset.
        """
        json_path = f"{self.base_path}.json"
        csv_path = f"{self.base_path}.csv"

        # Read and merge with existing results (for parallel training safety)
        if is_gcs_path(self.base_path):
            existing_content = gcs_read_text(json_path)
            if existing_content:
                try:
                    existing_data = json.loads(existing_content)
                    existing_results = {
                        r["experiment_id"]: ExperimentResult(**r)
                        for r in existing_data.get("results", [])
                    }
                    current_results = {r.experiment_id: r for r in self.results}

                    # Only add NEW completed results from GCS that we don't have as completed
                    for exp_id, existing in existing_results.items():
                        current = current_results.get(exp_id)
                        if existing.status == ExperimentStatus.COMPLETED:
                            if current is None:
                                # New experiment from GCS - add it
                                current_results[exp_id] = existing
                            elif current.status != ExperimentStatus.COMPLETED:
                                # GCS has completed, we don't - add from GCS
                                current_results[exp_id] = existing
                            # else: both completed - keep ours (more recent)

                    self.results = list(current_results.values())
                    logger.debug(
                        "Merged results: {n} experiments, {c} completed",
                        n=len(self.results),
                        c=len([r for r in self.results if r.status == ExperimentStatus.COMPLETED]),
                    )
                except (json.JSONDecodeError, KeyError):
                    pass  # Use current results as-is

        csv_content = self.to_csv()
        json_content = self.to_json()

        if is_gcs_path(self.base_path):
            gcs_write_text(csv_content, csv_path)
            gcs_write_text(json_content, json_path)
            logger.info("Saved results to GCS: {path}.[csv|json]", path=self.base_path)
        else:
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            Path(csv_path).write_text(csv_content)
            Path(json_path).write_text(json_content)
            logger.info("Saved results to: {path}.[csv|json]", path=self.base_path)

    def load(self) -> bool:
        """Load results from JSON file.

        Returns:
            True if results were loaded successfully.
        """
        json_path = f"{self.base_path}.json"

        try:
            if is_gcs_path(self.base_path):
                content = gcs_read_text(json_path)
                if content is None:
                    return False
            else:
                path = Path(json_path)
                if not path.exists():
                    return False
                content = path.read_text()

            data = json.loads(content)
            loaded_results = [ExperimentResult(**r) for r in data.get("results", [])]

            # Merge loaded results with existing (loaded takes precedence for same experiment_id)
            # This allows new experiments to be added while preserving existing results
            existing_ids = {r.experiment_id for r in self.results}
            loaded_ids = {r.experiment_id for r in loaded_results}

            # Update existing results with loaded data
            for loaded in loaded_results:
                self.add_result(loaded)

            # Log what was loaded
            new_count = len(loaded_ids - existing_ids)
            updated_count = len(loaded_ids & existing_ids)
            logger.info(
                "Loaded {n} experiment results from {path} ({new} new, {updated} updated)",
                n=len(loaded_results),
                path=json_path,
                new=new_count,
                updated=updated_count,
            )
            return True
        except Exception as e:
            logger.warning("Failed to load results from {path}: {err}", path=json_path, err=e)
            return False

    def print_summary(self) -> None:
        """Print a summary of results to the console."""
        completed = [r for r in self.results if r.status == ExperimentStatus.COMPLETED]
        pending = [r for r in self.results if r.status == ExperimentStatus.PENDING]
        failed = [r for r in self.results if r.status == ExperimentStatus.FAILED]

        logger.info("=" * 70)
        logger.info("EXPERIMENT RESULTS SUMMARY")
        logger.info("=" * 70)
        logger.info(
            "Total: {total} | Completed: {comp} | Pending: {pend} | Failed: {fail}",
            total=len(self.results),
            comp=len(completed),
            pend=len(pending),
            fail=len(failed),
        )
        logger.info("-" * 70)

        if completed:
            # Sort by validation accuracy (descending)
            sorted_results = sorted(completed, key=lambda r: r.val_acc or 0, reverse=True)

            logger.info(
                "{:<25} {:>12} {:>10} {:>10} {:>10}",
                "Experiment",
                "Augmentations",
                "Train%",
                "Val%",
                "Test%",
            )
            logger.info("-" * 70)

            for r in sorted_results:
                aug_str = "+".join(r.augmentations) if r.augmentations else "none"
                if len(aug_str) > 12:
                    aug_str = aug_str[:11] + "~"
                logger.info(
                    "{:<25} {:>12} {:>10.1f} {:>10.1f} {:>10.1f}",
                    r.experiment_id,
                    aug_str,
                    r.train_acc or 0,
                    r.val_acc or 0,
                    r.test_acc or 0,
                )

        logger.info("=" * 70)
