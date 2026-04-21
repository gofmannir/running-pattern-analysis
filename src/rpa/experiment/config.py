"""Configuration for augmentation experiments.

Defines experiment presets and data models for tracking experiment results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from rpa.augment import AugmentConfig


class AugmentationType(StrEnum):
    """The 6 augmentation types to study."""

    FLIP = "flip"
    BRIGHTNESS_CONTRAST = "brightness_contrast"
    BLUR = "blur"
    ROTATION = "rotation"
    SCALE = "scale"
    CUTOUT = "cutout"


# Standard probabilities for each augmentation type (from augment.py defaults)
STANDARD_PROBS: dict[AugmentationType, tuple[str, float]] = {
    AugmentationType.FLIP: ("flip_prob", 0.7),
    AugmentationType.BRIGHTNESS_CONTRAST: ("brightness_contrast_prob", 0.8),
    AugmentationType.BLUR: ("blur_prob", 0.6),
    AugmentationType.ROTATION: ("rotation_prob", 0.8),
    AugmentationType.SCALE: ("scale_prob", 0.6),
    AugmentationType.CUTOUT: ("cutout_prob", 0.6),
}


@dataclass
class AugmentationPreset:
    """Preset defining which augmentations are enabled for an experiment."""

    name: str
    enabled_augmentations: list[AugmentationType] = field(default_factory=list)
    description: str = ""

    def __post_init__(self) -> None:
        """Generate description if not provided."""
        if not self.description:
            if not self.enabled_augmentations:
                self.description = "Baseline (grayscale only)"
            else:
                aug_names = [aug.value for aug in self.enabled_augmentations]
                self.description = " + ".join(aug_names)

    def to_augment_config(self, versions_per_video: int = 25) -> AugmentConfig:
        """Convert to AugmentConfig for augment.py.

        Args:
            versions_per_video: Number of augmented versions per video.

        Returns:
            AugmentConfig with only specified augmentations enabled.
        """
        from rpa.augment import AugmentConfig

        # For baseline (no augmentations), only need 1 version since all would be identical
        actual_versions = 1 if not self.enabled_augmentations else versions_per_video

        # Start with all probabilities at 0.0
        config_kwargs: dict[str, float | int] = {
            "versions_per_video": actual_versions,
            "flip_prob": 0.0,
            "brightness_contrast_prob": 0.0,
            "blur_prob": 0.0,
            "rotation_prob": 0.0,
            "scale_prob": 0.0,
            "cutout_prob": 0.0,
        }

        # Enable selected augmentations with standard probabilities
        for aug_type in self.enabled_augmentations:
            param_name, prob = STANDARD_PROBS[aug_type]
            config_kwargs[param_name] = prob

        return AugmentConfig(**config_kwargs)  # type: ignore[arg-type]


class ExperimentStatus(StrEnum):
    """Status of an experiment."""

    PENDING = "pending"
    AUGMENTING = "augmenting"
    AUGMENTED = "augmented"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentResult(BaseModel):
    """Results for a single experiment."""

    experiment_id: str
    augmentations: list[str]
    augmentation_description: str

    # Sample counts
    original_train_count: int = 0
    augmented_train_count: int = 0
    versions_per_video: int = 25

    # Training metrics
    train_acc: float | None = None
    val_acc: float | None = None
    test_acc: float | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    test_loss: float | None = None
    best_val_acc: float | None = None

    # Metadata
    training_time_seconds: float | None = None
    status: ExperimentStatus = ExperimentStatus.PENDING
    error_message: str | None = None
    model_path: str | None = None
    dataset_path: str | None = None


class ExperimentSuiteConfig(BaseModel):
    """Configuration for the entire experiment suite."""

    suite_id: str
    gcs_bucket: str
    base_split_json: str = ""  # Only needed for training phase
    raw_input_path: str | None = None  # GCS path to raw videos for augmentation
    versions_per_video: int = 25
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_workers: int = 0


# =============================================================================
# EXPERIMENT PRESETS (33 total)
# =============================================================================

# Baseline (1)
_BASELINE = [
    AugmentationPreset("baseline", []),
]

# Single augmentations (6)
_SINGLES = [
    AugmentationPreset("flip_only", [AugmentationType.FLIP]),
    AugmentationPreset("bc_only", [AugmentationType.BRIGHTNESS_CONTRAST]),
    AugmentationPreset("blur_only", [AugmentationType.BLUR]),
    AugmentationPreset("rotation_only", [AugmentationType.ROTATION]),
    AugmentationPreset("scale_only", [AugmentationType.SCALE]),
    AugmentationPreset("cutout_only", [AugmentationType.CUTOUT]),
]

# Selected pairs (12) - removed blur_scale, blur_cutout, scale_cutout
_PAIRS = [
    AugmentationPreset("flip_bc", [AugmentationType.FLIP, AugmentationType.BRIGHTNESS_CONTRAST]),
    AugmentationPreset("flip_blur", [AugmentationType.FLIP, AugmentationType.BLUR]),
    AugmentationPreset("flip_rotation", [AugmentationType.FLIP, AugmentationType.ROTATION]),
    AugmentationPreset("flip_scale", [AugmentationType.FLIP, AugmentationType.SCALE]),
    AugmentationPreset("flip_cutout", [AugmentationType.FLIP, AugmentationType.CUTOUT]),
    AugmentationPreset("bc_blur", [AugmentationType.BRIGHTNESS_CONTRAST, AugmentationType.BLUR]),
    AugmentationPreset("bc_rotation", [AugmentationType.BRIGHTNESS_CONTRAST, AugmentationType.ROTATION]),
    AugmentationPreset("bc_scale", [AugmentationType.BRIGHTNESS_CONTRAST, AugmentationType.SCALE]),
    AugmentationPreset("bc_cutout", [AugmentationType.BRIGHTNESS_CONTRAST, AugmentationType.CUTOUT]),
    AugmentationPreset("blur_rotation", [AugmentationType.BLUR, AugmentationType.ROTATION]),
    AugmentationPreset("rotation_scale", [AugmentationType.ROTATION, AugmentationType.SCALE]),
    AugmentationPreset("rotation_cutout", [AugmentationType.ROTATION, AugmentationType.CUTOUT]),
]

# Selected triples (10)
_TRIPLES = [
    AugmentationPreset(
        "flip_bc_blur", [AugmentationType.FLIP, AugmentationType.BRIGHTNESS_CONTRAST, AugmentationType.BLUR]
    ),
    AugmentationPreset(
        "flip_bc_rotation", [AugmentationType.FLIP, AugmentationType.BRIGHTNESS_CONTRAST, AugmentationType.ROTATION]
    ),
    AugmentationPreset(
        "flip_bc_cutout", [AugmentationType.FLIP, AugmentationType.BRIGHTNESS_CONTRAST, AugmentationType.CUTOUT]
    ),
    AugmentationPreset(
        "flip_rotation_scale", [AugmentationType.FLIP, AugmentationType.ROTATION, AugmentationType.SCALE]
    ),
    AugmentationPreset("flip_blur_cutout", [AugmentationType.FLIP, AugmentationType.BLUR, AugmentationType.CUTOUT]),
    AugmentationPreset(
        "bc_blur_rotation", [AugmentationType.BRIGHTNESS_CONTRAST, AugmentationType.BLUR, AugmentationType.ROTATION]
    ),
    AugmentationPreset(
        "bc_blur_cutout", [AugmentationType.BRIGHTNESS_CONTRAST, AugmentationType.BLUR, AugmentationType.CUTOUT]
    ),
    AugmentationPreset(
        "bc_rotation_scale", [AugmentationType.BRIGHTNESS_CONTRAST, AugmentationType.ROTATION, AugmentationType.SCALE]
    ),
    AugmentationPreset(
        "blur_rotation_scale", [AugmentationType.BLUR, AugmentationType.ROTATION, AugmentationType.SCALE]
    ),
    AugmentationPreset(
        "rotation_scale_cutout", [AugmentationType.ROTATION, AugmentationType.SCALE, AugmentationType.CUTOUT]
    ),
]

# All augmentations (1)
_ALL = [
    AugmentationPreset("all_augmentations", list(AugmentationType)),
]

# Combined list of all 33 experiments
EXPERIMENT_PRESETS: list[AugmentationPreset] = _BASELINE + _SINGLES + _PAIRS + _TRIPLES + _ALL


def get_preset_by_name(name: str) -> AugmentationPreset | None:
    """Get an experiment preset by its name.

    Args:
        name: The experiment name to look up.

    Returns:
        The matching AugmentationPreset, or None if not found.
    """
    for preset in EXPERIMENT_PRESETS:
        if preset.name == name:
            return preset
    return None


def list_experiment_names() -> list[str]:
    """Get list of all experiment names.

    Returns:
        List of experiment IDs.
    """
    return [preset.name for preset in EXPERIMENT_PRESETS]
