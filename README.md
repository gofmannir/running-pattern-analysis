# Running Pattern Analysis (RPA)

A machine learning pipeline for classifying running foot strike patterns from video using VideoMAE.

## Pipeline Overview

```
Raw Videos → Process & Clip → Dataset Split → Augmentation → Training → Inference
     │              │               │              │            │           │
     │              │               │              │            │           └── predict()
     │              │               │              │            └── train.py
     │              │               │              └── augment.py
     │              │               └── dataset_split.py
     │              └── batch_process.py / process_runners.py
     └── Tagged video files with labels
```

---

## Pipeline Visualization

### 1. Raw Lap Video

Full lap video captured from trackside camera (~30 seconds, 60fps). Below is a 3-second preview:


### 2. Extracted 16-Frame Clip

The raw video is processed to detect and track runners, extracting stabilized foot crops as 16-frame clips (224x224):


**Sampled frames from a single clip:**


### 3. Data Augmentation

Each training clip generates 25 augmented versions with random combinations of transformations:

**Available Augmentations:**


**Example: 9 Augmented Versions of the Same Clip:**


| Augmentation | Probability | Parameters |
|--------------|-------------|------------|
| Grayscale | 100% | Always applied |
| Horizontal Flip | 70% | Mirror left-right |
| Brightness/Contrast | 80% | ±25% each |
| Gaussian Blur | 60% | σ: 0.1-1.0 |
| Rotation | 80% | -5° to +5° |
| Scale/Crop | 60% | 0.9x - 1.1x |
| Cutout | 60% | 15% area, upper half only |

---

## Quick Start

```bash
# 1. Setup
make init

# 2. Process videos (creates 16-frame clips)
uv run python -m rpa.batch_process \
    --input-dir /path/to/tagged_videos \
    --output-dir /path/to/clips

# 3. Split dataset by runner (prevents data leakage)
uv run python -m rpa.dataset_split \
    --input-dir /path/to/clips \
    --output dataset_split.json \
    --remap-labels 2:0

# 4. Train locally
make train

# 5. Run inference
uv run python -m rpa.inference \
    --model_dir trained_model/best_model \
    --video_dir /path/to/test_videos
```

---

## Detailed Pipeline

### Step 1: Video Processing

Extract stabilized foot crops from raw videos. Creates 16-frame clips at 224x224 resolution.

#### Single Video Processing

```bash
uv run python -m rpa.process_runners \
    --input /path/to/video.mp4 \
    --output-dir /path/to/output \
    --roi-height-ratio 0.50 \
    --min-roi-size 150 \
    --height-smoothing-window 5 \
    --min-frames 40 \
    --min-speed-ratio 0.5 \
    --top-n-fastest 2 \
    --min-ankle-variance 400
```

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--roi-height-ratio` | 0.40 | Crop size as % of person's bbox height |
| `--min-roi-size` | 128 | Minimum crop size in pixels |
| `--min-frames` | 20 | Minimum frames to keep a track |
| `--slice-len` | 16 | Frames per output clip (VideoMAE requirement) |
| `--stride` | 16 | Sliding window stride between clips |
| `--min-speed-ratio` | 0.5 | Keep tracks with speed >= this ratio of max |
| `--top-n-fastest` | 2 | Keep only top N fastest tracks |
| `--min-ankle-variance` | 400 | Filter out standing/walking people |

#### Batch Processing

Process all videos in a directory structure: (the script has hardcoded the args for the process_video params)

```bash
uv run python -m rpa.batch_process \
    --input-dir /path/to/tagged_data_videos \
    --output-dir /path/to/training_clips \
    --workers 7
```

**Input Structure:**
```
tagged_data_videos/
├── 04HN_RUN2_CAM1/
│   └── clips/
│       ├── lap_006_000.mp4  # Label 000 (forefoot)
│       └── lap_007_001.mp4  # Label 001 (heel strike)
├── 15VR_RUN2_CAM2/
│   └── clips/
│       └── ...
```

**Output Naming:**
```
{camera}_{lap}_CUT_{clip}_{label}.mp4
Example: 04HN_RUN2_CAM1_lap_006_CUT_001_001.mp4
```

---

### Step 2: Dataset Split

Split dataset by **runner ID** to prevent data leakage (same runner shouldn't appear in train and test).

```bash
uv run python -m rpa.dataset_split \
    --input-dir /path/to/training_clips \
    --output dataset_split.json \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --remap-labels 2:0
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-ratio` | 0.7 | Fraction for training set |
| `--val-ratio` | 0.15 | Fraction for validation set |
| `--test-ratio` | 0.15 | Fraction for test set |
| `--seed` | 42 | Random seed for reproducibility |
| `--remap-labels` | None | Remap labels (e.g., `2:0` for binary) |
| `--no-stratify` | False | Disable label stratification |

**Output:** `dataset_split.json` with train/val/test video paths

---

### Step 3: Data Augmentation

Generate augmented versions of training videos (grayscale, flip, rotation, blur, etc.).

#### Local Augmentation

```bash
uv run python -m rpa.augment \
    --input /path/to/videos \
    --output /path/to/augmented \
    --versions 25 \
    --workers 4
```

#### GCS Augmentation (on VM)

```bash
# Using Makefile target
make augment-gcs

# Or directly
uv run python -m rpa.augment \
    --input gs://rpa-dataset-nirgofman/raw/ \
    --output gs://rpa-dataset-nirgofman/augmented/ \
    --versions 25 \
    --workers 4 \
    --checkpoint gs://rpa-dataset-nirgofman/augmented/.checkpoint.json
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--versions` | 25 | Number of augmented versions per video |
| `--workers` | 1 | Parallel workers |
| `--checkpoint` | None | Resume from checkpoint (local or GCS path) |

**Augmentations Applied:**
- Grayscale conversion (always)
- Horizontal flip (70%)
- Brightness/contrast (80%)
- Gaussian blur (60%)
- Rotation ±5° (80%)
- Scale 0.9-1.1x (60%)
- Cutout in upper half (60%)

---

### Step 4: Create GCS Split (for VM Training)

Map training data to augmented GCS paths, val/test to raw GCS paths:

```bash
# Using Makefile target
make create-gcs-split

# Or directly
uv run python -m rpa.create_augmented_split \
    --original-split dataset_split.json \
    --bucket gs://rpa-dataset-nirgofman \
    --output dataset_split_gcs.json \
    --versions 25
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--versions` | 25 | Expected augmented versions per video |
| `--strict` | False | Fail if any augmented versions missing |

**Output:** `dataset_split_gcs.json`
- Train → `gs://bucket/augmented/` paths
- Val/Test → `gs://bucket/raw/` paths

---

### Step 5: Training

Train VideoMAE model for foot strike classification.

#### Local Training

```bash
make train

# Or with custom parameters
uv run python -m rpa.train \
    --split-json dataset_split.json \
    --output-dir trained_model \
    --remap-labels 2:0 \
    --epochs 10 \
    --batch-size 8 \
    --lr 5e-5
```

#### GCS Training (on VM)

```bash
make train-gcs

# Or directly
uv run python -m rpa.train \
    --split-json dataset_split_gcs.json \
    --output-dir trained_model_gcs \
    --remap-labels 2:0 \
    --epochs 10 \
    --batch-size 8 \
    --no-augmentation  # Videos are pre-augmented
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 10 | Training epochs |
| `--batch-size` | 4 | Batch size |
| `--lr` | 5e-5 | Learning rate |
| `--remap-labels` | None | Label remapping (e.g., `2:0`) |
| `--no-augmentation` | False | Disable runtime augmentations |
| `--no-grayscale` | False | Keep RGB (not recommended) |

**Output:**
```
trained_model/
├── checkpoint-epoch-1/
├── checkpoint-epoch-2/
├── ...
└── best_model/
    ├── config.json
    ├── model.safetensors
    └── label_mapping.json
```

---

### Step 6: Inference

Run predictions on new videos.

```bash
# Single video
uv run python -m rpa.inference \
    --model_dir trained_model/best_model \
    --video /path/to/video.mp4

# Multiple videos
uv run python -m rpa.inference \
    --model_dir trained_model/best_model \
    --video video1.mp4 video2.mp4 video3.mp4

# Directory of videos
uv run python -m rpa.inference \
    --model_dir trained_model/best_model \
    --video_dir /path/to/test_clips

# Save results to JSON
uv run python -m rpa.inference \
    --model_dir trained_model/best_model \
    --video_dir /path/to/test_clips \
    --output predictions.json
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--model_dir` | Path to trained model directory |
| `--video` | Path(s) to video file(s) |
| `--video_dir` | Directory containing .mp4 files |
| `--output` | Save predictions to JSON file |

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make init` | Create venv and install dependencies |
| `make check` | Run linting and type checking |
| `make fix` | Auto-fix formatting issues |
| `make train` | Train locally with dataset_split.json |
| `make train-gcs` | Train on GCS data (streams from bucket) |
| `make augment-gcs` | Run augmentation on GCS bucket |
| `make augment-local` | Test augmentation locally |
| `make create-gcs-split` | Generate GCS-aware split JSON |
| `make ssh-vm` | SSH to GCP training VM |
| `make sync-vm` | Push code to VM via git |

---

## GCP VM Workflow

```bash
# 1. Push code to VM
make sync-vm

# 2. SSH to VM
make ssh-vm

# 3. On VM: Run augmentation
cd ~/rpa
make augment-gcs

# 4. On VM: Create split and train
make create-gcs-split
make train-gcs

# 5. Download trained model (from local machine)
gcloud compute scp --zone=us-central1-a \
    rpa-training-gpu:~/rpa/trained_model_gcs ./trained_model_gcs \
    --recurse
```

---

## File Naming Convention

```
{runner}_{run}_{camera}_{lap}_CUT_{clip}_{label}.mp4
   │       │       │       │         │       │
   │       │       │       │         │       └── Label (000=forefoot, 001=heel)
   │       │       │       │         └── Clip number from sliding window
   │       │       │       └── Lap identifier
   │       │       └── Camera angle (CAM1, CAM2, CAM3)
   │       └── Run number
   └── Runner ID (e.g., 04HN, 15VR)

Augmented: {base}_v{NNN}_{label}.mp4
Example: 04HN_RUN2_CAM1_lap_006_CUT_001_v015_001.mp4
```

---

## Project Structure

```
rpa/
├── src/rpa/
│   ├── process_runners.py    # Single video processing
│   ├── batch_process.py      # Batch processing with labels
│   ├── dataset_split.py      # Runner-based train/val/test split
│   ├── dataset_stats.py      # Dataset statistics
│   ├── augment.py            # Offline video augmentation
│   ├── create_augmented_split.py  # GCS split generation
│   ├── train.py              # VideoMAE training
│   └── inference.py          # Model inference
├── Makefile                  # Build/run targets
├── pyproject.toml            # Project config
└── dataset_split.json        # Generated split file
```

---

## Labels

| Label | Description |
|-------|-------------|
| 0 | Forefoot strike |
| 1 | Heel strike |
| 2 | (Remapped to 0 for binary classification) |
