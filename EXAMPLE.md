# Example — running inference on a YouTube runner clip

End-to-end walkthrough using the pretrained foot-strike model and the example video included in this repo (`examples/runner_video.mp4`). The video was downloaded from https://www.youtube.com/watch?v=w_g1i6tzNGk.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager (`brew install uv` or `pip install uv`)
- `curl` and `unzip` (preinstalled on macOS / most Linux)
- ~1 GB free disk (model + dependencies)

Works on macOS (including Apple Silicon via MPS), Linux, and any system with PyTorch support.

## Three-step flow

```bash
# 1. Install Python dependencies into .venv (creates the venv if missing)
make init

# 2. Download the trained VideoMAE checkpoint (~305 MB) into ./model/
make download-model

# 3. Run inference on the example video
make inference VIDEO=examples/runner_video.mp4
```

That's it. The final command prints per-clip predictions and a summary block like:

```
============================================================
INFERENCE SUMMARY
============================================================
Video:                runner_video.mp4
Runners detected:     1
Primary track_id:     1
Clips used:           33
Predicted pattern:    1
Confidence:           94.2%
Class probabilities:
  0: 5.8%
  1: 94.2%
============================================================
```

**Class mapping:** `0 = forefoot` (flat-foot + toes-first merged), `1 = heel strike`.

The runner in the example clip is classified as a **heel striker** with 94.2% confidence, aggregated across 33 temporal windows.

## What happens under the hood

The `make inference` target runs a single Python script (`src/rpa/inference.py`) that chains:

1. **YOLOv8-Pose detection** — detects all people per frame, extracts 17 keypoints each
2. **Runner filtering** — 5-stage pipeline removes spectators, officials, walkers
3. **ROI extraction** — crops a foot-region window around each runner, stabilizes across frames
4. **Temporal slicing** — chops the stabilized video into 16-frame clips at 224×224
5. **VideoMAE classification** — runs each clip through the fine-tuned model, collects softmax
6. **Aggregation** — mean softmax across clips → argmax → final per-runner prediction

## Running on your own video

```bash
make inference VIDEO=/path/to/your_video.mp4 OUTPUT=results.json
```

- `VIDEO` — path to a raw trackside video (any length, any resolution — the pipeline resizes)
- `OUTPUT` *(optional)* — JSON report with per-clip probabilities
- `WORKDIR` *(optional)* — where intermediate extracted clips land
- `KEEP=1` *(optional)* — keep the `workdir` after the run to inspect the 16-frame clips

For all CLI knobs (confidence thresholds, clip stride, ROI ratio, etc.):

```bash
uv run python -m rpa.inference --help
```

## Troubleshooting

**`make download-model` fails mid-download** — it's a single `curl` call. Re-run the target; it's idempotent.

**`make inference` reports "No runners detected"** — the pipeline filters out videos with no runners meeting the gait criteria. Check that the video shows a person actually running (not walking, not still). Try lowering thresholds:

```bash
make inference VIDEO=your.mp4 \
    MODEL_DIR=./model
# edit the target if you need: --conf-threshold 0.15 --min-frames 10 --min-speed-ratio 0.3
```

Or call the script directly:

```bash
uv run python -m rpa.inference \
    --model-dir ./model \
    --video your.mp4 \
    --conf-threshold 0.15 \
    --min-frames 10
```

**GPU/MPS not used** — the script auto-selects device (CUDA > MPS > CPU). Check `torch.cuda.is_available()` / `torch.backends.mps.is_available()`.

## Model info

- Architecture: VideoMAE-Base (ViT-Base, 86.5M params)
- Fine-tuned from: `MCG-NJU/videomae-base-finetuned-kinetics`
- Training: `bc_only` experiment (brightness/contrast augmentation only)
- Test accuracy: **76.7%** on the RPA test set
- Checkpoint released: [v0.1.0](https://github.com/gofmannir/running-pattern-analysis/releases/tag/v0.1.0)
