PROJ_VENV=$(CURDIR)/.venv

MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

.PHONY: help
help:
	@echo "Available targets:"
	@$(MAKE) -pRrq -f $(MAKEFILE_LIST) : 2>/dev/null |\
	  awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}'|\
	  egrep -v -e '^[^[:alnum:]]' -e '^$@' |\
	  sort |\
	  awk '{print "  " $$0}'

$(PROJ_VENV):
	uv venv "$@"

.PHONY: init clean
init: $(PROJ_VENV)
	uv sync

clean:
	rm -rf "$(PROJ_VENV)"

.PHONY: check check-py
check-py:
	uv run ruff check
	uv run ruff format --check
	uv run mypy .
check: check-py

.PHONY: format
format:
	uv run ruff format

.PHONY: fix
fix: format
	uv run ruff check --fix

.PHONY: train
train:
	uv run python -m rpa.train \
		--split-json dataset_split.json \
		--output-dir trained_model \
		--remap-labels 2:0 \
		--epochs 5 \
		--batch-size 8

# Train with pre-augmented dataset (no runtime augmentation)
.PHONY: train-augmented
train-augmented:
	uv run python -m rpa.train \
		--split-json dataset_split_augmented.json \
		--output-dir trained_model_augmented \
		--remap-labels 2:0 \
		--epochs 5 \
		--batch-size 8 \
		--no-augmentation

# Augment dataset on GCS (run on VM for better bandwidth)
GCS_BUCKET=gs://rpa-dataset-nirgofman
AUGMENT_WORKERS ?= 8
.PHONY: augment-gcs
augment-gcs:
	uv run python -m rpa.augment \
		--input $(GCS_BUCKET)/raw/ \
		--output $(GCS_BUCKET)/augmented/ \
		--versions 25 \
		--workers $(AUGMENT_WORKERS) \
		--checkpoint $(GCS_BUCKET)/augmented/.checkpoint.json

# =============================================================================
# GCP VM TARGETS (GPU VM - for training)
# =============================================================================
VM_NAME=rpa-training-gpu
VM_ZONE=us-central1-a

# SSH to GPU VM
.PHONY: ssh-vm
ssh-vm:
	gcloud compute ssh $(VM_NAME) --zone=$(VM_ZONE)

# Sync code to GPU VM (via git push)
.PHONY: sync-vm
sync-vm:
	git push vm main --force

# =============================================================================
# CPU VM TARGETS (for augmentation - 8 vCPU, 4GB RAM, on-demand)
# =============================================================================
CPU_VM_NAME=rpa-augment-cpu
CPU_VM_ZONE=us-central1-a

# SSH to CPU VM
.PHONY: ssh-cpu
ssh-cpu:
	gcloud compute ssh $(CPU_VM_NAME) --zone=$(CPU_VM_ZONE)

# Sync code to CPU VM (via git push)
.PHONY: sync-cpu
sync-cpu:
	git push cpu main --force

# Setup git remote for CPU VM (run after VM is created)
.PHONY: setup-cpu-remote
setup-cpu-remote:
	@echo "Getting CPU VM external IP..."
	$(eval CPU_IP := $(shell gcloud compute instances describe $(CPU_VM_NAME) --zone=$(CPU_VM_ZONE) --format='get(networkInterfaces[0].accessConfigs[0].natIP)'))
	@echo "CPU VM IP: $(CPU_IP)"
	@echo "Setting up git remote 'cpu'..."
	-git remote remove cpu 2>/dev/null || true
	git remote add cpu ssh://nirgofman@$(CPU_IP)/home/nirgofman/rpa
	@echo "Remote 'cpu' added. You can now run: make sync-cpu"

# Run augmentation on CPU VM
.PHONY: run-augment-cpu
run-augment-cpu:
	gcloud compute ssh $(CPU_VM_NAME) --zone=$(CPU_VM_ZONE) -- \
		"cd ~/rpa && make augment-gcs"

# Terraform targets for CPU VM
.PHONY: tf-cpu-init
tf-cpu-init:
	cd terraform/cpu-vm && terraform init

.PHONY: tf-cpu-plan
tf-cpu-plan:
	cd terraform/cpu-vm && terraform plan

.PHONY: tf-cpu-apply
tf-cpu-apply:
	cd terraform/cpu-vm && terraform apply
	@echo ""
	@echo "VM created! Next steps:"
	@echo "  1. SSH to setup: make ssh-cpu"
	@echo "  2. Setup git remote: make setup-cpu-remote"
	@echo "  3. Sync code: make sync-cpu"

.PHONY: tf-cpu-destroy
tf-cpu-destroy:
	cd terraform/cpu-vm && terraform destroy

# Download the released VideoMAE model checkpoint from GitHub releases.
# Unzips into $(MODEL_DIR) so `make inference` finds it with zero config.
MODEL_URL ?= https://github.com/gofmannir/running-pattern-analysis/releases/download/v0.1.0/model.zip
MODEL_DIR ?= ./model
.PHONY: download-model
download-model:
	mkdir -p $(MODEL_DIR)
	curl -fL --progress-bar -o /tmp/rpa_model.zip "$(MODEL_URL)"
	unzip -o /tmp/rpa_model.zip -d $(MODEL_DIR)
	rm /tmp/rpa_model.zip
	@echo "Model extracted to $(MODEL_DIR)/"
	@ls -lh $(MODEL_DIR)

# Run end-to-end inference on a RAW trackside video.
# Chains: YOLO detection -> runner filtering -> ROI extraction -> VideoMAE classification.
#
# Required:
#   VIDEO=<path to raw .mp4>
#   MODEL_DIR=<path to model dir with config.json, label_mapping.json, model.safetensors>
#   (MODEL_DIR defaults to ./trained_model/best_model if not given)
#
# Optional:
#   OUTPUT=<path>   Save the JSON report (default: stdout only)
#   WORKDIR=<path>  Where intermediate clips go (default: auto temp dir)
#   KEEP=1          Keep the workdir after running (clips + stabilized video for inspection)
#
# Examples:
#   make inference VIDEO=examples/runner_video.mp4
#   make inference VIDEO=examples/runner_video.mp4 MODEL_DIR=./model OUTPUT=results.json KEEP=1
.PHONY: inference
inference:
ifndef VIDEO
	@echo "Usage: make inference MODEL_DIR=<path> VIDEO=<path>"
	@echo "Optional: OUTPUT=<json path>  WORKDIR=<dir>  KEEP=1"
	@false
else
	uv run python -m rpa.inference \
		--model-dir $(MODEL_DIR) \
		--video $(VIDEO) \
		$(if $(OUTPUT),--output $(OUTPUT)) \
		$(if $(WORKDIR),--workdir $(WORKDIR)) \
		$(if $(KEEP),--keep-intermediates)
endif

# Augment local directory (for testing)
.PHONY: augment-local
augment-local:
	uv run python -m rpa.augment \
		--input /tmp/test_videos/ \
		--output /tmp/test_augmented/ \
		--versions 5 \
		--workers 2

# Create GCS-aware split with augmented train data
.PHONY: create-gcs-split
create-gcs-split:
	uv run python -m rpa.create_augmented_split \
		--original-split dataset_split.json \
		--bucket $(GCS_BUCKET) \
		--output dataset_split_gcs.json \
		--versions 25

# Train on GCS data (augmented train, raw val/test)
.PHONY: train-gcs
train-gcs:
	uv run python -m rpa.train \
		--split-json dataset_split_gcs.json \
		--output-dir trained_model_gcs \
		--remap-labels 2:0 \
		--epochs 10 \
		--batch-size 8 \
		--no-augmentation

# =============================================================================
# AUGMENTATION EXPERIMENT FRAMEWORK
# =============================================================================
# Suite ID for experiment runs (override with SUITE_ID=xxx)
SUITE_ID ?= aug_study_v1

# Phase 1: Generate all augmented datasets (run once, takes time)
.PHONY: experiment-augment
experiment-augment:
	uv run python -m rpa.experiment augment \
		--bucket $(GCS_BUCKET) \
		--raw-input $(GCS_BUCKET)/raw/ \
		--suite-id $(SUITE_ID) \
		--versions 25

# Phase 1: Resume augmentation if interrupted
.PHONY: experiment-augment-resume
experiment-augment-resume:
	uv run python -m rpa.experiment augment \
		--bucket $(GCS_BUCKET) \
		--raw-input $(GCS_BUCKET)/raw/ \
		--suite-id $(SUITE_ID) \
		--resume

# Phase 1: Check augmentation status
.PHONY: experiment-augment-status
experiment-augment-status:
	uv run python -m rpa.experiment augment-status \
		--bucket $(GCS_BUCKET) \
		--suite-id $(SUITE_ID)

# Phase 2: Train a specific experiment (run manually for each)
# Usage: make experiment-train EXP_ID=baseline
EXP_ID ?= baseline
.PHONY: experiment-train
experiment-train:
	uv run python -m rpa.experiment train \
		--bucket $(GCS_BUCKET) \
		--suite-id $(SUITE_ID) \
		--experiment $(EXP_ID) \
		--epochs 10

# Phase 2: Train all experiments sequentially
.PHONY: experiment-train-all
experiment-train-all:
	uv run python -m rpa.experiment train-all \
		--bucket $(GCS_BUCKET) \
		--suite-id $(SUITE_ID) \
		--epochs 10

# Phase 2: List experiments and their training status
.PHONY: experiment-list
experiment-list:
	uv run python -m rpa.experiment list \
		--bucket $(GCS_BUCKET) \
		--suite-id $(SUITE_ID)

# Phase 3: Generate results table (CSV + JSON)
.PHONY: experiment-results
experiment-results:
	uv run python -m rpa.experiment results \
		--bucket $(GCS_BUCKET) \
		--suite-id $(SUITE_ID) \
		--output ./experiment_results_$(SUITE_ID)

# Phase 3: Show results summary in console
.PHONY: experiment-summary
experiment-summary:
	uv run python -m rpa.experiment summary \
		--bucket $(GCS_BUCKET) \
		--suite-id $(SUITE_ID)

# List all available experiment presets
.PHONY: experiment-presets
experiment-presets:
	uv run python -m rpa.experiment list-presets

# =============================================================================
# END EXPERIMENT FRAMEWORK
# =============================================================================

# File browser for GCS bucket (requires Docker)
.PHONY: filestash
filestash:
	@echo "Starting Filestash file browser..."
	@echo "Open http://localhost:8334 in your browser"
	@echo "Bucket: $(GCS_BUCKET)"
	docker run --rm -d \
		-p 8334:8334 \
		--name filestash \
		-v $(HOME)/rpa-gcs-key.json:/app/data/state/config/gcs-key.json:ro \
		-e APPLICATION_URL=http://localhost:8334 \
		machines/filestash
	@echo "Container started. Configure GCS backend in admin panel."

.PHONY: filestash-stop
filestash-stop:
	docker stop filestash 2>/dev/null || true

# Terraform targets for GCS service account
.PHONY: tf-init
tf-init:
	cd terraform && terraform init

.PHONY: tf-plan
tf-plan:
	cd terraform && terraform plan

.PHONY: tf-apply
tf-apply:
	cd terraform && terraform apply
	@echo "Service account key saved to: terraform/rpa-gcs-key.json"

.PHONY: tf-destroy
tf-destroy:
	cd terraform && terraform destroy
