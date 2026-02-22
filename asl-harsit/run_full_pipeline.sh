#!/bin/bash

echo "=================================================="
echo "Starting Full ASL Pipeline: Static Extraction & Training"
echo "=================================================="

# Go to the script directory
cd "$(dirname "$0")"

PYTHON="../.venv_stable/bin/python"
DATASET_DIR="$HOME/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5"

echo "[1/2] Extracting static frames from WLASL dataset..."
$PYTHON extract_wlasl.py \
    --dataset_dir "$DATASET_DIR" \
    --output_dir data \
    --max_words 100 \
    > full_extraction.log 2>&1

if [ $? -eq 0 ]; then
    echo "Extraction successful. See full_extraction.log for details."
else
    echo "Extraction FAILED. Check full_extraction.log for errors."
    exit 1
fi

echo "[2/2] Training the Dense word model..."
$PYTHON train_model.py --type word > full_training.log 2>&1

if [ $? -eq 0 ]; then
    echo "Training successful. Models saved to models/!"
else
    echo "Training FAILED. Check full_training.log for errors."
    exit 1
fi

echo "=================================================="
echo "PIPELINE FULLY COMPLETE!"
echo "=================================================="
