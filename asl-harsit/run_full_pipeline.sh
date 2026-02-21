#!/bin/bash

echo "=================================================="
echo "Starting Full ASL Pipeline: Extraction & Training"
echo "=================================================="

# Go to the script directory
cd "$(dirname "$0")"

# 1. Run full extraction
echo "[1/2] Extracting all sequences from WLASL dataset..."
../.venv_stable/bin/python extract_wlasl.py --dataset_dir ~/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5 --output_dir data --max_words 2000 > full_extraction.log 2>&1

if [ $? -eq 0 ]; then
    echo "Extraction successful. See full_extraction.log for details."
else
    echo "Extraction failed. Check full_extraction.log."
    exit 1
fi

# 2. Train the model
echo "[2/2] Training the LSTM word model on the new dataset..."
../.venv_stable/bin/python train_model.py --type word > full_training.log 2>&1

if [ $? -eq 0 ]; then
    echo "Training successful. Sequence mode models saved to models/!"
    echo "=================================================="
    echo "PIPELINE FULLY COMPLETE!"
    echo "=================================================="
else
    echo "Training failed. Check full_training.log."
    exit 1
fi
