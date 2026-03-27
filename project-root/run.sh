#!/bin/bash

echo "Running pipeline..."
python src/data_pipeline.py

echo "Training model..."
python src/train.py

echo "Evaluating..."
python src/eval.py

echo "Done."