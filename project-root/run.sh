#!/bin/bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running project..."
python train.py

echo "Done."