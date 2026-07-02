#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script: generates synthetic data and quick model artifacts required by the API.
# Usage: ./scripts/bootstrap_models.sh

PYTHON=${PYTHON:-python}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

echo "Creating models/saved directories..."
mkdir -p models/saved/recommendation_models

# Generate data if missing
if [ ! -f data/raw/orders.csv ]; then
  echo "Generating synthetic dataset..."
  $PYTHON src/data/generate_dataset.py
else
  echo "data/raw/orders.csv already exists — skipping generation"
fi

# Run the Python bootstrap that trains quick models and saves artifacts
echo "Running bootstrap_models.py to create model artifacts (this may download small transformer tokenizer)..."
$PYTHON scripts/bootstrap_models.py

echo "Bootstrap complete. Models saved under models/saved/"
