#!/bin/bash
set -e

# Change to the repository root (the script is inside scripts/)
cd "$(dirname "$0")/.."

echo "Running unit tests..."
pytest backend/tests/ -v

echo ""
echo "Running evaluation framework..."
python -m backend.evaluation.run_full_evaluation
