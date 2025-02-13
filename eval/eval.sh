#!/bin/bash

# Check if OPENROUTER_API_KEY is set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY environment variable is not set"
    echo "Please set it using: export OPENROUTER_API_KEY=your-api-key"
    exit 1
fi

# Configuration
OUTPUT_DIR="results"

# List of models to evaluate
MODELS=(
    "google/gemini-2.0-flash-001"
)

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluations
for model in "${MODELS[@]}"; do
    echo "Evaluating $model..."
    python eval.py \
        --model "$model" \
        --config "eval_basic.json" \
        --output-dir "$OUTPUT_DIR"
done

echo "All evaluations completed!"
