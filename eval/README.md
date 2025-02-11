# Model Evaluation Framework

A simple asynchronous framework for evaluating language models on reasoning tasks using the OpenRouter API.

## Overview

This framework provides tools to evaluate language models on the reasoning_gym datasets. It supports:
- Concurrent evaluation of multiple questions and datasets
- Customizable dataset configurations
- Automatic result aggregation and summary generation
- Rate limiting for API calls

## Setup

1. Set your OpenRouter API key as an environment variable:
```bash
export OPENROUTER_API_KEY=your-api-key
```

2. Prepare your dataset configuration in JSON format (e.g., `eval_basic.json`):
```json
[
  {
    "name": "dataset_name",
    "parameter1": "value1",
    "parameter2": "value2"
  }
]
```

## Usage

### Running Evaluations

You can run evaluations in two ways:

1. Using the provided bash script:
```bash
./run_eval.sh
```

2. Running the Python script directly:
```bash
python eval.py --model "model-name" --config "eval_basic.json" --output-dir "results"
```

### Command Line Arguments

- `--model`: Model identifier (required)
- `--config`: Path to JSON configuration file (required)
- `--output-dir`: Directory for saving results (default: "results")
- `--max-concurrent`: Maximum number of concurrent API calls (default: 10)

## Output

The framework generates two types of output files:

1. Detailed results: `evaluation_{model}_{timestamp}.json`
   - Contains full response data and scoring for each question

2. Summary: `summary_{model}_{timestamp}.json`
   - Contains aggregated metrics for each dataset

## Structure

```
.
├── eval.py              # Main evaluation script
├── run_eval.sh          # Bash script for running evaluations
├── eval_basic.json      # Dataset configuration file
└── results/             # Output directory
```
