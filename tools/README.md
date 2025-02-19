# Reasoning Gym Tools

This directory contains additional tools for working with Reasoning Gym:

## Server

A FastAPI server that manages reasoning gym experiments, allowing runtime configuration and monitoring.

### Starting the Server

1. Install server dependencies:
```bash
pip install -e ".[server]"
```

2. Set the API key environment variable:
```bash
export REASONING_GYM_API_KEY=your-secret-key
```

3. Start the server:
```bash
uvicorn tools.server.server:app
```

The server will be available at http://localhost:8000. You can access the API documentation at http://localhost:8000/docs.

## RGC (Reasoning Gym Client)

A command-line interface for interacting with the Reasoning Gym server.

### Installation

```bash
pip install -e ".[cli]"
```

### Usage

First, set the API key to match your server:
```bash
export REASONING_GYM_API_KEY=your-secret-key
```

Then you can use the CLI:

```bash
# List all commands
rgc --help

# List experiments
rgc experiments list

# Create a new experiment interactively
rgc experiments create my-experiment

# Create from config file
rgc experiments create my-experiment -f config.yaml

# Show experiment details
rgc experiments show my-experiment

# Edit dataset configuration
rgc config edit my-experiment chain_sum
```

### Example Configuration File

Here's an example `config.yaml` for creating an experiment:

```yaml
size: 500
seed: 42
datasets:
  chain_sum:
    weight: 1.0
    config:
      min_terms: 2
      max_terms: 4
      min_digits: 1
      max_digits: 2
      allow_negation: false
```
