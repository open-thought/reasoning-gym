# Word Ladder Puzzle Dataset Generator

## Overview

This project generates a dataset of word ladder puzzles and (optionally) submits chain-of-thought reasoning requests using Anthropic's Message Batches API. Each puzzle is stored as a JSON object with the following keys: `question`, `answer`, `metadata`, and `reasoning` (initially `null`).

The project consists of several key components:

- **`main.py`**:
  Orchestrates the overall flow. It performs the following tasks:
  1. Generates a dataset of word ladder puzzles by calling functions from `utils/create_word_ladders.py`.
  2. (Optionally) Triggers the reasoning request process to augment puzzles with chain-of-thought reasoning via `utils/generate_reasoning.py`.
  3. (Planned) Additional steps such as checking results or uploading the final dataset.

  The configuration for the dataset parameters (e.g., word length, chain length, and dataset size) is centralized here, making it easy to adjust the settings as needed.

- **`utils/create_word_ladders.py`**:
  Contains functions to create and validate a word ladder dataset. It leverages underlying modules (e.g., `reasoning_gym`) to generate individual puzzles and ensures uniqueness across the dataset.

- **`utils/generate_reasoning.py`**:
  Reads the generated dataset (in JSONL format), then filters out puzzles that already have reasoning. For puzzles missing chain-of-thought data, it splits them into batches (with a default batch size that you can adjust) and submits each batch to Anthropic's Message Batches API. Each API request includes the puzzle along with a custom system prompt (read from `system_prompt.txt`), and the resulting metadata is stored for later retrieval and analysis.

- **`usage_stats.py`**:
  Analyzes API response files to compute detailed usage statistics. This script:
  - Extracts token usage metrics such as `input_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, and `output_tokens`.
  - Calculates costs based on pricing data and shows the savings achieved through prompt caching.
  - Forecasts costs for various quantities of jobs (e.g., 2,000, 4,000, 10,000, 20,000, and 50,000 puzzles) using the observed average token usage.
  This is especially useful for monitoring your API spend and ensuring that your usage stays within budget.

## Warning

**Caution:**
Running large batches of requests via the Anthropic API (especially in `generate_reasoning.py`) can incur significant costs in Anthropic credits. **Please review and understand your API quota and budgeting before running the API call.** If you are just testing or working with a demo dataset, ensure you adjust the batch size or dataset size appropriately to avoid unexpected charges.

## Prerequisites

- **Python Version:** Python 3.7+
- **Dependencies:**
  - `tqdm`
  - `anthropic`
  - `reasoning_gym`
- **Environment Variables:**
  For generating reasoning batches, set your Anthropic API key:
  ```bash
  export ANTROPIC_API_KEY=your_api_key_here
  ```

## Directory Structure

```
examples/word_ladder/
├── main.py
├── utils/
│   ├── create_word_ladders.py
│   ├── generate_reasoning.py
│   └── system_prompt.txt
├── usage_stats.py
```


## Configuration

The dataset generation parameters are centralized in `main.py` under the `config` dictionary. You can adjust settings like:

- **Word Length:**
  - `min_word_length`
  - `max_word_length`

- **Chain Length:**
  - `min_chain_length` (e.g., set to -1 for the shortest possible chain)
  - `max_chain_length`

- **Dataset Size:**
  - `size` — the number of puzzles to generate (e.g., `1000` for a demo)

## How to Run

1. **Generate the Dataset**

   Run the main script:
   ```bash
   python3 main.py
   ```
   This does the following:
   - Generates a unique JSONL file containing the word ladder puzzles in the `output` folder.
   - Calls functions from `utils/create_word_ladders.py` to create the puzzles.
   - Optionally (if enabled), submits the puzzles for chain-of-thought reasoning via the API.

2. **Submit Reasoning Batches (Optional)**

   To generate chain-of-thought reasoning for puzzles:
   - Verify that `ANTHROPIC_API_KEY` is set.
   - Confirm that `system_prompt.txt` is present in the `/examples/word_ladder` folder and contains the desired system prompt.
   - In `main.py`, uncomment the reasoning submission section to enable the API call, or run directly:
   ```bash
   python3 utils/generate_reasoning.py
   ```

   **Warning:** Be aware that submitting large batches can quickly incur high costs in Anthropic credits.

3. **Compute Usage Statistics**

   After running batches through the API, you can analyze the cost and token usage statistics with:
   ```bash
   python3 usage_stats.py path/to/msgbatch_results.jsonl
   ```
   This script provides detailed costing information, token usage per query, savings from caching, and forecasting for future job batches.

## Output

- All generated datasets and batch metadata files are stored in the `/examples/word_ladder/output` folder.
- After submitting reasoning batches via Anthropic's API, you can monitor progress and download the batch results from the Anthropic web dashboard.
- Use `usage_stats.py` to compute detailed statistics and forecast future costs based on your current usage and token pricing.

## Troubleshooting

- **File Paths:**
  Verify that `system_prompt.txt` is in the `/examples/word_ladder` folder as expected. The modules use paths relative to their location.

- **Environment Variables:**
  Make sure your `ANTHROPIC_API_KEY` is set correctly when submitting API requests.

- **Output Directory Permissions:**
  Ensure the `output` directory exists and is writable by your user.

- **Cost Monitoring:**
  Check your Anthropic API usage and account balance before running large batches to avoid unexpected costs.

## License

This project is licensed under the MIT License.
