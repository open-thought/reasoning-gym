# Model Evaluation Framework

A simple asynchronous framework for evaluating language models on reasoning tasks using the OpenRouter API.

## Evaluation Results Repository

In order to keep the main repo clean and not clutter it with evaluation traces from different models, we store all evaluation results in a separate repository: [reasoning-gym-eval](https://github.com/open-thought/reasoning-gym-eval).

If you run evaluations and want to contribute your results, please create a pull request in the [reasoning-gym-eval](https://github.com/open-thought/reasoning-gym-eval) repository, not in the main reasoning-gym repo.

## Overview

This framework provides tools to evaluate language models on the reasoning_gym datasets. It supports:
- Concurrent evaluation of multiple questions and datasets
- Customizable dataset configurations
- Automatic result aggregation and summary generation
- Rate limiting for API calls

## Setup

1. Install reasoning-gym in development mode:
```bash
pip install -e ..
```

2. Install the additional dependencies required for evaluation:
```bash
pip install -r requirements-eval.txt
```

3. Set your OpenRouter API key as an environment variable:
```bash
export OPENROUTER_API_KEY=your-api-key
```


4. Prepare your dataset configuration in YAML format (see examples in `yaml/algorithmic.yaml` or `yaml/logic.yaml`):
```yaml
model: model-name
category: category-name
datasets:
  - dataset1
  - dataset2
eval_dir: eval/r1
dataset_size: 50
dataset_seed: 42
developer_role: system

```
For example the following file will run an evaluation for deepseek r1 for algorithmic datasets.
``` yaml
model: deepseek/deepseek-r1
category: algorithmic
datasets:
  - ab
  -  base_conversion
  -  binary_matrix
  -  caesar_cipher
  -  count_primes
  -  game_of_life
  -  graph_color
  -  group_anagrams
  -  isomorphic_strings
  -  letter_counting
  -  letter_jumble
  -  manipulate_matrix
  -  number_filtering
  -  number_sorting
  -  palindrome
  -  pool_matrix
  -  ransom_note
  -  rotate_matrix
  -  sentence_reordering
  -  spell_backward
  -  spiral_matrix
  -  string_insertion
  -  string_manipulation
  -  string_synthesis
  -  word_ladder
  -  word_sequence_reversal
  -  word_sorting
eval_dir: eval/r1
dataset_size: 50
dataset_seed: 45
developer_role: system

```

 The following would run Claude 3.5 on the algorithmic dataset.
```yaml
model: anthropic/claude-3.5-sonnet
category: algorithmic
provider: Anthropic
datasets:
  -  count_primes
  -  game_of_life
  -  graph_color
  -  group_anagrams
  -  isomorphic_strings
  -  letter_counting
  -  letter_jumble
  -  manipulate_matrix
  -  number_filtering
  -  number_sorting
  -  palindrome
  -  pool_matrix
  -  ransom_note
  -  rotate_matrix
  -  sentence_reordering
  -  spell_backward
  -  spiral_matrix
  -  string_insertion
  -  string_manipulation
  -  string_synthesis
  -  word_ladder
  -  word_sequence_reversal
  -  word_sorting
eval_dir: eval/r1
dataset_size: 50
dataset_seed: 45
developer_role: system
```
Here you specify individual model and provider

### Running Evaluations

To run evaluations
```
python eval.py --yaml <path-to yaml file>
```
e.g
```
python eval.py --yaml yaml/algorithmic.yaml
```


The results of individual model on a dataset will be stored in a new folder in the directory E.g `r1/algorithmic/proposition_logic.json`
