#!/usr/bin/env python3
"""
This script reads a JSONL file that contains messages with usage statistics.
For each JSON record, it expects to find the token usage information under:
    record["result"]["message"]["usage"]

It then calculates and prints statistics for each usage token field:
    - input_tokens
    - cache_creation_input_tokens
    - cache_read_input_tokens
    - output_tokens
+pricing calculations
+calculates the savings from caching (vs if we hadn't done any caching)
+forecasts costs for 10,000, 20,000 and 50,000 jobs based on tokens per query

Usage:
    python usage_stats.py path/to/msgbatch_01X9LgZNVkLFhzrrBd9LNgWb_results.jsonl
"""

import argparse
import json
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description="Compute usage token statistics from a JSONL file.")
    parser.add_argument("file", help="Path to the JSONL file containing usage token data.")
    args = parser.parse_args()

    # Usage token fields that we want to track
    usage_fields = [
        "input_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "output_tokens",
    ]

    # Pricing for Sonnet, 2 Feb 2025
    base_input_rate = 1.50
    pricing = {
        "input_tokens": base_input_rate,
        "cache_creation_input_tokens": base_input_rate * 1.25,  # More expensive for initial computation
        "cache_read_input_tokens": base_input_rate * 0.1,  # Cheaper for cache-read tokens
        "output_tokens": 7.50,
    }

    # A dictionary to store lists of values for each usage field
    usage_data = {key: [] for key in usage_fields}
    total_lines = 0
    error_count = 0

    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Warning] Failed to parse JSON on line {total_lines}.")
                error_count += 1
                continue

            # Navigate to the usage stats
            try:
                usage = record["result"]["message"]["usage"]
            except KeyError:
                print(f"[Warning] Missing usage field in line {total_lines}.")
                error_count += 1
                continue

            # Extract token values from the usage data
            for key in usage_fields:
                # Defaulting to 0 if the token field is missing or non-numeric
                try:
                    token_value = int(usage.get(key, 0))
                except (ValueError, TypeError):
                    token_value = 0
                usage_data[key].append(token_value)

    print(f"\nProcessed {total_lines} lines with {error_count} error(s).\n")
    print("Usage Tokens Statistics:")
    print("-" * 40)

    grand_total_cost = 0.0
    # Calculate and print stats for each token type
    for key in usage_fields:
        values = usage_data[key]
        if values:
            total = sum(values)
            count = len(values)
            min_val = min(values)
            max_val = max(values)
            avg = mean(values)
            # Calculate pricing cost scaling by tokens per million
            cost = total / 1_000_000 * pricing[key]
            grand_total_cost += cost

            print(f"{key}:")
            print(f"  Total    = {total}")
            print(f"  Count    = {count}")
            print(f"  Min      = {min_val}")
            print(f"  Max      = {max_val}")
            print(f"  Mean     = {avg:.2f}")
            print(f"  Cost     = ${cost:.2f}\n")
        else:
            print(f"{key}: No data found.\n")

    print("-" * 40)
    print(f"Grand Total Estimated Cost: ${grand_total_cost:.2f}")

    # -----------------------------------------------
    # Calculate caching savings (for input-related tokens)
    # Without caching, all tokens would have been charged at the standard input rate.
    #
    # Baseline cost (if no caching were used):
    #   = (input_tokens + cache_creation_input_tokens + cache_read_input_tokens)
    #     / 1_000_000 * base_input_rate
    #
    # Actual cost (with caching):
    #   = input_tokens * base_input_rate +
    #     cache_creation_input_tokens * (base_input_rate * 1.25) +
    #     cache_read_input_tokens * (base_input_rate * 0.1)
    #
    # Savings from caching is then the difference.
    sum_input = sum(usage_data["input_tokens"])
    sum_cache_creation = sum(usage_data["cache_creation_input_tokens"])
    sum_cache_read = sum(usage_data["cache_read_input_tokens"])

    baseline_input_cost = (sum_input + sum_cache_creation + sum_cache_read) / 1_000_000 * pricing["input_tokens"]
    actual_input_cost = (
        (sum_input) / 1_000_000 * pricing["input_tokens"]
        + (sum_cache_creation) / 1_000_000 * pricing["cache_creation_input_tokens"]
        + (sum_cache_read) / 1_000_000 * pricing["cache_read_input_tokens"]
    )
    caching_savings = baseline_input_cost - actual_input_cost

    print(f"Caching Savings (input-related tokens): ${caching_savings:.2f}")

    # -----------------------------------------------
    # Forecast future cost estimates based on the average tokens per job.
    #
    # We'll compute the average tokens per job (i.e. tokens per query) for:
    #   - input_tokens
    #   - cache_creation_input_tokens
    #   - cache_read_input_tokens
    #   - output_tokens
    #
    # Then we forecast, for example, for 10,000, 20,000, and 50,000 jobs:
    #   - Apply the relevant pricing to compute the cost per token type.
    #   - Also compute the baseline cost for input-related tokens and the savings
    #     from caching.
    if usage_data["input_tokens"]:
        job_count = len(usage_data["input_tokens"])
        avg_input_tokens = sum(usage_data["input_tokens"]) / job_count
        avg_cache_creation_tokens = sum(usage_data["cache_creation_input_tokens"]) / job_count
        avg_cache_read_tokens = sum(usage_data["cache_read_input_tokens"]) / job_count
        avg_output_tokens = sum(usage_data["output_tokens"]) / job_count

        print("\nAverage Tokens per Job:")
        print(f"  input_tokens                = {avg_input_tokens:.2f}")
        print(f"  cache_creation_input_tokens = {avg_cache_creation_tokens:.2f}")
        print(f"  cache_read_input_tokens     = {avg_cache_read_tokens:.2f}")
        print(f"  output_tokens               = {avg_output_tokens:.2f}")

        forecast_jobs = [2000, 4000, 10000, 20000, 50000]
        print("\nForecasting Future Job Costs:")
        for jobs in forecast_jobs:
            # Forecast token usage for the job count by multiplying the per-job averages.
            forecast_input = avg_input_tokens * jobs
            forecast_cache_creation = avg_cache_creation_tokens * jobs
            forecast_cache_read = avg_cache_read_tokens * jobs
            forecast_output = avg_output_tokens * jobs

            # Forecast actual cost (with caching applied for input tokens):
            actual_input_cost_forecast = (
                (forecast_input) / 1_000_000 * pricing["input_tokens"]
                + (forecast_cache_creation) / 1_000_000 * pricing["cache_creation_input_tokens"]
                + (forecast_cache_read) / 1_000_000 * pricing["cache_read_input_tokens"]
            )

            # Without caching, all input-related tokens would be at base_input_rate:
            baseline_input_cost_forecast = (
                (forecast_input + forecast_cache_creation + forecast_cache_read) / 1_000_000 * pricing["input_tokens"]
            )

            caching_savings_forecast = baseline_input_cost_forecast - actual_input_cost_forecast

            forecast_output_cost = forecast_output / 1_000_000 * pricing["output_tokens"]
            total_forecast_cost = actual_input_cost_forecast + forecast_output_cost

            print(f"\nFor {jobs:,} jobs:")
            print("  Forecasted Token Usage:")
            print(f"    input_tokens                = {forecast_input:,.0f}")
            print(f"    cache_creation_input_tokens = {forecast_cache_creation:,.0f}")
            print(f"    cache_read_input_tokens     = {forecast_cache_read:,.0f}")
            print(f"    output_tokens               = {forecast_output:,.0f}")
            print("  Estimated Costs:")
            print(f"    Input Cost (with caching)   = ${actual_input_cost_forecast:,.2f}")
            print(f"    Output Cost                 = ${forecast_output_cost:,.2f}")
            print(f"    Grand Total Cost            = ${total_forecast_cost:,.2f}")
            print(f"    Caching Savings (input)     = ${caching_savings_forecast:,.2f}")
    else:
        print("No valid jobs to forecast future costs.")


if __name__ == "__main__":
    main()
