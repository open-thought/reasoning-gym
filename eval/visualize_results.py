#!/usr/bin/env python
"""
Visualization script for reasoning gym evaluation results.

This script generates visualizations from evaluation results stored in summary.json files.

Usage:
    python visualize_results.py --results-dir results/ [options]

Options:
    --output-dir DIR          Directory to save visualizations (default: visualizations)
    --plots PLOTS             Comma-separated list of plots to generate (default: all)
                              Available: radar,bar,violin,heatmap,dashboard,distribution,top_datasets
    --top-n N                 Number of datasets to show in top datasets plot (default: 15)
    --top-mode MODE           Mode for top datasets plot: hardest, easiest, variable (default: hardest)
    --format FORMAT           Output format for plots: png, pdf, svg (default: png)
    --dpi DPI                 DPI for output images (default: 300)
    --no-show                 Don't display plots, just save them
    --debug                   Enable debug logging
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("visualize_results")


def load_summaries(results_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all summary.json files from subdirectories.

    Args:
        results_dir: Directory containing model evaluation results

    Returns:
        Dictionary mapping model names to their summary data
    """
    summaries = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        logger.error(f"Results directory {results_dir} does not exist")
        return {}

    # Find all summary.json files
    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue

        summary_path = model_dir / "summary.json"
        if not summary_path.exists():
            logger.warning(f"No summary.json found in {model_dir}")
            continue

        try:
            # Extract model name from directory name (remove timestamp)
            model_name = re.sub(r"_\d{8}_\d{6}$", "", model_dir.name)
            # Replace underscores with slashes in model name for better display
            model_name = model_name.replace("_", "/")

            with open(summary_path, "r") as f:
                summary_data = json.load(f)

            # Check if summary has required fields
            if "dataset_best_scores" not in summary_data:
                logger.warning(f"Summary in {model_dir} is missing required fields")
                continue

            summaries[model_name] = summary_data
            logger.info(f"Loaded summary for {model_name}")

        except Exception as e:
            logger.error(f"Error loading summary from {model_dir}: {str(e)}")

    if not summaries:
        logger.error("No valid summary files found")

    return summaries


def get_dataset_categories(results_dir: str, summaries: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Group datasets by their categories based on directory structure.

    Args:
        results_dir: Directory containing model evaluation results
        summaries: Dictionary of model summaries

    Returns:
        Dictionary mapping category names to lists of dataset names
    """
    categories = {}
    results_path = Path(results_dir)

    # Get all dataset names from the first summary
    if not summaries:
        return {}

    first_summary = next(iter(summaries.values()))
    all_datasets = set(first_summary["dataset_best_scores"].keys())

    # Find categories by looking at directory structure
    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue

        # Look for category directories
        for category_dir in model_dir.iterdir():
            if not category_dir.is_dir():
                continue

            category_name = category_dir.name
            if category_name not in categories:
                categories[category_name] = []

            # Find all dataset JSON files in this category
            for dataset_file in category_dir.glob("*.json"):
                dataset_name = dataset_file.stem
                if dataset_name in all_datasets and dataset_name not in categories[category_name]:
                    categories[category_name].append(dataset_name)

    # Check if we found categories for all datasets
    categorized_datasets = set()
    for datasets in categories.values():
        categorized_datasets.update(datasets)

    uncategorized = all_datasets - categorized_datasets
    if uncategorized:
        logger.warning(f"Found {len(uncategorized)} datasets without categories")
        categories["uncategorized"] = list(uncategorized)

    return categories


def create_category_radar(summaries: Dict[str, Dict[str, Any]], categories: Dict[str, List[str]]) -> Figure:
    """Create a radar chart showing performance by category.

    Args:
        summaries: Dictionary of model summaries
        categories: Dictionary mapping categories to dataset lists

    Returns:
        Matplotlib figure
    """
    # Calculate average score per category for each model
    category_scores = {}
    for model_name, summary in summaries.items():
        category_scores[model_name] = {}
        for category, datasets in categories.items():
            scores = [summary["dataset_best_scores"].get(dataset, 0) for dataset in datasets]
            if scores:  # Avoid division by zero
                category_scores[model_name][category] = np.mean(scores)
            else:
                category_scores[model_name][category] = 0

    # Create radar chart
    categories_list = sorted(categories.keys())
    angles = np.linspace(0, 2 * np.pi, len(categories_list), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

    # Use a color cycle for different models
    colors = plt.cm.tab10.colors

    for i, (model_name, scores) in enumerate(category_scores.items()):
        color = colors[i % len(colors)]
        values = [scores[cat] for cat in categories_list]
        values += values[:1]  # Close the loop

        ax.plot(angles, values, linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories_list)

    # Add radial grid lines at 0.2, 0.4, 0.6, 0.8
    ax.set_rticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])
    ax.set_rlabel_position(0)  # Move radial labels away from plotted line

    # Add legend and title
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.title("Model Performance by Category", size=15)

    return fig


def create_overall_performance_bar(summaries: Dict[str, Dict[str, Any]]) -> Figure:
    """Create a bar chart of overall model performance.

    Args:
        summaries: Dictionary of model summaries

    Returns:
        Matplotlib figure
    """
    # Calculate overall average score for each model
    overall_scores = {}
    for model_name, summary in summaries.items():
        scores = list(summary["dataset_best_scores"].values())
        overall_scores[model_name] = np.mean(scores)

    # Sort models by performance
    sorted_models = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    models = [m[0] for m in sorted_models]
    scores = [m[1] for m in sorted_models]

    # Use a color gradient based on performance
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))

    bars = ax.bar(models, scores, color=colors)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.2%}", ha="center", va="bottom")

    ax.set_ylabel("Average Score")
    ax.set_ylim(0, max(scores) * 1.1)  # Add some space for labels
    plt.xticks(rotation=45, ha="right")

    plt.title("Overall Model Performance", size=15)
    plt.tight_layout()

    return fig


def create_top_datasets_comparison(summaries: Dict[str, Dict[str, Any]], n: int = 15, mode: str = "hardest") -> Figure:
    """Create a bar chart comparing performance on top N datasets.

    Args:
        summaries: Dictionary of model summaries
        n: Number of datasets to show
        mode: Selection mode - 'hardest', 'easiest', or 'variable'

    Returns:
        Matplotlib figure
    """
    if not summaries:
        logger.error("No summaries provided")
        return plt.figure()

    # Calculate average score across all models for each dataset
    dataset_avg_scores = {}
    for dataset in next(iter(summaries.values()))["dataset_best_scores"].keys():
        scores = [summary["dataset_best_scores"].get(dataset, 0) for summary in summaries.values()]
        dataset_avg_scores[dataset] = np.mean(scores)

    # Select top N datasets based on mode
    if mode == "hardest":
        # Select datasets with lowest average scores
        selected_datasets = sorted(dataset_avg_scores.items(), key=lambda x: x[1])[:n]
    elif mode == "easiest":
        # Select datasets with highest average scores
        selected_datasets = sorted(dataset_avg_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    else:  # 'variable'
        # Select datasets with highest variance in scores
        dataset_variances = {}
        for dataset in next(iter(summaries.values()))["dataset_best_scores"].keys():
            scores = [summary["dataset_best_scores"].get(dataset, 0) for summary in summaries.values()]
            dataset_variances[dataset] = np.var(scores)
        selected_datasets = sorted(dataset_variances.items(), key=lambda x: x[1], reverse=True)[:n]
        selected_datasets = [(dataset, dataset_avg_scores[dataset]) for dataset, _ in selected_datasets]

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, n * 0.5))

    datasets = [d[0] for d in selected_datasets]
    x = np.arange(len(datasets))
    width = 0.8 / len(summaries)

    # Use a color cycle for different models
    colors = plt.cm.tab10.colors

    for i, (model_name, summary) in enumerate(summaries.items()):
        scores = [summary["dataset_best_scores"].get(dataset, 0) for dataset, _ in selected_datasets]
        ax.barh(
            x + i * width - 0.4 + width / 2, scores, width, label=model_name, color=colors[i % len(colors)], alpha=0.8
        )

    ax.set_yticks(x)
    ax.set_yticklabels(datasets)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1)

    # Add legend and title
    plt.legend(loc="upper right")
    title = f'Model Performance on {n} {"Hardest" if mode=="hardest" else "Easiest" if mode=="easiest" else "Most Variable"} Datasets'
    plt.title(title, size=15)

    plt.tight_layout()
    return fig


def create_performance_distribution_violin(summaries: Dict[str, Dict[str, Any]]) -> Figure:
    """Create a violin plot showing score distribution for each model.

    Args:
        summaries: Dictionary of model summaries

    Returns:
        Matplotlib figure
    """
    # Prepare data for violin plot
    data = []
    labels = []

    for model_name, summary in summaries.items():
        scores = list(summary["dataset_best_scores"].values())
        data.append(scores)
        labels.append(model_name)

    # Create violin plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Use a color cycle
    colors = plt.cm.tab10.colors

    parts = ax.violinplot(data, showmeans=True, showmedians=True)

    # Customize violin plot
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)

    # Add labels
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Score Distribution")
    ax.set_ylim(0, 1)

    # Add grid for better readability
    ax.yaxis.grid(True)

    # Add mean and median to legend
    legend_elements = [
        Patch(facecolor="black", edgecolor="black", label="Mean", alpha=0.3),
        Patch(facecolor="white", edgecolor="black", label="Median"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.title("Distribution of Scores Across All Datasets", size=15)
    plt.tight_layout()

    return fig


def create_performance_heatmap(summaries: Dict[str, Dict[str, Any]], categories: Dict[str, List[str]]) -> Figure:
    """Create a heatmap of model performance across datasets.

    Args:
        summaries: Dictionary of model summaries
        categories: Dictionary mapping categories to dataset lists

    Returns:
        Matplotlib figure
    """
    if not summaries:
        logger.error("No summaries provided")
        return plt.figure()

    # Get all dataset names
    all_datasets = []
    for category, datasets in sorted(categories.items()):
        all_datasets.extend(sorted(datasets))

    # Sort models by overall performance
    overall_scores = {}
    for model_name, summary in summaries.items():
        scores = list(summary["dataset_best_scores"].values())
        overall_scores[model_name] = np.mean(scores)
    models = [item[0] for item in sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)]

    # Create score matrix
    score_matrix = np.zeros((len(models), len(all_datasets)))

    for i, model in enumerate(models):
        for j, dataset in enumerate(all_datasets):
            score_matrix[i, j] = summaries[model]["dataset_best_scores"].get(dataset, 0)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(20, len(all_datasets) * 0.25), max(8, len(models) * 0.5)))

    im = ax.imshow(score_matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Score", rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(all_datasets)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(all_datasets, rotation=90, fontsize=8)
    ax.set_yticklabels(models)

    # Add category separators and labels
    current_idx = 0
    for category, datasets in sorted(categories.items()):
        if datasets:
            # Add vertical line after each category
            next_idx = current_idx + len(datasets)
            if next_idx < len(all_datasets):
                ax.axvline(x=next_idx - 0.5, color="white", linestyle="-", linewidth=2)

            # Add category label
            middle_idx = current_idx + len(datasets) / 2 - 0.5
            ax.text(
                middle_idx,
                -0.5,
                category,
                ha="center",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

            current_idx = next_idx

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, len(all_datasets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)

    plt.title("Model Performance Heatmap", size=15)
    plt.tight_layout()

    return fig


def create_dashboard(summaries: Dict[str, Dict[str, Any]], categories: Dict[str, List[str]]) -> Figure:
    """Create a comprehensive dashboard with multiple visualizations.

    Args:
        summaries: Dictionary of model summaries
        categories: Dictionary mapping categories to dataset lists

    Returns:
        Matplotlib figure
    """
    if not summaries:
        logger.error("No summaries provided")
        return plt.figure()

    fig = plt.figure(figsize=(20, 15))

    # 1. Overall performance comparison
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    models = []
    scores = []
    for model_name, summary in summaries.items():
        models.append(model_name)
        scores.append(np.mean(list(summary["dataset_best_scores"].values())))

    # Sort by performance
    sorted_indices = np.argsort(scores)[::-1]
    models = [models[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]

    # Use a color gradient based on performance
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))

    bars = ax1.bar(models, scores, color=colors)
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.2%}", ha="center", va="bottom", fontsize=8
        )

    ax1.set_ylabel("Average Score")
    ax1.set_ylim(0, max(scores) * 1.1)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax1.set_title("Overall Model Performance", size=12)

    # 2. Top 10 hardest datasets comparison
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    # Calculate average score across all models for each dataset
    dataset_avg_scores = {}
    for dataset in next(iter(summaries.values()))["dataset_best_scores"].keys():
        scores = [summary["dataset_best_scores"].get(dataset, 0) for summary in summaries.values()]
        dataset_avg_scores[dataset] = np.mean(scores)

    # Select 10 hardest datasets
    hardest_datasets = sorted(dataset_avg_scores.items(), key=lambda x: x[1])[:10]

    datasets = [d[0] for d in hardest_datasets]
    x = np.arange(len(datasets))
    width = 0.8 / len(summaries)

    # Use a color cycle for different models
    colors = plt.cm.tab10.colors

    for i, (model_name, summary) in enumerate(summaries.items()):
        scores = [summary["dataset_best_scores"].get(dataset, 0) for dataset, _ in hardest_datasets]
        ax2.barh(
            x + i * width - 0.4 + width / 2, scores, width, label=model_name, color=colors[i % len(colors)], alpha=0.8
        )

    ax2.set_yticks(x)
    ax2.set_yticklabels(datasets, fontsize=8)
    ax2.set_xlabel("Score")
    ax2.set_xlim(0, 1)
    ax2.set_title("Performance on 10 Hardest Datasets", size=12)
    ax2.legend(fontsize=8)

    # 3. Category radar chart
    ax3 = plt.subplot2grid((2, 2), (1, 0), polar=True)

    # Calculate average score per category for each model
    category_scores = {}
    for model_name, summary in summaries.items():
        category_scores[model_name] = {}
        for category, datasets in categories.items():
            scores = [summary["dataset_best_scores"].get(dataset, 0) for dataset in datasets]
            if scores:  # Avoid division by zero
                category_scores[model_name][category] = np.mean(scores)
            else:
                category_scores[model_name][category] = 0

    # Create radar chart
    categories_list = sorted(categories.keys())
    angles = np.linspace(0, 2 * np.pi, len(categories_list), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    for i, (model_name, scores) in enumerate(category_scores.items()):
        color = colors[i % len(colors)]
        values = [scores.get(cat, 0) for cat in categories_list]
        values += values[:1]  # Close the loop

        ax3.plot(angles, values, linewidth=2, label=model_name, color=color)
        ax3.fill(angles, values, alpha=0.1, color=color)

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories_list, fontsize=8)
    ax3.set_title("Performance by Category", size=12)

    # 4. Performance distribution violin plot
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    data = []
    labels = []

    for model_name, summary in summaries.items():
        scores = list(summary["dataset_best_scores"].values())
        data.append(scores)
        labels.append(model_name)

    parts = ax4.violinplot(data, showmeans=True, showmedians=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)

    ax4.set_xticks(np.arange(1, len(labels) + 1))
    ax4.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax4.set_ylabel("Score Distribution")
    ax4.set_ylim(0, 1)
    ax4.yaxis.grid(True)
    ax4.set_title("Distribution of Scores", size=12)

    plt.tight_layout()
    plt.suptitle("Model Evaluation Dashboard", size=16, y=0.98)
    plt.subplots_adjust(top=0.9)

    return fig


def create_comparison_plot(
    summaries: Dict[str, Dict[str, Any]],
    other_summaries: Dict[str, Dict[str, Any]],
    categories: Optional[Dict[str, List[str]]] = None,
) -> Figure:
    """
    Build a heat-map of per-category score differences (scaled to –100 … 100).

    Rows  : model IDs present in both `summaries` and `other_summaries`
    Cols  : category names (`categories`)
    Value : 100 * (mean(score in summaries) − mean(score in other_summaries))

    A numeric annotation (rounded to 2 dp) is rendered in every cell.
    """
    if not summaries or not other_summaries:
        logger.error("No summaries provided for comparison")
        return plt.figure()

    if categories is None:
        all_ds = next(iter(summaries.values()))["dataset_best_scores"].keys()
        categories = {"all": list(all_ds)}

    # models appearing in both result sets
    common_models = [m for m in summaries if m in other_summaries]
    if not common_models:
        logger.error("No overlapping model IDs between the two result sets.")
        return plt.figure()

    # sort models by overall performance
    overall_scores = {}
    for model_name, summary in summaries.items():
        scores = list(summary["dataset_best_scores"].values())
        overall_scores[model_name] = np.mean(scores)
    models = [item[0] for item in sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)]
    common_models = [m for m in models if m in common_models]

    category_list = sorted(categories.keys())
    diff_matrix = np.zeros((len(common_models), len(category_list)))

    # compute 100 × Δ
    for i, model in enumerate(common_models):
        cur_scores = summaries[model]["dataset_best_scores"]
        base_scores = other_summaries[model]["dataset_best_scores"]

        for j, cat in enumerate(category_list):
            ds = categories[cat]
            cur_mean = np.mean([cur_scores.get(d, 0.0) for d in ds]) if ds else 0.0
            base_mean = np.mean([base_scores.get(d, 0.0) for d in ds]) if ds else 0.0
            diff_matrix[i, j] = 100 * (cur_mean - base_mean)  # scale to -100 … 100

    # ---------------------------------------------------------------- Plot
    fig, ax = plt.subplots(figsize=(max(8, len(category_list) * 1.2), max(6, len(common_models) * 0.5)))

    im = ax.imshow(diff_matrix, cmap="coolwarm", aspect="auto", vmin=-100, vmax=100)

    # colour-bar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Δ score (percentage-points)", rotation=-90, va="bottom")

    # ticks / labels
    ax.set_xticks(np.arange(len(category_list)), labels=category_list, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(common_models)), labels=common_models)

    # grid for readability
    ax.set_xticks(np.arange(-0.5, len(category_list), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(common_models), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)

    # annotate each cell
    for i in range(len(common_models)):
        for j in range(len(category_list)):
            value = diff_matrix[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="black" if abs(value) < 50 else "white",
                fontsize=8,
            )

    ax.set_title("Per-Category Performance Δ (hard - easy)", fontsize=14)
    plt.tight_layout()
    return fig


def save_figure(fig: Figure, output_dir: str, name: str, fmt: str = "png", dpi: int = 300) -> str:
    """Save a figure to a file.

    Args:
        fig: Matplotlib figure to save
        output_dir: Directory to save the figure
        name: Base name for the figure file
        fmt: File format (png, pdf, svg)
        dpi: DPI for raster formats

    Returns:
        Path to the saved file
    """

    # Create filename
    filename = f"{name.replace('/', '-')}.{fmt}"
    filepath = output_dir / filename

    # Save figure
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved {filepath}")

    return filepath


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate visualizations from evaluation results")
    parser.add_argument("--results-dir", required=True, help="Directory containing evaluation results")
    parser.add_argument("--output-dir", default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--plots", default="all", help="Comma-separated list of plots to generate")
    parser.add_argument("--top-n", type=int, default=15, help="Number of datasets to show in top datasets plot")
    parser.add_argument(
        "--top-mode", default="hardest", choices=["hardest", "easiest", "variable"], help="Mode for top datasets plot"
    )
    parser.add_argument("--compare-results-dir", help="Directory to compare results with", default=None)
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Output format for plots")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output images")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots, just save them")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Load summaries
    logger.info(f"Loading summaries from {args.results_dir}")
    summaries = load_summaries(args.results_dir)

    args.output_dir = Path(args.output_dir)
    if not args.output_dir.exists():
        logger.info(f"Creating output directory {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if not summaries:
        logger.error("No valid summaries found. Exiting.")
        return 1

    logger.info(f"Found {len(summaries)} model summaries")

    # Get dataset categories
    categories = get_dataset_categories(args.results_dir, summaries)
    logger.info(f"Found {len(categories)} dataset categories")

    # Determine which plots to generate
    if args.plots.lower() == "all":
        plots_to_generate = ["radar", "bar", "violin", "heatmap", "dashboard", "top_datasets", "compare"]
    else:
        plots_to_generate = [p.strip().lower() for p in args.plots.split(",")]

    logger.info(f"Generating plots: {', '.join(plots_to_generate)}")

    # Generate and save plots
    for plot_type in plots_to_generate:
        try:
            if plot_type == "radar":
                fig = create_category_radar(summaries, categories)
                save_figure(fig, args.output_dir, "category_radar", args.format, args.dpi)

            elif plot_type == "bar":
                fig = create_overall_performance_bar(summaries)
                save_figure(fig, args.output_dir, "overall_performance", args.format, args.dpi)

            elif plot_type == "violin":
                fig = create_performance_distribution_violin(summaries)
                save_figure(fig, args.output_dir, "score_distribution", args.format, args.dpi)

            elif plot_type == "heatmap":
                fig = create_performance_heatmap(summaries, categories)
                save_figure(fig, args.output_dir, "performance_heatmap", args.format, args.dpi)

            elif plot_type == "dashboard":
                fig = create_dashboard(summaries, categories)
                save_figure(fig, args.output_dir, "evaluation_dashboard", args.format, args.dpi)

            elif plot_type == "top_datasets":
                fig = create_top_datasets_comparison(summaries, args.top_n, args.top_mode)
                save_figure(fig, args.output_dir, f"top_{args.top_n}_{args.top_mode}_datasets", args.format, args.dpi)

            elif plot_type == "compare":
                assert args.compare_results_dir, "Comparison directory is required for compare plot"
                other_summaries = load_summaries(args.compare_results_dir)
                if not other_summaries:
                    logger.error("No valid summaries found in comparison directory. Exiting.")
                    return 1
                fig = create_comparison_plot(summaries, other_summaries, categories)
                save_figure(fig, args.output_dir, "model_category_delta_heatmap", args.format, args.dpi)

            else:
                logger.warning(f"Unknown plot type: {plot_type}")
                continue

            # Show plot if requested
            if not args.no_show:
                plt.show()
            else:
                plt.close(fig)

        except Exception as e:
            logger.error(f"Error generating {plot_type} plot: {str(e)}")
            if args.debug:
                import traceback

                traceback.print_exc()

    logger.info(f"All visualizations saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    import sys

    sys.exit(exit_code)
