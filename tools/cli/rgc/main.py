"""Main entry point for the Reasoning Gym CLI."""

import os
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

from tools.server.models import DatasetConfigUpdate, ExperimentCreate

# Initialize Typer apps
app = typer.Typer(
    name="rgc",
    help="Reasoning Gym CLI - Manage and monitor reasoning gym experiments",
    add_completion=True,
)
experiments_app = typer.Typer(help="Manage experiments")
config_app = typer.Typer(help="Manage configurations")

app.add_typer(experiments_app, name="experiments")
app.add_typer(config_app, name="config")


@app.command("health")
def check_health():
    """Check server connection and health status."""
    try:
        if client.check_health():
            console.print("[green]Server is healthy[/]")
        else:
            console.print("[red]Server is not responding correctly[/]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error connecting to server: {e}[/]")
        raise typer.Exit(1)


# Initialize client and console
from .client import RGClient

client = RGClient()
console = Console()


@experiments_app.command("list")
def list_experiments():
    """List all registered experiments with their status."""
    table = Table(title="Registered Experiments")
    table.add_column("Name", style="cyan")
    table.add_column("Datasets", style="magenta")
    table.add_column("Size", style="blue")
    table.add_column("Seed", style="green")

    try:
        experiments = client.list_experiments()
        for exp_name in experiments.experiments:
            try:
                config = client.get_experiment_config(exp_name)
                datasets = ", ".join(config.datasets.keys())
                table.add_row(exp_name, datasets, str(config.size), str(config.seed or ""))
            except Exception as e:
                console.print(f"[yellow]Warning: Could not get config for {exp_name}: {e}[/]")
                table.add_row(exp_name, "?", "?", "?")
    except Exception as e:
        console.print(f"[red]Error listing experiments: {e}[/]")
        raise typer.Exit(1)

    console.print(table)


@experiments_app.command("create")
def create_experiment(
    name: str = typer.Argument(..., help="Name of the experiment"),
    config_file: Optional[str] = typer.Option(None, "--file", "-f", help="YAML configuration file"),
):
    """Create a new experiment."""
    if config_file:
        try:
            with open(config_file, "r") as f:
                exp_config = yaml.safe_load(f)
            config = ExperimentCreate(**exp_config)
            response = client.create_experiment(name, config)
            console.print(f"[green]Created experiment[/] [cyan]{response.name}[/]")
        except Exception as e:
            console.print(f"[red]Error creating experiment: {e}[/]")
            raise typer.Exit(1)
    else:
        # Interactive creation
        size = Prompt.ask("Dataset size", default="500")
        seed = Prompt.ask("Random seed (optional)", default="")

        datasets = {}
        while Confirm.ask("Add dataset?"):
            ds_name = Prompt.ask("Dataset name")
            weight = float(Prompt.ask("Weight", default="1.0"))

            # Get dataset-specific config
            console.print("\nEnter dataset configuration:")
            config = {}
            while Confirm.ask("Add config parameter?"):
                key = Prompt.ask("Parameter name")
                value = Prompt.ask("Parameter value")
                try:
                    # Try to convert to appropriate type
                    if value.isdigit():
                        value = int(value)
                    elif value.lower() in ("true", "false"):
                        value = value.lower() == "true"
                    elif "." in value and value.replace(".", "").isdigit():
                        value = float(value)
                except ValueError:
                    pass
                config[key] = value

            datasets[ds_name] = {"weight": weight, "config": config}

        # Create experiment config
        exp_config = {"name": name, "size": int(size), "seed": int(seed) if seed else None, "datasets": datasets}

        # Show final config
        console.print("\nFinal configuration:")
        console.print(Syntax(yaml.dump(exp_config), "yaml"))

        if Confirm.ask("Create experiment with this configuration?"):
            try:
                config = ExperimentCreate(**exp_config)
                response = client.create_experiment(name, config)
                console.print(f"[green]Created experiment[/] [cyan]{response.name}[/]")
            except Exception as e:
                console.print(f"[red]Error creating experiment: {e}[/]")
                raise typer.Exit(1)
        else:
            console.print("[yellow]Experiment creation cancelled[/]")
            raise typer.Exit()


@experiments_app.command("delete")
def delete_experiment(
    name: str = typer.Argument(..., help="Name of the experiment to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without confirmation"),
):
    """Delete an experiment."""
    if not force and not Confirm.ask(f"Delete experiment [cyan]{name}[/]?"):
        raise typer.Exit()

    try:
        client.delete_experiment(name)
        console.print(f"[green]Deleted experiment[/] [cyan]{name}[/]")
    except Exception as e:
        console.print(f"[red]Error deleting experiment: {e}[/]")
        raise typer.Exit(1)


@experiments_app.command("show")
def show_experiment(
    name: str = typer.Argument(..., help="Name of the experiment"),
):
    """Show experiment details."""
    try:
        config = client.get_experiment_config(name)
        console.print(Syntax(yaml.dump(config.model_dump()), "yaml"))
    except Exception as e:
        console.print(f"[red]Error getting experiment config: {e}[/]")
        raise typer.Exit(1)


@config_app.command("edit")
def edit_config(
    experiment: str = typer.Argument(..., help="Name of the experiment"),
    dataset: str = typer.Argument(..., help="Name of the dataset to edit"),
):
    """Interactive configuration editor."""
    try:
        exp_config = client.get_experiment_config(experiment)
        if dataset not in exp_config.datasets:
            console.print(f"[red]Dataset {dataset} not found in experiment[/]")
            raise typer.Exit(1)
        current_config = exp_config.datasets[dataset]["config"]

        console.print(f"\nCurrent configuration for [cyan]{dataset}[/]:")
        console.print(Syntax(yaml.dump(current_config), "yaml"))

        # Interactive editing
        new_config = {}
        for key, value in current_config.items():
            new_value = Prompt.ask(f"{key}", default=str(value), show_default=True)

            # Try to convert to appropriate type
            try:
                if isinstance(value, bool):
                    new_value = new_value.lower() == "true"
                elif isinstance(value, int):
                    new_value = int(new_value)
                elif isinstance(value, float):
                    new_value = float(new_value)
            except ValueError:
                console.print(f"[yellow]Warning: Could not convert {new_value} to {type(value)}[/]")

            new_config[key] = new_value

        # Show changes
        console.print("\nNew configuration:")
        console.print(Syntax(yaml.dump(new_config), "yaml"))

        if Confirm.ask("Apply these changes?"):
            try:
                config_update = DatasetConfigUpdate(config=new_config)
                client.update_dataset_config(experiment, dataset, config_update)
                console.print("[green]Configuration updated successfully[/]")
            except Exception as e:
                console.print(f"[red]Error updating configuration: {e}[/]")
                raise typer.Exit(1)
        else:
            console.print("[yellow]Update cancelled[/]")

    except Exception as e:
        console.print(f"[red]Error getting experiment configuration: {e}[/]")
        raise typer.Exit(1)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
