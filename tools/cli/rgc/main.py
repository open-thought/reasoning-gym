"""Main entry point for the Reasoning Gym CLI."""

from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

# Initialize Typer app
app = typer.Typer(
    name="rgc", help="Reasoning Gym CLI - Manage and monitor reasoning gym experiments", add_completion=True
)

# Initialize Rich console
console = Console()

# Experiments group
experiments_app = typer.Typer(help="Manage experiments")
app.add_typer(experiments_app, name="experiments")

# Config group
config_app = typer.Typer(help="Manage configurations")
app.add_typer(config_app, name="config")


@experiments_app.command("list")
def list_experiments():
    """List all registered experiments with their status."""
    table = Table(title="Registered Experiments")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Datasets", style="magenta")

    # TODO: Implement actual experiment listing
    table.add_row("example-exp", "Running", "chain_sum")

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
                config = yaml.safe_load(f)
            console.print(f"Created experiment [cyan]{name}[/] from [green]{config_file}[/]")
        except Exception as e:
            console.print(f"[red]Error loading config file:[/] {str(e)}")
    else:
        console.print("[yellow]Interactive creation not implemented yet[/]")


@experiments_app.command("delete")
def delete_experiment(
    name: str = typer.Argument(..., help="Name of the experiment to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without confirmation"),
):
    """Delete an experiment."""
    if not force:
        delete = typer.confirm(f"Delete experiment {name}?")
        if not delete:
            raise typer.Abort()

    # TODO: Implement actual deletion
    console.print(f"[green]Deleted experiment[/] [cyan]{name}[/]")


@experiments_app.command("show")
def show_experiment(
    name: str = typer.Argument(..., help="Name of the experiment"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch mode - update continuously"),
):
    """Show experiment details."""
    # TODO: Implement actual experiment details
    console.print(f"[cyan]Experiment:[/] {name}")
    if watch:
        console.print("[yellow]Watch mode not implemented yet[/]")


@config_app.command("get")
def get_config(experiment: str = typer.Argument(..., help="Name of the experiment")):
    """Get current configuration."""
    # TODO: Implement actual config retrieval
    console.print(f"[cyan]Configuration for {experiment}[/]")
    console.print("[yellow]Not implemented yet[/]")


@config_app.command("edit")
def edit_config(experiment: str = typer.Argument(..., help="Name of the experiment")):
    """Interactive configuration editor."""
    console.print(f"[cyan]Editing configuration for {experiment}[/]")
    console.print("[yellow]Interactive editing not implemented yet[/]")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
"""Main entry point for the Reasoning Gym CLI."""

import os
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

from reasoning_gym.composite import CompositeConfig, DatasetSpec

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

# Initialize Rich console
console = Console()

# Default server settings
DEFAULT_SERVER = "http://localhost:8000"
API_KEY = os.getenv("REASONING_GYM_API_KEY", "default-key")


def get_headers():
    """Get headers for API requests."""
    return {"X-API-Key": API_KEY}


@experiments_app.command("list")
def list_experiments():
    """List all registered experiments with their status."""
    table = Table(title="Registered Experiments")
    table.add_column("Name", style="cyan")
    table.add_column("Datasets", style="magenta")
    table.add_column("Size", style="blue")
    table.add_column("Seed", style="green")

    # TODO: Implement actual API call
    # For now show example data
    table.add_row(
        "example-exp",
        "chain_sum",
        "500",
        "42"
    )
    
    console.print(table)


@experiments_app.command("create")
def create_experiment(
    name: str = typer.Argument(..., help="Name of the experiment"),
    config_file: Optional[str] = typer.Option(
        None,
        "--file", "-f",
        help="YAML configuration file"
    ),
):
    """Create a new experiment."""
    if config_file:
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            console.print(f"Created experiment [cyan]{name}[/] from [green]{config_file}[/]")
        except Exception as e:
            console.print(f"[red]Error loading config file:[/] {str(e)}")
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
            
            datasets[ds_name] = {
                "weight": weight,
                "config": config
            }
        
        # Create experiment config
        exp_config = {
            "name": name,
            "size": int(size),
            "seed": int(seed) if seed else None,
            "datasets": datasets
        }
        
        # Show final config
        console.print("\nFinal configuration:")
        console.print(Syntax(yaml.dump(exp_config), "yaml"))
        
        if Confirm.ask("Create experiment with this configuration?"):
            # TODO: Implement actual API call
            console.print(f"[green]Created experiment[/] [cyan]{name}[/]")
        else:
            console.print("[yellow]Experiment creation cancelled[/]")
            raise typer.Exit()


@experiments_app.command("delete")
def delete_experiment(
    name: str = typer.Argument(..., help="Name of the experiment to delete"),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force deletion without confirmation"
    ),
):
    """Delete an experiment."""
    if not force and not Confirm.ask(f"Delete experiment [cyan]{name}[/]?"):
        raise typer.Exit()
    
    # TODO: Implement actual API call
    console.print(f"[green]Deleted experiment[/] [cyan]{name}[/]")


@experiments_app.command("show")
def show_experiment(
    name: str = typer.Argument(..., help="Name of the experiment"),
):
    """Show experiment details."""
    # TODO: Implement actual API call
    # For now show example data
    example_config = {
        "name": name,
        "size": 500,
        "seed": 42,
        "datasets": {
            "chain_sum": {
                "weight": 1.0,
                "config": {
                    "min_terms": 2,
                    "max_terms": 4,
                    "min_digits": 1,
                    "max_digits": 2,
                    "allow_negation": False
                }
            }
        }
    }
    
    console.print(Syntax(yaml.dump(example_config), "yaml"))


@config_app.command("get")
def get_config(
    experiment: str = typer.Argument(..., help="Name of the experiment"),
):
    """Get current configuration."""
    # TODO: Implement actual API call
    console.print(f"[cyan]Configuration for {experiment}[/]")
    console.print("[yellow]Not implemented yet[/]")


@config_app.command("edit")
def edit_config(
    experiment: str = typer.Argument(..., help="Name of the experiment"),
    dataset: str = typer.Argument(..., help="Name of the dataset to edit"),
):
    """Interactive configuration editor."""
    # TODO: Implement actual API call to get current config
    current_config = {
        "min_terms": 2,
        "max_terms": 4,
        "min_digits": 1,
        "max_digits": 2,
        "allow_negation": False
    }
    
    console.print(f"\nCurrent configuration for [cyan]{dataset}[/]:")
    console.print(Syntax(yaml.dump(current_config), "yaml"))
    
    # Interactive editing
    new_config = {}
    for key, value in current_config.items():
        new_value = Prompt.ask(
            f"{key}",
            default=str(value),
            show_default=True
        )
        
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
        # TODO: Implement actual API call
        console.print("[green]Configuration updated successfully[/]")
    else:
        console.print("[yellow]Update cancelled[/]")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
