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
