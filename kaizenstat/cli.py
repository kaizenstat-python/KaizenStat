import typer
import sys
import os
import json
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

app = typer.Typer(
    help="🧠 KaizenStat - Intelligent AutoML & Data Diagnostics | https://www.kaizenstat.com",
    add_completion=False,
)

def header():
    console.print(
        Panel.fit(
            "[bold cyan]🧠 KaizenStat CLI[/bold cyan]\n[dim]Intelligent AutoML & Data Quality[/dim]",
            border_style="cyan"
        )
    )

@app.command()
def analyze(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Argument(..., help="Name of target column"),
):
    """
    Analyze dataset and generate AI insights.
    """
    header()
    console.print("🔍 [bold cyan]Analyzing dataset...[/bold cyan]")
    
    from kaizenstat.core import KaizenStat
    try:
        KaizenStat.analyze(file, target)
        console.print("[bold green]✔ Analysis complete[/bold green]")
    except Exception as e:
        console.print(f"[bold bright_red]✖ Error during analysis:[/] {e}")
        sys.exit(1)

@app.command()
def ask(
    query: str = typer.Argument(..., help="The question to ask the AI engine"),
    followup: bool = typer.Option(False, "--followup", "-f", help="Maintain conversation history memory"),
):
    """
    Ask questions about your dataset.
    """
    from kaizenstat.core import KaizenStat
    console.print(f"\n[bold bright_yellow]❓ {query}[/bold bright_yellow]\n")
    try:
        if followup:
            KaizenStat.ask_followup(query)
        else:
            KaizenStat.ask(query)
    except Exception as e:
        console.print(f"[bold bright_red]✖ Error querying AI:[/] {e}")
        sys.exit(1)

@app.command()
def improve():
    """
    Get next best actions and improvement plan.
    """
    console.print("🚀 [bold cyan]Generating improvement plan...[/bold cyan]")
    from kaizenstat.core import KaizenStat
    try:
        KaizenStat.improve()
        console.print("[bold green]✔ Plan ready[/bold green]")
    except Exception as e:
        console.print(f"[bold bright_red]✖ Error generating plan:[/] {e}")
        sys.exit(1)

@app.command()
def status():
    """
    Show current system and dataset state.
    """
    try:
        with open(".kaizenstat_context.json") as f:
            data = json.load(f)
    except Exception:
        console.print("[bold bright_red]✖ No active dataset context found. Run 'kz auto' or 'kz analyze' first.[/bold bright_red]")
        return

    console.print("\n[bold cyan]📊 Current Status[/bold cyan]\n")
    console.print(f"Model: [bold green]{data.get('model', 'None')}[/bold green]")
    console.print(f"Score: [bold green]{data.get('score', 0.0):.4f}[/bold green]")
    issues = data.get('issues', [])
    main_problem = ", ".join(issues) if issues else "None"
    console.print(f"Issues: [bold bright_red]{main_problem}[/bold bright_red]")
    shape = data.get('shape', [0, 0])
    console.print(f"Dataset Shape: [bold cyan]{shape[0]} rows x {shape[1]} columns[/bold cyan]")

@app.command()
def reset():
    """
    Reset conversational memory and active dataset context.
    """
    files = [
        ".kaizenstat_context.json",
        ".kaizenstat_history.json",
    ]
    removed = []
    for f in files:
        if os.path.exists(f):
            try:
                os.remove(f)
                removed.append(f)
            except Exception:
                pass
    if removed:
        console.print("[bold green]✔ Memory cleared[/bold green]")
    else:
        console.print("[bold cyan]ℹ Memory already clean[/bold cyan]")

@app.command()
def audit(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Option(None, "--target", "-t", help="Target column name (optional)"),
):
    """
    Audit dataset for issues.
    """
    header()
    from kaizenstat.core import KaizenStat
    try:
        df = pd.read_csv(file)
        KaizenStat.audit(df, target=target)
    except Exception as e:
        console.print(f"[bold bright_red]✖ Error during audit:[/] {e}")
        sys.exit(1)

@app.command()
def heal(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Option(None, "--target", "-t", help="Target column name (optional)"),
    output: str = typer.Option(None, "--output", "-o", help="Output path for cleaned CSV"),
):
    """
    Auto-clean and repair dataset.
    """
    header()
    from kaizenstat.core import KaizenStat
    try:
        df = pd.read_csv(file)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]🩹 Cleaning and healing dataset...[/bold cyan]"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("", total=None)
            healed_df = KaizenStat.heal(df, target=target)
            
        output_path = output or (
            file[:-4] + "_healed.csv"
            if file.endswith(".csv")
            else file + "_healed.csv"
        )
        healed_df.to_csv(output_path, index=False)
        console.print(f"[bold green]✔ Saved healed dataset →[/bold green] {output_path}")
    except Exception as e:
        console.print(f"[bold bright_red]✖ Error during healing:[/] {e}")
        sys.exit(1)

@app.command()
def benchmark(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Argument(..., help="Name of target column"),
):
    """
    Benchmark ML models on dataset.
    """
    header()
    from kaizenstat.core import KaizenStat
    try:
        df = pd.read_csv(file)
        KaizenStat.benchmark(df, target=target)
    except Exception as e:
        console.print(f"[bold bright_red]✖ Error during benchmark:[/] {e}")
        sys.exit(1)

@app.command()
def auto(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Argument(..., help="Name of target column"),
):
    """
    Full pipeline: load, audit, heal, and benchmark in a single run.
    """
    from kaizenstat.core import KaizenStat
    try:
        KaizenStat.auto(file, target)
        console.print("[bold green]✔ Pipeline run complete[/bold green]")
    except Exception as e:
        console.print(f"[bold bright_red]✖ Error running pipeline:[/] {e}")
        sys.exit(1)

@app.command()
def explain(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Argument(..., help="Name of target column"),
):
    """
    Generate a plain-English executive summary.
    """
    header()
    console.print("[bold cyan]ℹ Generating explanation report...[/bold cyan]")
    from kaizenstat.core import KaizenStat
    try:
        KaizenStat.explain(file, target)
    except Exception as e:
        console.print(f"[bold bright_red]✖ Error generating explanation:[/] {e}")
        sys.exit(1)

@app.command()
def codegen(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Argument(..., help="Name of target column"),
    output: str = typer.Option("pipeline.py", "--output", "-o", help="Output .py file path"),
):
    """
    Generate standalone Python script reproducing the pipeline.
    """
    header()
    from kaizenstat.core import KaizenStat
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]📝 Generating training script to {output}...[/bold cyan]"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("", total=None)
            KaizenStat.codegen(file, target, output_path=output)
        console.print(f"[bold green]✔ Code generated successfully[/bold green]")
    except Exception as e:
        console.print(f"[bold bright_red]✖ Error generating code:[/] {e}")
        sys.exit(1)

@app.command(name="export-model")
def export_model(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Argument(..., help="Name of target column"),
    output: str = typer.Option("model.joblib", "--output", "-o", help="Output .joblib file path"),
):
    """
    Train and export the best model to disk.
    """
    header()
    from kaizenstat.core import KaizenStat
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]💾 Training and exporting best model to {output}...[/bold cyan]"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("", total=None)
            KaizenStat.auto(file, target)
            KaizenStat.save_model(path=output)
        console.print(f"[bold green]✔ Model exported successfully[/bold green]")
    except Exception as e:
        console.print(f"[bold bright_red]✖ Error exporting model:[/] {e}")
        sys.exit(1)

@app.command()
def report(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Argument(..., help="Name of target column"),
    output: str = typer.Option("report.html", "--output", "-o", help="Output .html file path"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Automatically open report in browser"),
    serve: bool = typer.Option(False, "--serve", "-s", help="Serve report locally on a temporary port"),
):
    """
    Generate interactive HTML profiling report.
    """
    header()
    from kaizenstat.core import KaizenStat
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]📊 Building HTML profiling report to {output}...[/bold cyan]"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("", total=None)
            KaizenStat.report(file, target, output_path=output, open_browser=open_browser and not serve)
        console.print(f"[bold green]✔ Report generated successfully[/bold green]")
        
        if serve:
            KaizenStat.serve_report(output)
    except Exception as e:
        console.print(f"[bold bright_red]✖ Error generating HTML report:[/] {e}")
        sys.exit(1)

@app.command()
def serve(
    file: str = typer.Argument(..., help="Path to CSV dataset"),
    target: str = typer.Argument(..., help="Name of target column"),
    port: int = typer.Option(8501, "--port", "-p", help="Port to run Streamlit server on"),
):
    """
    Launch local Streamlit web application.
    """
    header()
    from kaizenstat.core import KaizenStat
    try:
        KaizenStat.serve(file, target, port=port)
    except Exception as e:
        console.print(f"[bold bright_red]✖ Error launching Streamlit dashboard:[/] {e}")
        sys.exit(1)

def main():
    app()

if __name__ == "__main__":
    main()
