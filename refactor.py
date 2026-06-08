import re
import sys

def main():
    with open("kaizenstat/core.py", "r") as f:
        code = f.read()

    # Add rich imports at the top
    imports = """from typing import Optional, Dict, List, Union

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

def section(title):
    console.print(
        Panel.fit(
            f"[bold cyan]{title}[/bold cyan]",
            border_style="cyan"
        )
    )
"""
    code = code.replace("from typing import Optional, Dict, List, Union", imports)

    # 1. Update Audit
    # We replace from print("\n🔍 KAIZENSTAT AUDIT\n") down to KaizenStat._last_audit_findings = findings
    audit_new = """        section("KAIZENSTAT AUDIT")

        findings = {
            "shape": df.shape,
            "duplicates": int(df.duplicated().sum()),
            "missing_values": int(df.isna().sum().sum()),
        }

        table = Table(box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")

        table.add_row("Shape", f"{df.shape[0]} rows × {df.shape[1]} columns")
        table.add_row("Duplicates", str(findings['duplicates']))
        table.add_row("Missing Values", str(findings['missing_values']))

        num_df = df.select_dtypes(include=[np.number])
        inf_count = int(np.isinf(num_df.values).sum()) if not num_df.empty else 0
        findings["infinite_values"] = inf_count
        table.add_row("Infinite Values", str(inf_count))

        constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
        findings["constant_columns"] = constant_cols
        table.add_row("Constant Columns", str(constant_cols) if constant_cols else 'None')

        id_cols = _detect_id_columns(df)
        findings["id_columns"] = id_cols
        if id_cols:
            table.add_row("ID-like Columns", f"[yellow]⚠️ {id_cols}[/yellow]")

        dt_cols = _detect_datetime_columns(df)
        findings["datetime_columns"] = dt_cols
        if dt_cols:
            table.add_row("Datetime Columns", f"[yellow]⚠️ {dt_cols}[/yellow]")

        if target:
            target_missing = int(df[target].isna().sum())
            findings["target_missing"] = target_missing
            table.add_row("Target Missing", str(target_missing))

            task_type = "Classification" if _detect_task_type(df[target].dropna()) else "Regression"
            findings["task_type"] = task_type
            table.add_row("Detected Task", f"[bold green]{task_type}[/bold green]")

            if task_type == "Classification":
                imbalanced = _detect_imbalance(df[target].dropna())
                findings["imbalanced"] = imbalanced
                if imbalanced:
                    table.add_row("Class Imbalance", "[yellow]⚠️ Detected (majority > 65%)[/yellow]")

        console.print(table)
        KaizenStat._last_audit_findings = findings"""
    
    # We find the start and end of audit
    audit_start = code.find('print("\\n🔍 KAIZENSTAT AUDIT\\n")')
    audit_end = code.find('KaizenStat._last_audit_findings = findings') + len('KaizenStat._last_audit_findings = findings')
    code = code[:audit_start] + audit_new + code[audit_end:]


    # 2. Update Heal
    heal_new = """        table = Table(title="🩹 Heal Report", box=box.ROUNDED)
        table.add_column("Status", style="bold")
        table.add_column("Action", style="white")

        if dropped_cols:
            for col, reason in dropped_cols:
                table.add_row("[red]✗ Dropped[/red]", f"'{col}' ({reason})")
        for action in actions:
            table.add_row("[green]✓ Fixed[/green]", action)
        if not dropped_cols and not actions:
            table.add_row("[green]✓ Perfect[/green]", "Dataset was already clean")

        console.print(table)
        KaizenStat._last_dropped_cols = dropped_cols"""
    
    heal_start = code.find('print("\\n🩹 HEAL REPORT:")')
    heal_end = code.find('KaizenStat._last_dropped_cols = dropped_cols') + len('KaizenStat._last_dropped_cols = dropped_cols')
    code = code[:heal_start] + heal_new + code[heal_end:]


    # 3. Update Benchmark
    # We replace from print(f"\n⚡ Running on: {device.upper()}") down to the loop that prints results
    
    # First, let's just do the benchmark prints before loop
    code = code.replace('print(f"\\n⚡ Running on: {device.upper()}")', 'section(f"⚡ RUNNING ON: {device.upper()}")')
    code = code.replace('print(f"  Task: {task_str}")', 'console.print(f"  [bold]Task:[/] [green]{task_str}[/green]")')
    code = code.replace('print(f"  ✓ Encoded target labels: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")', 'console.print(f"  [green]✓ Encoded target labels:[/] {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")')
    code = code.replace('print("  ⚠️  Imbalanced classes → applying class_weight=\'balanced\'")', 'console.print("  [yellow]⚠️ Imbalanced classes → applying class_weight=\'balanced\'[/yellow]")')
    code = code.replace('print("  ℹ️  Categorical-heavy dataset → tree models preferred")', 'console.print("  [cyan]ℹ️ Categorical-heavy dataset → tree models preferred[/cyan]")')
    
    # Replace the print loop for benchmark
    # Find:
    '''        print("\\n🏆 MODEL LEADERBOARD (Accuracy)" if is_classification else "\\n🏆 MODEL LEADERBOARD (R2 Score)")
        for i, res in enumerate(results):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"  {medal} #{i+1} {res['Model']:<20} Score: {res['Score']:.4f} ± {res['Std']:.4f}  |  {res['Time(s)']:.3f}s")

        print(f"\\n🏆 BEST MODEL: {results[0]['Model']} (Score: {results[0]['Score']:.4f})")'''
    bench_print_str = """        table = Table(title=f"🏆 MODEL LEADERBOARD ({scoring.capitalize()})", box=box.ROUNDED)
        table.add_column("Rank", justify="center")
        table.add_column("Model", style="cyan")
        table.add_column("Score", style="bold green")
        table.add_column("Std Dev", style="dim")
        table.add_column("Time(s)", style="yellow")

        for i, res in enumerate(results):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
            table.add_row(
                medal,
                res['Model'],
                f"{res['Score']:.4f}",
                f"± {res['Std']:.4f}",
                f"{res['Time(s)']:.3f}s"
            )

        console.print(table)
        console.print(f"\\n[bold green]🏆 BEST MODEL:[/] {results[0]['Model']} (Score: {results[0]['Score']:.4f})")"""
    
    old_bench_print = 'print("\\n🏆 MODEL LEADERBOARD (Accuracy)" if is_classification else "\\n🏆 MODEL LEADERBOARD (R2 Score)")'
    bench_print_start = code.find(old_bench_print)
    bench_print_end = code.find('KaizenStat._last_benchmark_results = df_results')
    
    # Actually the loop looks like:
    # print("\n🏆 MODEL LEADERBOARD (Accuracy)" if is_classification else "\n🏆 MODEL LEADERBOARD (R2 Score)")
    # for i, res in enumerate(results):
    #     medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
    #     print(f"  {medal} #{i+1} {res['Model']:<20} Score: {res['Score']:.4f} ± {res['Std']:.4f}  |  {res['Time(s)']:.3f}s")
    #
    # print(f"\n🏆 BEST MODEL: {results[0]['Model']} (Score: {results[0]['Score']:.4f})")
    
    # We'll use regex to replace it
    bench_pattern = r'print\("\\n🏆 MODEL LEADERBOARD \([^\n]+.*?🏆 BEST MODEL:[^\n]+\)'
    code = re.sub(bench_pattern, bench_print_str, code, flags=re.DOTALL)

    with open("kaizenstat/core.py", "w") as f:
        f.write(code)

if __name__ == "__main__":
    main()
