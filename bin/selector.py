import argparse
from rich.console import Console
import ROOT

def filter_with_rdf(input_file: str,
                    output_file: str,
                    selection: str,
                    tree_name: str = "kin") -> None:
    """
    Use ROOT RDataFrame + implicit multithreading to apply a selection
    string (e.g. "branchName > 10") and write the filtered events.

    Args:
      input_file:  Path to source .root file
      output_file: Path for new .root file
      selection:   RDataFrame Filter expression (e.g. "pt > 0.5 && eta < 2.4")
      tree_name:   Name of the TTree inside the file
    """
    console = Console()

    if not selection or not selection.strip():
        raise ValueError("Selection string must be a non-empty expression")

    # Enable ROOT’s implicit multithreading (one thread per core)
    ROOT.ROOT.EnableImplicitMT()

    console.print(f"[blue]Loading tree '{tree_name}' from {input_file}…[/blue]")
    df = ROOT.RDataFrame(tree_name, input_file)

    # Pre-selection count
    n_in = int(df.Count().GetValue())
    console.print(f"[cyan]Events before selection:[/cyan] [bold]{n_in:,}[/bold]")

    # Apply user selection
    console.print(f"[blue]Applying selection:[/blue] [italic]{selection}[/italic]")
    df_filtered = df.Filter(selection, "selection")

    # Post-selection count
    n_out = int(df_filtered.Count().GetValue())
    n_lost = n_in - n_out
    pct_kept = (n_out / n_in * 100.0) if n_in > 0 else 0.0
    pct_lost = (n_lost / n_in * 100.0) if n_in > 0 else 0.0

    console.print(
        f"[cyan]Events after selection:[/cyan] [bold]{n_out:,}[/bold] "
        f"([green]{pct_kept:.2f}% kept[/green], [red]{pct_lost:.2f}% lost[/red])"
    )

    console.print(f"[yellow]Writing filtered tree to {output_file}…[/yellow]")
    df_filtered.Snapshot(tree_name, output_file)
    console.print(f"[bold green]✓ Finished writing filtered dataset[/bold green]")


def main():
    parser = argparse.ArgumentParser(
        description="Filter a ROOT tree via RDataFrame selection expression, i.e. 'pol_angle == 0'"
    )
    parser.add_argument("input_file",  help="Path to input ROOT file")
    parser.add_argument("output_file", help="Path to output ROOT file")
    parser.add_argument("selection",
                        help="Selection string, e.g. 'branchName > 10 && flag == 1'")
    parser.add_argument("--tree",   default="kin",
                        help="Name of the tree (default: kin)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        ROOT.gErrorIgnoreLevel = ROOT.kINFO

    try:
        filter_with_rdf(
            input_file=args.input_file,
            output_file=args.output_file,
            selection=args.selection,
            tree_name=args.tree,
        )
    except Exception as e:
        Console().print(f"[bold red]Error: {e}[/bold red]")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
