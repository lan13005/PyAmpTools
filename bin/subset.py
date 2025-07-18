#!/usr/bin/env python3
import argparse
from rich.console import Console
import ROOT

def subset_with_rdf(input_file: str,
                    output_file: str,
                    fraction: float,
                    tree_name: str = "kin",
                    seed: int = None) -> None:
    """
    Use ROOT RDataFrame + implicit multithreading to
    write a random subset of events.

    Args:
      input_file:  Path to source .root file
      output_file: Path for new .root file
      fraction:    Fraction of events to keep (0.0 - 1.0)
      tree_name:   Name of the TTree inside the file
      seed:        Optional integer seed for reproducibility
    """
    console = Console()
    if seed is not None:
        ROOT.gRandom.SetSeed(seed)
        console.print(f"[green]Using seed: {seed}[/green]")

    # Enable ROOT’s implicit multithreading (one thread per core)
    ROOT.ROOT.EnableImplicitMT()

    console.print(f"[blue]Loading tree '{tree_name}' from {input_file}…[/blue]")
    df = ROOT.RDataFrame(tree_name, input_file)

    console.print(f"[blue]Applying random filter: keep ~{fraction:.1%} of events[/blue]")
    df_filtered = df.Filter(
        f"gRandom->Uniform() < {fraction}",
        f"Select ~{fraction:.1%} fraction"
    )

    console.print(f"[yellow]Writing filtered tree to {output_file}…[/yellow]")
    df_filtered.Snapshot(tree_name, output_file)
    console.print(f"[bold green]✓ Finished writing subset[/bold green]")


def main():
    parser = argparse.ArgumentParser(
        description="Randomly subset a ROOT tree via RDataFrame + MT"
    )
    parser.add_argument("input_file",  help="Path to input ROOT file")
    parser.add_argument("output_file", help="Path to output ROOT file")
    parser.add_argument("fraction",    type=float,
                        help="Fraction of events to keep (0.0–1.0)")
    parser.add_argument("--tree",   default="kin",
                        help="Name of the tree (default: kin)")
    parser.add_argument("--seed",   type=int,
                        help="Random seed (optional)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        ROOT.gErrorIgnoreLevel = ROOT.kINFO

    try:
        subset_with_rdf(
            input_file=args.input_file,
            output_file=args.output_file,
            fraction=args.fraction,
            tree_name=args.tree,
            seed=args.seed
        )
    except Exception as e:
        Console().print(f"[bold red]Error: {e}[/bold red]")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
