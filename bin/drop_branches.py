#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List
import ROOT
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Prevent ROOT from hijacking your CLI args
ROOT.PyConfig.IgnoreCommandLineOptions = True

console = Console()

def parse_branch_indexes(index_strings: List[str], max_index: int) -> List[int]:
    """
    Parse branch index strings into a list of integer indexes.

    Supports individual indexes and ranges:
    - "1"      -> [1]
    - "3-5"    -> [3, 4, 5]
    - "1-"     -> [1, 2, ..., max_index]
    - "1 3-5 10-12" -> [1, 3, 4, 5, 10, 11, 12]

    Open-ended ranges use the discovered maximum branch index.
    """
    indexes = []
    for s in index_strings:
        s = s.strip()
        if '-' in s:
            # Support open-ended ranges like "1-" meaning to the end
            start_str, end_str = s.split('-', 1)
            try:
                start = int(start_str)
            except ValueError:
                raise ValueError(f"Invalid start index in range '{s}'")

            if end_str == "":
                end = max_index
            else:
                try:
                    end = int(end_str)
                except ValueError:
                    raise ValueError(f"Invalid end index in range '{s}'")

            if start > end:
                raise ValueError(f"Invalid range: {s} (start > end)")
            indexes.extend(range(start, end + 1))
        else:
            try:
                indexes.append(int(s))
            except ValueError:
                raise ValueError(f"Invalid index '{s}': must be an integer")
    return sorted(set(indexes))


def drop_branches_cli(input_file: str,
                      output_file: str,
                      branch_indexes: List[str],
                      tree_name: str = "kin") -> None:
    """
    Drop specified branches (by index) from a ROOT file and save to a new file,
    using PyROOT's CloneTree to preserve original structure.
    """
    input_path  = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    console.print(f"[blue]Opening {input_path}…[/blue]")

    # Open input file and tree
    fin = ROOT.TFile.Open(str(input_path), "READ")
    if not fin or fin.IsZombie():
        raise IOError(f"Failed to open input file: {input_path}")
    tin = fin.Get(tree_name)
    if not tin:
        raise ValueError(f"Tree '{tree_name}' not found; available: {list(fin.GetListOfKeys())}")

    # List all branch names
    all_branches = [b.GetName() for b in tin.GetListOfBranches()]
    console.print(f"[green]Found {len(all_branches)} branches in '{tree_name}'[/green]")

    # Parse 0-based indexes now that we know the maximum index
    max_idx = len(all_branches) - 1
    indexes_to_drop = parse_branch_indexes(branch_indexes, max_idx)
    console.print(f"[blue]Dropping branch indexes: {indexes_to_drop}[/blue]")

    # Validate indexes
    bad = [i for i in indexes_to_drop if i < 0 or i > max_idx]
    if bad:
        raise ValueError(f"Invalid branch indexes {bad} (must be 0–{max_idx})")

    # Map to names and disable them
    names_to_drop = [all_branches[i] for i in indexes_to_drop]
    for name in names_to_drop:
        tin.SetBranchStatus(name, 0)
    console.print(f"[blue]Disabled branches: {names_to_drop}[/blue]")

    # Prepare output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fout = ROOT.TFile.Open(str(output_path), "RECREATE")
    if not fout or fout.IsZombie():
        raise IOError(f"Failed to create output file: {output_path}")

    # Clone the tree with only the enabled branches
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as prog:
        task = prog.add_task("Cloning tree…", total=None)
        tout = tin.CloneTree(-1, "fast")  # -1 = all entries, "fast" = bypass buffers
        prog.update(task, description="✓ Clone complete")

    # Write and close
    fout.WriteTObject(tout, tree_name)
    fout.Close()
    fin.Close()

    console.print(f"[bold green]✓ Successfully wrote {output_path}[/bold green]")
    console.print(f"[green]Dropped {len(names_to_drop)} branches[/green]")
    console.print(f"[green]Kept    {len(all_branches) - len(names_to_drop)} branches[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Drop selected branches (by index) from a ROOT file using PyROOT."
    )
    parser.add_argument("input_file",  help="Path to input ROOT file")
    parser.add_argument("output_file", help="Path to output ROOT file")
    parser.add_argument("branches", nargs="+",
                        help="Branch indexes to drop (e.g. '1' '3-5' '10-12')")
    parser.add_argument("--tree", default="kin",
                        help="Name of the TTree (default: kin)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()
    try:
        drop_branches_cli(
            input_file=args.input_file,
            output_file=args.output_file,
            branch_indexes=args.branches,
            tree_name=args.tree
        )
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
