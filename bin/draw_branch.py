#!/usr/bin/env python3
"""
Script to draw histograms of branches from ROOT files using uproot.

This script allows users to create histograms from specific branches in ROOT files,
with support for subsampling to minimize memory usage and improve performance.
"""

import argparse
import uproot as up
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import track
import os
from typing import List, Optional
import mplhep as hep


def _normalize_selection_expression(selection: str) -> str:
    """
    Normalize a ROOT RDataFrame-style boolean expression to uproot/numexpr syntax.

    This converts logical operators to their element-wise counterparts and
    maps unary negation to bitwise not, which is what numexpr expects.

    Examples:
      - "a > 0 && b < 1" -> "a > 0 & b < 1"
      - "x == 1 || y != 2" -> "x == 1 | y != 2"
      - "!(flag)" -> "~(flag)"
      - "not good && (pt > 0.2 or q == 1)" -> "~ good & (pt > 0.2 | q == 1)"
    """
    import re

    expr = selection
    # Replace word-bound logical operators first to avoid partial replacements
    expr = re.sub(r"\band\b", "&", expr)
    expr = re.sub(r"\bor\b", "|", expr)
    expr = re.sub(r"\bnot\b", "~", expr)
    # Replace C/C++ style logical operators
    expr = expr.replace("&&", "&").replace("||", "|")
    # Replace unary ! with ~ when it appears before a variable or (
    expr = re.sub(r"!\s*(?=\w|\()", "~", expr)
    return expr


def draw_branch_histogram(
    file_names: List[str],
    branch_names: List[str],
    subsample_fraction: float = 1.0,
    bins: int = 100,
    output_dir: str = ".",
    show_plot: bool = False,
    selection: Optional[str] = None,
    weight_branch: Optional[str] = None,
    density: bool = False,
    log_y: bool = False,
) -> None:
    """
    Draw histograms for specified branches from a ROOT file.
    
    Args:
        file_names: List of paths to the ROOT files
        branch_names: List of branch names to histogram
        subsample_fraction: Fraction of events to load (0.0 to 1.0)
        bins: Number of bins for the histogram
        output_dir: Directory to save histogram plots
        show_plot: Whether to display the plot interactively
        selection: Optional RDataFrame-style selection string to filter events
        weight_branch: Optional branch name containing event weights
        density: If True, plot normalized densities instead of raw counts
        log_y: If True, use logarithmic scaling on the y-axis
        
    Raises:
        ValueError: If branch is not found or subsample_fraction is invalid
        FileNotFoundError: If the ROOT file doesn't exist
    """
    for file_name in file_names:
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File {file_name} not found")
    
    if not 0.0 < subsample_fraction <= 1.0:
        raise ValueError("Subsample fraction must be between 0.0 and 1.0")
    
    def check_branch_exists(tree, branch_name: str) -> None:
        """Check if a branch exists in the tree."""
        if branch_name not in tree.keys():
            raise ValueError(f"Branch {branch_name} not found in file {file_name}")
    
    console = Console()
    
    # Prepare containers for data per branch across files
    per_branch_data = {bn: [] for bn in branch_names}

    # Helper to truncate labels
    def _label_for_file(path: str) -> str:
        base = os.path.basename(path)
        return base if len(base) <= 40 else base[:18] + "…" + base[-18:]

    # First pass: load arrays per file and collect data per branch
    for file_name in file_names:
        with up.open(file_name) as f:
            nkeys = len(f.keys())
            if nkeys == 0:
                console.print(f"[red]Warning: No trees found in {file_name}[/red]")
                continue
            if nkeys > 1:
                console.print(f"[yellow]Warning: More than one tree in file, selecting first[/yellow]")
            tree = f[f.keys()[0]]
            total_entries = tree.num_entries
            entries_to_load = int(total_entries * subsample_fraction)
            console.print(f"[blue]{os.path.basename(file_name)}: Loading {entries_to_load:,} / {total_entries:,} ({subsample_fraction:.1%})[/blue]")

            # Existence checks
            for branch_name in branch_names:
                check_branch_exists(tree, branch_name)
            if weight_branch and weight_branch.strip():
                check_branch_exists(tree, weight_branch)

            # Selection
            cut_expression = None
            if selection and selection.strip():
                cut_expression = _normalize_selection_expression(selection)

            # Branch list to fetch
            fetch_branches = list(branch_names)
            if weight_branch and weight_branch not in fetch_branches:
                fetch_branches.append(weight_branch)

            arrays = tree.arrays(
                fetch_branches,
                cut=cut_expression,
                entry_stop=entries_to_load if subsample_fraction < 1.0 else None,
            )

            # Store data per branch
            for branch_name in branch_names:
                branch_data = arrays[branch_name]
                weights_data = arrays[weight_branch] if (weight_branch and weight_branch in arrays.fields) else None

                data = branch_data.to_numpy() if hasattr(branch_data, 'to_numpy') else np.array(branch_data)
                if weights_data is not None:
                    weights = weights_data.to_numpy() if hasattr(weights_data, 'to_numpy') else np.array(weights_data)
                else:
                    weights = None

                finite_mask = np.isfinite(data)
                if weights is not None:
                    finite_mask &= np.isfinite(weights)
                data = data[finite_mask]
                if weights is not None:
                    weights = weights[finite_mask]

                per_branch_data[branch_name].append({
                    'label': _label_for_file(file_name),
                    'data': data,
                    'weights': weights,
                })

    # Create subplots once per branch
    n_branches = len(branch_names)
    if n_branches == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        axes = [ax]
    else:
        n_cols = min(2, n_branches)
        n_rows = (n_branches + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.flatten()

    # For each branch, compute common edges and overlay per-file histograms
    for i, branch_name in enumerate(track(branch_names, description="Plotting overlays...")):
        ax = axes[i]
        datasets = per_branch_data.get(branch_name, [])
        if len(datasets) == 0:
            ax.text(0.5, 0.5, f'No data for {branch_name}', ha='center', va='center', transform=ax.transAxes)
            continue

        # Determine common bin edges across files
        global_min = min(np.min(d['data']) for d in datasets if len(d['data']) > 0)
        global_max = max(np.max(d['data']) for d in datasets if len(d['data']) > 0)
        if not np.isfinite(global_min) or not np.isfinite(global_max) or global_min == global_max:
            # Fallback to default edges
            global_min, global_max = 0.0, 1.0
        edges = np.linspace(global_min, global_max, bins + 1)
        bin_width = edges[1] - edges[0]

        # Plot each file's histogram on the same axes
        for d in datasets:
            if len(d['data']) == 0:
                continue
            counts, _ = np.histogram(
                d['data'], bins=edges, weights=d['weights'], density=density,
            )
            hep.histplot(counts, edges, ax=ax, label=d['label'])

        ax.set_xlabel(branch_name)
        if density:
            ax.set_ylabel(f'Density / {bin_width:.3f})')
        else:
            ax.set_ylabel(f'Intensity / {bin_width:.3f})')
        if log_y:
            ax.set_yscale('log')
        ax.grid(False)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for i in range(n_branches, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_dir, f"drawn_histograms.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    console.print(f"[green]Histogram saved to: {output_file}[/green]")

    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """Main function to handle command line arguments and execute the script."""
    parser = argparse.ArgumentParser(
        description="Draw histograms of branches from ROOT files using uproot (overlays multiple files)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Draw histogram of a single branch
  python draw_branch.py data.root -b momentum

  # Draw histograms of multiple branches with 50% subsampling
  python draw_branch.py data1.root data2.root -b momentum energy -f 0.5

  # Apply filters and save to specific directory
  python draw_branch.py data1.root data2.root -b momentum -f 0.1 -o plots/ -s "pt > 0.5 && eta < 2.4" -d

  # Show interactive plot
  python draw_branch.py data.root -b momentum --show
        """
    )
    
    parser.add_argument(
        "file_locations", 
        nargs="+", 
        help="ROOT file(s) to process (supports wildcards). Multiple files overlayed"
    )
    parser.add_argument(
        "-b", "--branches", 
        nargs="+", 
        required=True,
        help="Branch names to histogram"
    )
    parser.add_argument(
        "-f", "--fraction", 
        type=float, 
        default=1.0,
        help="Subsample fraction (0.0 to 1.0, default: 1.0)"
    )
    parser.add_argument(
        "--bins", 
        type=int, 
        default=100,
        help="Number of histogram bins (default: 100)"
    )
    parser.add_argument(
        "-o", "--output-dir", 
        default=".",
        help="Output directory for plots (default: current directory)"
    )
    parser.add_argument(
        "-d", "--density",
        action="store_true",
        help="Plot normalized densities (area=1) instead of raw intensities",
    )
    parser.add_argument(
        "-w", "--weight-branch",
        default="",
        help="Optional branch name to use as event weights",
    )
    parser.add_argument(
        "-s", "--selection",
        default="",
        help=(
            "RDataFrame-style selection expression, e.g. 'pt > 0.5 && eta < 2.4'. "
            "Operators '&&', '||', '!' as well as 'and', 'or', 'not' are supported."
        ),
    )
    parser.add_argument(
        "--show", 
        action="store_true",
        help="Show interactive plots"
    )
    parser.add_argument(
        "-log",
        action="store_true",
        help="Use logarithmic scale on the y-axis",
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("Subsample fraction must be between 0.0 and 1.0")
    
    selection = args.selection if isinstance(args.selection, str) else ""
    weight_branch = args.weight_branch if isinstance(args.weight_branch, str) else ""
    density = bool(args.density)
    log_y = bool(args.log)
    
    console = Console()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.verbose:
        console.print(f"[blue]Processing {len(args.file_locations)} ROOT file(s)...[/blue]")
        for f in args.file_locations:
            console.print(f"[blue]Processing file: {f}[/blue]")
        if selection.strip():
            console.print(f"[blue]Selection:[/blue] [italic]{selection}[/italic]")
        console.print(f"[blue]Branches to histogram: {args.branches}[/blue]")
        console.print(f"[blue]Subsample fraction: {args.fraction}[/blue]")
        if weight_branch.strip():
            console.print(f"[blue]Weight branch: {weight_branch}[/blue]")
        console.print(f"[blue]Output directory: {args.output_dir}[/blue]")
        console.print(f"[blue]Density: {density}[/blue]")
        console.print(f"[blue]Log Y-scale: {log_y}[/blue]")
    
    # Perform a single overlay plot across all files
    try:
        draw_branch_histogram(
            file_names=args.file_locations,
            branch_names=args.branches,
            subsample_fraction=args.fraction,
            bins=args.bins,
            output_dir=args.output_dir,
            show_plot=args.show,
            selection=selection,
            weight_branch=weight_branch or None,
            density=density,
            log_y=log_y,
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if args.verbose:
            import traceback
            console.print(f"[red]{traceback.format_exc()}[/red]")


if __name__ == "__main__":
    main()
