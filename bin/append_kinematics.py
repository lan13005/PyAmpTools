#!/usr/bin/env python3
import argparse
from rich.console import Console
from pyamptools.utility.general import append_kinematics

console = Console()

def append_kinematics_cli(input_file: str,
                         output_file: str = None,
                         tree_name: str = "kin",
                         beam_angle: float = None,
                         overwrite: bool = False,
                         particles: list = None) -> None:
    """
    Append various derived kinematics to a ROOT file using the append_kinematics function
    
    Args:
        input_file: Path to source .root file
        output_file: Path for output .root file (if None, overwrites input file)
        tree_name: Name of the TTree inside the file
        beam_angle: Beam angle in degrees or branch name for beam angles
        overwrite: Force recalculation of derived kinematics even if they already exist
        particles: List of particle names in the order they appear in the final state. 
                  Must be one of: {RECOIL, X1, X2, X3}. For 4-particle final states, 
                  RECOIL and X3 will be combined into BARYRECOIL.
    """
    try:
        console.print(f"[blue]Processing kinematics for {input_file}...[/blue]")
        
        df, kin_quantities = append_kinematics(
            infile=input_file,
            output_location=output_file,
            treeName=tree_name,
            console=console,
            beam_angle=beam_angle,
            overwrite=overwrite,
            particles=particles
        )
        
        if df is not None:
            console.print(f"[bold green]✓ Successfully appended kinematics[/bold green]")
            console.print(f"[green]Calculated quantities: {list(kin_quantities.keys())}[/green]")
        else:
            console.print(f"[yellow]Kinematics already present, no changes made[/yellow]")
            
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Append various derived kinematics to a ROOT file. For list of quantities, see `append_kinematics` in `pyamptools/utility/general.py`"
    )
    parser.add_argument("input_file", help="Path to input ROOT file")
    parser.add_argument("-o", "--output", help="Path to output ROOT file (if not specified, overwrites input file)")
    parser.add_argument("--tree", default="kin", help="Name of the tree (default: kin)")
    parser.add_argument("--beam-angle", type=float, help="Beam angle in degrees or branch name for beam angles")
    parser.add_argument("--overwrite", action="store_true", help="Force recalculation of derived kinematics even if they already exist")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--particles", nargs="+", choices=["RECOIL", "X1", "X2", "X3"], 
                       help="List of particle names in the order they appear in the final state. Each must be either: {RECOIL, X1, X2, X3}")
    
    args = parser.parse_args()
    
    if args.particles is not None and len(args.particles) not in [3, 4]:
        console.print(f"[bold red]Error: --particles must be either 3 or 4 particles if not None[/bold red]")
        return 1

    try:
        from pyamptools import atiSetup
        import ROOT
        atiSetup.setup(globals(), use_fsroot=True)
        ROOT.ROOT.EnableImplicitMT()
        from pyamptools.utility.rdf_macros import loadMacros
        loadMacros()

        append_kinematics_cli(
            input_file=args.input_file,
            output_file=args.output,
            tree_name=args.tree,
            beam_angle=args.beam_angle,
            overwrite=args.overwrite,
            particles=args.particles
        )
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
