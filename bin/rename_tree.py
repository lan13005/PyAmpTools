import argparse
from rich.console import Console
import ROOT


def rename_ttree(input_file: str, old_tree_name: str, new_tree_name: str, force: bool = False) -> None:
    """
    Rename a TTree inside an existing ROOT file.

    Args:
        input_file: Path to the ROOT file to modify.
        old_tree_name: Current name of the TTree to rename.
        new_tree_name: Desired new name for the TTree.
        force: If True, overwrite existing object with the new name if it exists.

    Raises:
        FileNotFoundError: If the input file cannot be opened.
        ValueError: If the old tree does not exist, names are invalid, or overwrite is unsafe.
        RuntimeError: If writing the renamed tree back to file fails.
    """
    console = Console()

    if not old_tree_name or not new_tree_name:
        raise ValueError("Old and new tree names must be non-empty strings")
    if old_tree_name == new_tree_name:
        raise ValueError("Old and new tree names are identical; nothing to do")

    console.print(f"[blue]Opening file[/blue] [bold]{input_file}[/bold] in UPDATE mode…")
    root_file = ROOT.TFile(input_file, "UPDATE")
    if not root_file or root_file.IsZombie():
        raise FileNotFoundError(f"Unable to open ROOT file: {input_file}")

    try:
        console.print(f"[blue]Fetching tree[/blue] [bold]{old_tree_name}[/bold]…")
        tree = root_file.Get(old_tree_name)
        if not tree:
            raise ValueError(f"TTree '{old_tree_name}' not found in '{input_file}'")

        # Check for potential overwrite of an existing object with the new name
        existing = root_file.Get(new_tree_name)
        if existing and not force:
            raise ValueError(
                f"An object named '{new_tree_name}' already exists in the file. "
                "Use --force to overwrite."
            )

        # Perform rename and write back
        console.print(f"[yellow]Renaming[/yellow] '{old_tree_name}' -> '{new_tree_name}'…")
        tree.SetName(new_tree_name)
        # Overwrite any existing key with same name if --force, else ROOT decides
        write_code = tree.Write("", ROOT.TObject.kOverwrite if force else 0)
        if write_code <= 0:
            raise RuntimeError("Failed to write renamed TTree back to the file")

        console.print(
            f"[bold green]✓ Renamed TTree '{old_tree_name}' to '{new_tree_name}' in '{input_file}'.[/bold green]"
        )
    finally:
        root_file.Close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rename a TTree inside a ROOT file. Example: \n"
            "  rename_tree.py input.root oldTree newTree"
        )
    )
    parser.add_argument("input_file", help="Path to input ROOT file")
    parser.add_argument("old_name", help="Current name of the TTree")
    parser.add_argument("new_name", help="New name for the TTree")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing object with new name if it already exists",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable ROOT verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        ROOT.gErrorIgnoreLevel = ROOT.kInfo

    try:
        rename_ttree(
            input_file=args.input_file,
            old_tree_name=args.old_name,
            new_tree_name=args.new_name,
            force=args.force,
        )
        return 0
    except Exception as exc:
        Console().print(f"[bold red]Error:[/bold red] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


