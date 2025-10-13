#!/usr/bin/env python

import argparse
import os
import subprocess
import sys

import argcomplete
from rich.console import Console
from rich.text import Text

##################################################################
# pa = pyamptools...
# This script attempts to be a dispatch system that organizes the
# various scripts in the repository to be accessible under a single
# executable
##################################################################

console = Console(width=120)

def main():
    """Dispatch function that calls the appropriate script and passes remaining arguments to it"""

    # check if full path
    if "PYAMPTOOLS_HOME" not in os.environ:
        console.print("PYAMPTOOLS_HOME environment variable not set. Please try to set it again using set_environment.sh", style="bold red")
        sys.exit(1)
    else:
        if not os.path.isabs(os.environ["PYAMPTOOLS_HOME"]):
            console.print(f"PYAMPTOOLS_HOME environment variable is not an absolute path. PYAMPTOOLS_HOME={os.environ['PYAMPTOOLS_HOME']}", style="bold red")
            sys.exit(1)
        if not os.path.exists(os.environ["PYAMPTOOLS_HOME"]):
            console.print(f"PYAMPTOOLS_HOME environment variable does not exist. PYAMPTOOLS_HOME={os.environ['PYAMPTOOLS_HOME']}", style="bold red")
            sys.exit(1)

    PYAMPTOOLS_HOME = os.environ["PYAMPTOOLS_HOME"]

    # Format descriptions with colored AmpTools tags
    def format_description(desc):
        if "[AmpTools]" in desc:
            return desc.replace("[AmpTools]", "[yellow](AmpTools)[/yellow]")
        return desc

    func_map = {
        # 'command' : (path, description)
        "nentries": (f"{PYAMPTOOLS_HOME}/bin/get_nentries.py", "Print entries in List[ROOT files] (* wildcard), flag for branch integration (i.e. Weight)"),
        "subset": (f"{PYAMPTOOLS_HOME}/bin/subset.py", "Subset a root file by randomly selecting a fraction of the events"),
        "select": (f"{PYAMPTOOLS_HOME}/bin/selector.py", "Filter a root file by a selection expression, RDataFrame.Filter() style"),
        "draw_branch": (f"{PYAMPTOOLS_HOME}/bin/draw_branch.py", "Draw a branch(es) from a root file potentially weighted"),
        "rename_tree": (f"{PYAMPTOOLS_HOME}/bin/rename_tree.py", "Rename a TTree inside a ROOT file"),
        "drop_branches": (f"{PYAMPTOOLS_HOME}/bin/drop_branches.py", "Drop branches from a root file"),
        "append_kinematics": (f"{PYAMPTOOLS_HOME}/bin/append_kinematics.py", "Append derived kinematics to root file, i.e. mMassX, to a root file"),
        "convert_fsroot": (f"{PYAMPTOOLS_HOME}/bin/convert_fsroot.py", "Convert a fsroot file to expected ROOT file format"),
        "fit": (f"{PYAMPTOOLS_HOME}/src/pyamptools/mle.py", "[AmpTools] Perform a set of MLE fits given an amptools config file"),
        "fitfrac": (f"{PYAMPTOOLS_HOME}/src/pyamptools/extract_ff.py", "[AmpTools] Extract fit fractions from a given amptools FitResults file"),
        "gen_amp": (f"{PYAMPTOOLS_HOME}/bin/gen_amp.py", "[AmpTools] Generate data for a given configuration file"),
        "gen_vec_ps": (f"{PYAMPTOOLS_HOME}/bin/gen_vec_ps.py", "[AmpTools] Generate vector-pseduoscalar data for a given configuration file"),
        "ift_pkl_summary": (f"{PYAMPTOOLS_HOME}/bin/ift_pkl_summary.py", "Summarize the contents of an IFT results pickle file"),
        # "dx_normint": (f"{PYAMPTOOLS_HOME}/bin/dx_normint.py", "Make diagnostic heatmaps for (norm)alizaton and (amp)litude integrals. Can tracks matrix elements over all mass bins"),
        "calc_ps": (f"{PYAMPTOOLS_HOME}/bin/calc_ps.py", "Calculate the phase space factor for IFT fits"),
    }

    analysis_map = {
        # 'command' : (path, description)
        "from_default": (f"{PYAMPTOOLS_HOME}/bin/from_default.py", "Copy default yaml to a given location"),
        "run_priorSim": (f"{PYAMPTOOLS_HOME}/bin/run_priorSim.py", "Draw sample from NIFTy prior, generate simulated data, and split into kinematic bins"),
        "run_cfgGen": (f"{PYAMPTOOLS_HOME}/bin/run_cfgGen.py", "[AmpTools] Generate an AmpTools fit configuration file"),
        "run_divideData": (f"{PYAMPTOOLS_HOME}/bin/run_divideData.py", "Divide data into kinematic bins (separate folders)"),
        "run_processEvents": (f"{PYAMPTOOLS_HOME}/bin/run_processEvents.py", "[AmpTools] Process bins: dump AmpVecs + NormInts to pkl files"),
        "run_fit": (f"{PYAMPTOOLS_HOME}/bin/run_fit.py", "[AmpTools] Run MLE fits over kinematic bins using AmpTools"),
        "run_mle": (f"{PYAMPTOOLS_HOME}/bin/run_mle.py", "Run MLE fits over kinematic bins using variety of optimizers (minuit, lbfgs, ...)"),
        "run_mcmc": (f"{PYAMPTOOLS_HOME}/bin/run_mcmc.py", "Run MCMC fits over kinematic bins using numpyro NUTS sampler"),
        "run_ift": (f"{PYAMPTOOLS_HOME}/bin/run_ift.py", "Run IFT fit over kinematic bins"),
        "run_momentInverter": (f"{PYAMPTOOLS_HOME}/bin/run_momentInverter.py", "Run moment inverter"),
        "run_resultMan": (f"{PYAMPTOOLS_HOME}/bin/run_resultMan.py", "Run result manager commands, make all default plots"),
        "dash_ift_cmp": (f"{PYAMPTOOLS_HOME}/bin/dash_ift_cmp.py", "Compare multiple IFT fits (intensity and phase) using dash package"),
    }

    availability_map = {**func_map, **analysis_map}

    choices = availability_map.keys()

    # Custom formatter class to improve help message formatting
    class HelpOnErrorParser(argparse.ArgumentParser):
        def error(self, message):
            console.print(f"Error: {message}", style="bold red")
            self.print_help()
            sys.exit(2)

        def print_help(self, file=None):
            help_text = self.format_help()
            console.print(help_text)

        def format_help(self):
            help_message = super().format_help()
            command_help = "\n[bold green]Commands:[/bold green]\n"
            for command, (path, description) in func_map.items():
                description = format_description(description)
                command_help += f"  * {command:25} {description}\n"
            command_help += "\n  [bold green]==== YAML based commands below (takes a YAML file argument to configure setup) ====[/bold green]\n"
            for command, (path, description) in analysis_map.items():
                description = format_description(description)
                command_help += f"  * {command:25} {description}\n"
            return help_message + "\n" + command_help

        def format_files(self):
            file_help = "[bold green]Command file locations:[/bold green]\n"
            for command, (path, description) in func_map.items():
                file_help += f"  * {command:25} {path}\n"
            file_help += "\n  [bold green]==== YAML based command files ====[/bold green]\n"
            for command, (path, description) in analysis_map.items():
                file_help += f"  * {command:25} {path}\n"
            return file_help

    parser = HelpOnErrorParser(description="Dispatch pyamptools commands. Select a command from the Commands section below. Remaining arguments will be passed to the selected command (i.e. -h for help)")
    parser.add_argument("-f", "--files", action="store_true", help="show command file locations and exit ")
    parser.add_argument("command", nargs="?", choices=choices, help=argparse.SUPPRESS)
    parser.add_argument("command_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    # tab-autocomplete so you can type 'pa ' into terminal and tab for available functions
    #   help is implicit
    argcomplete.autocomplete(parser, exclude=["-h", "--help"])

    _args = parser.parse_args()

    if _args.files:
        console.print(parser.format_files())
        sys.exit(0)

    if _args.command is None:
        parser.print_help()
        sys.exit(0)

    cmd = _args.command
    cmd_args = _args.command_args
    cmd_path = availability_map[cmd][0]

    assert cmd in availability_map.keys(), f"Command {cmd} not recognized. Must be one of: {availability_map.keys()}"

    # Call the script with additional arguments
    command = ["python"] + [cmd_path] + cmd_args
    console.rule()
    console.print(f'Running command: {" ".join(command)}', style="bold blue")
    console.rule()
    subprocess.run(command)


if __name__ == "__main__":
    main()
