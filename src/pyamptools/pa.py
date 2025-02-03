#!/usr/bin/env python

import argparse
import os
import subprocess
import sys

import argcomplete

##################################################################
# pa = pyamptools...
# This script attempts to be a dispatch system that organizes the
# various scripts in the repository to be accessible under a single
# executable
##################################################################


def main():
    """Dispatch function that calls the appropriate script and passes remaining arguments to it"""

    # check if full path
    if "PYAMPTOOLS_HOME" not in os.environ:
        print("PYAMPTOOLS_HOME environment variable not set. Please try to set it again using set_environment.sh")
        sys.exit(1)
    else:
        if not os.path.isabs(os.environ["PYAMPTOOLS_HOME"]):
            print(f"PYAMPTOOLS_HOME environment variable is not an absolute path. PYAMPTOOLS_HOME={os.environ['PYAMPTOOLS_HOME']}")
            sys.exit(1)
        if not os.path.exists(os.environ["PYAMPTOOLS_HOME"]):
            print(f"PYAMPTOOLS_HOME environment variable does not exist. PYAMPTOOLS_HOME={os.environ['PYAMPTOOLS_HOME']}")
            sys.exit(1)

    PYAMPTOOLS_HOME = os.environ["PYAMPTOOLS_HOME"]

    # print(f"PYAMPTOOLS_HOME: {PYAMPTOOLS_HOME}")

    func_map = {
        # 'command' : (path, description)
        "fit": (f"{PYAMPTOOLS_HOME}/src/pyamptools/mle.py", "Perform a set of MLE fits given an amptools config file"),
        "fitfrac": (f"{PYAMPTOOLS_HOME}/src/pyamptools/extract_ff.py", "Extract fit fractions from a given amptools FitResults file"),
        "mcmc": (f"{PYAMPTOOLS_HOME}/src/pyamptools/mcmc.py", "Perform a MCMC fit given an amptools config file"),
        "gen_amp": (f"{PYAMPTOOLS_HOME}/bin/gen_amp.py", "Generate data for a given configuration file"),
        "gen_vec_ps": (f"{PYAMPTOOLS_HOME}/bin/gen_vec_ps.py", "Generate vector-pseduoscalar data for a given configuration file"),
        "nentries": (f"{PYAMPTOOLS_HOME}/bin/get_nentries.py", "Print number of entries in a list of ROOT files (* wildcard supported)"),
        "calc_ps": (f"{PYAMPTOOLS_HOME}/bin/calc_ps.py", "Calculate the phase space factor for IFT fits"),
        "dash_ift_cmp": (f"{PYAMPTOOLS_HOME}/bin/dash_ift_cmp.py", "Compare multiple IFT fits (intensity and phase) using dash package"),
        "dx_normint": (f"{PYAMPTOOLS_HOME}/bin/dx_normint.py", "Make diagnostic heatmaps for (norm)alizaton and (amp)litude integrals. Can tracks matrix elements over all mass bins"),
    }

    analysis_map = {
        # 'command' : (path, description)
        "run_cfgGen": (f"{PYAMPTOOLS_HOME}/bin/run_cfgGen.py", "Generate an AmpTools fit configuration file"),
        "run_divideData": (f"{PYAMPTOOLS_HOME}/bin/run_divideData.py", "Divide data into kinematic bins (separate folders)"),
        "run_processEvents": (f"{PYAMPTOOLS_HOME}/bin/run_processEvents.py", "Process binned datasets to dump AmpTools' ampvecs data structure and normalization integrals into pkl files"),
        "run_mle": (f"{PYAMPTOOLS_HOME}/bin/run_mle.py", "Run MLE fits over kinematic bins using AmpTools"),
        "run_ift": (f"{PYAMPTOOLS_HOME}/bin/run_ift.py", "Run IFT fit over kinematic bins. MLE fits must be run first!"),
        "run_momentPlotter": (f"{PYAMPTOOLS_HOME}/bin/run_momentPlotter.py", "After running IFT + MLE fits we can attempt to plot all the moments"),
        "run_resultDump": (f"{PYAMPTOOLS_HOME}/bin/run_resultDump.py", "Dump IFT and AmpTools results to csv files"),
    }

    availability_map = {**func_map, **analysis_map}

    choices = availability_map.keys()

    # Custom formatter class to improve help message formatting
    class HelpOnErrorParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write(f"Error: {message}\n")
            self.print_help()
            sys.exit(2)

        def format_help(self):
            help_message = super().format_help()
            command_help = "\nCommands:\n"
            for command, (path, description) in func_map.items():
                command_help += f"  * {command:25} {description}\n"
            command_help += "\n  ==== YAML based commands below (takes a single YAML file argument to configure setup) ====\n"
            for command, (path, description) in analysis_map.items():
                command_help += f"  * {command:25} {description}\n"
            return help_message + "\n" + command_help

        def format_files(self):
            file_help = "Command file locations:\n"
            for command, (path, description) in func_map.items():
                file_help += f"  * {command:25} {path}\n"
            file_help += "\n  ==== YAML based command files ====\n"
            for command, (path, description) in analysis_map.items():
                file_help += f"  * {command:25} {path}\n"
            return file_help

    parser = HelpOnErrorParser(description="Dispatch pyamptools commands. Select a command from the Commands section below. Remaning arguments will be passed to the selected command.")
    parser.add_argument("-f", "--files", action="store_true", help="show command file locations and exit ")
    parser.add_argument("command", nargs="?", choices=choices, help=argparse.SUPPRESS)
    parser.add_argument("command_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    # tab-autocomplete so you can type 'pa ' into terminal and tab for available functions
    #   help is implicit
    argcomplete.autocomplete(parser, exclude=["-h", "--help"])

    _args = parser.parse_args()

    if _args.files:
        print(parser.format_files())
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
    print("\n======================================================================================================")
    print(f'Running command: {" ".join(command)}')
    print("======================================================================================================\n")
    subprocess.run(command)


if __name__ == "__main__":
    main()
