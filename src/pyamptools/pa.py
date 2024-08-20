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
        "dx_normint": (f"{PYAMPTOOLS_HOME}/bin/dx_normint.py", "Make diagnostic heatmaps for (norm)alizaton and (amp)litude integrals. Can tracks matrix elements over all mass bins"),
    }

    analysis_map = {
        # 'command' : (path, description)
        "run_cfgGen": (f"{PYAMPTOOLS_HOME}/bin/run_cfgGen.py", "Generate AmpTools fit configuration files"),
        "run_divideData": (f"{PYAMPTOOLS_HOME}/bin/run_divideData.py", "Binned analysis: Divide data into mass bins (separate folders)"),
        "run_processEvents": (f"{PYAMPTOOLS_HOME}/bin/processEvents.py", "Binned analysis: Process binned datasets and generate ampvecs and normints"),
        "run_mle": (f"{PYAMPTOOLS_HOME}/bin/run_mle.py", "Binned analysis: Run MLE fits over bins"),
        "run_ift": (f"{PYAMPTOOLS_HOME}/bin/run_ift.py", "(IN DEVELOPMENT) Binned analysis: Run IFT fit over bins"),
        "run_iftsyst": (f"{PYAMPTOOLS_HOME}/bin/run_systematics.py", "(IN DEVELOPMENT) Binned analysis: Run IFT systematic variations"),
        "run_submit": (f"{PYAMPTOOLS_HOME}/bin/run_submit.py", "Binned analysis: Submit jobs to the batch system to perform complete binned analysis"),
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
                command_help += f"  * {command:15} {description}\n"
            command_help += "\n  ==== YAML based commands below (takes a single YAML file argument to configure setup) ====\n"
            for command, (path, description) in analysis_map.items():
                command_help += f"  * {command:15} {description}\n"
            return help_message + "\n" + command_help

    parser = HelpOnErrorParser(description="Dispatch pyamptools commands. Select a command from the Commands section below. Remaning arguments will be passed to the selected command.")
    parser.add_argument("command", choices=choices, help=argparse.SUPPRESS)
    parser.add_argument("command_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    # tab-autocomplete so you can type 'pa ' into terminal and tab for available functions
    #   help is implicit
    argcomplete.autocomplete(parser, exclude=["-h", "--help"])

    _args = parser.parse_args()

    cmd = _args.command
    cmd_args = _args.command_args
    cmd_path = availability_map[cmd][0]

    assert cmd in availability_map.keys(), f"Command {cmd} not recognized. Must be one of: {availability_map.keys()}"

    # Call the script with additional arguments
    command = ["python"] + [cmd_path] + cmd_args
    # print("\n======================================================================================================")
    # print(f'Running command: {" ".join(command)}')
    # print("======================================================================================================\n")
    subprocess.run(command)


if __name__ == "__main__":
    main()
