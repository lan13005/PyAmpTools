#!/usr/bin/env python

import argparse
import argcomplete
import subprocess
import os
import sys

##################################################################
# pa = pyamptools...
# This script attempts to be a dispatch system that organizes the
# various scripts in the repository to be accessible under a single
# executable
##################################################################


def main():
    """Dispatch function that calls the appropriate script and passes remaining arguments to it"""

    REPO_HOME = os.environ["REPO_HOME"]

    print(f"REPO_HOME: {REPO_HOME}")

    func_map = {
        # 'command' : (path, description)
        "fit": (f"{REPO_HOME}/src/pyamptools/mle.py", "Perform a set of MLE fits given an amptools config file"),
        "fitfrac": (f"{REPO_HOME}/src/pyamptools/extract_ff.py", "Extract fit fractions from a given amptools FitResults file"),
        "mcmc": (f"{REPO_HOME}/src/pyamptools/mcmc.py", "Perform a MCMC fit given an amptools config file"),
        "gen_amp": (f"{REPO_HOME}/bin/gen_amp.py", "Generate data for a given configuration file"),
        "gen_vec_ps": (f"{REPO_HOME}/bin/gen_vec_ps.py", "Generate vector-pseduoscalar data for a given configuration file"),
        "nentries": (f"{REPO_HOME}/bin/get_nentries.py", "Print number of entries in a list of ROOT files (* wildcard supported)"),
    }

    analysis_map = {
        # 'command' : (path, description)
        "run_cfgGen": (f"{REPO_HOME}/bin/run_cfgGen.py", "Generate configuration files"),
        "run_divideData": (f"{REPO_HOME}/bin/run_divideData.py", "Binned analysis: Divide data into mass bins"),
        "run_mle": (f"{REPO_HOME}/bin/run_mle.py", "Binned analysis: Run MLE fits over bins"),
        "run_ift": (f"{REPO_HOME}/bin/run_ift.py", "(IN DEVELOPMENT) Binned analysis: Run IFT fit over bins"),
        "run_iftsyst": (f"{REPO_HOME}/bin/run_systematics.py", "(IN DEVELOPMENT) Binned analysis: Run IFT systematic variations"),
        "run_submit": (f"{REPO_HOME}/bin/run_submit.py", "Binned analysis: Submit jobs to the batch system to perform complete binned analysis"),
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
            command_help += "  ==== YAML based commands below (takes a single YAML file argument to configure setup) ====\n"
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
    print("\n======================================================================================================")
    print(f'Running command: {" ".join(command)}')
    print("======================================================================================================\n")
    subprocess.run(command)


if __name__ == "__main__":
    main()
